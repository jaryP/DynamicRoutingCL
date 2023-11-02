import itertools
from copy import deepcopy
from typing import Sequence, Tuple, Any, Union, Optional, Callable, List

import numpy as np
import torch
from avalanche.models import MultiTaskModule
from torch import nn, Tensor
from torch.nn.modules.batchnorm import _NormBase
from torch.utils.data import DataLoader

from models.routing.layers import BlockRoutingLayer


class FactoryWrapper(nn.Module):
    def __init__(self, module: nn.Module, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._reset_parameters = len(list(module.parameters())) > 0
        self.to_route = self._reset_parameters and not isinstance(module, _NormBase)

        self.module = module

    def __call__(self, *args, **kwargs):
        new_module = deepcopy(self.module)

        if self._reset_parameters:
            for m in new_module.modules():
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()

        for m in new_module.children():
            if hasattr(m, 'reset_running_stats'):
                m.reset_running_stats()

        return new_module


class RoutingModel(MultiTaskModule):
    def __init__(self,
                 layers: List[nn.Module],
                 layers_block_n: Union[Sequence[int], int],
                 backbone_output_dim: int,
                 *,
                 head_topology: Optional[Callable] = None,
                 cumulative: bool = False,
                 freeze_past_tasks: bool = False,
                 freeze_future_logits: bool = True,
                 freeze_projectors: bool = False,
                 future_paths_to_sample: int = 5,
                 sample_wise_future_sampling: bool = False,
                 pre_process_module: nn.Module = None,
                 path_selection_strategy: str = 'random',
                 prediction_mode: str = 'task',
                 **kwargs):

        super().__init__()

        self.forced_future = 0

        self.internal_features = None
        self.current_features = None
        self.past_batch = None

        self.head_topology = head_topology
        self.backbone_output_dim = backbone_output_dim

        self.use_future = True
        self.cumulative = cumulative

        self.freeze_future_logits = freeze_future_logits
        self.freeze_past_tasks = freeze_past_tasks
        self.path_selection_strategy = path_selection_strategy
        self.prediction_mode = prediction_mode
        self.future_paths_to_sample = future_paths_to_sample
        self.sample_wise_future_sampling = sample_wise_future_sampling
        self.freeze_projectors = freeze_projectors

        assert path_selection_strategy in ['random', 'usage',
                                           'inverse_usage',
                                           'negative_usage',
                                           'gradient']

        assert prediction_mode in ['class', 'task']

        self.adapt = True

        self.layers = nn.ModuleList()
        self.heads = nn.ModuleList()

        self.pre_process = pre_process_module

        blocks_f = [FactoryWrapper(m) for m in layers]
        trainable_blocks = len([f for f in blocks_f if f.to_route])

        if not isinstance(layers_block_n, Sequence):
            layers_block_n = [layers_block_n] * trainable_blocks
        else:
            assert len(
                layers_block_n) == trainable_blocks, f'The number of blocks per each layer must be equal to the number of trainable layers ({trainable_blocks}, {len(layers_block_n)})'

        blocks_it = iter(layers_block_n)

        for block_factory in blocks_f:
            if block_factory.to_route:
                n_blocks = next(blocks_it)
                self.layers.append(BlockRoutingLayer(factory=block_factory,
                                                     n_blocks=n_blocks))
            else:
                self.layers.append(block_factory())

        self.classifiers = nn.ParameterDict()

        self.gates = nn.ModuleDict()
        self.translate = nn.ModuleDict()

        self.heads = nn.ModuleDict()
        self.centroids_scaler = nn.ParameterDict()

        paths = list(itertools.product(*[range(len(l.blocks))
                                         for l in self.layers
                                         if isinstance(l, BlockRoutingLayer)]))
        for i, p in enumerate(paths):
            self.heads[str(i)] = nn.Sequential(nn.Flatten(1),
                                               nn.Sequential(nn.Linear(
                                                   self.backbone_output_dim,
                                                   1)),
                                               )
            paths[i] = (p, i)

        self.available_paths = paths
        self.assigned_paths = {}
        self.n_classes_seen_so_far = 0

        if freeze_past_tasks or freeze_future_logits:
            for l in self.layers:
                if isinstance(l, BlockRoutingLayer):
                    l.freeze_blocks()

            for p in self.heads.parameters():
                p.requires_grad_(False)

    def eval_adaptation(self, experience):
        self.forced_future = 0

        v = len(experience.classes_seen_so_far) - self.n_classes_seen_so_far
        if v > 0:
            self.forced_future = v

    def count_parameters(self):
        used_blocks = dict()
        layers = [l for l in self.layers if isinstance(l, BlockRoutingLayer)]
        for c, (p, v) in self.assigned_paths.items():
            for i, b in enumerate(p):
                key = f'{i}_{b}'
                if key in used_blocks:
                    continue
                block = layers[i].blocks[str(b)]
                params = sum(p.numel() for p in block.parameters()
                             if p.requires_grad)
                used_blocks[key] = params

        return sum(used_blocks.values())

    # def train_adaptation(self, experience):
    #     if not self.adapt:
    #         return
    #     self.forced_future = 0
    #
    #     task_classes = len(experience.classes_in_this_experience)
    #     self.n_classes_seen_so_far += task_classes
    #     to_samples = task_classes if self.prediction_mode == 'class' else 1
    #
    #     if self.path_selection_strategy == 'random' or len(
    #             self.assigned_paths) == 0:
    #         selected_paths = np.random.choice(
    #             np.arange(len(self.available_paths)),
    #             to_samples,
    #             replace=False)
    #         paths = [self.available_paths[i] for i in selected_paths]
    #
    #     elif 'usage' in self.path_selection_strategy:
    #         probs = []
    #
    #         used_blocks = set()
    #         for c, (p, v) in self.assigned_paths.items():
    #             for i, b in enumerate(p):
    #                 used_blocks.add(f'{i}_{b}')
    #
    #         for p, v in self.available_paths:
    #             c = 0
    #             for i, b in enumerate(p):
    #                 s = f'{i}_{b}'
    #                 if s in used_blocks:
    #                     c += 1
    #
    #             c = c / len(p)
    #             probs.append(c)
    #
    #         probs = np.asarray(probs)
    #
    #         if self.path_selection_strategy == 'negative_usage':
    #             probs -= max(probs)
    #         elif self.path_selection_strategy == 'inverse_usage':
    #             probs[probs > 0] = 1 / probs[probs > 0]
    #
    #         probs = probs / sum(probs)
    #
    #         selected_paths = np.random.choice(
    #             np.arange(len(self.available_paths)),
    #             to_samples,
    #             replace=False,
    #             p=probs)
    #
    #         paths = [self.available_paths[i] for i in selected_paths]
    #
    #     elif self.path_selection_strategy == 'gradient':
    #
    #         d = DataLoader(experience.dataset, batch_size=128, shuffle=True)
    #         x, labels, _ = next(iter(d))
    #         x = x.to(next(self.parameters()).device)
    #
    #         if self.prediction_mode == 'class':
    #             xs = [x[labels == l] for l in torch.unique(labels)]
    #         else:
    #             xs = [x]
    #
    #         paths = []
    #         with torch.enable_grad():
    #             for x in xs:
    #                 past_paths = list(self.assigned_paths.values())
    #                 features, past_logits, _ = self.features(x,
    #                                                          paths_to_use=past_paths)
    #
    #                 past_max = past_logits.max(-1).values.detach()
    #                 best_val = (np.inf, None)
    #
    #                 for i in range(0, len(self.available_paths), 10):
    #                     pts = self.available_paths[i:i + 10]
    #
    #                     features, logits, _ = self.features(x, paths_to_use=pts)
    #
    #                     alphas = nn.Parameter(torch.ones((1, logits.shape[-1]),
    #                                                      device=next(
    #                                                          self.parameters()).device))
    #                     logits = logits * alphas
    #
    #                     diff = past_max[:, None] - logits
    #                     diff = torch.maximum(torch.zeros_like(diff), diff)
    #
    #                     grad = torch.autograd.grad(diff.mean(), alphas,
    #                                                retain_graph=False,
    #                                                create_graph=False)[0]
    #                     # grad = - grad
    #                     (mn, val) = [v.item() for v in grad.min(-1)]
    #                     val = pts[val]
    #                     if mn < best_val[0] and val not in paths:
    #                         best_val = (mn, val)
    #
    #                 paths.append(best_val[1])
    #     else:
    #         assert False
    #
    #     if self.freeze_past_tasks:
    #         for pt, v in self.assigned_paths.values():
    #
    #             for p in self.heads[str(v)].parameters():
    #                 p.requires_grad_(False)
    #
    #             layers = [l for l in self.layers if
    #                       isinstance(l, BlockRoutingLayer)]
    #
    #             for b, l in zip(pt, layers):
    #                 l.freeze_block(b)
    #
    #     z = experience.classes_in_this_experience \
    #         if self.prediction_mode == 'class' else [
    #         len(self.assigned_paths) + 1]
    #
    #     for c, p in zip(z, paths):
    #         self.available_paths.remove(p)
    #         self.assigned_paths[c] = p
    #
    #         if self.prediction_mode == 'task':
    #             l = nn.Linear(self.backbone_output_dim, task_classes)
    #             self.heads[str(p[1])] = l
    #
    #         if self.freeze_past_tasks or self.freeze_future_logits:
    #             layers = [l for l in self.layers if
    #                       isinstance(l, BlockRoutingLayer)]
    #
    #             for b, l in zip(p[0], layers):
    #                 l.freeze_block(b, False)
    #
    #             for p in self.heads[str(p[1])].parameters():
    #                 p.requires_grad_(True)
    #
    #     if self.freeze_projectors:
    #         for _, v in self.assigned_paths.values():
    #             for p in self.heads[str(v)].parameters():
    #                 p.requires_grad_(False)
    #
    #     print(self.assigned_paths)

    def train_adaptation(self, experience):
        if not self.adapt:
            return
        self.forced_future = 0
        device = next(self.parameters()).device

        task_classes = len(experience.classes_in_this_experience)
        self.n_classes_seen_so_far += task_classes
        to_samples = task_classes if self.prediction_mode == 'class' else 1

        head = nn.Linear(self.backbone_output_dim, task_classes).to(device)
        probs = []

        used_blocks = set()
        for c, (p, v) in self.assigned_paths.items():
            for i, b in enumerate(p):
                used_blocks.add(f'{i}_{b}')

        for p, v in self.available_paths:
            c = 0
            for i, b in enumerate(p):
                s = f'{i}_{b}'
                if s in used_blocks:
                    c += 1

            c = c / len(p)
            probs.append(c)

        probs = np.asarray(probs)
        mx = max(probs)
        indexes = np.argwhere(probs == mx).reshape(-1)

        x, y, _ = next(iter(DataLoader(experience.dataset, 256, shuffle=True)))
        if self.past_batch is not None:
            x = torch.cat((x, self.past_batch[0]))
            y = torch.cat((y, self.past_batch[1]))

        x, y = x.to(device), y.to(device)

        paths = [self.available_paths[i] for i in indexes]
        scores = [0] * len(paths)
        past_paths = list(self.assigned_paths.values())

        for i, p in enumerate(paths):
            f = self.features(x, paths_to_use=[p])[0]
            l = head(f).squeeze(1)

            if len(past_paths) > 0:
                pl = self.features(x, paths_to_use=past_paths)[1]
                l = torch.cat((pl, l), 1)

            loss = nn.functional.cross_entropy(l, y)
            scores[i] = loss.item()

        assigned_path = paths[np.argmin(scores)]
        self.available_paths.remove(assigned_path)
        self.assigned_paths[len(self.assigned_paths) + 1] = assigned_path
        self.heads[str(assigned_path[1])] = head

        if self.freeze_past_tasks or self.freeze_future_logits:
            layers = [l for l in self.layers if
                      isinstance(l, BlockRoutingLayer)]

            for b, l in zip(assigned_path[0], layers):
                l.freeze_block(b, False)

            for p in self.heads[str(assigned_path[1])].parameters():
                p.requires_grad_(True)

        if self.freeze_projectors:
            for _, v in self.assigned_paths.values():
                for p in self.heads[str(v)].parameters():
                    p.requires_grad_(False)

        print(self.assigned_paths)

        return
        if (probs == mx).sum() > 1:
            pass
        else:
            pass

        if self.path_selection_strategy == 'random' or len(
                self.assigned_paths) == 0:
            selected_paths = np.random.choice(
                np.arange(len(self.available_paths)),
                to_samples,
                replace=False)
            paths = [self.available_paths[i] for i in selected_paths]

        elif 'usage' in self.path_selection_strategy:
            probs = []

            used_blocks = set()
            for c, (p, v) in self.assigned_paths.items():
                for i, b in enumerate(p):
                    used_blocks.add(f'{i}_{b}')

            for p, v in self.available_paths:
                c = 0
                for i, b in enumerate(p):
                    s = f'{i}_{b}'
                    if s in used_blocks:
                        c += 1

                c = c / len(p)
                probs.append(c)

            probs = np.asarray(probs)

            if self.path_selection_strategy == 'negative_usage':
                probs -= max(probs)
            elif self.path_selection_strategy == 'inverse_usage':
                probs[probs > 0] = 1 / probs[probs > 0]

            probs = probs / sum(probs)

            selected_paths = np.random.choice(
                np.arange(len(self.available_paths)),
                to_samples,
                replace=False,
                p=probs)

            paths = [self.available_paths[i] for i in selected_paths]

        elif self.path_selection_strategy == 'gradient':

            d = DataLoader(experience.dataset, batch_size=128, shuffle=True)
            x, labels, _ = next(iter(d))
            x = x.to(next(self.parameters()).device)

            if self.prediction_mode == 'class':
                xs = [x[labels == l] for l in torch.unique(labels)]
            else:
                xs = [x]

            paths = []
            with torch.enable_grad():
                for x in xs:
                    past_paths = list(self.assigned_paths.values())
                    features, past_logits, _ = self.features(x,
                                                             paths_to_use=past_paths)

                    past_max = past_logits.max(-1).values.detach()
                    best_val = (np.inf, None)

                    for i in range(0, len(self.available_paths), 10):
                        pts = self.available_paths[i:i + 10]

                        features, logits, _ = self.features(x, paths_to_use=pts)

                        alphas = nn.Parameter(torch.ones((1, logits.shape[-1]),
                                                         device=next(
                                                             self.parameters()).device))
                        logits = logits * alphas

                        diff = past_max[:, None] - logits
                        diff = torch.maximum(torch.zeros_like(diff), diff)

                        grad = torch.autograd.grad(diff.mean(), alphas,
                                                   retain_graph=False,
                                                   create_graph=False)[0]
                        # grad = - grad
                        (mn, val) = [v.item() for v in grad.min(-1)]
                        val = pts[val]
                        if mn < best_val[0] and val not in paths:
                            best_val = (mn, val)

                    paths.append(best_val[1])
        else:
            assert False

        if self.freeze_past_tasks:
            for pt, v in self.assigned_paths.values():

                for p in self.heads[str(v)].parameters():
                    p.requires_grad_(False)

                layers = [l for l in self.layers if
                          isinstance(l, BlockRoutingLayer)]

                for b, l in zip(pt, layers):
                    l.freeze_block(b)

        z = experience.classes_in_this_experience \
            if self.prediction_mode == 'class' else [
            len(self.assigned_paths) + 1]

        for c, p in zip(z, paths):
            self.available_paths.remove(p)
            self.assigned_paths[c] = p

            if self.prediction_mode == 'task':
                l = nn.Linear(self.backbone_output_dim, task_classes)
                self.heads[str(p[1])] = l

            if self.freeze_past_tasks or self.freeze_future_logits:
                layers = [l for l in self.layers if
                          isinstance(l, BlockRoutingLayer)]

                for b, l in zip(p[0], layers):
                    l.freeze_block(b, False)

                for p in self.heads[str(p[1])].parameters():
                    p.requires_grad_(True)

        if self.freeze_projectors:
            for _, v in self.assigned_paths.values():
                for p in self.heads[str(v)].parameters():
                    p.requires_grad_(False)

        print(self.assigned_paths)

    def forward(self,
                x: torch.Tensor,
                task_labels=None,
                other_paths: list = None,
                **kwargs) \
            -> Tuple[
                Tensor, Union[Tensor, Any], Optional[None], Optional[None]]:

        base_paths = list(self.assigned_paths.values())
        random_paths = []

        random_features = None
        random_logits = None

        features = None
        logits = None

        # if self.sample_wise_future_sampling and self.use_future and not self.forced_future > 0:
        #
        #     features, logits, all_features = self.features(x, paths_to_use=base_paths)
        #
        #     if self.training:
        #         random_features = []
        #         random_logits = []
        #
        #         for _x in x:
        #             sampled_paths = np.random.choice(
        #                 np.arange(len(self.available_paths)),
        #                 self.future_paths_to_sample, replace=True)
        #
        #             random_paths = [self.available_paths[p] for p in sampled_paths]
        #
        #             f, _ = self.features(_x[None], paths_to_use=random_paths)
        #
        #             lg = []
        #
        #             for i, (_, v) in enumerate(random_paths):
        #                 l = self.heads[str(v)](f[:, i])
        #
        #                 if self.freeze_past_tasks and str(
        #                         v) in self.centroids_scaler:
        #                     l = l / self.centroids_scaler[str(v)]
        #                 lg.append(l)
        #
        #             lg = torch.cat(lg, -1)
        #             random_logits.append(lg)
        #             random_features.append(f)
        #
        #         random_logits = torch.cat(random_logits, 0)
        #         random_features = torch.cat(random_features, 0)

        # else:

        if self.training and self.use_future:
            sampled_paths = np.random.choice(
                np.arange(len(self.available_paths)),
                self.future_paths_to_sample, replace=True)
            random_paths = [self.available_paths[p] for p in sampled_paths]

        if self.forced_future > 0:
            sampled_paths = np.random.choice(
                np.arange(len(self.available_paths)),
                self.forced_future, replace=True)
            random_paths += [self.available_paths[p] for p in sampled_paths]

        self.current_random_paths = random_paths

        if len(base_paths) > 0:
            features, logits, _ = self.features(x, paths_to_use=base_paths)

        if len(random_paths) > 0:
            random_features, random_logits, _ = (
                self.features(x, paths_to_use=random_paths))

        if features is None:
            features, logits = random_features, random_logits
            random_features, random_logits = None, None
        elif self.forced_future > 0:
            features = torch.cat((features, random_features), 1)
            logits = torch.cat((logits, random_logits), 1)
            random_features, random_logits = None, None

        if other_paths is not None and len(other_paths) > 0:
            f, l, _ = self.features(x, paths_to_use=other_paths)
            features = torch.cat((features, f), 1)
            logits = torch.cat((logits, l), 1)

        return logits, features, random_logits, random_features

    def features(self, x, *, paths_to_use=None, **kwargs):
        if paths_to_use is not None:
            if isinstance(paths_to_use, Sequence):
                if any(isinstance(v, int) for v in paths_to_use):
                    assert False
            elif paths_to_use == 'all':
                paths_to_use = self.available_paths

            to_iter = paths_to_use
        else:
            to_iter = self.assigned_paths.values()

        feats = []

        paths = list(zip(*[p for p, _ in to_iter]))
        paths_iterable = iter(paths)

        if self.pre_process is not None:
            x = self.pre_process(x)

        _x = [x] * len(paths[0])

        for l in self.layers[:-1]:
            if isinstance(l, BlockRoutingLayer):
                _p = next(paths_iterable)
                _x, f = l(_x, _p)
                feats.append(f)
            else:
                _x = [l(a) for a in _x]

        ll = self.layers[-1]
        if isinstance(ll, BlockRoutingLayer):
            _p = next(paths_iterable)
            _x, f = ll(_x, _p)
            feats.append(f)
        else:
            f = [ll(a) for a in _x]

        features = torch.stack(f, 1)

        logits = []
        for i, (_, v) in enumerate(paths_to_use):
            l = self.heads[str(v)](features[:, i])

            if self.freeze_past_tasks and str(v) in self.centroids_scaler:
                l = l / self.centroids_scaler[str(v)]
            logits.append(l)

        logits = torch.cat(logits, 1)

        return features, logits, feats


class CondensedRoutingModel(MultiTaskModule):
    def __init__(self,
                 layers: List[nn.Module],
                 layers_block_n: Union[Sequence[int], int],
                 backbone_output_dim: int,
                 *,
                 head_topology: Optional[Callable] = None,
                 cumulative: bool = False,
                 freeze_past_tasks: bool = False,
                 freeze_future_logits: bool = True,
                 freeze_projectors: bool = False,
                 future_paths_to_sample: int = 5,
                 sample_wise_future_sampling: bool = False,
                 pre_process_module: nn.Module = None,
                 path_selection_strategy: str = 'random',
                 prediction_mode: str = 'task',
                 **kwargs):

        super().__init__()

        self.use_condensed_model = False
        self.forced_future = 0

        self.internal_features = None
        self.current_features = None
        self.head_topology = head_topology
        self.backbone_output_dim = backbone_output_dim

        self.use_future = True
        self.cumulative = cumulative

        self.freeze_future_logits = freeze_future_logits
        self.freeze_past_tasks = freeze_past_tasks
        self.path_selection_strategy = path_selection_strategy
        self.prediction_mode = prediction_mode
        self.future_paths_to_sample = future_paths_to_sample
        self.sample_wise_future_sampling = sample_wise_future_sampling
        self.freeze_projectors = freeze_projectors

        assert path_selection_strategy in ['random', 'usage',
                                           'inverse_usage',
                                           'negative_usage',
                                           'gradient']

        assert prediction_mode in ['class', 'task']

        self.adapt = True

        self.layers = nn.ModuleList()
        self.heads = nn.ModuleList()
        self.condensed_model = nn.Sequential()

        self.pre_process = pre_process_module

        blocks_f = [FactoryWrapper(m) for m in layers]
        trainable_blocks = len([f for f in blocks_f if f.to_route])

        if not isinstance(layers_block_n, Sequence):
            layers_block_n = [layers_block_n] * trainable_blocks
        else:
            assert len(
                layers_block_n) == trainable_blocks, f'The number of blocks per each layer must be equal to the number of trainable layers ({trainable_blocks}, {len(layers_block_n)})'

        blocks_it = iter(layers_block_n)

        for block_factory in blocks_f:
            if block_factory.to_route:
                n_blocks = next(blocks_it)
                self.layers.append(BlockRoutingLayer(factory=block_factory,
                                                     n_blocks=n_blocks))
                self.condensed_model.append(block_factory())
            else:
                self.layers.append(block_factory())
                self.condensed_model.append(block_factory())

        self.condensed_model.append(nn.Linear(self.backbone_output_dim, 1))

        self.classifiers = nn.ParameterDict()

        self.gates = nn.ModuleDict()
        self.translate = nn.ModuleDict()

        self.heads = nn.ModuleDict()
        self.centroids_scaler = nn.ParameterDict()

        paths = list(itertools.product(*[range(len(l.blocks))
                                         for l in self.layers
                                         if isinstance(l, BlockRoutingLayer)]))
        for i, p in enumerate(paths):
            self.heads[str(i)] = nn.Sequential(nn.Flatten(1),
                                               nn.Sequential(nn.Linear(
                                                   self.backbone_output_dim,
                                                   1)),
                                               )
            paths[i] = (p, i)

        self.available_paths = paths
        self.assigned_paths = {}
        self.n_classes_seen_so_far = 0

        if freeze_past_tasks or freeze_future_logits:
            for l in self.layers:
                if isinstance(l, BlockRoutingLayer):
                    l.freeze_blocks()

            for p in self.heads.parameters():
                p.requires_grad_(False)

    def eval_adaptation(self, experience):
        self.forced_future = 0

        v = len(experience.classes_seen_so_far) - self.n_classes_seen_so_far
        if v > 0:
            self.forced_future = v

    def count_parameters(self):
        used_blocks = dict()
        layers = [l for l in self.layers if isinstance(l, BlockRoutingLayer)]
        for c, (p, v) in self.assigned_paths.items():
            for i, b in enumerate(p):
                key = f'{i}_{b}'
                if key in used_blocks:
                    continue
                block = layers[i].blocks[str(b)]
                params = sum(p.numel() for p in block.parameters()
                             if p.requires_grad)
                used_blocks[key] = params

        return sum(used_blocks.values())

    def reallocate_paths(self, paths):
        assert all(p not in self.available_paths for p in paths)

        self.available_paths.extend(paths.values())
        self.assigned_paths = {}

        for pt, v in paths.values():

            layer = nn.Sequential(nn.Flatten(1),
                          nn.Sequential(nn.Linear(
                              self.backbone_output_dim,
                              1)),
                          )

            for p in layer.parameters():
                p.requires_grad_(False)

            self.heads[str(v)] = layer

            layers = [l for l in self.layers if
                      isinstance(l, BlockRoutingLayer)]

            for b, l in zip(pt, layers):
                l.freeze_block(b)

    def train_adaptation(self, experience):
        if not self.adapt:
            return
        self.forced_future = 0

        # if self.n_classes_seen_so_far > 0:
        #     self.use_condensed_model = True

        task_classes = len(experience.classes_in_this_experience)
        self.n_classes_seen_so_far += task_classes
        to_samples = task_classes if self.prediction_mode == 'class' else 1

        if self.path_selection_strategy == 'random' or len(
                self.assigned_paths) == 0:
            selected_paths = np.random.choice(
                np.arange(len(self.available_paths)),
                to_samples,
                replace=False)
            paths = [self.available_paths[i] for i in selected_paths]

        elif 'usage' in self.path_selection_strategy:
            probs = []

            used_blocks = set()
            for c, (p, v) in self.assigned_paths.items():
                for i, b in enumerate(p):
                    used_blocks.add(f'{i}_{b}')

            for p, v in self.available_paths:
                c = 0
                for i, b in enumerate(p):
                    s = f'{i}_{b}'
                    if s in used_blocks:
                        c += 1

                c = c / len(p)
                probs.append(c)

            probs = np.asarray(probs)

            if self.path_selection_strategy == 'negative_usage':
                probs -= max(probs)
            elif self.path_selection_strategy == 'inverse_usage':
                probs[probs > 0] = 1 / probs[probs > 0]

            probs = probs / sum(probs)

            selected_paths = np.random.choice(
                np.arange(len(self.available_paths)),
                to_samples,
                replace=False,
                p=probs)

            paths = [self.available_paths[i] for i in selected_paths]

        elif self.path_selection_strategy == 'gradient':

            d = DataLoader(experience.dataset, batch_size=128, shuffle=True)
            x, labels, _ = next(iter(d))
            x = x.to(next(self.parameters()).device)

            if self.prediction_mode == 'class':
                xs = [x[labels == l] for l in torch.unique(labels)]
            else:
                xs = [x]

            paths = []
            with torch.enable_grad():
                for x in xs:
                    past_paths = list(self.assigned_paths.values())
                    features, past_logits, _ = self.features(x,
                                                             paths_to_use=past_paths)

                    past_max = past_logits.max(-1).values.detach()
                    best_val = (np.inf, None)

                    for i in range(0, len(self.available_paths), 10):
                        pts = self.available_paths[i:i + 10]

                        features, logits, _ = self.features(x, paths_to_use=pts)

                        alphas = nn.Parameter(torch.ones((1, logits.shape[-1]),
                                                         device=next(
                                                             self.parameters()).device))
                        logits = logits * alphas

                        diff = past_max[:, None] - logits
                        diff = torch.maximum(torch.zeros_like(diff), diff)

                        grad = torch.autograd.grad(diff.mean(), alphas,
                                                   retain_graph=False,
                                                   create_graph=False)[0]
                        # grad = - grad
                        (mn, val) = [v.item() for v in grad.min(-1)]
                        val = pts[val]
                        if mn < best_val[0] and val not in paths:
                            best_val = (mn, val)

                    paths.append(best_val[1])
        else:
            assert False

        if self.freeze_past_tasks:
            for pt, v in self.assigned_paths.values():

                for p in self.heads[str(v)].parameters():
                    p.requires_grad_(False)

                layers = [l for l in self.layers if
                          isinstance(l, BlockRoutingLayer)]

                for b, l in zip(pt, layers):
                    l.freeze_block(b)

        z = experience.classes_in_this_experience \
            if self.prediction_mode == 'class' else [
            len(self.assigned_paths) + 1]

        for c, p in zip(z, paths):
            self.available_paths.remove(p)
            self.assigned_paths[c] = p

            if self.prediction_mode == 'task':
                l = nn.Linear(self.backbone_output_dim, task_classes)
                self.heads[str(p[1])] = l

            if self.freeze_past_tasks or self.freeze_future_logits:
                layers = [l for l in self.layers if
                          isinstance(l, BlockRoutingLayer)]

                for b, l in zip(p[0], layers):
                    l.freeze_block(b, False)

                for p in self.heads[str(p[1])].parameters():
                    p.requires_grad_(True)

        if self.freeze_projectors:
            for _, v in self.assigned_paths.values():
                for p in self.heads[str(v)].parameters():
                    p.requires_grad_(False)

        print(self.assigned_paths)

    def forward(self,
                x: torch.Tensor,
                task_labels=None,
                other_paths: list = None,
                **kwargs) \
            -> Tuple[
                Tensor, Union[Tensor, Any], Optional[None], Optional[None]]:

        # if self.use_condensed_model and not self.training:
        #     base_paths = []
        # else:
        base_paths = list(self.assigned_paths.values())

        random_paths = []

        random_features = None
        random_logits = None

        features = None
        logits = None

        # if self.sample_wise_future_sampling and self.use_future and not self.forced_future > 0:
        #
        #     features, logits, all_features = self.features(x, paths_to_use=base_paths)
        #
        #     if self.training:
        #         random_features = []
        #         random_logits = []
        #
        #         for _x in x:
        #             sampled_paths = np.random.choice(
        #                 np.arange(len(self.available_paths)),
        #                 self.future_paths_to_sample, replace=True)
        #
        #             random_paths = [self.available_paths[p] for p in sampled_paths]
        #
        #             f, _ = self.features(_x[None], paths_to_use=random_paths)
        #
        #             lg = []
        #
        #             for i, (_, v) in enumerate(random_paths):
        #                 l = self.heads[str(v)](f[:, i])
        #
        #                 if self.freeze_past_tasks and str(
        #                         v) in self.centroids_scaler:
        #                     l = l / self.centroids_scaler[str(v)]
        #                 lg.append(l)
        #
        #             lg = torch.cat(lg, -1)
        #             random_logits.append(lg)
        #             random_features.append(f)
        #
        #         random_logits = torch.cat(random_logits, 0)
        #         random_features = torch.cat(random_features, 0)

        # else:

        if self.training and self.use_future:
            sampled_paths = np.random.choice(
                np.arange(len(self.available_paths)),
                self.future_paths_to_sample, replace=True)
            random_paths = [self.available_paths[p] for p in sampled_paths]

        if self.forced_future > 0:
            sampled_paths = np.random.choice(
                np.arange(len(self.available_paths)),
                self.forced_future, replace=True)
            random_paths += [self.available_paths[p] for p in sampled_paths]

        self.current_random_paths = random_paths

        if len(base_paths) > 0:
            features, logits, _ = self.features(x, paths_to_use=base_paths)

        if len(random_paths) > 0:
            random_features, random_logits, _ = (
                self.features(x, paths_to_use=random_paths))

        if features is None:
            features, logits = random_features, random_logits
            random_features, random_logits = None, None
        elif self.forced_future > 0:
            features = torch.cat((features, random_features), 1)
            logits = torch.cat((logits, random_logits), 1)
            random_features, random_logits = None, None

        if other_paths is not None and len(other_paths) > 0:
            f, l, _ = self.features(x, paths_to_use=other_paths)
            features = torch.cat((features, f), 1)
            logits = torch.cat((logits, l), 1)

        if self.use_condensed_model:
            clogits = self.condensed_model(x)
            if logits is not None:
                logits = torch.cat((clogits, logits), -1)
            else:
                logits = clogits

        return logits, features, random_logits, random_features

    def features(self, x, *, paths_to_use=None, **kwargs):
        if paths_to_use is not None:
            if isinstance(paths_to_use, Sequence):
                if any(isinstance(v, int) for v in paths_to_use):
                    assert False
            elif paths_to_use == 'all':
                paths_to_use = self.available_paths

            to_iter = paths_to_use
        else:
            to_iter = self.assigned_paths.values()

        feats = []

        paths = list(zip(*[p for p, _ in to_iter]))
        paths_iterable = iter(paths)

        if self.pre_process is not None:
            x = self.pre_process(x)

        _x = [x] * len(paths[0])

        for l in self.layers[:-1]:
            if isinstance(l, BlockRoutingLayer):
                _p = next(paths_iterable)
                _x, f = l(_x, _p)
                feats.append(f)
            else:
                _x = [l(a) for a in _x]

        ll = self.layers[-1]
        if isinstance(ll, BlockRoutingLayer):
            _p = next(paths_iterable)
            _x, f = ll(_x, _p)
            feats.append(f)
        else:
            f = [ll(a) for a in _x]

        features = torch.stack(f, 1)

        logits = []
        for i, (_, v) in enumerate(paths_to_use):
            l = self.heads[str(v)](features[:, i])

            if self.freeze_past_tasks and str(v) in self.centroids_scaler:
                l = l / self.centroids_scaler[str(v)]
            logits.append(l)

        logits = torch.cat(logits, 1)

        return features, logits, feats

