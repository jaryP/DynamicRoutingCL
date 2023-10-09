import itertools
from typing import Sequence, Tuple, Any, Union, Optional

import numpy as np
import torch
from avalanche.models import MultiTaskModule
from torch import nn, Tensor
from torch.utils.data import DataLoader

from layers import BlockRoutingLayer


class RoutingModel(MultiTaskModule):
    def __init__(self,
                 model_dimension,
                 n_blocks_in_layer,
                 block_type,
                 cumulative=False,
                 freeze_past_tasks=False,
                 freeze_future_logits=True,
                 freeze_projectors=False,
                 future_paths_to_sample=5,
                 sample_wise_future_sampling=False,
                 path_selection_strategy='random',
                 prediction_mode='task',
                 **kwargs):

        super().__init__()

        assert model_dimension in ['tiny', 'small']
        assert block_type in ['conv', 'small']

        self.forced_future = 0

        self.internal_features = None
        self.current_features = None
        self.centroids = None

        self.use_future = True
        self.cumulative = cumulative

        self.freeze_future_logits = freeze_future_logits
        self.freeze_past_tasks = freeze_past_tasks
        self.path_selection_strategy = path_selection_strategy
        self.prediction_mode = prediction_mode
        self.future_paths_to_sample = future_paths_to_sample
        self.sample_wise_future_sampling = sample_wise_future_sampling
        self.freeze_projectors = freeze_projectors

        assert path_selection_strategy in ['random', 'usage', 'gradient']
        # assert path_selection_strategy in ['random', 'usage']
        assert prediction_mode in ['class', 'task']

        self.distance = 'cosine'
        self.adapt = True

        self.layers = nn.ModuleList()

        if model_dimension == 'tiny':
            self.layers.append(BlockRoutingLayer(3, 32, n_blocks=n_blocks_in_layer, block_type=block_type))
            self.layers.append(BlockRoutingLayer(32, 64, n_blocks=n_blocks_in_layer, block_type=block_type))
            self.layers.append(BlockRoutingLayer(64, 128, project_dim=128, n_blocks=n_blocks_in_layer, block_type=block_type))

        if cumulative:
            self.in_features = 32 + 64 + 128
        else:
            self.in_features = 128

        self.classifiers = nn.ParameterDict()

        self.gates = nn.ModuleDict()
        self.translate = nn.ModuleDict()

        self.centroids = nn.ModuleDict()
        self.centroids_scaler = nn.ParameterDict()

        paths = list(itertools.product(*[range(len(l.blocks))
                                         for l in self.layers]))
        for i, p in enumerate(paths):
            self.centroids[str(i)] = nn.Sequential(nn.Flatten(1),
                                                    # nn.ReLU(),
                                                    # CosineLinearLayer(self.in_features),
                                                    # ConcatLinearLayer(self.in_features, 100)
                                                    nn.Sequential(nn.ReLU(),
                                                                  nn.Linear(
                                                                      self.in_features,
                                                                      1)),
                                                    # nn.Sigmoid()
                                                    )
            paths[i] = (p, i)

        # while len(paths) < 100:
        #     b = [np.random.randint(0, l) for l in layers_blocks]
        #     if b not in paths:
        #         ln = len(paths)
        #         v = (b, ln)
        #         self.centroids[str(ln)] = nn.Sequential(nn.Flatten(1),
        #                                                 # nn.ReLU(),
        #                                                 # CosineLinearLayer(self.in_features),
        #                                                 # ConcatLinearLayer(self.in_features, 100)
        #                                                 nn.Sequential(nn.ReLU(), nn.Linear(self.in_features, 1)),
        #                                                 # nn.Sigmoid()
        #                                                 )
        #         paths.append(v)

        self.available_paths = paths
        self.assigned_paths = {}
        self.n_classes_seen_so_far = 0

        if freeze_past_tasks or freeze_future_logits:
            for l in self.layers:
                l.freeze_blocks()

            for p in self.centroids.parameters():
                p.requires_grad_(False)

    def eval_adaptation(self, experience):
        self.forced_future = 0

        v = len(experience.classes_seen_so_far) - self.n_classes_seen_so_far
        if v > 0:
            self.forced_future = v

    def train_adaptation(self, experience):
        if not self.adapt:
            return
        self.forced_future = 0

        task_classes = len(experience.classes_in_this_experience)
        self.n_classes_seen_so_far += task_classes
        to_samples = task_classes if self.prediction_mode == 'class' else 1

        if self.path_selection_strategy == 'random' or len(self.assigned_paths) == 0:
            selected_paths = np.random.choice(np.arange(len(self.available_paths)),
                                              to_samples,
                                              replace=False)
            paths = [self.available_paths[i] for i in selected_paths]

        elif self.path_selection_strategy == 'usage':
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
            probs = probs / sum(probs)
            selected_paths = np.random.choice(np.arange(len(self.available_paths)),
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
                    features, past_logits, _ = self.features(x, paths_to_use=past_paths)

                    past_max = past_logits.max(-1).values.detach()
                    best_val = (np.inf, None)

                    for i in range(0, len(self.available_paths), 10):
                        pts = self.available_paths[i:i+10]

                        features, logits, _ = self.features(x, paths_to_use=pts)

                        alphas = nn.Parameter(torch.ones((1, logits.shape[-1]),
                                                         device=next(self.parameters()).device))
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

                for p in self.centroids[str(v)].parameters():
                    p.requires_grad_(False)

                for b, l in zip(pt, self.layers):
                    l.freeze_block(b)

                # self.centroids_scaler[str(v)] = nn.Parameter(torch.tensor([1.0]))

            # print(self.centroids[str(v)])

        # for b, l in zip(zip(*[p[0] for p in paths]), self.layers):
        #     l.activate_blocks(b)

        z = experience.classes_in_this_experience \
            if self.prediction_mode == 'class' else experience.task_labels

        for c, p in zip(z, paths):
            self.available_paths.remove(p)
            self.assigned_paths[c] = p

            if self.prediction_mode == 'task':
                l = nn.Linear(self.in_features, task_classes)
                self.centroids[str(p[1])] = l

            if self.freeze_past_tasks or self.freeze_future_logits:
                for b, l in zip(p[0], self.layers):
                    l.freeze_block(b, False)

                for p in self.centroids[str(p[1])].parameters():
                    p.requires_grad_(True)

        if self.freeze_projectors:
            for _, v in self.assigned_paths.values():
                for p in self.centroids[str(v)].parameters():
                    p.requires_grad_(False)

        print(self.assigned_paths)

    def forward(self,
                x: torch.Tensor,
                task_labels = None,
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

        if self.sample_wise_future_sampling and self.use_future and not self.forced_future > 0:

            features, logits, all_features = self.features(x, paths_to_use=base_paths)

            if self.training:
                random_features = []
                random_logits = []

                for _x in x:
                    sampled_paths = np.random.choice(
                        np.arange(len(self.available_paths)),
                        self.future_paths_to_sample, replace=True)

                    random_paths = [self.available_paths[p] for p in sampled_paths]

                    f, _ = self.features(_x[None], paths_to_use=random_paths)

                    lg = []

                    for i, (_, v) in enumerate(random_paths):
                        l = self.centroids[str(v)](f[:, i])

                        if self.freeze_past_tasks and str(
                                v) in self.centroids_scaler:
                            l = l / self.centroids_scaler[str(v)]
                        lg.append(l)

                    lg = torch.cat(lg, -1)
                    random_logits.append(lg)
                    random_features.append(f)

                random_logits = torch.cat(random_logits, 0)
                random_features = torch.cat(random_features, 0)

        else:
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
            # all_paths = base_paths + random_paths

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
            #     paths_to_use = list(paths_to_use)
            # elif (isinstance(paths_to_use, list)
            #       and any(isinstance(v, int) for v in paths_to_use)):
            #
            #     assert False
            elif paths_to_use == 'all':
                paths_to_use = self.available_paths

            to_iter = paths_to_use
        else:
            to_iter = self.assigned_paths.values()

        feats = []

        paths = list(zip(*[p for p, _ in to_iter]))
        paths_iterable = iter(paths)
        _x = [x] * len(paths[0])

        for l in self.layers[:-1]:
            _p = next(paths_iterable)
            _x, f = l(_x, _p).relu()
            # _x = _x.relu()
            feats.append(f)

        _x, f = self.layers[-1](_x, next(paths_iterable))
        if self.cumulative:
           feats.append(f)
        else:
            feats.append(_x)

        if self.cumulative:
            feats = [torch.cat(l, -1 )for l in list(zip(*feats))]
            features = torch.stack(feats, 1)
        else:
            features = torch.stack(f, 1)
            # logits = torch.stack(f, 1).relu()

        logits = []
        for i, (_, v) in enumerate(paths_to_use):
            l = self.centroids[str(v)](features[:, i])

            if self.freeze_past_tasks and str(v) in self.centroids_scaler:
                l = l / self.centroids_scaler[str(v)]
            logits.append(l)

        logits = torch.cat(logits, 1)

        return features, logits, feats
