from builtins import enumerate
from copy import deepcopy
from itertools import chain
from typing import Optional, Sequence, Iterable, Union
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import distinctipy

import collections
import numpy as np
import torch
import torch.nn as nn
import torchvision
from avalanche.benchmarks import SplitMNIST, SplitCIFAR10, nc_benchmark, \
    CLExperience
from avalanche.benchmarks.datasets.external_datasets import get_cifar10_dataset
from avalanche.benchmarks.utils import AvalancheDataset, \
    AvalancheConcatDataset, concat_datasets
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.core import SupervisedPlugin
from avalanche.evaluation.metrics import accuracy_metrics, bwt_metrics
from avalanche.logging import TextLogger
from avalanche.models import MultiTaskModule, IncrementalClassifier, \
    avalanche_forward, MultiHeadClassifier
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.templates.base import ExpSequence, \
    _group_experiences_by_stream
from avalanche.training.utils import trigger_plugins
from torch import cosine_similarity
from torch.nn import CrossEntropyLoss
from torch.nn.functional import binary_cross_entropy_with_logits, \
    binary_cross_entropy
from torch.optim import Adam, Optimizer, SGD
from collections import defaultdict

from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from layers import AbsDynamicLayer, DynamicMoERoutingLayerCE, \
    DynamicMoERoutingLayerCE1, RoutingLayer
from utils import calculate_similarity, CumulativeMultiHeadClassifier


def clone_module(module, memo=None):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py)
    **Description**
    Creates a copy of a module, whose parameters/buffers/submodules
    are created using PyTorch's torch.clone().
    This implies that the computational graph is kept, and you can compute
    the derivatives of the new modules' parameters w.r.t the original
    parameters.
    **Arguments**
    * **module** (Module) - Module to be cloned.
    **Return**
    * (Module) - The cloned module.
    **Example**
    ~~~python
    net = nn.Sequential(Linear(20, 10), nn.ReLU(), nn.Linear(10, 2))
    clone = clone_module(net)
    error = loss(clone(X), y)
    error.backward()  # Gradients are back-propagate all the way to net.
    ~~~
    """
    # NOTE: This function might break in future versions of PyTorch.

    # TODO: This function might require that module.forward()
    #       was called in order to work properly, if forward() instanciates
    #       new variables.
    # TODO: We can probably get away with a shallowcopy.
    #       However, since shallow copy does not recurse, we need to write a
    #       recursive version of shallow copy.
    # NOTE: This can probably be implemented more cleanly with
    #       clone = recursive_shallow_copy(model)
    #       clone._apply(lambda t: t.clone())

    if memo is None:
        # Maps original data_ptr to the cloned tensor.
        # Useful when a Module uses parameters from another Module; see:
        # https://github.com/learnables/learn2learn/issues/174
        memo = {}

    # First, create a copy of the module.
    # Adapted from:
    # https://github.com/pytorch/pytorch/blob/65bad41cbec096aa767b3752843eddebf845726f/torch/nn/modules/module.py#L1171
    if not isinstance(module, torch.nn.Module):
        return module
    clone = module.__new__(type(module))
    clone.__dict__ = module.__dict__.copy()
    clone._parameters = clone._parameters.copy()
    clone._buffers = clone._buffers.copy()
    clone._modules = clone._modules.copy()

    # Second, re-write all parameters
    if hasattr(clone, '_parameters'):
        for param_key in module._parameters:
            if module._parameters[param_key] is not None:
                param = module._parameters[param_key]
                param_ptr = param.data_ptr
                if param_ptr in memo:
                    clone._parameters[param_key] = memo[param_ptr]
                else:
                    cloned = param.clone().detach()
                    clone._parameters[param_key] = cloned
                    memo[param_ptr] = cloned

    # Third, handle the buffers if necessary
    if hasattr(clone, '_buffers'):
        for buffer_key in module._buffers:
            if clone._buffers[buffer_key] is not None and \
                    clone._buffers[buffer_key].requires_grad:
                buff = module._buffers[buffer_key]
                buff_ptr = buff.data_ptr
                if buff_ptr in memo:
                    clone._buffers[buffer_key] = memo[buff_ptr]
                else:
                    cloned = buff.clone().detach()
                    clone._buffers[buffer_key] = cloned
                    memo[param_ptr] = cloned

    # Then, recurse for each submodule
    if hasattr(clone, '_modules'):
        for module_key in clone._modules:
            clone._modules[module_key] = clone_module(
                module._modules[module_key],
                memo=memo,
            )

    # Finally, rebuild the flattened parameters for RNNs
    # See this issue for more details:
    # https://github.com/learnables/learn2learn/issues/139
    if hasattr(clone, 'flatten_parameters'):
        clone = clone._apply(lambda x: x)
    return clone


class CentroidsMatching(SupervisedPlugin):
    def __init__(self,
                 sit=True,
                 top_k=1,
                 per_sample_routing_reg=False,
                 centroids_merging_strategy=None,
                 **kwargs):

        super().__init__()

        self.distributions = {}
        self.patterns_per_experience = 500
        self.top_k = top_k
        self.per_sample_routing_reg = per_sample_routing_reg

        self.tasks_centroids = []

        self.past_model = None
        self.past_scaler = None

        self.current_iteration_centroids = None
        self.similarity = 'euclidean'
        self.multimodal_merging = centroids_merging_strategy

        self.sit = sit
        self.tasks_nclasses = {}

        self.past_routing = {}
        self.past_embeddings = {}

        self.memory_x, self.memory_y, self.memory_tid = {}, {}, {}
        self.past_dataset = {}

        self.layers_centroids = defaultdict(dict)

        self.centroids = []

    def before_train_dataset_adaptation(self, strategy, *args, **kwargs):
        # if self.per_sample_routing_reg:
        if strategy.experience.current_experience == 0:
            return

        self.past_model = deepcopy(strategy.model)

    def before_training_exp(self, strategy: SupervisedTemplate,
                            **kwargs):

        classes = strategy.experience.classes_in_this_experience
        self.tasks_nclasses[strategy.experience.task_label] = classes

        tid = strategy.experience.current_experience

        if tid > 0:
            self.past_model = deepcopy(strategy.model)

        paths = strategy.model.available_paths
        selected_path = np.random.randint(0, paths)

        for layer in strategy.model.layers:
            p_id = layer.get_unassigned_paths()[selected_path]
            layer.assign_path_to_task(p_id, tid)

        strategy.model.calculate_available_paths()

        if tid > 0:
            av = concat_datasets(list(self.past_dataset.values()))
            self.av = av

            strategy.dataloader = ReplayDataLoader(
                strategy.adapted_dataset,
                av,
                oversample_small_tasks=True,
                batch_size=strategy.train_mb_size,
                batch_size_mem=strategy.train_mb_size,
                task_balanced_dataloader=True,
                num_workers=0,
                shuffle=True,
                drop_last=False,
            )

    def before_training_epoch(self, strategy, *args, **kwargs):

        for name, module in strategy.model.named_modules():
            if isinstance(module, (AbsDynamicLayer)):
                module.similarity_statistics = []

    def after_finetuning_exp(self, strategy: 'BaseStrategy', **kwargs):
        return
        tid = strategy.experience.current_experience

        for module in strategy.model.modules():
            if isinstance(module, (AbsDynamicLayer)):
                module.freeze_blocks(tid)

        dataset = strategy.experience.dataset
        dataset_idx = np.arange(len(dataset))
        np.random.shuffle(dataset_idx)

        idx_to_get = dataset_idx[:self.patterns_per_experience]
        memory = dataset.train().subset(idx_to_get)
        self.past_dataset[tid] = memory

        self.past_model = clone_module(strategy.model)

    @torch.no_grad()
    def after_training_exp(self, strategy, **kwargs):
        tid = strategy.experience.current_experience

        dev_x, _, _ = next(
            iter(DataLoader(strategy.experience.dev_dataset.eval(),
                            len(strategy.experience.dev_dataset))))
        dev_x = dev_x.to(strategy.device)

        current_centroids = strategy.model.pp(dev_x, tid).mean(0, keepdims=True)

        self.centroids.append(current_centroids)

        # for module in strategy.model.modules():
        #     if isinstance(module, RoutingLayer):
        #         module.freeze_blocks(tid)

        for i, d in self.past_dataset.items():
            dev_x, _, _ = next(
                iter(DataLoader(d.eval(), len(d))))
            dev_x = dev_x.to(strategy.device)

            current_centroids = strategy.model.pp(dev_x, i).mean(0,
                                                                 keepdims=True)

            self.centroids[i] = current_centroids

        self.past_dataset[tid] = strategy.experience.dev_dataset

        dataset = strategy.experience.dataset
        dataset_idx = np.arange(len(dataset))
        np.random.shuffle(dataset_idx)

        idx_to_get = dataset_idx[:self.patterns_per_experience]
        memory = dataset.train().subset(idx_to_get)
        self.past_dataset[tid] = memory

        return

        for module in strategy.model.modules():
            if isinstance(module, (AbsDynamicLayer)):
                module.freeze_logits(strategy.experience,
                                     strategy,
                                     tid,
                                     top_k=self.top_k)

        # if tid > 0:

    def before_backward(self, strategy, *args, **kwargs):
        # return
        # # self.before_backward_kl(strategy, args, kwargs)
        # self.before_backward_distance(strategy, args, kwargs)

        tid = strategy.experience.current_experience

        if tid == 0:
            return

        tids = strategy.mb_task_id
        x = strategy.mb_x
        all_tasks = torch.unique(tids)
        current_features = strategy.model.current_features[tids != tid]

        all_loss = 0
        for t in all_tasks:
            task_loss = 0

            t_mask = tids == t
            x_t = x[t_mask]

            # other_indexes = [l for i in range(tid + 1)
            #                  for l in self.tasks_nclasses[i] if i != t.item()]
            # other_indexes = torch.tensor(other_indexes, device=x.device)

            for t1 in all_tasks:
                if t1 == t:
                    continue

                i = max(self.tasks_nclasses[t1.item()]) + 1

                preds = strategy.model(x_t, t1.item())
                if preds.shape[-1] != i:
                    # preds = preds[:, :i]
                    preds = preds[:, self.tasks_nclasses[t1.item()]]
                # preds = preds.index_select(-1, other_indexes)
                preds = torch.softmax(preds, -1)
                
                h = -(preds.log() * preds).sum(-1)
                h = h / np.log(preds.shape[-1])
                # h = h[~torch.isnan(h)]
                # h = torch.nan_to_num(h, 1)
                task_loss += (1 - h).mean()

            all_loss += task_loss

        all_loss = all_loss / (tid + 1)
        strategy.loss += all_loss * 1

        x = x[tids != tid]
        tids = tids[tids != tid]

        self.past_model(x, tids)
        past_features = self.past_model.current_features

        # loss = nn.functional.mse_loss(current_features, past_features)
        loss = nn.functional.cosine_similarity(current_features, past_features)
        loss = (loss + 1) / 2
        loss = 1 - loss
        loss = loss.mean()

        strategy.loss += loss * 1

    def before_backward_distance(self, strategy, *args, **kwargs):
        tid = strategy.experience.current_experience
        if tid == 0:
            return

        tids = strategy.mb_task_id

        # current_features = strategy.model.current_features
        # dev_x, _, _ = next(iter(DataLoader(strategy.experience.dev_dataset,
        #                                    len(strategy.experience.dev_dataset))))
        # dev_x = dev_x.to(strategy.device)
        #
        # current_centroids = strategy.model.features(dev_x, tid)
        # current_centroids = current_centroids.flatten(1).mean(0, keepdims=True)
        #
        # dist = torch.norm(current_features - current_centroids, 2, -1)
        # dist = torch.maximum(torch.zeros_like(dist), dist - 0.5)
        #
        # positive_distance = dist.mean()
        # # loss += dist.mean()
        #
        # strategy.loss += positive_distance
        # current_features = strategy.model.current_features
        # dev_x, _, _ = next(iter(DataLoader(strategy.experience.dev_dataset,
        #                                    len(strategy.experience.dev_dataset))))
        # dev_x = dev_x.to(strategy.device)
        #
        # current_centroids = strategy.model.features(dev_x, tid)
        # current_centroids = current_centroids.mean(0, keepdims=True)
        #
        # dist = torch.norm(current_features - current_centroids, 2, -1)
        # dist = torch.maximum(torch.zeros_like(dist), dist - 0.5)
        #
        # positive_distance = dist.mean()

        m = torch.zeros((tid + 1, tid + 1), device=x.device)

        for it in range(tid + 1):
            mask = tids == it
            tx = x[mask]

            for ot in range(it, tid + 1):
                # current_centroids = ot
                features = strategy.model.pp(tx, ot)

                if ot == tid:
                    with torch.no_grad():
                        dev_x, _, _ = next(
                            iter(DataLoader(strategy.experience.dev_dataset,
                                            len(strategy.experience.dev_dataset))))
                        dev_x = dev_x.to(strategy.device)

                        current_centroids = strategy.model.pp(dev_x, ot)
                        current_centroids = current_centroids.mean(0,
                                                                   keepdims=True)
                else:
                    current_centroids = self.centroids[ot]

                features = nn.functional.normalize(features, 2, -1)
                # current_centroids = nn.functional.normalize(current_centroids, 2, -1)

                # v = torch.norm(features - current_centroids, 2, -1)

                v = torch.cosine_similarity(features, current_centroids, -1)

                v = (v + 1) / 2
                # if it == ot:
                #     v = 1 / v
                # v = torch.maximum(torch.zeros_like(v), v - 0.5)
                # else:
                # v = 1 / v
                # v = torch.maximum(torch.zeros_like(v), 0.5 - v)

                v = v.mean()
                m[it, ot] = v
                m[ot, it] = v

        # ds = torch.diag(m)
        # ds = torch.maximum(torch.zeros_like(ds), ds - 0.5)
        # nz = torch.count_nonzero(ds)
        # if nz > 0:
        #     ds = ds.sum() / nz
        #     strategy.loss += ds
        #
        # ss = 0
        # if tid > 0:
        #     i, j = torch.triu_indices(len(m), len(m), 1)
        #     dists = m[i, j]
        #     if torch.count_nonzero(dists) > 0:
        #         sims = 1 / dists
        #         # sims = torch.maximum(torch.zeros_like(sims), sims - 0.5)
        #         ss = sims.mean() * 0.1
        #
        # strategy.loss += ss

        # m = torch.eye(len(m), device=m.device) - m
        norm = torch.linalg.matrix_norm(torch.eye(len(m), device=m.device) - m)
        strategy.loss += norm

        # print(m)
        return

        for t in torch.unique(tids):
            mask = tids == t
            t = t.item()

            features = strategy.model.current_features[mask]

            if t == tid:
                dev_x, _, _ = next(
                    iter(DataLoader(strategy.experience.dev_dataset,
                                    len(strategy.experience.dev_dataset))))
                dev_x = dev_x.to(strategy.device)

                current_centroids = strategy.model.features(dev_x, tid)
                current_centroids = current_centroids.mean(0, keepdims=True)
            else:
                current_centroids = self.centroids

        # loss += dist.mean()

        negative_r_distance = 0

        # for i in range(tid):
        #     current_centroids = self.centroids[i]
        #     f = strategy.model.features(x, i)
        #     dist = torch.norm(f - current_centroids, 2, -1)
        #     sim = 1 / dist
        #     # sim = torch.maximum(torch.zeros_like(sim), sim - 0.5)
        #
        #     negative_r_distance += sim.mean()

        # negative_r_distance = 0

        # r_x = torch.randn(500, 3, 32, 32, device=strategy.device)
        # d = 0
        #
        # paths = strategy.model.get_random_paths(5)
        # for p in paths:
        #     d += 1
        #
        #     f = strategy.model.features(r_x, None, p).mean(0, keepdims=True)
        #     f1 = strategy.model.features(x, None, p).flatten(1)
        #
        #     dist = torch.norm(f1 - f, 2, -1)
        #     sim = 1 / dist
        #     sim = torch.maximum(torch.zeros_like(sim), sim - 0.5)
        #
        #     negative_r_distance += sim.mean()
        #
        # strategy.loss += (negative_r_distance / d) * 0.1
        #
        strategy.loss += positive_distance * 0.1
        # strategy.loss += negative_r_distance * 0.01

        if tid == 0:
            return

        distances = 0
        sims = 0

        for i in range(tid):
            _x, _, _ = next(iter(DataLoader(self.past_dataset[i],
                                            len(x), shuffle=True)))
            _x = _x.to(x.device)

            for j in range(tid + 1):
                if j == tid:
                    cc = current_centroids
                else:
                    cc = self.centroids[j]

                f = strategy.model.features(x, j)

                dist = torch.norm(f - cc, 2, -1)

                if j == i:
                    dist = torch.maximum(torch.zeros_like(dist), dist - 0.5)
                    distances += dist.mean()
                else:
                    sim = 1 / dist
                    sim = torch.maximum(torch.zeros_like(sim), sim - 0.5)
                    sims += sim.mean()

        strategy.loss += sims * 0.01 + distances * 0.01
        return

        features = []

        negative_r_distance = 0
        for i in range(tid):
            f = strategy.model.features(x, i)
            dist = torch.norm(f - self.centroids[i], 2, -1)
            sim = 1 / dist
            # sim = torch.maximum(torch.zeros_like(sim), sim - 0.5)

            negative_r_distance += sim.mean()

            # features.append(torch.norm(f - self.centroids[i], 2, -1))

        negative_r_distance = negative_r_distance / d
        strategy.loss += negative_r_distance * 1

        return
        dev_x, _, _ = next(iter(DataLoader(strategy.experience.dev_dataset,
                                           len(strategy.experience.dev_dataset))))
        dev_x = dev_x.to(strategy.device)

        current_centroids = strategy.model.features(dev_x, tid)
        current_centroids = current_centroids.mean(0, keepdims=True)

        features.append(torch.norm(strategy.model.current_features -
                                   current_centroids, 2, -1))

        features = -torch.stack(features, -1)
        distr = torch.log_softmax(features, -1)
        loss = -distr[:, tid].squeeze().mean(0)

        strategy.loss += loss * 5
        return

        positive_w = 1
        negative_r_w = 0
        negative_t_w = 1

        x = strategy.mb_x
        t = strategy.mb_task_id

        past_centroids = torch.cat(self.centroids)
        past_xs = []
        past_ts = []

        for i in range(tid):
            _x, _, _ = next(iter(DataLoader(self.past_dataset[i],
                                            len(x), shuffle=True)))
            _x = _x.to(x.device)
            past_xs.append(_x)

            past_ts.extend([i] * len(_x))

        past_xs = torch.cat(past_xs, 0)
        past_ts = torch.tensor(past_ts, device=x.device)

        # current_features = strategy.model.current_features
        dev_x, _, _ = next(iter(DataLoader(strategy.experience.dev_dataset,
                                           len(strategy.experience.dev_dataset))))
        dev_x = dev_x.to(strategy.device)

        current_centroids = strategy.model.features(dev_x, tid)
        current_centroids = current_centroids.flatten(1).mean(0, keepdims=True)

        all_centroids = self.centroids + [current_centroids]
        xs = torch.cat((x, past_xs), 0)
        ts = torch.cat((t, past_ts), 0)

        features = []
        for i in range(tid + 1):
            f = strategy.model.features(xs, i)
            features.append(torch.norm(f - all_centroids[i], 2, -1))

        features = -torch.stack(features, -1)

        distr = torch.log_softmax(features, -1)
        loss = -distr.gather(1, ts[:, None]).squeeze().mean(0)

        strategy.loss += loss * 5

        past_reg = 0
        for i in range(tid):
            x, _, _ = next(iter(DataLoader(self.past_dataset[i],
                                           len(x), shuffle=True)))
            x = x.to(strategy.device)

            past_centroids = self.past_model.features(x, i).flatten(1)
            current_centroids = strategy.model.features(x, i).flatten(1)

            sim = nn.functional.cosine_similarity(past_centroids,
                                                  current_centroids)
            dist = 1 - sim

            # dist = nn.functional.mse_loss(current_centroids, past_centroids)

            past_reg += dist.mean()

        strategy.loss += past_reg * 5

        return
        # current_features = current_features.flatten(1)

        # curre = 0
        dist = torch.norm(current_features - current_centroids, 2, -1)
        dist = torch.maximum(torch.zeros_like(dist), dist - 0.5)

        positive_distance = dist.mean()
        # loss += dist.mean()

        d = 1
        negative_r_distance = 0
        negative_t_distance = 0

        if negative_r_w > 0:
            paths = strategy.model.get_random_paths(5)
            for p in paths:
                d += 1

                f = strategy.model.features(dev_x, None, p)
                f = f.mean(0, keepdims=True).flatten(1)

                f1 = strategy.model.features(x, None, p).flatten(1)

                dist = torch.norm(f1 - f, 2, -1)
                sim = 1 / dist
                # sim = torch.maximum(torch.zeros_like(sim), sim - 0.5)

                negative_r_distance += sim.mean()

        # for f in strategy.model.random_path_forward(dev_x, 5):
        #     d += 1
        #     # for _ in range(5):
        #     #     f = strategy.model.random_path_forward(dev_x)
        #     f = f.mean(0, keepdims=True).flatten(1)
        #
        #     dist = torch.norm(current_features - f, 2, -1)
        #     sim = 1 / dist
        #     sim = torch.maximum(torch.zeros_like(sim), sim - 0.5)
        #
        #     negative_r_distance += sim.mean()

        for i in range(tid):
            d += 1
            f = strategy.model.features(x, i).flatten(1)
            # f = f.mean(0, keepdims=True).flatten(1        )

            x, _, _ = next(iter(DataLoader(self.past_dataset[i],
                                           len(x), shuffle=True)))
            x = x.to(strategy.device)

            f1 = strategy.model.features(x, i).flatten(1).mean(0, keepdims=True)

            # f1 = strategy.model.features(x, i, None).flatten(1)
            # f1 = self.centroids[i]

            # dist = torch.norm(current_features - f, 2, -1)
            dist = torch.norm(f1 - f, 2, -1)

            sim = 1 / dist
            sim = torch.maximum(torch.zeros_like(sim), sim - 0.5)
            negative_t_distance += sim.mean()

        loss = (positive_distance * positive_w +
                negative_r_distance * negative_r_w +
                negative_t_distance * negative_t_w) / d

        strategy.loss += loss * 0.1

        past_reg = 0
        for i in range(tid):
            x, _, _ = next(iter(DataLoader(self.past_dataset[i],
                                           len(x), shuffle=True)))
            x = x.to(strategy.device)

            past_centroids = self.past_model.features(x, i).flatten(1)
            current_centroids = strategy.model.features(x, i).flatten(1)

            # sim = nn.functional.cosine_similarity(past_centroids, current_centroids)
            # dist = 1 - sim

            dist = nn.functional.mse_loss(current_centroids, past_centroids)

            past_reg += dist.mean()

        strategy.loss += past_reg * 1

        return
        entropies = []
        classification_losses = []
        distance_losses = []
        mask = strategy.mb_task_id == tid

        # if tid == 0 and strategy.is_finetuning:
        #     return

        if tid > 0:
            self.past_model.eval()

            _x = strategy.mb_x[~mask]
            # past_routing = self.past_model.routing_model(_x)

            # current_routing = strategy.mb_output
            #
            # norm = torch.norm(past_routing - current_routing, 2, -1).mean()
            #     # strategy.loss += norm
            #     # print(norm)
            #
            #     _x = strategy.mb_x[~mask]
            #     t = strategy.mb_task_id[~mask]
            #     # _x = strategy.mb_x

            with torch.no_grad():
                self.past_model(_x, strategy.mb_task_id[~mask])
                past_routing = {}

                for name, module in self.past_model.named_modules():
                    if isinstance(module, (
                            AbsDynamicLayer)):
                        past_routing[name] = torch.argmax(
                            module.last_distribution[0], -1)
                        # past_routing[name] = torch.softmax(
                        #     module.last_distribution[1],
                        #     -1)

            d = 0
            for name, module in strategy.model.named_modules():
                if isinstance(module, (
                        AbsDynamicLayer)):
                    # distr = torch.log_softmax(
                    #     module.last_distribution[1][~mask], -1)
                    # similarity = cosine_similarity(
                    #     module.current_routing[~mask],
                    #     past_routing[name])
                    # loss = nn.functional.kl_div(distr, past_routing[name])
                    loss = nn.functional.cross_entropy(
                        module.last_distribution[1][~mask],
                        past_routing[name])
                    # distance = 1 - similarity
                    # distance = distance.mean(0)

                    # distance = nn.functional.mse_loss(
                    #     module.current_routing[~mask],
                    #     past_routing[name])
                    d += loss

            distance = d / 3

            strategy.loss += distance * 0

        for name, module in strategy.model.named_modules():
            if isinstance(module, (AbsDynamicLayer)):
                weights, similarity = module.last_distribution

                # if strategy.is_finetuning:
                #     toiter =
                # else:
                #     toiter = range(tid)
                # if not strategy.is_finetuning:
                #     distr = torch.softmax(similarity[mask].mean(0), -1)
                #     h = -(distr.log() * distr).sum(-1) / np.log(
                #         distr.shape[-1])
                #     h = -h
                #     entropies.append(h)
                # for i in range(tid + 1):
                #     distr = torch.softmax(
                #         similarity[strategy.mb_task_id == i].mean(0), -1)
                #     h = -(distr.log() * distr).sum(-1) / np.log(
                #         distr.shape[-1])
                #
                #     if i == tid and not strategy.is_finetuning:
                #         h = -h
                #
                #     es += h
                #
                # es = es / (tid + 1)
                #
                # entropies.append(es)

                if tid == 0:
                    labels = strategy.mb_y
                else:
                    offsets = [0] + list(self.tasks_nclasses.values())
                    offsets = np.cumsum(offsets)

                    offsets = [offsets[t.item()]
                               for t in strategy.mb_task_id]
                    offsets = torch.tensor(offsets, dtype=torch.long,
                                           device=strategy.mb_y.device)

                    labels = strategy.mb_y + offsets

                all_h = 0
                # uniques = torch.unique(labels)
                uniques = torch.unique(strategy.mb_task_id)
                centroids = []

                for i in uniques:
                    # mask = labels == i
                    mask = strategy.mb_task_id == i
                    # distr = torch.softmax(weights[mask].mean(0), -1)
                    distr = weights[mask].mean(0)
                    distr = distr[distr > 0]

                    if len(distr) > 0:
                        h = -(distr.log() * distr).sum(-1) / np.log(
                            weights.shape[-1])
                        all_h += h

                entropies.append(all_h / len(uniques))

                if tid > 0 or strategy.is_finetuning:
                    for i in uniques:
                        centroids.append(module.current_routing
                                         [strategy.mb_task_id == i].mean(0))
                        block_output, other_outputs = module.current_output
                        other_outputs = other_outputs.mean(0)

                        block_output, other_outputs = torch.flatten(
                            block_output, 1), torch.flatten(other_outputs, 1)

                        similarity = nn.functional.cosine_similarity(
                            block_output, other_outputs)
                        similarity = (similarity + 1).mean()
                        distance_losses.append(similarity)

                    # centroids = torch.stack(centroids, 0)
                    #
                    # similarity = torch.cosine_similarity(centroids[..., None, :, :],
                    #                                      centroids[..., :, None, :],
                    #                                      dim=-1)
                    #
                    # # centroids = nn.functional.normalize(centroids, 2, -1)
                    # # distance = torch.norm(centroids[..., None, :, :]
                    # #                       - centroids[..., :, None, :], 2, -1)
                    # # similarity = 1 / (distance + 1)
                    # similarity = (similarity + 1)
                    # similarity = torch.triu(similarity, 1)
                    # d = ((len(centroids) - 1) * len(centroids)) / 2
                    # similarity = similarity.sum() / d
                    # # similarity = -torch.log(similarity)
                    #
                    # distance_losses.append(similarity)

                # print(similarity)

                # if tid > 0:
                #     # labels = [module.get_task_blocks(t.item())
                #     #           for t in strategy.mb_task_id[~mask]]
                #     # labels = torch.stack(labels, 0)
                #     centroids = []
                #     for y in uniques:
                #         distr = torch.softmax(similarity[labels == y].mean(0),
                #                               -1)
                #
                #     centroids = [
                #         module.current_routing[strategy.mb_task_id == i].mean(0)
                #         for i in range(tid + 1)]
                #
                # #     centroids = torch.stack(centroids, 0)
                # #
                # #     # distances = nn.PairwiseDistance(p=2)(centroids, centroids)
                # #     distances = torch.cdist(centroids, centroids)
                # #     distances = torch.triu(distances)
                # #
                # #     d = (len(centroids) - 1) * len(centroids)
                # #     d = d / 2
                # #
                #
                # if strategy.is_finetuning:
                #     labels = [module.get_task_blocks(t.item())
                #               for t in strategy.mb_task_id]
                #     labels = torch.stack(labels, 0)
                #     distr = torch.log_softmax(similarity, -1)
                #     loss = -distr.gather(1, labels).squeeze()
                #
                #     # loss[~mask] *= 10
                #
                #     # loss = nn.functional.cross_entropy(similarity,
                #     #                                    labels.squeeze())
                #     classification_losses.append(loss.mean(0))
                #
                # elif tid > 0:
                #     # h = -h
                #     # if tid > 0:
                #     # h[mask] = -h[mask]
                #
                #     # distr = torch.softmax(similarity, -1)
                #     # h = -(distr.log() * distr).sum(-1) / np.log(
                #     #     distr.shape[-1])
                #     #
                #     # if tid > 0:
                #     #     h = h[mask]
                #     # entropies.append(-h)
                #     # if tid > 0:
                #
                #     # losses = torch.zeros_like(strategy.mb_task_id)
                #
                #     labels = [module.get_task_blocks(t.item())
                #               for t in strategy.mb_task_id[~mask]]
                #     labels = torch.stack(labels, 0)
                #     # loss = -torch.log_softmax(similarity[~mask], -1) \
                #     #     .gather(1, labels).squeeze().mean(0)
                #     #
                #     # preds = similarity[~mask].argmax(-1)
                #
                #     # losses[:len(loss)] = loss
                #
                #     # loss = -torch.log_softmax(similarity[~mask], -1)\
                #     #     .gather(1, labels).squeeze()
                #     # loss = loss[~mask]
                #
                #     # classification_losses.append(loss)
                #
                #     d = torch.log_softmax(similarity[mask], -1) \
                #         .index_select(-1, torch.unique(labels))
                #     d = d.squeeze().mean(-1).mean(0)
                #
                #     distance_losses.append(d)
                #
                #     # losses[len(loss):] = d
                #     # a = 0

        # else:
        #     outputs = {}
        #     past_output = self.past_model(strategy.mb_x, strategy.mb_task_id)
        #
        #     for name, module in self.past_model.named_modules():
        #         if isinstance(module, (MoERoutingLayer,
        #                                AbsDynamicLayer)):
        #             weights, similarity = module.last_distribution
        #             outputs[name] = weights
        #
        #     mask = strategy.mb_task_id == tid
        #
        #     for name, module in strategy.model.named_modules():
        #         if isinstance(module, (MoERoutingLayer,
        #                                AbsDynamicLayer)):
        #             weights, similarity = module.last_distribution
        #
        #             if strategy.is_finetuning and tid > 0:
        #                 distr = torch.log_softmax(similarity, -1)
        #                 distr = distr * outputs[name]
        #                 distr = distr.sum(-1) / outputs[name].sum(-1)
        #                 loss = - distr
        #
        #                 classification_losses.append(loss)
        #             else:
        #                 distr = torch.softmax(similarity, -1)
        #                 h = -(distr.log() * distr).sum(-1) / np.log(
        #                     weights.shape[-1])
        #                 if tid > 0:
        #                     h = h[mask]
        #                 entropies.append(-h)
        #
        #                 if tid > 0:
        #                     sim = -torch.log_softmax(similarity[~mask], -1)
        #                     past_selection = outputs[name][~mask]
        #
        #                     sim = sim * past_selection
        #                     loss = sim.sum(-1) / past_selection.sum(-1)
        #                     classification_losses.append(loss)
        #
        #                     labels = [module.get_task_blocks(t.item())
        #                               for t in strategy.mb_task_id[~mask]]
        #                     labels = torch.stack(labels, 0)
        #                     # loss = -torch.log_softmax(similarity[~mask], -1) \
        #                     #     .gather(1, labels).squeeze()
        #                     # loss = loss[~mask]
        #
        #                     d = torch.log_softmax(similarity[mask], -1) \
        #                         .index_select(-1, torch.unique(labels))
        #                     # d = -torch.log_softmax(similarity[~mask], -1).mean(-1)
        #                     distance_losses.append(d.mean(-1))

        if len(entropies) > 0:
            entropy = sum(entropies) / len(entropies)
            # entropy = sum(entropies)
            # entropy = entropy.mean(0)
        else:
            entropy = 0

        if len(classification_losses) > 0:
            class_loss = sum(classification_losses) / len(
                classification_losses)
            class_loss = class_loss.mean(0)
        else:
            class_loss = 0

        if len(distance_losses) > 0:
            distance_loss = sum(distance_losses) / len(distance_losses)
            distance_loss = distance_loss.mean(0)
        else:
            distance_loss = 0

        strategy.loss += entropy * 1e-4 + class_loss * 0 + distance_loss * 100

        return

        # return
        # self._bb_sigmoid(strategy)
        self._bb_kl_div(strategy)
        # self.___before_backward(strategy)

    def _before_backward_distance(self, strategy, *args, **kwargs):

        positive_w = 1
        negative_r_w = 0
        negative_t_w = 1

        x = strategy.mb_x
        tid = strategy.experience.current_experience

        current_features = strategy.model.current_features
        dev_x, _, _ = next(iter(DataLoader(strategy.experience.dev_dataset,
                                           len(strategy.experience.dev_dataset))))
        dev_x = dev_x.to(strategy.device)

        current_centroids = strategy.model.features(dev_x, tid)
        current_centroids = current_centroids.flatten(1).mean(0, keepdims=True)

        current_features = current_features.flatten(1)

        # curre = 0
        dist = torch.norm(current_features - current_centroids, 2, -1)
        dist = torch.maximum(torch.zeros_like(dist), dist - 0.5)

        positive_distance = dist.mean()
        # loss += dist.mean()

        d = 1
        negative_r_distance = 0
        negative_t_distance = 0

        if negative_r_w > 0:
            paths = strategy.model.get_random_paths(5)
            for p in paths:
                d += 1

                f = strategy.model.features(dev_x, None, p)
                f = f.mean(0, keepdims=True).flatten(1)

                f1 = strategy.model.features(x, None, p).flatten(1)

                dist = torch.norm(f1 - f, 2, -1)
                sim = 1 / dist
                # sim = torch.maximum(torch.zeros_like(sim), sim - 0.5)

                negative_r_distance += sim.mean()

        # for f in strategy.model.random_path_forward(dev_x, 5):
        #     d += 1
        #     # for _ in range(5):
        #     #     f = strategy.model.random_path_forward(dev_x)
        #     f = f.mean(0, keepdims=True).flatten(1)
        #
        #     dist = torch.norm(current_features - f, 2, -1)
        #     sim = 1 / dist
        #     sim = torch.maximum(torch.zeros_like(sim), sim - 0.5)
        #
        #     negative_r_distance += sim.mean()

        for i in range(tid):
            d += 1
            f = strategy.model.features(x, i).flatten(1)
            # f = f.mean(0, keepdims=True).flatten(1        )

            x, _, _ = next(iter(DataLoader(self.past_dataset[i],
                                           len(x), shuffle=True)))
            x = x.to(strategy.device)

            f1 = strategy.model.features(x, i).flatten(1).mean(0, keepdims=True)

            # f1 = strategy.model.features(x, i, None).flatten(1)
            # f1 = self.centroids[i]

            # dist = torch.norm(current_features - f, 2, -1)
            dist = torch.norm(f1 - f, 2, -1)

            sim = 1 / dist
            sim = torch.maximum(torch.zeros_like(sim), sim - 0.5)
            negative_t_distance += sim.mean()

        loss = (positive_distance * positive_w +
                negative_r_distance * negative_r_w +
                negative_t_distance * negative_t_w) / d

        strategy.loss += loss * 0.1

        past_reg = 0
        for i in range(tid):
            x, _, _ = next(iter(DataLoader(self.past_dataset[i],
                                           len(x), shuffle=True)))
            x = x.to(strategy.device)

            past_centroids = self.past_model.features(x, i).flatten(1)
            current_centroids = strategy.model.features(x, i).flatten(1)

            # sim = nn.functional.cosine_similarity(past_centroids, current_centroids)
            # dist = 1 - sim

            dist = nn.functional.mse_loss(current_centroids, past_centroids)

            past_reg += dist.mean()

        strategy.loss += past_reg * 1

        return
        entropies = []
        classification_losses = []
        distance_losses = []
        mask = strategy.mb_task_id == tid

        # if tid == 0 and strategy.is_finetuning:
        #     return

        if tid > 0:
            self.past_model.eval()

            _x = strategy.mb_x[~mask]
            # past_routing = self.past_model.routing_model(_x)

            # current_routing = strategy.mb_output
            #
            # norm = torch.norm(past_routing - current_routing, 2, -1).mean()
            #     # strategy.loss += norm
            #     # print(norm)
            #
            #     _x = strategy.mb_x[~mask]
            #     t = strategy.mb_task_id[~mask]
            #     # _x = strategy.mb_x

            with torch.no_grad():
                self.past_model(_x, strategy.mb_task_id[~mask])
                past_routing = {}

                for name, module in self.past_model.named_modules():
                    if isinstance(module, (
                            AbsDynamicLayer)):
                        past_routing[name] = torch.argmax(
                            module.last_distribution[0], -1)
                        # past_routing[name] = torch.softmax(
                        #     module.last_distribution[1],
                        #     -1)

            d = 0
            for name, module in strategy.model.named_modules():
                if isinstance(module, (
                        AbsDynamicLayer)):
                    # distr = torch.log_softmax(
                    #     module.last_distribution[1][~mask], -1)
                    # similarity = cosine_similarity(
                    #     module.current_routing[~mask],
                    #     past_routing[name])
                    # loss = nn.functional.kl_div(distr, past_routing[name])
                    loss = nn.functional.cross_entropy(
                        module.last_distribution[1][~mask],
                        past_routing[name])
                    # distance = 1 - similarity
                    # distance = distance.mean(0)

                    # distance = nn.functional.mse_loss(
                    #     module.current_routing[~mask],
                    #     past_routing[name])
                    d += loss

            distance = d / 3

            strategy.loss += distance * 0

        for name, module in strategy.model.named_modules():
            if isinstance(module, (AbsDynamicLayer)):
                weights, similarity = module.last_distribution

                # if strategy.is_finetuning:
                #     toiter =
                # else:
                #     toiter = range(tid)
                # if not strategy.is_finetuning:
                #     distr = torch.softmax(similarity[mask].mean(0), -1)
                #     h = -(distr.log() * distr).sum(-1) / np.log(
                #         distr.shape[-1])
                #     h = -h
                #     entropies.append(h)
                # for i in range(tid + 1):
                #     distr = torch.softmax(
                #         similarity[strategy.mb_task_id == i].mean(0), -1)
                #     h = -(distr.log() * distr).sum(-1) / np.log(
                #         distr.shape[-1])
                #
                #     if i == tid and not strategy.is_finetuning:
                #         h = -h
                #
                #     es += h
                #
                # es = es / (tid + 1)
                #
                # entropies.append(es)

                if tid == 0:
                    labels = strategy.mb_y
                else:
                    offsets = [0] + list(self.tasks_nclasses.values())
                    offsets = np.cumsum(offsets)

                    offsets = [offsets[t.item()]
                               for t in strategy.mb_task_id]
                    offsets = torch.tensor(offsets, dtype=torch.long,
                                           device=strategy.mb_y.device)

                    labels = strategy.mb_y + offsets

                all_h = 0
                # uniques = torch.unique(labels)
                uniques = torch.unique(strategy.mb_task_id)
                centroids = []

                for i in uniques:
                    # mask = labels == i
                    mask = strategy.mb_task_id == i
                    # distr = torch.softmax(weights[mask].mean(0), -1)
                    distr = weights[mask].mean(0)
                    distr = distr[distr > 0]

                    if len(distr) > 0:
                        h = -(distr.log() * distr).sum(-1) / np.log(
                            weights.shape[-1])
                        all_h += h

                entropies.append(all_h / len(uniques))

                if tid > 0 or strategy.is_finetuning:
                    for i in uniques:
                        centroids.append(module.current_routing
                                         [strategy.mb_task_id == i].mean(0))
                        block_output, other_outputs = module.current_output
                        other_outputs = other_outputs.mean(0)

                        block_output, other_outputs = torch.flatten(
                            block_output, 1), torch.flatten(other_outputs, 1)

                        similarity = nn.functional.cosine_similarity(
                            block_output, other_outputs)
                        similarity = (similarity + 1).mean()
                        distance_losses.append(similarity)

                    # centroids = torch.stack(centroids, 0)
                    #
                    # similarity = torch.cosine_similarity(centroids[..., None, :, :],
                    #                                      centroids[..., :, None, :],
                    #                                      dim=-1)
                    #
                    # # centroids = nn.functional.normalize(centroids, 2, -1)
                    # # distance = torch.norm(centroids[..., None, :, :]
                    # #                       - centroids[..., :, None, :], 2, -1)
                    # # similarity = 1 / (distance + 1)
                    # similarity = (similarity + 1)
                    # similarity = torch.triu(similarity, 1)
                    # d = ((len(centroids) - 1) * len(centroids)) / 2
                    # similarity = similarity.sum() / d
                    # # similarity = -torch.log(similarity)
                    #
                    # distance_losses.append(similarity)

                # print(similarity)

                # if tid > 0:
                #     # labels = [module.get_task_blocks(t.item())
                #     #           for t in strategy.mb_task_id[~mask]]
                #     # labels = torch.stack(labels, 0)
                #     centroids = []
                #     for y in uniques:
                #         distr = torch.softmax(similarity[labels == y].mean(0),
                #                               -1)
                #
                #     centroids = [
                #         module.current_routing[strategy.mb_task_id == i].mean(0)
                #         for i in range(tid + 1)]
                #
                # #     centroids = torch.stack(centroids, 0)
                # #
                # #     # distances = nn.PairwiseDistance(p=2)(centroids, centroids)
                # #     distances = torch.cdist(centroids, centroids)
                # #     distances = torch.triu(distances)
                # #
                # #     d = (len(centroids) - 1) * len(centroids)
                # #     d = d / 2
                # #
                #
                # if strategy.is_finetuning:
                #     labels = [module.get_task_blocks(t.item())
                #               for t in strategy.mb_task_id]
                #     labels = torch.stack(labels, 0)
                #     distr = torch.log_softmax(similarity, -1)
                #     loss = -distr.gather(1, labels).squeeze()
                #
                #     # loss[~mask] *= 10
                #
                #     # loss = nn.functional.cross_entropy(similarity,
                #     #                                    labels.squeeze())
                #     classification_losses.append(loss.mean(0))
                #
                # elif tid > 0:
                #     # h = -h
                #     # if tid > 0:
                #     # h[mask] = -h[mask]
                #
                #     # distr = torch.softmax(similarity, -1)
                #     # h = -(distr.log() * distr).sum(-1) / np.log(
                #     #     distr.shape[-1])
                #     #
                #     # if tid > 0:
                #     #     h = h[mask]
                #     # entropies.append(-h)
                #     # if tid > 0:
                #
                #     # losses = torch.zeros_like(strategy.mb_task_id)
                #
                #     labels = [module.get_task_blocks(t.item())
                #               for t in strategy.mb_task_id[~mask]]
                #     labels = torch.stack(labels, 0)
                #     # loss = -torch.log_softmax(similarity[~mask], -1) \
                #     #     .gather(1, labels).squeeze().mean(0)
                #     #
                #     # preds = similarity[~mask].argmax(-1)
                #
                #     # losses[:len(loss)] = loss
                #
                #     # loss = -torch.log_softmax(similarity[~mask], -1)\
                #     #     .gather(1, labels).squeeze()
                #     # loss = loss[~mask]
                #
                #     # classification_losses.append(loss)
                #
                #     d = torch.log_softmax(similarity[mask], -1) \
                #         .index_select(-1, torch.unique(labels))
                #     d = d.squeeze().mean(-1).mean(0)
                #
                #     distance_losses.append(d)
                #
                #     # losses[len(loss):] = d
                #     # a = 0

        # else:
        #     outputs = {}
        #     past_output = self.past_model(strategy.mb_x, strategy.mb_task_id)
        #
        #     for name, module in self.past_model.named_modules():
        #         if isinstance(module, (MoERoutingLayer,
        #                                AbsDynamicLayer)):
        #             weights, similarity = module.last_distribution
        #             outputs[name] = weights
        #
        #     mask = strategy.mb_task_id == tid
        #
        #     for name, module in strategy.model.named_modules():
        #         if isinstance(module, (MoERoutingLayer,
        #                                AbsDynamicLayer)):
        #             weights, similarity = module.last_distribution
        #
        #             if strategy.is_finetuning and tid > 0:
        #                 distr = torch.log_softmax(similarity, -1)
        #                 distr = distr * outputs[name]
        #                 distr = distr.sum(-1) / outputs[name].sum(-1)
        #                 loss = - distr
        #
        #                 classification_losses.append(loss)
        #             else:
        #                 distr = torch.softmax(similarity, -1)
        #                 h = -(distr.log() * distr).sum(-1) / np.log(
        #                     weights.shape[-1])
        #                 if tid > 0:
        #                     h = h[mask]
        #                 entropies.append(-h)
        #
        #                 if tid > 0:
        #                     sim = -torch.log_softmax(similarity[~mask], -1)
        #                     past_selection = outputs[name][~mask]
        #
        #                     sim = sim * past_selection
        #                     loss = sim.sum(-1) / past_selection.sum(-1)
        #                     classification_losses.append(loss)
        #
        #                     labels = [module.get_task_blocks(t.item())
        #                               for t in strategy.mb_task_id[~mask]]
        #                     labels = torch.stack(labels, 0)
        #                     # loss = -torch.log_softmax(similarity[~mask], -1) \
        #                     #     .gather(1, labels).squeeze()
        #                     # loss = loss[~mask]
        #
        #                     d = torch.log_softmax(similarity[mask], -1) \
        #                         .index_select(-1, torch.unique(labels))
        #                     # d = -torch.log_softmax(similarity[~mask], -1).mean(-1)
        #                     distance_losses.append(d.mean(-1))

        if len(entropies) > 0:
            entropy = sum(entropies) / len(entropies)
            # entropy = sum(entropies)
            # entropy = entropy.mean(0)
        else:
            entropy = 0

        if len(classification_losses) > 0:
            class_loss = sum(classification_losses) / len(
                classification_losses)
            class_loss = class_loss.mean(0)
        else:
            class_loss = 0

        if len(distance_losses) > 0:
            distance_loss = sum(distance_losses) / len(distance_losses)
            distance_loss = distance_loss.mean(0)
        else:
            distance_loss = 0

        strategy.loss += entropy * 1e-4 + class_loss * 0 + distance_loss * 100

        return

        # return
        # self._bb_sigmoid(strategy)
        self._bb_kl_div(strategy)
        # self.___before_backward(strategy)

    # def _before_backward_distance(self, strategy, *args, **kwargs):
    #     tid = strategy.experience.current_experience
    #
    #     entropies = []
    #     classification_losses = []
    #     distance_losses = []
    #     mask = strategy.mb_task_id == tid
    #
    #     # if tid == 0 and strategy.is_finetuning:
    #     #     return
    #
    #     if tid > 0:
    #         self.past_model.eval()
    #
    #         _x = strategy.mb_x[~mask]
    #
    #         with torch.no_grad():
    #             self.past_model(_x, strategy.mb_task_id[~mask])
    #             past_routing = {}
    #
    #             for name, module in self.past_model.named_modules():
    #                 if isinstance(module, (AbsDynamicLayer)):
    #                     past_routing[name] = module.current_routing
    #
    #         d = 0
    #         for name, module in strategy.model.named_modules():
    #             if isinstance(module, (AbsDynamicLayer)):
    #                 similarity = cosine_similarity(
    #                     module.current_routing[~mask],
    #                     past_routing[name])
    #                 distance = 1 - similarity
    #                 distance = distance.mean(0)
    #
    #                 # distance = nn.functional.mse_loss(
    #                 #     module.current_routing[~mask],
    #                 #     past_routing[name])
    #                 d += distance
    #
    #         distance = d / 3
    #
    #         strategy.loss += distance * 20
    #
    #     # if not self.per_sample_routing_reg or tid == 0:
    #
    #     for name, module in strategy.model.named_modules():
    #         if isinstance(module, (AbsDynamicLayer)):
    #             weights, similarity = module.last_distribution
    #
    #             # if strategy.is_finetuning:
    #             #     toiter =
    #             # else:
    #             #     toiter = range(tid)
    #             # if not strategy.is_finetuning:
    #             #     distr = torch.softmax(similarity[mask].mean(0), -1)
    #             #     h = -(distr.log() * distr).sum(-1) / np.log(
    #             #         distr.shape[-1])
    #             #     h = -h
    #             #     entropies.append(h)
    #             # for i in range(tid + 1):
    #             #     distr = torch.softmax(
    #             #         similarity[strategy.mb_task_id == i].mean(0), -1)
    #             #     h = -(distr.log() * distr).sum(-1) / np.log(
    #             #         distr.shape[-1])
    #             #
    #             #     if i == tid and not strategy.is_finetuning:
    #             #         h = -h
    #             #
    #             #     es += h
    #             #
    #             # es = es / (tid + 1)
    #             #
    #             # entropies.append(es)
    #
    #             if not strategy.is_finetuning:
    #                 current_distr = torch.softmax(similarity[mask].mean(0), -1)
    #                 current_h = -(current_distr.log() * current_distr).sum(
    #                     -1) / np.log(current_distr.shape[-1])
    #
    #                 entropies.append(current_h)
    #
    #             if tid > 0 and not strategy.is_finetuning:
    #                 # labels = [module.get_task_blocks(t.item())
    #                 #           for t in strategy.mb_task_id[~mask]]
    #                 # d = torch.log_softmax(similarity[mask], -1) \
    #                 #     .index_select(-1, torch.unique(labels))
    #                 # d = d.squeeze().mean(-1).mean(0)
    #                 #
    #                 # distance_losses.append(d)
    #                 #
    #                 #
    #                 # if tid > 0 or strategy.is_finetuning:
    #                 #     if strategy.is_finetuning:
    #                 #         labels = [module.get_task_blocks(t.item())
    #                 #                   for t in strategy.mb_task_id]
    #                 #         sim = similarity
    #                 #     else:
    #                 #         labels = [module.get_task_blocks(t.item())
    #                 #                   for t in strategy.mb_task_id[~mask]]
    #                 #         sim = similarity[~mask]
    #                 #
    #                 #         past_routing = torch.unique(torch.cat(labels, 0))
    #                 #         dist_labels = nn.functional.one_hot(past_routing,
    #                 #                                        similarity.shape[-1]).float()
    #                 #
    #                 #         dist_labels = 1 - dist_labels
    #                 #         dist_labels = dist_labels / dist_labels.sum()
    #                 #         loss = nn.functional.kl_div(torch.log_softmax(similarity[mask], -1),
    #                 #                                     dist_labels)
    #                 #         distance_losses.append(loss)
    #                 #
    #                 #     labels = torch.cat(labels, 0)
    #                 #     labels = nn.functional.one_hot(labels,
    #                 #                                    similarity.shape[-1]).float()
    #                 #
    #                 #     loss = nn.functional.kl_div(torch.log_softmax(sim, -1),
    #                 #                          labels)
    #                 #
    #                 #     # distr = torch.log_softmax(similarity, -1)
    #                 #     # loss = -distr.gather(1, labels).squeeze()
    #                 #
    #                 #     # loss[~mask] *= 10
    #                 #
    #                 #     # loss = nn.functional.cross_entropy(similarity,
    #                 #     #                                    labels.squeeze())
    #                 #     classification_losses.append(loss)
    #
    #                 # elif tid > 0:
    #                 #     # current_distr = torch.softmax(similarity[mask].mean(0), -1)
    #                 #     # current_h = -(current_distr.log() * current_distr).sum(-1) \
    #                 #     #             / np.log(current_distr.shape[-1])
    #                 #     #
    #                 #     # # current_distr = torch.softmax(similarity[~mask].mean(0), -1)
    #                 #     #
    #                 #     # entropies.append(-current_h)
    #                 #
    #                 #     labels = [module.get_task_blocks(t.item())
    #                 #               for t in strategy.mb_task_id[~mask]]
    #                 #     labels = torch.stack(labels, 0)
    #                 #     loss = -torch.log_softmax(similarity[~mask], -1) \
    #                 #         .gather(1, labels).squeeze().mean(0)
    #                 #
    #                 #     classification_losses.append(loss)
    #                 #
    #                 #     # d = torch.log_softmax(similarity[mask], -1) \
    #                 #     #     .index_select(-1, torch.unique(labels))
    #                 #     # d = d.squeeze().mean(-1).mean(0)
    #                 #     #
    #                 #     # distance_losses.append(d)
    #                 #
    #                 centroids = [
    #                     module.current_routing[strategy.mb_task_id == i].mean(0)
    #                     for i in range(tid + 1)]
    #             #     centroids = torch.stack(centroids, 0)
    #             #
    #             #     # distances = nn.PairwiseDistance(p=2)(centroids, centroids)
    #             #     distances = torch.cdist(centroids, centroids)
    #             #     distances = torch.triu(distances)
    #             #
    #             #     d = (len(centroids) - 1) * len(centroids)
    #             #     d = d / 2
    #             #
    #             #     distances = distances.sum() / d
    #             #     distances = 1 / (distances + 1)
    #             #
    #             #     distance_losses.append(distances)
    #
    #     # else:
    #     #     outputs = {}
    #     #     past_output = self.past_model(strategy.mb_x, strategy.mb_task_id)
    #     #
    #     #     for name, module in self.past_model.named_modules():
    #     #         if isinstance(module, (MoERoutingLayer,
    #     #                                AbsDynamicLayer)):
    #     #             weights, similarity = module.last_distribution
    #     #             outputs[name] = weights
    #     #
    #     #     mask = strategy.mb_task_id == tid
    #     #
    #     #     for name, module in strategy.model.named_modules():
    #     #         if isinstance(module, (MoERoutingLayer,
    #     #                                AbsDynamicLayer)):
    #     #             weights, similarity = module.last_distribution
    #     #
    #     #             if strategy.is_finetuning and tid > 0:
    #     #                 distr = torch.log_softmax(similarity, -1)
    #     #                 distr = distr * outputs[name]
    #     #                 distr = distr.sum(-1) / outputs[name].sum(-1)
    #     #                 loss = - distr
    #     #
    #     #                 classification_losses.append(loss)
    #     #             else:
    #     #                 distr = torch.softmax(similarity, -1)
    #     #                 h = -(distr.log() * distr).sum(-1) / np.log(
    #     #                     weights.shape[-1])
    #     #                 if tid > 0:
    #     #                     h = h[mask]
    #     #                 entropies.append(-h)
    #     #
    #     #                 if tid > 0:
    #     #                     sim = -torch.log_softmax(similarity[~mask], -1)
    #     #                     past_selection = outputs[name][~mask]
    #     #
    #     #                     sim = sim * past_selection
    #     #                     loss = sim.sum(-1) / past_selection.sum(-1)
    #     #                     classification_losses.append(loss)
    #     #
    #     #                     labels = [module.get_task_blocks(t.item())
    #     #                               for t in strategy.mb_task_id[~mask]]
    #     #                     labels = torch.stack(labels, 0)
    #     #                     # loss = -torch.log_softmax(similarity[~mask], -1) \
    #     #                     #     .gather(1, labels).squeeze()
    #     #                     # loss = loss[~mask]
    #     #
    #     #                     d = torch.log_softmax(similarity[mask], -1) \
    #     #                         .index_select(-1, torch.unique(labels))
    #     #                     # d = -torch.log_softmax(similarity[~mask], -1).mean(-1)
    #     #                     distance_losses.append(d.mean(-1))
    #
    #     if len(entropies) > 0:
    #         entropy = sum(entropies) / len(entropies)
    #         entropy = entropy.mean(0)
    #     else:
    #         entropy = 0
    #
    #     if len(classification_losses) > 0:
    #         class_loss = sum(classification_losses) / len(
    #             classification_losses)
    #         class_loss = class_loss.mean(0)
    #     else:
    #         class_loss = 0
    #
    #     if len(distance_losses) > 0:
    #         distance_loss = sum(distance_losses) / len(distance_losses)
    #         distance_loss = distance_loss.mean(0)
    #     else:
    #         distance_loss = 0
    #
    #     strategy.loss += entropy * 0.01 + class_loss * 1 + distance_loss * 1
    #
    #     return
    #
    #     # return
    #     # self._bb_sigmoid(strategy)
    #     self._bb_kl_div(strategy)
    #     # self.___before_backward(strategy)


class Trainer(SupervisedTemplate):
    def __init__(self, model,
                 optimizer: Optimizer, criterion,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 top_k: int = 1,
                 fine_tuning_epochs: int = 1,
                 eval_mb_size: int = None, device=None,
                 plugins=None,

                 evaluator: EvaluationPlugin = default_evaluator, eval_every=-1,
                 annealing_epochs=3,
                 contrastive_w=0.1, contrastive_margin=0.5,
                 tau_decay=0.95, initial_tau=10, final_tau=1
                 ):

        rp = CentroidsMatching(annealing_epochs=annealing_epochs,
                               top_k=top_k,
                               fine_tuning_epochs=fine_tuning_epochs,
                               contrastive_w=contrastive_w,
                               contrastive_margin=contrastive_margin,
                               tau_decay=tau_decay, initial_tau=initial_tau,
                               final_tau=final_tau)

        self.is_finetuning = False
        self.base_plugin = rp

        self.main_pi = rp

        if plugins is None:
            plugins = [rp]
        else:
            plugins.append(rp)

        self.fine_tuning_epochs = fine_tuning_epochs
        self.dev_split_size = 100

        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device,
            plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)

    def _after_finetuning_exp(self, **kwargs):
        trigger_plugins(self, "after_finetuning_exp", **kwargs)

    def train(
            self,
            experiences: Union[CLExperience, ExpSequence],
            eval_streams: Optional[
                Sequence[Union[CLExperience, ExpSequence]]
            ] = None,
            **kwargs,
    ):

        self.is_training = True
        self._stop_training = False
        self.is_finetuning = False

        self.model.train()
        self.model.to(self.device)

        # Normalize training and eval data.
        if not isinstance(experiences, Iterable):
            experiences = [experiences]
        if eval_streams is None:
            eval_streams = [experiences]

        self._eval_streams = _group_experiences_by_stream(eval_streams)

        self._before_training(**kwargs)

        for self.experience in experiences:
            self._before_training_exp(**kwargs)
            self._train_exp(self.experience, eval_streams, **kwargs)
            self._after_training_exp(**kwargs)

            self.make_optimizer()
            self._finetune_exp(self.experience, eval_streams, **kwargs)
            self._after_finetuning_exp(**kwargs)

        self._after_training(**kwargs)

    def training_epoch_finetuning(self, **kwargs):
        for self.mbatch in self.dataloader:
            if self._stop_training:
                break

            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            self.optimizer.zero_grad()
            self.loss = 0

            # Forward
            self._before_forward(**kwargs)
            self.mb_output = self.forward()
            self._after_forward(**kwargs)

            # Loss & Backward
            self.loss += self.criterion()

            self._before_backward(**kwargs)
            self.backward()
            self._after_backward(**kwargs)

            # Optimization step
            self._before_update(**kwargs)
            self.optimizer_step()
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)

    def _finetune_exp(self, experience, eval_streams=None, **kwargs):
        self.is_finetuning = True

        if eval_streams is None:
            eval_streams = [experience]
        for i, exp in enumerate(eval_streams):
            if not isinstance(exp, Iterable):
                eval_streams[i] = [exp]

        for _ in range(self.fine_tuning_epochs):
            self._before_training_epoch(**kwargs)

            if self._stop_training:  # Early stopping
                self._stop_training = False
                break

            self.training_epoch(**kwargs)
            self._after_training_epoch(**kwargs)

    def train_dataset_adaptation(self, **kwargs):
        """ Initialize `self.adapted_dataset`. """

        if not hasattr(self.experience, 'dev_dataset'):
            dataset = self.experience.dataset

            idx = np.arange(len(dataset))
            np.random.shuffle(idx)

            if isinstance(self.dev_split_size, int):
                dev_i = self.dev_split_size
            else:
                dev_i = int(len(idx) * self.dev_split_size)

            dev_idx = idx[:dev_i]
            train_idx = idx[dev_i:]

            self.experience.dataset = dataset.train().subset(train_idx)
            # self.experience.dev_dataset = dataset.eval().subset(dev_idx)
            self.experience.dev_dataset = dataset.train().subset(dev_idx)

        self.adapted_dataset = self.experience.dataset


class CustomBlockModel(MultiTaskModule):
    def __init__(self):
        super().__init__()

        self.current_features = None
        self.current_routing = None
        self.hidden_features = None

        # self.embeddings = nn.Embedding(2, 100)
        self.embeddings = None

        self.rho = torch.nn.init.orthogonal_(torch.empty(100, 256)).to(device)

        self.layers = nn.ModuleList()

        self.layers.append(RoutingLayer(3, 32))
        self.layers.append(RoutingLayer(32, 64))
        self.layers.append(RoutingLayer(64, 128))

        self.mx = nn.AdaptiveAvgPool2d(2)
        self.in_features = 128 * 4

        self.task_classifier = None
        self.p = nn.Sequential(nn.Linear(512, 256),
                               nn.ReLU(),
                               nn.Linear(256, 256))

        self.classifiers = IncrementalClassifier(self.in_features,
                                                 initial_out_features=2,
                                                 masking=False)

        self.classifiers = MultiHeadClassifier(self.in_features,
                                               initial_out_features=2,
                                               masking=False)

        self.classifiers = CumulativeMultiHeadClassifier(self.in_features,
                                                         initial_out_features=2, )

        layers_blocks = [len(l.blocks) for l in self.layers]
        paths = []

        while len(paths) < 20:
            b = [np.random.randint(0, l) for l in layers_blocks]
            if paths not in b:
                paths.append(b)

                for layer, block_id in zip(self.layers, b):
                    layer.add_path(block_id)

        self.available_paths = len(self.layers[0].paths)

        # self.routing_model = torchvision.models.resnet18(pretrained=True)
        # self.routing_model.fc = EmptyModule()

        # self.routing_model = torch.hub.load('pytorch/vision:v0.10.0',
        #                                     'resnet18', pretrained=True)
        #
        # class empty(nn.Module):
        #     def __init__(self, *args, **kwargs):
        #         super().__init__(*args, **kwargs)
        #
        #     def forward(self, x):
        #         return x
        #
        # self.routing_model.fc = empty()

    def train_adaptation(self, experience):
        available_paths = [len(l.paths) for l in self.layers]
        if len(set(available_paths)) > 1:
            raise ValueError('Unbalanced paths')

        self.available_paths = available_paths[0]

    def calculate_available_paths(self):
        available_paths = [len(l.paths) for l in self.layers]
        if len(set(available_paths)) > 1:
            raise ValueError('Unbalanced paths')

        self.available_paths = available_paths[0]

    # def adaptation(self, dataset: CLExperience):
    #     if self.task_classifier is None:
    #         self.task_classifier = nn.Linear(256, 100)
    #     else:
    #         in_features = self.task_classifier.in_features
    #         old_nclasses = self.task_classifier.out_features
    #         new_nclasses = dataset.task_label + 1
    #
    #         if new_nclasses <= old_nclasses:
    #             return
    #
    #         old_w, old_b = \
    #             self.task_classifier.weight, \
    #                 self.task_classifier.bias
    #
    #         self.task_classifier = torch.nn.Linear(in_features, new_nclasses)
    #         self.task_classifier.weight.data[:old_nclasses] = old_w.data
    #         self.task_classifier.bias.data[:old_nclasses] = old_b.data

    #
    #     self.embeddings.append(e)
    #
    #     """ If `dataset` contains new tasks, a new head is initialized.
    #
    #     :param dataset: data from the current experience.
    #     :return:
    #     """
    #
    #     super().adaptation(dataset)
    # task_labels = dataset.task_labels
    # if isinstance(task_labels, ConstantSequence):
    #     # task label is unique. Don't check duplicates.
    #     task_labels = [task_labels[0]]
    #
    # for tid in set(task_labels):
    #     tid = str(tid)  # need str keys
    #     if tid not in self.classifiers:
    #         self.classifiers[tid] = nn.Linear(self.in_features,
    #                                           len(set(
    #                                               dataset.classes_in_this_experience)) * 10)
    #         self.tasks_classes = len(
    #             set(dataset.classes_in_this_experience))

    def forward(self, x: torch.Tensor, task_labels: torch.Tensor, **kwargs) \
            -> torch.Tensor:

        if isinstance(task_labels, int):
            # fast path. mini-batch is single task.
            return self.forward_single_task(x, task_labels, **kwargs)
        else:
            unique_tasks = torch.unique(task_labels)

        out = None
        features = None

        for task in unique_tasks:
            task_mask = task_labels == task
            x_task = x[task_mask]
            out_task = self.forward_single_task(x_task,
                                                task.item(),
                                                **kwargs)

            if out is None:
                out = torch.empty(x.shape[0], *out_task.shape[1:],
                                  device=out_task.device)
                features = torch.empty(x.shape[0],
                                       *self.current_features.shape[1:],
                                       device=out_task.device)

            out[task_mask] = out_task
            features[task_mask] = self.current_features

        self.current_features = features

        # return out / torch.norm(out, 2, -1, keepdim=True)
        return out

    def features(self, x, task_labels=None, path_id=None, **kwargs):

        for l in self.layers[:-1]:
            x = torch.relu(l(x, task_labels, path_id))

        x = torch.relu(self.layers[-1](x, task_labels, path_id))

        x = self.mx(x).flatten(1)

        self.current_features = x

        return x

    def pp(self, x, task_labels=None, path_id=None, **kwargs):
        x = self.features(x, task_labels, path_id)
        return self.p(x)

    def forward_single_task(self, x, task_labels, **kwargs):

        features = self.features(x, task_labels)

        logits = self.classifiers(features, task_labels=task_labels)
        return logits

    def get_random_paths(self, n_paths=1,
                         replacement=False):
        return np.random.choice(np.asarray(self.available_paths),
                                size=n_paths,
                                replace=replacement)

    def random_path_forward(self,
                            x,
                            n_forwards=1,
                            replacement=False,
                            return_intermediate_outputs=False):

        # paths = self.available_paths

        paths = np.random.choice(np.asarray(self.available_paths),
                                 size=n_forwards,
                                 replace=replacement)

        results = [self.features(x, None, p) for p in paths]

        # for p in paths:
        #     inter = []
        #     _x = x
        #     for l in self.layers[:-1]:
        #         _x = l(_x, path_id=p)
        #         inter.append(_x)
        #
        #         _x = torch.relu(_x)
        #
        #     _x = self.layers[-1](_x, path_id=p)
        #     inter.append(_x)
        #
        #     _x = self.mx(_x)
        #
        #     results.append(_x)
        #
        #     # if return_intermediate_outputs:
        #     #     return x, inter

        return results


torch.manual_seed(0)
# random.seed(0)
np.random.seed(0)

device = '0'

if torch.cuda.is_available() and device != 'cpu':
    device = 'cuda:{}'.format(device)
    torch.cuda.set_device(device)
else:
    device = 'cpu'

device = torch.device(device)


def train(train_stream, test_stream, theta=1, train_epochs=1,
          fine_tune_epochs=1):
    complete_model = CustomBlockModel()

    # x = torch.randn(500, 3, 32, 32)
    # random_features = [f.detach().cpu().numpy()
    #                    for f in complete_model.random_path_forward(
    #         x, 10)]
    #
    # all_features = np.concatenate(random_features, 0)

    # all_features = np.concatenate(([features], random_features), 0)
    # all_features = all_features.reshape(6 * len(x), -1)

    # ys = [0] * len(all_features)
    #
    # for j in range(len(test_stream[:current_task_id + 1])):
    #     # if j == current_task_id:
    #     #     continue
    #
    #     features = complete_model.features(x.to(device), j).reshape(
    #         len(x), -1)
    #     # sims = nn.functional.mse_loss(features, centroids[j])
    #     sims = torch.norm(features - centroids[j], 2, -1).mean(0)
    #     print(i, j, sims)
    #
    #     features = features.cpu().numpy()
    #     ys += [j + 1] * len(features)
    #
    #     all_features = np.concatenate((all_features, features), 0)
    #
    # ys = np.asarray(ys)
    #
    # cc = centroids[i].cpu().numpy()
    # all_features = np.concatenate((all_features, cc.reshape(1, -1)),
    #                               0)

    # embs_t = TSNE(n_jobs=-1).fit_transform(all_features)
    #
    # # embs_t, cc = embs_t[:-1], embs_t[-1]
    #
    # fig, ax = plt.subplots()
    #
    # # for y in np.unique(ys):
    # scatter = plt.scatter(embs_t[:, 0],
    #                       embs_t[:, 1],
    #                       # color=colors[y],
    #                       s=5,
    #                       # label='Neg' if y == 0 else str(y)
    #                       )

    # legend1 = ax.legend(handles=scatter.legend_elements()[0],
    #     # handles=*scatter.legend_elements(),
    #                     labels=['Neg'] + list(map(str, range(current_task_id + 1))),
    #                     loc="lower left",
    #                     title="Classes")

    # scatter = plt.scatter(cc[0], cc[1], c='k', s=50, marker='X')

    # ax.add_artist(legend1)
    # ax.legend()
    # # fig.suptitle(f'TSNE on task {i} after task {current_task_id}')
    # plt.show()

    # print(complete_model.embeddings.weight)

    criterion = CrossEntropyLoss()

    optimizer = Adam(complete_model.parameters(), lr=0.001)
    # optimizer = SGD(complete_model.parameters(), lr=0.1)

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=False, epoch=False, experience=True,
                         stream=True),
        bwt_metrics(experience=True, stream=True),
        loggers=[
            TextLogger(),
            # CustomTextLogger(),
        ],
    )

    colors = distinctipy.get_colors(len(train_stream) + 1)

    trainer = Trainer(model=complete_model, optimizer=optimizer,
                      train_mb_size=32, top_k=2,
                      eval_mb_size=32, evaluator=eval_plugin,
                      initial_tau=10, fine_tuning_epochs=fine_tune_epochs,
                      criterion=criterion, device=device,
                      train_epochs=train_epochs)

    for current_task_id, tr in enumerate(train_stream):
        trainer.train(tr)
        trainer.eval(test_stream[:current_task_id + 1])

        for l in complete_model.layers:
            print(l.tasks_path)

        centroids = trainer.base_plugin.centroids
        with torch.no_grad():

            m = np.zeros((current_task_id + 1, current_task_id + 1))
            for i in range(len(test_stream[:current_task_id + 1])):
                x, y, t = next(iter(DataLoader(test_stream[i].dataset,
                                               batch_size=512,
                                               shuffle=True)))

                for j in range(len(test_stream[:current_task_id + 1])):
                    pred = complete_model(x.to(device), j)
                    # pred = pred[:, test_stream[i].classes_in_this_experience]

                    print(i, j, pred.mean(0))

                    preds = torch.softmax(pred, -1)
                    h = -(preds.log() * preds).sum(-1)
                    h = h / np.log(preds.shape[-1])

                    m[i, j] = h.mean().item()
                    print(h.mean())

            if current_task_id > 0:

                fig, ax = plt.subplots()
                im = ax.matshow(m)
                threshold = im.norm(m.max()) / 2.

                for i in range(len(m)):
                    for j in range(len(m)):
                        c = m[i, j]
                        ax.text(i, j, f'{c:.2f}', va='center', ha='center',
                                color=("black", "white")[int(c < 0.5)])

                plt.colorbar(im)
                plt.ylabel('Source task')
                plt.xlabel('Target head')
                plt.show()

            # for j in range(len(test_stream[:current_task_id + 1])):
            #     all_features = None
            #     ys = []
            #     for i in range(len(test_stream[:current_task_id + 1])):
            #         x, y, t = next(iter(DataLoader(test_stream[i].dataset,
            #                                        batch_size=512,
            #                                        shuffle=False)))
            #
            #         # # features = complete_model.features(x.to(device), i).cpu().numpy()
            #         # random_features = [f.cpu().numpy().reshape(len(x), -1)
            #         #                    for f in complete_model.random_path_forward(
            #         #         x.to(device), 2)]
            #         # all_features = np.concatenate(random_features, 0)
            #
            #         # all_features = np.concatenate(([features], random_features), 0)
            #         # all_features = all_features.reshape(6 * len(x), -1)
            #
            #         # ys = [0] * len(all_features)
            #
            #     # for j in range(len(test_stream[:current_task_id + 1])):
            #         # if j == current_task_id:
            #         #     continue
            #         pred = complete_model(x.to(device), j)
            #         print(pred.mean(0))
            #
            #         features = complete_model.pp(x.to(device), j)
            #         # sims = nn.functional.mse_loss(features, centroids[j])
            #         sims = torch.norm(features - centroids[j], 2, -1).mean(0)
            #         print(i, j, sims)
            #
            #         features = features.cpu().numpy()
            #         ys += [i + 1] * len(features)
            #
            #         if all_features is None:
            #             all_features=features
            #         else:
            #             all_features = np.concatenate((all_features, features), 0)
            #
            #     ys = np.asarray(ys)
            #
            #     cc = centroids[i].cpu().numpy()
            #     all_features = np.concatenate((all_features, cc.reshape(1, -1)),
            #                                   0)
            #
            #     embs_t = TSNE(n_jobs=-1).fit_transform(all_features)
            #
            #     embs_t, cc = embs_t[:-1], embs_t[-1]
            #
            #     fig, ax = plt.subplots()
            #
            #     for y in np.unique(ys):
            #         scatter = plt.scatter(embs_t[ys == y, 0],
            #                               embs_t[ys == y, 1],
            #                               color=colors[y],
            #                               s=5,
            #                               label='Neg' if y == 0 else str(y))
            #
            #         # legend1 = ax.legend(handles=scatter.legend_elements()[0],
            #         #     # handles=*scatter.legend_elements(),
            #         #                     labels=['Neg'] + list(map(str, range(current_task_id + 1))),
            #         #                     loc="lower left",
            #         #                     title="Classes")
            #
            #     scatter = plt.scatter(cc[0], cc[1], c='k', s=50, marker='X')
            #
            #     # ax.add_artist(legend1)
            #     ax.legend()
            #     fig.suptitle(f'TSNE on path {j} after task {current_task_id}')
            #     plt.show()

        continue

        trainer.model.eval()

        indexes = defaultdict(list)

        for n, m in trainer.model.named_buffers():
            if 'idx' not in n or 'global' in n:
                continue
            n = n.rsplit('.', 1)[0]
            indexes[n].extend(m.tolist())

        for k, v in indexes.items():
            print(k, collections.Counter(v))

        # if current_task_id == 0:
        #     continue

        with torch.no_grad():
            for i in range(len(test_stream[:current_task_id + 1])):
                # x, y, t = next(iter(DataLoader(test_stream[i].dataset,
                #                                batch_size=128,
                #                                shuffle=True)))
                x, y, t = next(
                    iter(DataLoader(trainer.base_plugin.past_dataset[i],
                                    batch_size=128,
                                    shuffle=True)))
                x = x.to(device)

                # for _ in range(5):
                pred = trainer.model(x, t.to(device))
                # print(torch.softmax(pred, -1))

                # print(trainer.model.
                #       task_classifier(trainer.model.current_routing).mean(0))
                for module in trainer.model.modules():
                    if isinstance(module, AbsDynamicLayer):
                        print(i, module.last_distribution[0].mean(0),
                              module.get_task_blocks(i))
                        # print(i, j, module.last_distribution[1].mean(0))
                print()

            for i in range(len(test_stream[:current_task_id + 1])):
                x, y, t = next(iter(DataLoader(test_stream[i].dataset,
                                               batch_size=128,
                                               shuffle=True)))
                # x, y, t = next(iter(DataLoader(trainer.base_plugin.past_dataset[i],
                #                                batch_size=128,
                #                                shuffle=True)))
                x = x.to(device)

                # for _ in range(5):
                pred = trainer.model(x, t.to(device))
                # print(torch.softmax(pred, -1))

                # print(trainer.model.
                #       task_classifier(trainer.model.current_routing).mean(0))
                for module in trainer.model.modules():
                    if isinstance(module, AbsDynamicLayer):
                        print(i, module.last_distribution[0].mean(0),
                              module.get_task_blocks(i))
                        # print(i, j, module.last_distribution[1].mean(0))
                print()

        a = 0
        # print(complete_model.embeddings.weight)

        # if current_task_id > 0:
        #     plot_probs(complete_model, train_stream[:current_task_id+1], test_stream[:current_task_id+1])
        #     evaluate_max_prob(complete_model, train_stream[:current_task_id+1], test_stream[:current_task_id+1])

    # print(eval_plugin.last_metric_results)

    indexes = defaultdict(list)

    for n, m in trainer.model.named_buffers():
        if 'idx' not in n or 'global' in n:
            continue
        n = n.rsplit('.', 1)[0]
        indexes[n].extend(m.tolist())

    for k, v in indexes.items():
        print(k, collections.Counter(v))

    print(sum(
        p.numel() for p in complete_model.parameters()))

    return complete_model


if __name__ == '__main__':
    cifar_train, cifar_test = get_cifar10_dataset(None)
    _default_cifar10_train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    _default_cifar10_eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    perm_mnist = nc_benchmark(
        train_dataset=cifar_train,
        test_dataset=cifar_test,
        n_experiences=5,
        shuffle=True,
        task_labels=True,
        class_ids_from_zero_in_each_exp=False,
        class_ids_from_zero_from_first_exp=True,
        train_transform=_default_cifar10_train_transform,
        eval_transform=_default_cifar10_eval_transform)

    cifar10_train_stream = perm_mnist.train_stream
    cifar10_test_stream = perm_mnist.test_stream

    train_tasks = cifar10_train_stream
    test_tasks = cifar10_test_stream

    model = train(train_epochs=5,
                  fine_tune_epochs=0,
                  train_stream=train_tasks,
                  test_stream=test_tasks)

    # print(model.embeddings.weight)

    with torch.no_grad():
        for i in range(len(train_tasks)):
            x, y, t = next(iter(DataLoader(test_tasks[i].dataset,
                                           batch_size=128,
                                           shuffle=True)))
            x = x.to(device)

            # for _ in range(5):
            pred = model(x, t)
            # print(torch.softmax(pred, -1))

            for module in model.modules():
                if isinstance(module, AbsDynamicLayer):
                    print(i, module.last_distribution[0].mean(0),
                          module.get_task_blocks(i))
                    # print(i, j, module.last_distribution[1].mean(0))
                    print()

                # for module in model.modules():
                #     if isinstance(module, MoERoutingLayer):
                #         print(
                #             f'Selection dataset {i} using task routing {j}',
                #             torch.unique(
                #                 module.last_distribution[0],
                #                 return_counts=True))
