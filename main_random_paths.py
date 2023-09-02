from builtins import enumerate
from copy import deepcopy
from itertools import chain
from typing import Optional, Sequence, Iterable, Union

from avalanche.training import ExperienceBalancedBuffer
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
        self.patterns_per_experience = 100
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

        self.storage_policy = ExperienceBalancedBuffer(
            max_size=200, adaptive_size=True)

    # def before_train_dataset_adaptation(self, strategy, *args, **kwargs):
    #     # if self.per_sample_routing_reg:
    #     if strategy.experience.current_experience == 0:
    #         return
    #
    #     self.past_model = deepcopy(strategy.model)

    def before_training_exp(self, strategy: SupervisedTemplate,
                            **kwargs):
        self.past_model = deepcopy(strategy.model)

        classes = strategy.experience.classes_in_this_experience
        self.tasks_nclasses[strategy.experience.task_label] = classes

        tid = strategy.experience.current_experience

        if tid > 0:
            self.past_model = clone_module(strategy.model)

        paths = strategy.model.available_paths
        selected_path = np.random.randint(0, paths)

        for layer in strategy.model.layers:
            p_id = layer.get_unassigned_paths()[selected_path]
            layer.assign_path_to_task(p_id, tid)

        strategy.model.calculate_available_paths()

        if tid > 0:
            # av = concat_datasets(list(self.past_dataset.values()))
            self.av = self.storage_policy.buffer
            
            strategy.dataloader = ReplayDataLoader(
                strategy.adapted_dataset,
                self.storage_policy.buffer,
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

    @torch.no_grad()
    def after_training_exp(self, strategy, **kwargs):
        self.storage_policy.update(strategy, **kwargs)
        # return
        tid = strategy.experience.current_experience

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

    def _before_backward(self, strategy, *args, **kwargs):
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

        matrix = torch.zeros(len(all_tasks), len(all_tasks), device=x.device)
        loss = 0

        ot_x, ot_y, ot_t = next(
            iter(DataLoader(self.av, batch_size=strategy.train_mb_size,
                            shuffle=True)))

        ot_x = ot_x.to(x.device)

        all_loss = 0
        for t in torch.unique(ot_t):
            if t == tid:
                continue

            task_loss = 0

            t_mask = ot_t == t
            x_t = ot_x[t_mask]

            preds = strategy.model(x_t, tid, mask=False)
            preds = torch.softmax(preds, -1)

            h = -(preds.log() * preds)
            h = h.sum(-1) / np.log(preds.shape[-1])

            loss += (1 - h).sum()

        loss = loss.mean() / (len(x) // 2)
        # distance = matrix - torch.eye(len(matrix), device=x.device)
        # norm = torch.linalg.matrix_norm(distance)
        strategy.loss += loss * 1

        # all_loss = all_loss / (tid + 1)
        # strategy.loss += all_loss * 1

        # mask = tids != tid
        # x = x[mask]
        # tids = tids[mask]

        with torch.no_grad():
            past_logits = self.past_model(ot_x, ot_t)

        strategy.model(ot_x, ot_t)
        # current_logits = strategy.model(ot_x, ot_t)
        #
        # loss = nn.functional.mse_loss(current_logits,
        #                               past_logits)

        loss = nn.functional.cosine_similarity(strategy.model.current_features,
                                               self.past_model.current_features,
                                               -1).mean()
        # loss = (loss + 1) / 2
        loss = 1 - loss
        # loss = loss.mean()

        strategy.loss += loss * 1

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

        matrix = torch.zeros(len(all_tasks), len(all_tasks), device=x.device)

        # all_outs = torch.cat([strategy.model(x, i) for i in range(tid)], -1)

        entropies = []
        for _x, _t in zip(x, tids):
            _t = _t.item()
            logits = [strategy.model(_x[None], t)
                      for t in range(tid + 1) if t != _t]

            logits = torch.cat(logits, -1)
            logits = torch.softmax(logits, -1)

            h = -(logits.log() * logits)
            h = -h.sum(-1)
            entropies.append(h)

        entropies = torch.cat(entropies, 0)
        loss = entropies.mean()

        strategy.loss += loss * 10

        # all_loss = 0
        # for t in all_tasks:
        #     # if t == tid:
        #     #     continue
        #
        #     task_loss = 0
        #
        #     t_mask = tids == t
        #     x_t = x[t_mask]
        #
        #     # other_indexes = [l for i in range(tid + 1)
        #     #                  for l in self.tasks_nclasses[i] if i != t.item()]
        #     # other_indexes = torch.tensor(other_indexes, device=x.device)
        #     # for t1 in range(t.item() + 1, tid + 1):
        #
        #     for t1 in all_tasks:
        #         if t == t1:
        #             continue
        #
        #         t1 = t1.item()
        #
        #         i = max(self.tasks_nclasses[t1]) + 1
        #
        #         preds = strategy.model(x_t, t1, mask=False)
        #
        #         # if preds.shape[-1] != i:
        #         #     indexes = [for idx ]
        #         #     preds = preds[:, :i]
        #         # preds = preds[:, self.tasks_nclasses[t1]]
        #         # preds = preds[:, :i]
        #         # preds = preds.index_select(-1, other_indexes)
        #
        #         preds = torch.softmax(preds, -1)
        #         mask = preds != 0.0
        #         d = mask.sum(-1)
        #
        #         # h = -(preds.log() * preds)
        #         # h = h.sum(-1) / np.log(preds.shape[-1])
        #         # h = h.sum(-1)
        #         # h = h[mask].sum(-1)
        #         # a = torch.any(d != preds.shape[-1])
        #         h = torch.where(mask,
        #                         -(preds.log() * preds), 0).sum(-1)
        #         h = h / torch.log(d)
        #
        #         # h = h[~torch.isnan(h)]
        #         # h = torch.nan_to_num(h, 1)
        #         h = 1 - h
        #         # if t1 != t:
        #         #     h = -h
        #         #     continue
        #         # h = - h
        #
        #         matrix[t.item(), t1] = h.mean()
        #
        #         task_loss += h.mean()
        #
        #     all_loss += task_loss
        #
        # # norm = matrix.sum() / (len(matrix) ** 2)
        #
        # distance = matrix
        # norm = torch.linalg.matrix_norm(distance)
        # strategy.loss += norm * 1

        # all_loss = all_loss / (tid + 1)
        # strategy.loss += all_loss * 1

        ot_x, ot_y, ot_t = next(
            iter(DataLoader(self.av, batch_size=strategy.train_mb_size,
                            shuffle=True)))

        ot_x = ot_x.to(strategy.device)
        # mask = tids != tid
        # x = x[mask]
        # tids = tids[mask]
        with torch.no_grad():
            past_logits = self.past_model(ot_x, ot_t)

        strategy.model(ot_x, ot_t)

        # current_logits = strategy.model(ot_x, ot_t)
        #
        # loss = nn.functional.mse_loss(current_logits,
        #                               past_logits)

        loss = nn.functional.cosine_similarity(strategy.model.current_features,
                                               self.past_model.current_features,
                                               -1).mean()
        # loss = (loss + 1) / 2
        loss = 1 - loss
        # loss = loss.mean()

        strategy.loss += loss * 1

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

    # def train_dataset_adaptation(self, **kwargs):
    #     """ Initialize `self.adapted_dataset`. """
    #
    #     if not hasattr(self.experience, 'dev_dataset'):
    #         dataset = self.experience.dataset
    #
    #         idx = np.arange(len(dataset))
    #         np.random.shuffle(idx)
    #
    #         if isinstance(self.dev_split_size, int):
    #             dev_i = self.dev_split_size
    #         else:
    #             dev_i = int(len(idx) * self.dev_split_size)
    #
    #         dev_idx = idx[:dev_i]
    #         train_idx = idx[dev_i:]
    #
    #         self.experience.dataset = dataset.train().subset(train_idx)
    #         # self.experience.dev_dataset = dataset.eval().subset(dev_idx)
    #         self.experience.dev_dataset = dataset.train().subset(dev_idx)
    #
    #     self.adapted_dataset = self.experience.dataset


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
        self.layers.append(RoutingLayer(128, 256))

        self.mx = nn.AdaptiveAvgPool2d(2)
        self.in_features = 256 * 4

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

        # self.classifiers = CumulativeMultiHeadClassifier(self.in_features,
        #                                                  initial_out_features=2)

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

    def forward(self, x: torch.Tensor, task_labels: torch.Tensor,
                mask=True,
                **kwargs) \
            -> torch.Tensor:

        if isinstance(task_labels, int):
            # fast path. mini-batch is single task.
            return self.forward_single_task(x, task_labels, mask, **kwargs)
        else:
            unique_tasks = torch.unique(task_labels)

        out = None
        features = None

        for task in unique_tasks:
            task_mask = task_labels == task
            x_task = x[task_mask]
            out_task = self.forward_single_task(x_task,
                                                task.item(),
                                                mask,
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

        # return out / (torch.norm(out, 2, -1, keepdim=True) * 10)
        return out

    def features(self, x, task_labels=None, path_id=None, **kwargs):

        for l in self.layers[:-1]:
            x = torch.relu(l(x, task_labels, path_id))

        x = self.layers[-1](x, task_labels, path_id).relu()

        x = self.mx(x).flatten(1)

        self.current_features = x

        return x

    def pp(self, x, task_labels=None, path_id=None, **kwargs):
        x = self.features(x, task_labels, path_id)
        return self.p(x)

    def forward_single_task(self, x, task_labels, mask, **kwargs):

        features = self.features(x, task_labels)

        # logits = self.classifiers(features, task_labels=task_labels, mask=mask)
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

    optimizer = Adam(complete_model.parameters(), lr=0.001, weight_decay=1e-4)
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

        if current_task_id == 0:
            continue

        with torch.no_grad():

            m = np.zeros((current_task_id + 1, current_task_id + 1))

            for i in range(len(test_stream[:current_task_id + 1])):
                t_t, t_c = 0, 0
                c_t, c_c = 0, 0
                labels = []

                vals = []
                rmx = []

                for j in range(len(test_stream[:current_task_id + 1])):
                    task_vals = []
                    task_max = []

                    # for x, y, _ in DataLoader(test_stream[i].dataset,
                    #                           batch_size=1000,
                    #                           shuffle=False):
                    for x, y, _ in DataLoader(trainer.base_plugin.past_dataset[i],
                                              batch_size=1000,
                                              shuffle=False):
                        x = x.to(device)
                        y = y.to(device)

                        if j == 0:
                            labels.append(y)

                        pred = complete_model(x, j)
                        # pred = pred[:, max(test_stream[j].classes_in_this_experience) + 1]
                        task_max.append(pred.argmax(-1))

                        pred = pred[:, test_stream[j].classes_in_this_experience]

                        preds = torch.softmax(pred, -1)
                        h = -(preds.log() * preds).sum(-1)
                        h = h / np.log(preds.shape[-1])

                        task_vals.append(h)

                        m[i, j] += h.sum().item()

                    task_vals = torch.cat(task_vals, 0)
                    task_max = torch.cat(task_max, 0)

                    vals.append(task_vals)
                    rmx.append(task_max)

                labels = torch.cat(labels, 0)
                vals = torch.stack(vals, -1)
                rmx = torch.stack(rmx, -1)

                task_pred = vals.argmin(-1)
                mask = task_pred == i

                # preds = torch.stack(rmx, -1)[task_pred]

                t_c += mask.sum().item()
                c_c += (rmx.gather(-1, task_pred[:, None]).squeeze() == labels).sum().item()

                t_t += len(labels)
                c_t += len(labels)

                print(i, t_t, t_c, t_c / t_t)
                print(i, c_t, c_c, c_c / c_t)
                print(i, torch.bincount(task_pred))

            if current_task_id > 0 or True:
                m = m / t_t

                fig, ax = plt.subplots()
                im = ax.matshow(m)
                threshold = im.norm(m.max()) / 2.

                for i in range(len(m)):
                    for j in range(len(m)):
                        t_c = m[j, i]
                        ax.text(i, j, f'{t_c:.2f}', va='center', ha='center',
                                color=("black", "white")[int(t_c < 0.65)])

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
                x, y, t_t = next(
                    iter(DataLoader(trainer.base_plugin.past_dataset[i],
                                    batch_size=128,
                                    shuffle=True)))
                x = x.to(device)

                # for _ in range(5):
                pred = trainer.model(x, t_t.to(device))
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
                x, y, t_t = next(iter(DataLoader(test_stream[i].dataset,
                                               batch_size=128,
                                               shuffle=True)))
                # x, y, t = next(iter(DataLoader(trainer.base_plugin.past_dataset[i],
                #                                batch_size=128,
                #                                shuffle=True)))
                x = x.to(device)

                # for _ in range(5):
                pred = trainer.model(x, t_t.to(device))
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
        class_ids_from_zero_in_each_exp=True,
        class_ids_from_zero_from_first_exp=False,
        train_transform=_default_cifar10_train_transform,
        eval_transform=_default_cifar10_eval_transform)
    cifar10_train_stream = perm_mnist.train_stream
    cifar10_test_stream = perm_mnist.test_stream

    train_tasks = cifar10_train_stream
    test_tasks = cifar10_test_stream

    model = train(train_epochs=2,
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
