from builtins import enumerate
from copy import deepcopy
from itertools import chain
from typing import Optional, Sequence, Iterable, Union

import collections
import numpy as np
import torch
import torch.nn as nn
import torchvision
from avalanche.benchmarks import SplitMNIST, SplitCIFAR10, nc_benchmark, \
    CLExperience
from avalanche.benchmarks.datasets.external_datasets import get_cifar10_dataset
from avalanche.benchmarks.utils import AvalancheDataset, \
    AvalancheConcatDataset
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.core import SupervisedPlugin
from avalanche.evaluation.metrics import accuracy_metrics, bwt_metrics
from avalanche.logging import TextLogger
from avalanche.models import MultiTaskModule, IncrementalClassifier, \
    avalanche_forward
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

from layers import MoERoutingLayer, DynamicMoERoutingLayer


# from utils import calculate_similarity, calculate_distance
avalanche_forward

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


class LogitNormLoss(nn.Module):

    def __init__(self, t=1.0):
        super(LogitNormLoss, self).__init__()
        self.t = t

    def forward(self, x, target):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-7
        norms *= self.t
        # logit_norm = torch.div(x, norms) / self.t
        logit_norm = torch.div(x, norms)
        return torch.nn.functional.cross_entropy(logit_norm, target)


class CentroidsMatching(SupervisedPlugin):
    def __init__(self,
                 sit=True,
                 top_k=1,
                 per_sample_routing_reg=False,
                 centroids_merging_strategy=None,
                 **kwargs):

        super().__init__()

        self.distributions = {}
        self.patterns_per_experience = 50
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

    def before_train_dataset_adaptation(self, strategy, *args, **kwargs):
        if self.per_sample_routing_reg:
            self.past_model = deepcopy(strategy.model)

    def before_training_exp(self, strategy: SupervisedTemplate,
                            **kwargs):

        n_classes = len(strategy.experience.classes_in_this_experience)
        self.tasks_nclasses[strategy.experience.task_label] = n_classes

        if strategy.experience.current_experience == 0:
            return

        # self.past_model = deepcopy(strategy.model)

        av = AvalancheDataset(list(self.past_dataset.values()))
        # av.concat()
        strategy.dataloader = ReplayDataLoader(
            strategy.adapted_dataset,
            av,
            oversample_small_tasks=True,
            batch_size=strategy.train_mb_size,
            batch_size_mem=strategy.train_mb_size,
            task_balanced_dataloader=False,
            num_workers=0,
            shuffle=True,
            drop_last=False,
        )

    def before_training_epoch(self, strategy, *args, **kwargs):

        for name, module in strategy.model.named_modules():
            if isinstance(module, (MoERoutingLayer, DynamicMoERoutingLayer)):
                module.similarity_statistics = []

    def after_finetuning_exp(self, strategy: 'BaseStrategy', **kwargs):
        tid = strategy.experience.current_experience

        for module in strategy.model.modules():
            if isinstance(module, (MoERoutingLayer, DynamicMoERoutingLayer)):
                module.freeze_blocks(tid)

        self.update_memory(
            strategy.experience.dataset.eval(),
            tid,
            strategy.train_mb_size,
        )

        values = []

        for x, _, _ in strategy.experience.dataset.eval():
            values.append(x)

        values = torch.stack(values, 0)
        values = torch.cat([torch.nn.functional.adaptive_avg_pool2d(values, 4),
                            torch.nn.functional.adaptive_max_pool2d(values, 4),
                            -torch.nn.functional.adaptive_max_pool2d(-values,
                                                                     4)],
                           2)

        # values = torch.nn.functional.adaptive_avg_pool2d(values, 4)
        values = torch.flatten(values, 1)

        means = values.mean(0)
        cov = torch.cov(values.T, correction=0)
        cov += torch.eye(values.shape[1])

        d = torch.distributions.multivariate_normal.MultivariateNormal(
            means,
            cov)

        self.distributions[tid] = d

    @torch.no_grad()
    def after_training_exp(self, strategy, **kwargs):

        tid = strategy.experience.current_experience

        for module in strategy.model.modules():
            if isinstance(module, (MoERoutingLayer, DynamicMoERoutingLayer)):
                module.freeze_logits(strategy.experience,
                                     strategy,
                                     tid,
                                     top_k=self.top_k)

        # if tid > 0:
        self.past_model = clone_module(strategy.model)

    @torch.no_grad()
    def update_memory(self, dataset, t, batch_size):
        collate_fn = (
            dataset.collate_fn if hasattr(dataset, "collate_fn") else None
        )

        dataset_idx = np.arange(len(dataset))
        np.random.shuffle(dataset_idx)

        idx_to_get = dataset_idx[:self.patterns_per_experience]
        memory = dataset.train().subset(idx_to_get)
        self.past_dataset[t] = memory

    def before_backward(self, strategy, *args, **kwargs):
        # self.before_backward_kl(strategy, args, kwargs)
        self.before_backward_distance(strategy, args, kwargs)

    def before_backward_distance(self, strategy, *args, **kwargs):
        tid = strategy.experience.current_experience

        entropies = []
        classification_losses = []
        distance_losses = []
        mask = strategy.mb_task_id == tid

        if tid == 0 and strategy.is_finetuning:
            return

        if tid > 0:
            _x = strategy.mb_x[~mask]
            past_routing = self.past_model.routing_model(_x)
            current_routing = strategy.model.current_routing[~mask]

            distance = 1 - cosine_similarity(current_routing, past_routing)

            strategy.loss += distance.mean(0) * 0.5

        if not self.per_sample_routing_reg or tid == 0:
            for name, module in strategy.model.named_modules():
                if isinstance(module, (MoERoutingLayer,
                                       DynamicMoERoutingLayer)):
                    weights, similarity = module.last_distribution

                    if strategy.is_finetuning:
                        labels = [module.get_task_blocks(t.item())
                                  for t in strategy.mb_task_id]
                        labels = torch.stack(labels, 0)
                        distr = torch.log_softmax(similarity, -1)
                        loss = -distr.gather(1, labels).squeeze()

                        classification_losses.append(loss)
                    else:
                        distr = torch.softmax(similarity, -1)
                        h = -(distr.log() * distr).sum(-1) / np.log(
                            distr.shape[-1])
                        if tid > 0:
                            h = h[mask]
                        entropies.append(h)

                        if tid > 0:
                            labels = [module.get_task_blocks(t.item())
                                      for t in strategy.mb_task_id[~mask]]
                            labels = torch.stack(labels, 0)
                            loss = -torch.log_softmax(similarity[~mask], -1) \
                                .gather(1, labels).squeeze()
                            # loss = loss[~mask]

                            classification_losses.append(loss)

                            d = distr.index_select(-1, torch.unique(labels))
                            d = d[mask].log().mean(-1)
                            distance_losses.append(d)
        else:
            outputs = {}
            past_output = self.past_model(strategy.mb_x, strategy.mb_task_id)

            for name, module in self.past_model.named_modules():
                if isinstance(module, (MoERoutingLayer,
                                       DynamicMoERoutingLayer)):
                    weights, similarity = module.last_distribution
                    outputs[name] = weights

            mask = strategy.mb_task_id == tid

            for name, module in strategy.model.named_modules():
                if isinstance(module, (MoERoutingLayer,
                                       DynamicMoERoutingLayer)):
                    weights, similarity = module.last_distribution

                    if strategy.is_finetuning and tid > 0:
                        distr = torch.log_softmax(similarity, -1)
                        distr = distr * outputs[name]
                        distr = distr.sum(-1) / outputs[name].sum(-1)
                        loss = - distr

                        classification_losses.append(loss)
                    else:
                        distr = torch.softmax(similarity, -1)
                        h = -(distr.log() * distr).sum(-1) / np.log(
                            weights.shape[-1])
                        if tid > 0:
                            h = h[mask]
                        entropies.append(h)

                        if tid > 0:
                            sim = -torch.log_softmax(similarity[~mask], -1)
                            past_selection = outputs[name][~mask]

                            sim = sim * past_selection
                            loss = sim.sum(-1) / past_selection.sum(-1)
                            classification_losses.append(loss)

                            labels = [module.get_task_blocks(t.item())
                                      for t in strategy.mb_task_id[~mask]]
                            labels = torch.stack(labels, 0)
                            # loss = -torch.log_softmax(similarity[~mask], -1) \
                            #     .gather(1, labels).squeeze()
                            # loss = loss[~mask]

                            d = torch.log_softmax(similarity[mask], -1) \
                                .index_select(-1, torch.unique(labels))
                            # d = -torch.log_softmax(similarity[~mask], -1).mean(-1)
                            distance_losses.append(d.mean(-1))

        if len(entropies) > 0:
            entropy = sum(entropies) / len(entropies)
            entropy = entropy.mean(0)
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

        strategy.loss += entropy + class_loss * 1 + distance_loss * 0

        return

        # return
        # self._bb_sigmoid(strategy)
        self._bb_kl_div(strategy)
        # self.___before_backward(strategy)

    def before_backward_kl(self, strategy, *args, **kwargs):
        tid = strategy.experience.current_experience

        entropies = []
        classification_losses = []
        distance_losses = []
        mask = strategy.mb_task_id == tid

        if not self.per_sample_routing_reg or tid == 0:
            for name, module in strategy.model.named_modules():
                if isinstance(module,
                              (MoERoutingLayer, DynamicMoERoutingLayer)):
                    weights, similarity = module.last_distribution

                    similarity = torch.log_softmax(similarity, -1)

                    if strategy.is_finetuning:
                        labels = [module._get_indexes(t.item())
                                  for t in strategy.mb_task_id]
                        labels = torch.stack(labels, 0)

                        one_hot_mask = nn.functional. \
                            one_hot(labels, similarity.shape[-1])
                        one_hot_mask_d = one_hot_mask \
                                         / one_hot_mask.sum(-1, keepdims=True)

                        loss = nn.functional.kl_div(similarity,
                                                    one_hot_mask_d,
                                                    reduction='mean')

                        # distr = torch.log_softmax(similarity, -1)
                        # loss = -similarity.gather(1, labels).squeeze()

                        classification_losses.append(loss)
                    else:
                        distr = torch.softmax(similarity, -1)
                        h = -(distr.log() * distr).sum(-1) / np.log(
                            distr.shape[-1])
                        if tid > 0:
                            h = h[mask]
                        entropies.append(h)

                        if tid > 0:
                            labels = [module._get_indexes(t.item())
                                      for t in strategy.mb_task_id[~mask]]
                            labels = torch.stack(labels, 0)
                            one_hot_mask = nn.functional. \
                                one_hot(labels, similarity.shape[-1])
                            one_hot_mask_d = one_hot_mask \
                                             / one_hot_mask.sum(-1,
                                                                keepdims=True)

                            kl = nn.functional.kl_div(similarity[~mask],
                                                      one_hot_mask_d,
                                                      reduction='mean')
                            loss = kl

                            classification_losses.append(loss)

                            # one_hot_mask = nn.functional.one_hot(
                            #     torch.unique(labels),
                            #     similarity.shape[-1])

                            one_hot_mask = torch.ones_like(similarity[0:1])
                            one_hot_mask[0, torch.unique(labels)] = 0.0
                            one_hot_mask_d = one_hot_mask \
                                             / one_hot_mask.sum(-1,
                                                                keepdims=True)

                            d = nn.functional.kl_div(similarity[mask],
                                                     one_hot_mask_d,
                                                     reduction='mean')

                            # d = distr.index_select(-1, torch.unique(labels))
                            # d = d[mask].log().mean(-1)
                            distance_losses.append(d)
        else:
            outputs = {}
            past_output = self.past_model(strategy.mb_x, strategy.mb_task_id)

            for name, module in self.past_model.named_modules():
                if isinstance(module, MoERoutingLayer):
                    weights, similarity = module.last_distribution
                    outputs[name] = weights

            mask = strategy.mb_task_id == tid

            for name, module in strategy.model.named_modules():
                if isinstance(module, MoERoutingLayer):
                    weights, similarity = module.last_distribution

                    if strategy.is_finetuning and tid > 0:
                        distr = torch.log_softmax(similarity, -1)
                        distr = distr * outputs[name]
                        distr = distr.sum(-1) / outputs[name].sum(-1)
                        loss = - distr

                        classification_losses.append(loss)
                    else:
                        # distr = torch.softmax(similarity, -1)
                        h = -(weights.log() * weights).sum(-1) / np.log(
                            weights.shape[-1])
                        if tid > 0:
                            h = h[mask]
                        entropies.append(h)

                        if tid > 0:
                            sim = -torch.log_softmax(similarity[~mask], -1)
                            past_selection = outputs[name][~mask]

                            sim = sim * past_selection
                            loss = sim.sum(-1) / past_selection.sum(-1)
                            classification_losses.append(loss)

                            labels = [module._get_indexes(t.item())
                                      for t in strategy.mb_task_id[~mask]]
                            labels = torch.stack(labels, 0)
                            # loss = -torch.log_softmax(similarity[~mask], -1) \
                            #     .gather(1, labels).squeeze()
                            # loss = loss[~mask]

                            d = torch.log_softmax(similarity[mask], -1) \
                                .index_select(-1, torch.unique(labels))
                            # d = -torch.log_softmax(similarity[~mask], -1).mean(-1)
                            distance_losses.append(d.mean(-1))

        if len(entropies) > 0:
            entropy = sum(entropies) / len(entropies)
            entropy = -entropy.mean(0)
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

        strategy.loss += class_loss + distance_loss * 10

        return

        # return
        # self._bb_sigmoid(strategy)
        self._bb_kl_div(strategy)
        # self.___before_backward(strategy)

    # def ___before_backward(self, strategy, *args, **kwargs):
    #     # return
    #     tid = strategy.experience.current_experience
    #
    #     if tid > 0:
    #         current_reg_loss = 0
    #
    #         distance_loss = []
    #         similarity_loss = []
    #         agglomerating_loss = []
    #
    #         for name, module in strategy.model.named_modules():
    #             if isinstance(module, MoERoutingLayer):
    #                 idxs = module.get_task_blocks(tid)
    #
    #                 past_idx = [j
    #                             for i in range(tid)
    #                             for j in module.get_task_blocks(i).tolist()]
    #
    #                 if idxs is not None:
    #                     idxs = idxs.tolist()
    #                     past_idx = list(set(past_idx) - set(idxs))
    #
    #                 # routing = module.current_routing.mean(0)
    #                 # b_embeddings = list(module.blocks_embeddings.values())
    #                 # b_embeddings = torch.cat(b_embeddings, 0)
    #
    #                 # b_embeddings = torch.nn.functional.normalize(b_embeddings, 2, -1)
    #                 # routing = torch.nn.functional.normalize(routing, 2, -1)
    #                 # distance = torch.norm(b_embeddings - routing, 2, -1)
    #                 # agglomerating_loss.append(distance)
    #
    #                 if len(past_idx) == 0:
    #                     continue
    #
    #                 _, similarity = module.last_distribution
    #                 distance = - similarity
    #
    #                 # b_embeddings = list(module.blocks_embeddings.values())
    #                 # b_embeddings = torch.cat(b_embeddings, 0)
    #
    #                 if idxs is not None:
    #                     # correct_centroid = b_embeddings[idxs].mean(0)
    #                     # distance = 1 - cosine_similarity(module.current_routing,
    #                     #                                  correct_centroid)
    #
    #                     distance_loss.append(distance.mean(-1))
    #
    #                 similarity = 1 / (1 + distance)
    #                 similarity_loss.append(
    #                     similarity[:, past_idx].mean(-1))
    #
    #         if len(distance_loss) > 0:
    #             distance_loss = sum(distance_loss).mean() / len(distance_loss)
    #         else:
    #             distance_loss = 0
    #
    #         similarity_loss = sum(similarity_loss).mean() / len(similarity_loss)
    #
    #         # agglomerating_loss = sum(agglomerating_loss) / len(agglomerating_loss)
    #         # agglomerating_loss = agglomerating_loss.mean(0)
    #
    #         # print('Current task loss',
    #         #       distance_loss,
    #         #       similarity_loss)
    #
    #         current_reg_loss += distance_loss / 2 + similarity_loss / 2
    #         strategy.loss += current_reg_loss
    #
    #     if len(self.distributions) > 0:
    #         embeddings = strategy.model.embeddings
    #         x = strategy.mb_x
    #
    #         past_reg_loss = 0
    #
    #         for t, d in self.distributions.items():
    #             routing = d.sample([len(x)]).to(x.device)
    #
    #             if embeddings is not None:
    #                 i = torch.full((len(x),), t, device=x.device)
    #                 e = embeddings(i)
    #
    #                 routing = torch.cat((routing, e), -1)
    #
    #             #     all_routing.append(routing)
    #             #
    #             # all_routing = torch.cat(all_routing, 0)
    #             distance_loss = []
    #             similarity_loss = []
    #             for name, module in strategy.model.named_modules():
    #                 if isinstance(module, MoERoutingLayer):
    #                     r = module.process_routing(routing)
    #
    #                     task_blocks = module.get_task_blocks(t)
    #                     other_blocks = list(
    #                         set(range(len(module.blocks_embeddings)))
    #                         - set(task_blocks.tolist()))
    #
    #                     b_embeddings = list(module.blocks_embeddings.values())
    #                     b_embeddings = torch.cat(b_embeddings, 0)
    #                     distance = calculate_distance(r, b_embeddings)
    #
    #                     # distance = 1 / (1 - similarity)
    #                     distance_loss.append(distance[:, task_blocks].mean(-1))
    #
    #                     if len(other_blocks) > 0:
    #                         similarity = 1 / (1 + distance)
    #                         similarity_loss.append(
    #                             similarity[:, other_blocks].mean(-1))
    #
    #             distance_loss = sum(distance_loss) / len(distance_loss)
    #             similarity_loss = sum(similarity_loss) / len(similarity_loss)
    #
    #             past_reg_loss += distance_loss / 2 + similarity_loss / 2
    #
    #         # print('past task loss',
    #         #       past_reg_loss.mean())
    #         strategy.loss += past_reg_loss.mean() / len(self.distributions)
    #
    #     return
    #
    #     def __get_loss(x, t, forward_current):
    #         loss = 0
    #         d = 0
    #         labels = {}
    #
    #         if forward_current:
    #             strategy.model(x, t)
    #
    #         self.past_model(x, t)
    #
    #         with torch.no_grad():
    #             for name, module in self.past_model.named_modules():
    #                 if isinstance(module, MoERoutingLayer):
    #                     idxs = module._get_indexes(t)
    #
    #                     n_blocks = len(module.blocks)
    #                     mask = torch.zeros(n_blocks, device=x.device)
    #
    #                     for i, k in enumerate(module.blocks.keys()):
    #                         if int(k) in idxs:
    #                             mask[i] = 1
    #
    #                     _, similarity = module.last_distribution
    #                     similarity = mask * similarity - (1 - mask) * 100
    #                     labels[name] = torch.argmax(similarity,
    #                                                 -1).unsqueeze(-1)
    #
    #         for name, module in strategy.model.named_modules():
    #             if isinstance(module, MoERoutingLayer):
    #                 d += 1
    #                 _, similarity = module.last_distribution
    #                 log_p_y = torch.log_softmax(similarity, dim=1)
    #                 pull_loss = -log_p_y.gather(1, labels[name])
    #                 loss += pull_loss
    #
    #         return loss / d
    #
    #     def get_loss(x, t, forward_current):
    #         loss = 0
    #         d = 0
    #         labels = {}
    #
    #         if forward_current:
    #             strategy.model(x, t)
    #
    #         # self.past_model(x, t)
    #         #
    #         # with torch.no_grad():
    #         #     for name, module in self.past_model.named_modules():
    #         #         if isinstance(module, MoERoutingLayer):
    #         #             idxs = module._get_indexes(t)
    #         #
    #         #             n_blocks = len(module.blocks)
    #         #             mask = torch.zeros(n_blocks, device=x.device)
    #         #
    #         #             for i, k in enumerate(module.blocks.keys()):
    #         #                 if int(k) in idxs:
    #         #                     mask[i] = 1
    #         #
    #         #             _, similarity = module.last_distribution
    #         #             similarity = mask * similarity - (1 - mask) * 100
    #         #             labels[name] = torch.argmax(similarity,
    #         #                                         -1).unsqueeze(-1)
    #
    #         for name, module in strategy.model.named_modules():
    #             if isinstance(module, MoERoutingLayer):
    #                 d += 1
    #
    #                 _, similarity = module.last_distribution
    #                 # mask = torch.zeros_like(similarity)
    #                 idxs = module._get_indexes(t)
    #                 new_idxs = []
    #                 for i, k in enumerate(module.blocks.keys()):
    #                     if int(k) in idxs:
    #                         new_idxs.append(i)
    #                 # new_idxs = torch.tensor(new_idxs, device=similarity.device)
    #
    #                 distance = 1 / (1 - similarity)
    #                 distance = distance[:, new_idxs]
    #                 loss += distance.mean(-1)
    #
    #                 # log_p_y = torch.log_softmax(similarity, dim=1)
    #                 # pull_loss = -log_p_y.gather(1, labels[name])
    #                 # loss += pull_loss
    #
    #         return loss / d
    #
    #     def get_loss(x, t, forward_current):
    #         loss = 0
    #         d = 0
    #         labels = {}
    #
    #         if forward_current:
    #             strategy.model(x, t)
    #
    #         # self.past_model(x, t)
    #         #
    #         # with torch.no_grad():
    #         #     for name, module in self.past_model.named_modules():
    #         #         if isinstance(module, MoERoutingLayer):
    #         #             idxs = module._get_indexes(t)
    #         #
    #         #             n_blocks = len(module.blocks)
    #         #             mask = torch.zeros(n_blocks, device=x.device)
    #         #
    #         #             for i, k in enumerate(module.blocks.keys()):
    #         #                 if int(k) in idxs:
    #         #                     mask[i] = 1
    #         #
    #         #             _, similarity = module.last_distribution
    #         #             similarity = mask * similarity - (1 - mask) * 100
    #         #             labels[name] = torch.argmax(similarity,
    #         #                                         -1).unsqueeze(-1)
    #
    #         for name, module in strategy.model.named_modules():
    #             if isinstance(module, MoERoutingLayer):
    #                 d += 1
    #
    #                 _, similarity = module.last_distribution
    #                 log_p_y = torch.log_softmax(similarity, dim=1)
    #
    #                 # mask = torch.zeros_like(similarity)
    #                 idxs = module.get_task_blocks(t)
    #
    #                 # new_idxs = torch.tensor(new_idxs, device=similarity.device)
    #                 # distance = 1 / (1 + similarity)
    #                 p, _ = torch.max(log_p_y[:, idxs], -1)
    #                 loss += -p.mean(-1)
    #
    #                 # log_p_y = torch.log_softmax(similarity, dim=1)
    #                 # pull_loss = -log_p_y.gather(1, labels[name])
    #                 # loss += pull_loss
    #
    #         return loss / d
    #
    #     d = 0
    #     entropy_loss = 0
    #
    #     for module in strategy.model.modules():
    #         if isinstance(module, MoERoutingLayer):
    #             d += 1
    #             distr, _ = module.last_distribution
    #             entropy = -(distr.log() * distr).sum(-1) \
    #                       / np.log(distr.shape[-1])
    #             entropy = entropy.mean(0)
    #             entropy_loss += entropy
    #
    #     entropy_loss = entropy_loss / d
    #
    #     # print(strategy.is_finetuning, entropy_loss.item())
    #
    #     strategy.loss += entropy_loss
    #
    #     return
    #     # if strategy.is_finetuning:
    #     #     return
    #
    #     tid = strategy.experience.current_experience
    #     current_loss = 0
    #
    #     if tid > 0 and strategy.is_finetuning:
    #         x = strategy.mb_x
    #         current_loss = get_loss(x, tid, forward_current=False)
    #         current_loss = current_loss.view(-1).mean()
    #
    #         strategy.loss += current_loss
    #
    #     # return
    #     past_loss = 0
    #     if len(self.past_dataset) > 0:
    #
    #         for t, d in self.past_dataset.items():
    #             x, y, _ = next(iter(DataLoader(d, shuffle=True,
    #                                            batch_size=len(strategy.mb_x))))
    #
    #             x = x.to(device)
    #             past_loss += get_loss(x, t, forward_current=True)
    #
    #         past_loss = past_loss / len(self.past_dataset)
    #         past_loss = past_loss.view(-1).mean()
    #         strategy.loss += past_loss
    #
    #     if past_loss + current_loss > 0:
    #         print(past_loss, current_loss)
    #
    #     return
    #
    #     # current_head_routing_loss = 0
    #     # for module in strategy.model.modules():
    #     #     if isinstance(module, HeadsRoutingLayer):
    #     #         _, pull_loss = module.current_batch_similarity_loss
    #     #         past_loss = 0
    #     #         if pull_loss is not None:
    #     #             pull_loss = pull_loss.mean(-1)
    #     #             past_loss += pull_loss.mean()
    #     #
    #     #             current_head_routing_loss = pull_loss
    #     #
    #     # strategy.loss += current_head_routing_loss
    #
    #     if tid == 0 and not strategy.is_finetuning:
    #         return
    #
    #     # if tid == 0:
    #     #     to_check = HeadsRoutingLayer
    #     # else:
    #     # to_check = (MoERoutingLayer, HeadsRoutingLayer)
    #
    #     current_losses = []
    #     current_loss = 0
    #     past_loss = 0.0
    #
    #     if strategy.is_finetuning:
    #         for module in strategy.model.modules():
    #             if isinstance(module, MoERoutingLayer):
    #                 push_loss, pull_loss = module.current_batch_similarity_loss
    #                 past_loss = 0
    #                 if pull_loss is not None:
    #                     pull_loss = pull_loss.mean(-1)
    #                     past_loss += pull_loss.mean()
    #                 if push_loss is not None:
    #                     push_loss = push_loss.mean(-1)
    #                     past_loss += push_loss.mean()
    #
    #                 current_losses.append(past_loss)
    #
    #         current_loss = sum(current_losses) / len(current_losses)
    #
    #     if len(self.memory_x) > 0:
    #         losses = []
    #         for t, x in self.memory_x.items():
    #             strategy.model(x.to(device), t)
    #
    #             for module in strategy.model.modules():
    #                 if isinstance(module, (MoERoutingLayer, HeadsRoutingLayer)):
    #                     push_loss, pull_loss = module.current_batch_similarity_loss
    #
    #                     past_loss = 0
    #                     if pull_loss is not None:
    #                         pull_loss = pull_loss.mean(-1)
    #                         past_loss += pull_loss.mean()
    #                     if push_loss is not None:
    #                         push_loss = push_loss.mean(-1)
    #                         past_loss += push_loss.mean()
    #
    #                     losses.append(past_loss)
    #
    #         past_loss = sum(losses) / len(losses)
    #
    #     # if current_loss + past_loss > 0:
    #     #     print(current_loss, past_loss)
    #
    #     strategy.loss += past_loss * 0 + current_loss * 1
    #
    #     return
    #
    #     if tid > 0:
    #
    #         current_x = strategy.mb_x
    #         current_features = strategy.model.hidden_features
    #         # print(current_features)
    #         # current_centroid = current_features.mean(0, keepdims=True)
    #
    #         # past vs current
    #
    #         past_vs_current_loss = 0
    #         current_vs_past_loss = 0
    #
    #         for evaluated_task in range(tid):
    #             current_x = self.memory_x[evaluated_task].to(strategy.device)
    #             strategy.model(current_x, tid)
    #             past_features = strategy.model.hidden_features
    #
    #             # Compare images in the memory to the centroids calculated using the samples in the current task
    #             for i, (pf, cf) in enumerate(
    #                     zip(past_features, current_features)):
    #                 cf = torch.flatten(cf, 1).mean(0, keepdims=True)
    #                 pf = torch.flatten(pf, 1)
    #
    #                 distance = -calculate_similarity(pf, cf)
    #                 similarity = 1 / (1 + distance)
    #
    #                 past_vs_current_loss += similarity.mean()
    #                 # print(evaluated_task, similarity.shape, pf.shape, cf.shape)
    #
    #             # Minimize the similarity between the current samples and the past centroids
    #             for i, (pf, cf) in enumerate(
    #                     zip(self.layers_centroids[evaluated_task].values(),
    #                         current_features)):
    #                 cf = torch.flatten(cf, 1)
    #                 pf = torch.flatten(pf, 1)
    #
    #                 distance = -calculate_similarity(cf, pf)
    #                 similarity = 1 / (1 + distance)
    #
    #                 current_vs_past_loss += similarity.mean()
    #
    #                 # print(tid, evaluated_task, similarity.shape, pf.shape, cf.shape)
    #
    #         past_vs_current_loss = past_vs_current_loss / tid
    #         current_vs_past_loss = current_vs_past_loss / tid
    #
    #         print(current_vs_past_loss, past_vs_current_loss,
    #               current_vs_past_loss + past_vs_current_loss)
    #
    #         strategy.loss += current_vs_past_loss + past_vs_current_loss

    # def __before_backward(self, strategy, *args, **kwargs):
    #     tid = strategy.experience.current_experience
    #
    #     # if tid > 0:
    #     #     current_reg_loss = 0
    #     #
    #     #     distance_loss = []
    #     #     similarity_loss = []
    #     #     agglomerating_loss = []
    #     #
    #     #     for name, module in strategy.model.named_modules():
    #     #         if isinstance(module, MoERoutingLayer):
    #     #             idxs = module.get_task_blocks(tid)
    #     #
    #     #             past_idx = [j
    #     #                         for i in range(tid)
    #     #                         for j in module.get_task_blocks(i).tolist()]
    #     #
    #     #             if idxs is not None:
    #     #                 idxs = idxs.tolist()
    #     #                 past_idx = list(set(past_idx) - set(idxs))
    #     #
    #     #             routing = module.current_routing.mean(0)
    #     #             b_embeddings = list(module.blocks_embeddings.values())
    #     #             b_embeddings = torch.cat(b_embeddings, 0)
    #     #
    #     #             b_embeddings = torch.nn.functional.normalize(b_embeddings, 2, -1)
    #     #             routing = torch.nn.functional.normalize(routing, 2, -1)
    #     #             distance = torch.norm(b_embeddings - routing, 2, -1)
    #     #             agglomerating_loss.append(distance)
    #     #
    #     #             if len(past_idx) == 0:
    #     #                 continue
    #     #
    #     #             _, similarity = module.last_distribution
    #     #             distance = - similarity
    #     #
    #     #             # b_embeddings = list(module.blocks_embeddings.values())
    #     #             # b_embeddings = torch.cat(b_embeddings, 0)
    #     #
    #     #             if idxs is not None:
    #     #                 # correct_centroid = b_embeddings[idxs].mean(0)
    #     #                 # distance = 1 - cosine_similarity(module.current_routing,
    #     #                 #                                  correct_centroid)
    #     #
    #     #                 distance_loss.append(distance[:, idxs].mean(-1))
    #     #
    #     #             similarity = 1 / (1 + distance)
    #     #             similarity_loss.append(
    #     #                 similarity[:, past_idx].mean(-1))
    #     #
    #     #     if len(distance_loss) > 0:
    #     #         distance_loss = sum(distance_loss).mean() / len(distance_loss)
    #     #     else:
    #     #         distance_loss = 0
    #     #
    #     #     similarity_loss = sum(similarity_loss).mean() / len(similarity_loss)
    #     #
    #     #     agglomerating_loss = sum(agglomerating_loss) / len(agglomerating_loss)
    #     #     agglomerating_loss = agglomerating_loss.mean(0)
    #     #
    #     #     # print('Current task loss',
    #     #     #       distance_loss,
    #     #     #       similarity_loss)
    #     #
    #     #     current_reg_loss += distance_loss + similarity_loss + agglomerating_loss
    #     #     strategy.loss += current_reg_loss
    #
    #     if len(self.distributions) > 0:
    #         embeddings = strategy.model.embeddings
    #         x = strategy.mb_x
    #
    #         past_reg_loss = []
    #
    #         for t, d in self.distributions.items():
    #             routing = d.sample([len(x)]).to(x.device)
    #
    #             distance_loss = []
    #             similarity_loss = []
    #             for name, module in strategy.model.named_modules():
    #                 if isinstance(module, MoERoutingLayer):
    #                     b_embeddings = list(module.blocks_embeddings.values())
    #                     b_embeddings = torch.cat(b_embeddings, 0)
    #
    #                     for t1 in range(tid + 1):
    #                         # if embeddings is not None:
    #                         i = torch.full((len(x),), t1, device=x.device)
    #                         e = embeddings(i)
    #
    #                         _routing = torch.cat((routing, e), -1)
    #
    #                         task_blocks = module.get_task_blocks(t1)
    #                         if task_blocks is None:
    #                             continue
    #                         # other_blocks = list(set(range(len(module.blocks_embeddings))) - set(task_blocks.tolist()))
    #                         distance = calculate_distance(_routing,
    #                                                       b_embeddings)
    #
    #                         if t1 == t:
    #                             distance_loss.append(
    #                                 distance[:, task_blocks].mean(-1) / (
    #                                             tid + 1))
    #                         else:
    #                             similarity = 1 / (1 + distance)
    #                             similarity_loss.append(
    #                                 similarity[:, task_blocks].mean(-1) / (
    #                                             tid + 1))
    #
    #                         # if len(other_blocks) > 0:
    #                         #     similarity = 1 / (1 + distance)
    #                         #     similarity_loss.append(similarity[:, other_blocks].mean(-1))
    #
    #             distance_loss = sum(distance_loss) / len(distance_loss)
    #             if len(similarity_loss) > 0:
    #                 similarity_loss = sum(similarity_loss) / len(
    #                     similarity_loss)
    #             else:
    #                 similarity_loss = 0
    #
    #             past_reg_loss.append(distance_loss / 2 + similarity_loss / 2)
    #
    #         # print('past task loss',
    #         #       past_reg_loss.mean())
    #
    #         strategy.loss += sum(past_reg_loss).mean() / len(self.distributions)

    def _bb_sigmoid(self, strategy, *args, **kwargs):
        tid = strategy.experience.current_experience

        if tid > 0:
            # if strategy.is_finetuning:
            #     bce_loss = []
            #     for name, module in strategy.model.named_modules():
            #         if isinstance(module, MoERoutingLayer):
            #             idxs = module.get_task_blocks(tid)
            #             weights, similarity = module.last_distribution
            #
            #             mask = torch.zeros_like(weights)
            #             mask[:, idxs] = 1.0
            #
            #             loss = binary_cross_entropy_with_logits(similarity, mask)
            #             bce_loss.append(loss)
            #
            #     bce_loss = sum(bce_loss) / len(bce_loss)
            #
            #     strategy.loss += bce_loss
            # else:

            bce_loss = []
            for name, module in strategy.routing_model.named_modules():
                if isinstance(module, MoERoutingLayer):
                    past_idxs = [v for t in range(tid)
                                 for v in module.get_task_blocks(t).tolist()]

                    weights, similarity = module.last_distribution
                    mask = torch.ones_like(weights)
                    mask[:, past_idxs] = 0

                    loss = binary_cross_entropy_with_logits(similarity, mask)
                    bce_loss.append(loss)

            bce_loss = sum(bce_loss) / len(bce_loss)

            strategy.loss += bce_loss

        if len(self.distributions) > 0:
            embeddings = strategy.routing_model.embeddings
            x = strategy.mb_x

            past_reg_loss = 0

            for t, d in self.past_dataset.items():
                x, y, _ = next(iter(DataLoader(d, shuffle=True,
                                               batch_size=len(strategy.mb_x))))

                x = x.to(device)
                x = torch.cat(
                    [torch.nn.functional.adaptive_avg_pool2d(x, 4),
                     torch.nn.functional.adaptive_max_pool2d(x, 4),
                     -torch.nn.functional.adaptive_max_pool2d(-x, 4)],
                    2)
                routing = torch.flatten(x, 1)

                # for t, d in self.distributions.items():
                # routing = d.sample([len(x)]).to(x.device)

                if embeddings is not None:
                    i = torch.full((len(x),), t, device=x.device)
                    e = embeddings(i)

                    routing = torch.cat((routing, e), -1)

                bce_loss = []

                for name, module in strategy.routing_model.named_modules():
                    if isinstance(module, MoERoutingLayer):
                        r = module.process_routing(routing)

                        idxs = module.get_task_blocks(t)

                        _, similarity = module.get_routing_weights(r,
                                                                   augment=False)

                        mask = torch.zeros_like(similarity)
                        mask[:, idxs] = 1.0

                        loss = binary_cross_entropy_with_logits(similarity,
                                                                mask)
                        bce_loss.append(loss)

                bce_loss = sum(bce_loss) / len(bce_loss)
                past_reg_loss += bce_loss

            past_reg_loss = past_reg_loss / len(self.distributions)

            strategy.loss += past_reg_loss * 100

    def _bb_kl_div(self, strategy, *args, **kwargs):
        tid = strategy.experience.current_experience

        if tid > 0 and strategy.is_finetuning:
            bce_loss = []
            for name, module in strategy.routing_model.named_modules():
                if isinstance(module, MoERoutingLayer):
                    past_idxs = [v for t in range(tid)
                                 for v in module.get_task_blocks(t).tolist()]

                    weights, similarity = module.last_distribution

                    with torch.no_grad():
                        idxs = module.get_task_blocks(tid)
                        mask = torch.zeros_like(similarity)
                        mask[:, idxs] = 1.0
                        mask = torch.softmax(mask, -1)

                    kl = nn.functional.kl_div(torch.log_softmax(similarity, -1),
                                              mask)
                    # kl = weights * (weights / mask).log()
                    # kl = kl.sum(-1)

                    # loss = binary_cross_entropy_with_logits(similarity, mask)
                    bce_loss.append(kl)

            bce_loss = sum(bce_loss) / len(bce_loss)

            strategy.loss += bce_loss * 1000

        if len(self.distributions) > 0 and strategy.is_finetuning:
            embeddings = strategy.routing_model.embeddings
            x = strategy.mb_x

            past_reg_loss = 0
            all_entropy_loss = 0

            for t, d in self.past_dataset.items():
                x, y, _ = next(iter(DataLoader(d, shuffle=True,
                                               batch_size=len(strategy.mb_x))))

                x = x.to(device)
                with torch.no_grad():
                    routing = strategy.routing_model.routing_model(x)

                # x = torch.cat(
                #     [torch.nn.functional.adaptive_avg_pool2d(x, 4),
                #      torch.nn.functional.adaptive_max_pool2d(x, 4),
                #      -torch.nn.functional.adaptive_max_pool2d(-x, 4)],
                #     2)
                # routing = torch.flatten(x, 1)

                # for t, d in self.distributions.items():
                # routing = d.sample([len(x)]).to(x.device)

                if embeddings is not None:
                    i = torch.full((len(x),), t, device=x.device)
                    e = embeddings(i)

                    routing = torch.cat((routing, e), -1)

                bce_loss = []
                entropy_loss = []

                for name, module in strategy.routing_model.named_modules():
                    if isinstance(module, MoERoutingLayer):
                        r = module.process_routing(routing, t)

                        distr, similarity = module.get_routing_weights(r,
                                                                       task=t,
                                                                       augment=False)

                        with torch.no_grad():
                            idxs = module.get_task_blocks(t)
                            mask = torch.zeros_like(similarity)
                            mask[:, idxs] = 1.0
                            mask = torch.softmax(mask, -1)

                        kl = nn.functional.kl_div(
                            torch.log_softmax(similarity, -1),
                            mask)

                        # loss = binary_cross_entropy_with_logits(similarity,
                        #                                         mask)
                        bce_loss.append(kl)

                bce_loss = sum(bce_loss) / len(bce_loss)
                past_reg_loss += bce_loss

                # entropy_loss = sum(entropy_loss) / len(entropy_loss)
                # all_entropy_loss += entropy_loss

            past_reg_loss = past_reg_loss / len(self.distributions)
            # all_entropy_loss = all_entropy_loss / len(self.distributions)

            strategy.loss += past_reg_loss * 100


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


class EmptyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class CustomBlockModel(MultiTaskModule):
    def __init__(self):
        super().__init__()

        self.current_routing = None
        self.hidden_features = None

        # self.embeddings = nn.Embedding(2, 20)
        self.embeddings = None

        self.cv1 = DynamicMoERoutingLayer(3, 32,
                                          input_routing_size=512,
                                          input_routing_f=nn.Sequential(
                                              # nn.Conv2d(3, 3, 3),
                                              nn.AdaptiveAvgPool2d(
                                                  4)
                                          ))

        self.cv2 = DynamicMoERoutingLayer(32, 64,
                                          input_routing_size=512,
                                          input_routing_f=nn.Sequential(
                                              # nn.Conv2d(3, 3, 3),1
                                              nn.AdaptiveAvgPool2d(
                                                  4)
                                          ))

        self.cv3 = DynamicMoERoutingLayer(64, 128,
                                          input_routing_size=512,
                                          input_routing_f=nn.Sequential(
                                              # nn.Conv2d(3, 3, 3),
                                              nn.AdaptiveAvgPool2d(
                                                  4)
                                          ))

        # self.cv2 = MoERoutingLayer(32, 64, n_blocks=10,
        #                            input_routing_size=3 * 16 * 3 + 20,
        #                            input_routing_f=nn.Sequential(
        #                                # nn.Conv2d(32, 32, 3),
        #                                nn.AdaptiveAvgPool2d(
        #                                    2)
        #                            )
        #                            )
        #
        # self.cv3 = MoERoutingLayer(64, 64, n_blocks=10,
        #                            input_routing_size=3 * 16 * 3 + 20,
        #                            input_routing_f=nn.Sequential(
        #                                # nn.Conv2d(64, 64, 3),
        #                                nn.AdaptiveAvgPool2d(
        #                                    2)
        #                            )
        #                            )
        #
        # self.cv4 = MoERoutingLayer(64, 128, n_blocks=10,
        #                            input_routing_size=3 * 16 * 3 + 20,
        #                            input_routing_f=nn.Sequential(
        #                                # nn.Conv2d(64, 64, 3),
        #                                nn.AdaptiveAvgPool2d(
        #                                    2)
        #                            )
        #                            )

        self.mx = nn.AdaptiveAvgPool2d(2)
        self.in_features = 128 * 3 * 3

        # self.classifiers = HeadsRoutingLayer(128 * 4,
        #                                      3 * 16 * 3 + 20,
        #                                      input_routing_f=nn.Sequential(
        #                                          # nn.Conv2d(128, 128, 1),
        #                                          nn.AdaptiveAvgPool2d(1)
        #                                      ))

        self.classifiers = IncrementalClassifier(self.in_features,
                                                 initial_out_features=2,
                                                 masking=False)

        # self.routing_model = torchvision.models.resnet18(pretrained=True)
        # self.routing_model.fc = EmptyModule()

        self.routing_model = nn.Sequential(nn.Conv2d(3, 32, 3, 1),
                                           nn.ReLU(),
                                           nn.AvgPool2d(2),
                                           nn.Conv2d(32, 64, 3, 1),
                                           nn.ReLU(),
                                           nn.AvgPool2d(2),
                                           nn.Conv2d(64, 128, 3, 1),
                                           nn.AdaptiveAvgPool2d(2),
                                           nn.Flatten(1)
                                           )

    # def adaptation(self, dataset: CLExperience):
    #
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

        routing = self.routing_model(x)
        self.current_routing = routing

        x = self.features(x, task_labels, routing_vector=routing, tau=None)

        # logits = self.classifiers[str(task_labels)](x)
        # x = self.mx(x)
        x = torch.flatten(x, 1)

        logits = self.classifiers(x)

        return logits

        if task_labels is None:
            return self.forward_all_tasks(x)

        if isinstance(task_labels, int):
            # fast path. mini-batch is single task.
            return self.forward_single_task(x, task_labels, **kwargs)
        else:
            unique_tasks = torch.unique(task_labels)

        out = None
        for task in unique_tasks:
            task_mask = task_labels == task
            x_task = x[task_mask]
            out_task = self.forward_single_task(x_task, task.item(), **kwargs)

            if out is None:
                out = torch.empty(x.shape[0], *out_task.shape[1:],
                                  device=out_task.device)
            out[task_mask] = out_task
        return out

    def get_activations(self, x, task_labels):
        return self.features(x, task_labels=task_labels, get_activations=True)[
            1]

    def features(self, x, task_labels, routing_vector, tau=None, **kwargs):
        x = torch.relu(
            self.cv1(x, task_labels, routing_vector=routing_vector, **kwargs))
        x = torch.relu(
            self.cv2(x, task_labels, routing_vector=routing_vector, **kwargs))
        x = self.cv3(x, task_labels, routing_vector=routing_vector, **kwargs)
        # x = torch.relu(
        #     self.cv2(x, task_labels, routing_vector=routing_vector, **kwargs))
        # x = torch.relu(
        #     self.cv3(x, task_labels, routing_vector=routing_vector, **kwargs))
        # x = self.cv4(x, task_labels, routing_vector=routing_vector, **kwargs)
        # x = self.mx(x)
        # x = torch.flatten(x, 1)

        return x

    def forward_single_task(self, x, task_labels, tau=None, **kwargs):
        # routing = torch.nn.functional.adaptive_avg_pool2d(x, 4)
        #
        # routing = torch.cat([torch.nn.functional.adaptive_avg_pool2d(x, 4),
        #                      torch.nn.functional.adaptive_max_pool2d(x, 4),
        #                      -torch.nn.functional.adaptive_max_pool2d(-x, 4)],
        #                     2)
        #
        # routing = torch.flatten(routing, 1)

        # with torch.no_grad():
        routing = self.routing_model(x)

        # if self.embeddings is not None:
        #     i = torch.full((len(x),), task_labels, device=x.device)
        #     e = self.embeddings(i)
        #
        #     routing = torch.cat((routing, e), -1)
        #
        # self.current_routing = routing

        x = self.features(x, task_labels, routing_vector=routing, tau=tau)

        # logits = self.classifiers[str(task_labels)](x)
        x = self.mx(x)

        logits = self.classifiers(x, task_labels)
        return logits

        return torch.nn.functional.normalize(logits, 2, -1)


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
    # print(complete_model.embeddings.weight)

    criterion = LogitNormLoss(theta)
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

    trainer = Trainer(model=complete_model, optimizer=optimizer,
                      train_mb_size=32, top_k=2,
                      eval_mb_size=32, evaluator=eval_plugin,
                      initial_tau=10, fine_tuning_epochs=fine_tune_epochs,
                      criterion=criterion, device=device,
                      train_epochs=train_epochs)

    for current_task_id, tr in enumerate(train_stream):
        trainer.train(tr)
        trainer.eval(test_stream[:current_task_id + 1])

        trainer.model.eval()

        indexes = defaultdict(list)

        for n, m in trainer.model.named_buffers():
            if 'idx' not in n or 'global' in n:
                continue
            n = n.rsplit('.', 1)[0]
            indexes[n].extend(m.tolist())

        for k, v in indexes.items():
            print(k, collections.Counter(v))

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

    model = train(train_epochs=1,
                  fine_tune_epochs=1,
                  train_stream=train_tasks,
                  test_stream=test_tasks)

    # print(model.embeddings.weight)

    with torch.no_grad():
        for i in range(len(train_tasks)):
            x = next(iter(DataLoader(train_tasks[i].dataset, batch_size=128,
                                     shuffle=True)))[0].to(device)

            # for _ in range(5):
            pred = model(x, None)
            # print(torch.softmax(pred, -1))

            for module in model.modules():
                if isinstance(module, MoERoutingLayer):
                    print(i, module.last_distribution[0].mean(0))
                    # print(i, j, module.last_distribution[1].mean(0))
                    print()

                # for module in model.modules():
                #     if isinstance(module, MoERoutingLayer):
                #         print(
                #             f'Selection dataset {i} using task routing {j}',
                #             torch.unique(
                #                 module.last_distribution[0],
                #                 return_counts=True))
