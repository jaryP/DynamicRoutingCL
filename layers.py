from abc import ABC, abstractmethod
from collections import Counter
from itertools import count
from typing import Callable

import numpy as np
import torch
import torchvision
from avalanche.benchmarks.utils import ConstantSequence
from avalanche.models import DynamicModule
from torch import nn as nn
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader

from utils import calculate_distance, calculate_similarity
from simple import SIMPLESampler


class ProcessRouting(nn.Module):
    model = torchvision.models.resnet18(pretrained=True)

    def forward(self, x):
        return

def set_requires_grad(m, requires_grad):
    for param in m.parameters():
        param.requires_grad_(requires_grad)


class AbsDynamicLayer(nn.Module, ABC):
    @abstractmethod
    def freeze_logits(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def freeze_blocks(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward(self, **kwargs):
        raise NotImplementedError


class MoERoutingLayer(AbsDynamicLayer, DynamicModule):
    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 input_routing_size: int,
                 input_routing_f: Callable,
                 random_path=False,
                 use_task_subspace=False,
                 deterministic=False,
                 loss='centroids',
                 **kwargs):

        super().__init__()

        self.sampler = SIMPLESampler(1)
        self.classifier = None
        self.current_routing = None
        self.embeddings_to_initialize = True
        self.current_batch_similarity_loss = None
        self.last_image_routing = None
        self.last_distribution = None
        self.current_input = None
        self.use_task_subspace = use_task_subspace
        self.random_path = random_path

        self.similarity_statistics = []
        self.deterministic = deterministic
        self.input_routing_f = input_routing_f
        self.loss = loss

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.embedding_size = input_routing_size

        self.blocks = nn.ModuleDict()
        self.blocks_embeddings = nn.ParameterDict()
        self.tasks_projector = nn.ModuleDict()

        # self.null_task_embedding = torch.zeros()

        self.routing_processing = nn.Sequential(
            nn.Linear(input_routing_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64))

        self.num_tasks = torch.nn.Parameter(torch.LongTensor([0]),
                                            requires_grad=False)

    def clear_blocks(self):
        indexes = []
        for n, m in self.named_buffers():
            if n.startswith('idx'):
                indexes.extend(m.tolist())

        indexes = set(indexes)

        for i in range(len(self.blocks)):
            if i not in indexes:
                i = str(i)
                del self.blocks[i]
                if self.loss == 'centroids':
                    del self.blocks_embeddings[i]

    def reset_blocks(self):
        indexes = []
        for n, m in self.named_buffers():
            if n.startswith('idx'):
                indexes.extend(m.tolist())

        indexes = set(indexes)

        for i in range(len(self.blocks)):
            if i not in indexes:
                i = str(i)
                self.blocks[i].reset_parameters()

    def _process_routing(self, x):
        x = self.input_routing_f(x)
        x = torch.flatten(x, 1)

        # std = x.std(0)
        # x = (x - x.mean(0)) / torch.where(torch.logical_or(std == 0,
        #                                                    torch.isnan(std)),
        #                                   1, std)

        return x

    def train_adaptation(self, experience):
        # if self.num_tasks:
        self.num_tasks += 1

        tid = str(experience.current_experience)

        self.similarity_statistics = []

        if self.use_task_subspace:
            projector = nn.Linear(self.embedding_size,
                                  self.embedding_size, bias=False)
            self.tasks_projector[str(tid)] = projector

        # return
        if tid == '0':
            for _ in range(10):
                self.add_block()

            for i, b in self.blocks_embeddings.items():
                self.blocks_embeddings[i] = nn.Parameter(
                    torch.normal(0, 1, [1, 64]))

        # else:
        #     self.reset_blocks()

        if self.loss == 'cross_entropy':
            self.classifier = nn.Linear(self.embedding_size,
                                        len(self.blocks))

    def add_block(self):
        b = nn.Conv2d(in_channels=self.input_channels,
                      out_channels=self.output_channels,
                      kernel_size=3,
                      stride=1)

        for i in count():
            i = str(i)
            # e = torch.normal(0, 0.1, [1, self.embedding_size])
            # self.blocks_embeddings[i] = e
            if i not in self.blocks:
                self.blocks[i] = b
                if self.loss == 'centroids':
                    e = torch.normal(0, 1, [1, 64])
                    self.blocks_embeddings[i] = nn.Parameter(e)
                return

    @torch.no_grad()
    def freeze_logits(self, dataset, strategy, task, top_k=1):
        top_k = 1

        device = next(self.parameters()).device

        selection = self.similarity_statistics
        selection = np.concatenate(selection, 0)

        topk_rows = np.argpartition(selection, -top_k)[:, -top_k:]
        indexes = next(zip(*Counter(topk_rows.ravel()).most_common(top_k)))

        indexes = torch.tensor(indexes, device=device)

        self.register_buffer(f'idx_{task}', indexes)

        # self.clear_blocks()

        new_idxs = []
        for i, k in enumerate(self.blocks.keys()):
            if int(k) in indexes:
                new_idxs.append(i)
        new_idxs = torch.tensor(new_idxs, device=device)

        if self.loss == 'cross_entropy':
            if task > 0:
                neurons_to_extract = torch.cat([getattr(self, f'global_idx_{i}')
                                                for i in range(task)] + [
                                                   new_idxs],
                                               0)
                neurons_to_extract = torch.unique(neurons_to_extract)
            else:
                neurons_to_extract = new_idxs
            new_classifier = nn.Linear(self.embedding_size,
                                       len(neurons_to_extract),
                                       device=device)

            new_classifier.weight.data = self.classifier.weight.data[
                neurons_to_extract]
            self.classifier = new_classifier

        self.register_buffer(f'global_idx_{task}', new_idxs)

        self.similarity_statistics = []

        return None, indexes

    @torch.no_grad()
    def freeze_blocks(self, task, top_k=2):
        # return

        def hook(grad_input):
            return torch.zeros_like(grad_input)

        idxs = self._get_indexes(task)
        # set_requires_grad(self.routing_nn, False)
        # self.entity_embeddings[str(task)].register_hook(hook)

        for i in idxs.tolist():
            i = str(i)
            # self.blocks_embeddings[i].register_hook(hook)
            set_requires_grad(self.blocks[i], False)

        self.similarity_statistics = []

    def _get_indexes(self, task):
        return getattr(self, f'idx_{task}', None)

    def get_task_blocks(self, task):
        return getattr(self, f'global_idx_{task}', None)

    def process_routing(self, r, task):
        r = self.routing_processing(r)
        if str(task) in self.tasks_projector:
            r = self.tasks_projector[str(task)](r)

        return r

    def get_routing_weights(self, x, task, embeddings=None, idxs=None,
                            augment=False):

        # if self.embeddings_to_initialize:
        #     self.embeddings_to_initialize = False
        #
        #     means = x.mean(0)
        #     cov = torch.cov(x.T, correction=0) + torch.eye(x.shape[-1],
        #                                                    device=x.device)
        #     d = torch.distributions.multivariate_normal.MultivariateNormal(means, cov)
        #
        #     for k in self.blocks_embeddings:
        #         if self.blocks_embeddings[k] is None:
        #             self.blocks_embeddings[k].data = d.rsample([1])
        # if self.loss == 'cross_entropy':
        #     similarity = self.classifier(x)
        #     s = similarity
        #
        #     if self.training or augment:
        #         s = (similarity + torch.randn_like(similarity) * 0.1) / 1
        #
        #     weights = torch.sigmoid(s)
        #
        # if self.loss == 'centroids':

        # if embeddings is None:
        #     embeddings = list(self.blocks_embeddings.values())

            # if (self.random_path or idxs is None) or not self.training:
            #     embeddings = list(self.blocks_embeddings.values())
            # else:
            #     idxs = idxs.tolist()
            #     embeddings = [self.blocks_embeddings[str(i)] for i in idxs]

        embeddings = [self.blocks_embeddings[str(i)]
                      for i in range(len(self.blocks_embeddings))]
        blocks_embeddings = torch.cat(embeddings, 0)

        # if str(task) in self.tasks_projector:
        #     x = self.tasks_projector[str(task)](
        #         blocks_embeddings)

        similarity = calculate_similarity(x, blocks_embeddings)

        if self.training:
            s = (similarity + torch.randn_like(similarity) * 0.1) / 1
        else:
            s = similarity

        weights = torch.softmax(s, -1)
        # weights = nn.functional.gumbel_softmax(logits=weights, hard=True)
        weights = self.sampler(weights)

        # s = similarity
        # if self.training or augment:
        #     s = (s + torch.randn_like(s) * 0.1) / 1
        #
        # weights = torch.softmax(s, -1)

        return weights, similarity

    def _similarity_loss(self, x, task):
        if not self.training:
            return None, None

        x = self._process_routing(x)
        return self._similarity_loss_w_routing_input(x, task)

    def _similarity_loss_w_routing_input(self, x, task):
        # if self.num_tasks == 1:
        #     return None, None

        push_loss = pull_loss = None

        idxs = self._get_indexes(task)

        if idxs is not None:
            with torch.no_grad():
                n_blocks = len(self.blocks)
                mask = torch.zeros(n_blocks, device=x.device)
                # mask[:, idxs] = 1

                for i, k in enumerate(self.blocks.keys()):
                    if int(k) in idxs:
                        mask[i] = 1

                embeddings = torch.cat(list(self.blocks_embeddings.values()), 0)
                distance = calculate_similarity(x, embeddings)
                distance = mask * distance + (1 - mask) * 100

                labels = torch.argmax(distance, -1)

            embeddings = torch.cat(list(self.blocks_embeddings.values()), 0)
            distances = calculate_similarity(x, embeddings)
            if self.loss == 'centroids':
                log_p_y = torch.log_softmax(distances, dim=1)
                pull_loss = -log_p_y.gather(1, labels.unsqueeze(-1))
            else:
                pull_loss = nn.functional.cross_entropy(distances, labels)

        # push_loss = None
        # pull_loss = None
        #
        # to_push = [i for tid in range(self.num_tasks.item() - 1)
        #            for i in self._get_indexes(tid).tolist()]
        #
        # if idxs is not None:
        #     idxs = idxs.tolist()
        #     embeddings = torch.cat([self.blocks_embeddings[str(i)]
        #                             for i in idxs], 0)
        #
        #     pull_loss = calculate_distance(x, embeddings)
        #
        #     to_push = list(set(to_push) - set(idxs))
        #
        #     if len(to_push) == 0:
        #         return push_loss, pull_loss
        #
        # embeddings = torch.cat([self.blocks_embeddings[str(i)]
        #                         for i in to_push], 0)
        #
        # distances = calculate_distance(x, embeddings)
        # push_loss = 1 / (1 + distances)

        return push_loss, pull_loss

    def forward(self, x, task, routing_vector, tau=None, **kwargs):

        self.current_input = x

        routing_vector = self.process_routing(routing_vector, task)

        self.current_routing = routing_vector

        # routing = self._process_routing(x)
        # idxs = self.get_task_blocks(task)
        idxs = None
        distribution, similarity = self.get_routing_weights(routing_vector,
                                                            task=task,
                                                            idxs=idxs)

        self.last_distribution = (distribution, similarity)
        self.similarity_statistics.append(similarity.detach().cpu().numpy())
        stacked_outputs = torch.stack([b(x) for b in self.blocks.values()], -1)

        # if (self.random_path or idxs is None) or not self.training:
        #     stacked_outputs = torch.stack([b(x) for b in self.blocks.values()],
        #                                   -1)
        # else:
        #     stacked_outputs = torch.stack([self.blocks[str(i)](x)
        #                                    for i in idxs.tolist()], -1)
        #
        #     if self.loss == 'cross_entropy' \
        #             and distribution.shape[-1] > len(idxs):
        #         distribution = distribution[:, self.get_task_blocks(task)]
        # if not self.training:
        # #     if idxs is not None:
        # #         mask = torch.zeros_like(distribution[0])
        # #         mask.scatter_(-1, idxs, 1)
        # #         distribution = distribution * mask
        # # else:
        #     topk = torch.topk(distribution, 1).indices
        #     mask = torch.zeros_like(distribution).scatter_(-1, topk, 1.0)
        #     distribution = distribution * mask

        # if self.random_path:
        #     # pass
        #     index = distribution.max(-1, keepdim=True)[1]
        #     y_hard = torch.zeros_like(similarity,
        #                               memory_format=torch.legacy_contiguous_format).scatter_(
        #         -1, index, 1.0)
        #     distribution = y_hard - distribution.detach() + distribution
        # elif not self.training:
        #     topk = torch.topk(distribution, len(idxs)).indices
        #     mask = torch.zeros_like(distribution).scatter_(-1, topk, 1.0)
        #     distribution = distribution * mask

        distribution = distribution.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        stacked_outputs = stacked_outputs * distribution

        d = distribution.sum(-1)
        os = stacked_outputs.sum(-1) / d

        return os


class DynamicMoERoutingLayer(AbsDynamicLayer, DynamicModule):
    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 input_routing_size: int,
                 freeze_embeddings=True,
                 **kwargs):

        super().__init__()

        self.sampler = SIMPLESampler(1)

        self.current_routing = None
        self.last_distribution = None
        self.current_input = None
        self.embeddings_initializer = None

        self.freeze_embeddings = freeze_embeddings

        self.similarity_statistics = []

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.embedding_size = input_routing_size

        self.blocks = nn.ModuleDict()
        self.blocks_embeddings = nn.ParameterDict()

        self.routing_processing = nn.Sequential(
            nn.Linear(input_routing_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64))

    def clear_blocks(self):

        if self.freeze_embeddings:
            ee = [b for b in self.blocks_embeddings.values()]
            ee = torch.cat(ee, 0)

            mn = ee.mean(0).detach()
            cov = torch.cov(ee.T) + torch.eye(len(mn), device=mn.device).detach()

            distribution = MultivariateNormal(mn, cov)
            self.embeddings_initializer = distribution

        indexes = []
        for n, m in self.named_buffers():
            if n.startswith('idx'):
                indexes.extend(m.tolist())

        indexes = set(indexes)

        # keys =
        for i in list(self.blocks.keys()):
            i = int(i)
            if i not in indexes:
                i = str(i)
                del self.blocks[i]
                del self.blocks_embeddings[i]

    def reset_blocks(self):
        indexes = []
        for n, m in self.named_buffers():
            if n.startswith('idx'):
                indexes.extend(m.tolist())

        indexes = set(indexes)

        for i in range(len(self.blocks)):
            if i not in indexes:
                i = str(i)
                self.blocks[i].reset_parameters()

    def train_adaptation(self, experience):
        self.num_tasks += 1

        tid = experience.current_experience

        self.similarity_statistics = []

        for _ in range(10):
            i = self.add_block()

        if not self.freeze_embeddings:
            for t in range(tid):
                idxs = self._get_indexes(t)
                for i in idxs:
                    i = str(i)
                    if self.embeddings_initializer is not None:
                        e = self.embeddings_initializer.rsample([1])
                    else:
                        e = torch.normal(0, 1, [1, 64])

                    self.blocks_embeddings[i] = nn.Parameter(e)

    def add_block(self):
        b = nn.Conv2d(in_channels=self.input_channels,
                      out_channels=self.output_channels,
                      kernel_size=3,
                      stride=2)

        for i in count():
            i = str(i)
            if i not in self.blocks:
                self.blocks[i] = b
                if self.embeddings_initializer is not None:
                    e = self.embeddings_initializer.rsample([1])
                else:
                    e = torch.normal(0, 1, [1, 64])
                self.blocks_embeddings[i] = nn.Parameter(e)
                return i

    @torch.no_grad()
    def freeze_logits(self, dataset, strategy, task, top_k=1):
        top_k = 1

        device = next(self.parameters()).device

        selection = self.similarity_statistics
        selection = np.concatenate(selection, 0)

        topk_rows = np.argpartition(selection, -top_k)[:, -top_k:]
        indexes = next(zip(*Counter(topk_rows.ravel()).most_common(top_k)))

        indexes = torch.tensor(indexes, device=device)

        self.register_buffer(f'idx_{task}', indexes)

        self.clear_blocks()

        new_idxs = []
        for i, k in enumerate(self.blocks.keys()):
            if int(k) in indexes:
                new_idxs.append(i)
        new_idxs = torch.tensor(new_idxs, device=device)

        self.register_buffer(f'global_idx_{task}', new_idxs)

        self.similarity_statistics = []

        return None, indexes

    @torch.no_grad()
    def freeze_blocks(self, task, top_k=2):
        def hook(grad_input):
            return torch.zeros_like(grad_input)

        idxs = self._get_indexes(task)
        # set_requires_grad(self.routing_nn, False)
        # self.entity_embeddings[str(task)].register_hook(hook)

        for i in idxs.tolist():
            i = str(i)
            if self.freeze_embeddings:
                self.blocks_embeddings[i].register_hook(hook)
            set_requires_grad(self.blocks[i], False)

        self.similarity_statistics = []

    def _get_indexes(self, task):
        return getattr(self, f'idx_{task}', None)

    def get_task_blocks(self, task):
        return getattr(self, f'global_idx_{task}', None)

    def process_routing(self, r, task):
        r = self.routing_processing(r)
        return r

    def get_routing_weights(self, x):

        blocks_embeddings = torch.cat(list(self.blocks_embeddings.values()), 0)

        similarity = calculate_similarity(x, blocks_embeddings)

        # if self.training:
        #     s = (similarity + torch.randn_like(similarity) * 0.1) / 1
        # else:
        s = similarity

        weights = torch.softmax(s, -1)
        weights = self.sampler(weights)

        return weights, similarity

    def forward(self, x, task, routing_vector, tau=None, **kwargs):

        self.current_input = x

        routing_vector = self.process_routing(routing_vector, task)

        self.current_routing = routing_vector

        distribution, similarity = self.get_routing_weights(routing_vector)

        self.last_distribution = (distribution, similarity)
        self.similarity_statistics.append(similarity.detach().cpu().numpy())
        stacked_outputs = torch.stack([b(x) for b in self.blocks.values()], -1)

        distribution = distribution.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        stacked_outputs = stacked_outputs * distribution

        d = distribution.sum(-1)
        os = stacked_outputs.sum(-1) / d

        return os


# class HeadsRoutingLayer(AbsDynamicLayer, DynamicModule):
#     def __init__(self,
#                  input_dimension: int,
#                  # output_channels: int,
#                  input_routing_size: int,
#                  input_routing_f: Callable,
#                  loss='cross_entropy',
#                  deterministic=False,
#                  **kwargs):
#
#         super().__init__()
#
#         self.in_features = input_dimension
#         self.current_batch_similarity_loss = None, None
#         self.last_image_routing = None
#         self.last_distribution = None
#         self.input = None
#         self.similarity_statistics = []
#         self.deterministic = deterministic
#         self.input_routing_f = input_routing_f
#         self.loss = loss
#
#         self.embedding_size = input_routing_size
#
#         self.heads = nn.ModuleDict()
#         self.blocks_embeddings = nn.ParameterDict()
#
#         for tid in range(10):
#             e = nn.Parameter(
#                 torch.normal(0, 1, [1, self.embedding_size]),
#                 requires_grad=True)
#
#             self.blocks_embeddings[str(tid)] = e
#
#         self.entity_embeddings = nn.ParameterDict()
#
#         self.num_tasks = torch.nn.Parameter(torch.LongTensor([0]),
#                                             requires_grad=False)
#
#     def train_adaptation(self, experience):
#
#         task_labels = experience.task_labels
#         if isinstance(task_labels, ConstantSequence):
#             task_labels = [task_labels[0]]
#
#         for tid in set(task_labels):
#             tid = str(tid)  # need str keys
#             n_classes = len(set(experience.classes_in_this_experience))
#             if tid not in self.heads:
#                 self.heads[tid] = nn.Linear(self.in_features, n_classes)
#
#             # if tid not in self.blocks_embeddings:
#             e = nn.Parameter(
#                 torch.normal(0, 1, [1, self.embedding_size]),
#                 requires_grad=True)
#
#             self.blocks_embeddings[tid] = e
#
#     @torch.no_grad()
#     def freeze_logits(self, dataset, strategy, task, top_k=2):
#         return
#
#     @torch.no_grad()
#     def freeze_blocks(self, task, top_k=2):
#         return
#
#     def _get_indexes(self, task):
#         return getattr(self, f'idx_{task}', None)
#
#     def _process_routing(self, x):
#         x = self.input_routing_f(x)
#         x = torch.flatten(x, 1)
#         x = (x - x.mean(0)) / torch.where(x.std(0) == 0, 1, x.std(0))
#
#         return x
#
#     def get_weights(self, x, task, idxs=None):
#         x = self._process_routing(x)
#
#         keys = self.heads.keys()
#         be = [self.blocks_embeddings[k] for k in keys]
#         blocks_embeddings = torch.cat(be, 0)
#
#         distances = calculate_similarity(x, blocks_embeddings)
#
#         mx = torch.argmax(distances, -1)
#         weights = torch.nn.functional.one_hot(mx, distances.shape[-1])
#
#         return mx, distances
#
#     def _similarity_loss(self, x, task):
#         if not self.training:
#             return None, None
#
#         x = self._process_routing(x)
#
#         return self._similarity_loss_w_routing_input(x, task)
#
#     def _similarity_loss_w_routing_input(self, x, task):
#         # if len(self.heads) == 1:
#         #     return None, None
#
#         push_loss = None
#         pull_loss = None
#
#         # idxs = self._get_indexes(task)
#
#         to_push = []
#         to_pull = []
#
#         distances = [calculate_similarity(x, e)
#                      for e in self.blocks_embeddings.values()]
#         distances = torch.cat(distances, -1)
#
#         if self.loss == 'centroids':
#             log_p_y = torch.log_softmax(distances, dim=1)
#             pull_loss = -log_p_y[:, task]
#         else:
#             pull_loss = nn.functional.cross_entropy(distances,
#                                                     torch.full(
#                                                         (len(distances),),
#                                                         task,
#                                                         device=distances.device))
#
#         # log_p_y = torch.log_softmax(distances, dim=1)
#         # loss = -log_p_y[:, task]
#         # pull_loss = loss
#
#         # for k, e in self.blocks_embeddings.items():
#         #     dist = calculate_distance(x, e)
#         #     if k == str(task):
#         #         to_pull.append(dist)
#         #     else:
#         #         sim = 1 / (1 + dist)
#         #         to_push.append(sim)
#         #
#         # # if idxs is not None:
#         # #     idxs = idxs.tolist()
#         # #     embeddings = torch.cat([self.blocks_embeddings[str(i)]
#         # #                             for i in idxs], 0)
#         # #
#         # #     pull_loss = calculate_distance(x, embeddings)
#         # #
#         # #     to_push = list(set(to_push) - set(idxs))
#         # #
#         # #     if len(to_push) == 0:
#         # #         return push_loss, pull_loss
#         # #
#         # # embeddings = torch.cat([self.blocks_embeddings[str(i)]
#         # #                         for i in to_push], 0)
#         # #
#         # # distances = calculate_distance(x, embeddings)
#         # # push_loss = 1 / (1 + distances)
#         #
#         # if len(to_push) > 0:
#         #     push_loss = torch.cat(to_push, -1)
#         #
#         # if len(to_push) > 1:
#         #     pull_loss = torch.cat(to_pull, -1)
#         # else:
#         #     pull_loss = to_pull[0]
#
#         return push_loss, pull_loss
#
#     def _save_routing_labels(selfx, task, **kwargs):
#         pass
#
#     def forward(self, x, task, tau=None, **kwargs):
#
#         self.input = x
#
#         routing_x = x
#         x = torch.flatten(x, 1)
#
#         # idxs = self._get_indexes(task)
#         # if task == 0:
#         os = self.heads[str(task)](x)
#         # else:
#         #     os = [self.heads[str(t)](x) for t in range(task+1)]
#         #     os = torch.cat(os, -1)
#
#         # if self.training:
#         #     os = self.heads[str(task)](x)
#         # else:
#         #     if len(self.heads) == 1:
#         #         os = next(iter(self.heads.values()))(x)
#         #     else:
#         #         os = []
#         #
#         #         distribution, distances = self.get_weights(routing_x,
#         #                                                    task,
#         #                                                    None)
#         #
#         #         correct = torch.zeros(len(distances), 1,
#         #                               device=distances.device)
#         #
#         #         for i, (_x, _d) in enumerate(zip(x, distribution)):
#         #             predicted_task = _d.item()
#         #             scores = self.heads[str(predicted_task)](_x)
#         #
#         #             if predicted_task != task:
#         #                 correct[i] = scores.max(-1)[0] + 1
#         #
#         #             os.append(scores)
#         #
#         #         mp = list(map(len, os))
#         #         mx = max(mp)
#         #         if min(mp) != mx:
#         #             os = [o if len(o) == mx else
#         #                   nn.functional.pad(o, (0, mx - len(o))) for o in os]
#         #
#         #         os = torch.stack(os, 0)
#         #         os = torch.cat((os, correct), -1)
#         #
#         # self.current_batch_similarity_loss = self._similarity_loss(routing_x, task)
#
#         return os
