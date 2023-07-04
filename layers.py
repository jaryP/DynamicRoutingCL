import warnings
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


class DynamicMoERoutingLayer(AbsDynamicLayer, DynamicModule):
    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 input_routing_size: int,
                 freeze_embeddings=False,
                 dynamic_entity_init=False,
                 dynamic_expansion=False,
                 **kwargs):

        super().__init__()

        self.sampler = SIMPLESampler(1)

        self.current_routing = None
        self.last_distribution = None
        self.current_input = None
        self.embeddings_initializer = None

        self.freeze_embeddings = freeze_embeddings
        self.dynamic_entity_init = dynamic_entity_init
        self.dynamic_expansion = dynamic_expansion

        self.similarity_statistics = []

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.embedding_size = input_routing_size

        self.blocks = nn.ModuleDict()
        self.blocks_embeddings = nn.ParameterDict()

        self.routing_processing = nn.Linear(input_routing_size, 128)

    def clear_blocks(self):

        indexes = []
        for n, m in self.named_buffers():
            if n.startswith('idx'):
                indexes.extend(map(str, m.tolist()))

        indexes = set(indexes)

        for i in list(self.blocks.keys()):
            if i not in indexes:
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
        tid = experience.current_experience

        if not self.freeze_embeddings:
            for t in range(tid):
                idxs = self._get_indexes(t).tolist()
                for i in idxs:
                    i = str(i)
                    if self.embeddings_initializer is not None:
                        e = self.embeddings_initializer.rsample([1])
                    else:
                        e = torch.normal(0, 0.1, [1, 128])

                    self.blocks_embeddings[i] = nn.Parameter(e)

        if not self.dynamic_expansion and tid > 0:
            return

        self.similarity_statistics = []

        for _ in range(10):
            i = self.add_block()

    def add_block(self):
        b = nn.Conv2d(in_channels=self.input_channels,
                      out_channels=self.output_channels,
                      kernel_size=3,
                      stride=1)

        for i in count():
            i = str(i)
            if i not in self.blocks:
                self.blocks[i] = b
                if self.embeddings_initializer is not None:
                    e = self.embeddings_initializer.rsample([1])
                else:
                    e = torch.normal(0, 0.1, [1, 128])
                self.blocks_embeddings[i] = nn.Parameter(e)
                return i

    @torch.no_grad()
    def freeze_logits(self, dataset, strategy, task, top_k=1):
        top_k = 1

        device = next(self.parameters()).device

        selection = self.similarity_statistics
        selection = [p[t == task] for p, t in selection]
        selection = np.concatenate(selection, 0)

        topk_rows = np.argpartition(selection, -top_k)[:, -top_k:]
        indexes = next(zip(*Counter(topk_rows.ravel()).most_common(top_k)))

        indexes = torch.tensor(indexes, device=device)

        self.register_buffer(f'idx_{task}', indexes)

        if self.dynamic_entity_init:
            ee = [b for b in self.blocks_embeddings.values()]
            ee = torch.cat(ee, 0)

            mn = ee.mean(0).detach()
            cov = torch.cov(ee.T) + torch.eye(len(mn),
                                              device=mn.device).detach()

            distribution = MultivariateNormal(mn, cov)
            self.embeddings_initializer = distribution

        if self.dynamic_expansion:
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

            for p in self.blocks[i].parameters():
                p.register_hook(hook)

            # for p in self.routing_processing.parameters():
            #     p.register_hook(hook)

            # set_requires_grad(self.blocks[i], False)

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

        if self.training:
            s = (similarity + torch.randn_like(similarity) * 0.2) / 1
            weights = torch.softmax(similarity, -1)
            # s = torch.nn.functional.normalize(s, 2, -1)
            # weights = self.sampler(s)
        else:
            idx = torch.topk(similarity, 1, -1).indices
            weights = torch.zeros_like(similarity)
            weights.scatter_(-1, idx, 1.0)

        # s = s / torch.norm(s, p=2, dim=-1, keepdim=True)

        # weights = torch.softmax(s, -1)
        # weights = self.sampler(weights)

        return weights, similarity

    def forward(self, x, task, routing_vector, prev_routing=None,
                tau=None, **kwargs):

        self.current_input = x

        routing_vector = self.process_routing(routing_vector, task)
        if prev_routing is not None:
            routing_vector += prev_routing

        self.current_routing = routing_vector

        if len(self.blocks) == 1:
            return [b(x) for b in self.blocks.values()][0]

        distribution, similarity = self.get_routing_weights(routing_vector)

        self.last_distribution = (distribution, similarity)
        self.similarity_statistics.append((similarity.detach().cpu().numpy(),
                                           task.cpu().numpy()))
        stacked_outputs = torch.stack([b(x) for b in self.blocks.values()], -1)

        distribution = distribution.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        stacked_outputs = stacked_outputs * distribution

        d = distribution.sum(-1)
        os = stacked_outputs.sum(-1) / d

        return os


class DynamicMoERoutingLayerCE(AbsDynamicLayer, DynamicModule):
    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 input_routing_size: int,
                 routing_model: nn.Module,
                 initial_blocks: int = 10,
                 freeze_embeddings=False,
                 dynamic_entity_init=False,
                 dynamic_expansion=False,
                 **kwargs):

        super().__init__()

        self.sampler = SIMPLESampler(1)

        self.current_routing = None
        self.last_distribution = None
        self.current_input = None
        self.embeddings_initializer = None

        self.freeze_embeddings = freeze_embeddings
        self.dynamic_entity_init = dynamic_entity_init
        self.dynamic_expansion = dynamic_expansion

        self.similarity_statistics = []

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.embedding_size = input_routing_size

        self.blocks = nn.ModuleDict()
        self.blocks_embeddings = nn.ParameterDict()

        self.routing_processing = nn.Linear(input_routing_size, initial_blocks,
                                            bias=False)

        self.routing_model = routing_model

        for _ in range(initial_blocks):
            self.add_block()

    def clear_blocks(self):

        indexes = []
        for n, m in self.named_buffers():
            if n.startswith('idx'):
                indexes.extend(map(str, m.tolist()))

        indexes = set(indexes)

        for i in list(self.blocks.keys()):
            if i not in indexes:
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
        tid = experience.current_experience

        if not self.dynamic_expansion:
            return

        self.similarity_statistics = []

        for _ in range(5):
            i = self.add_block()

        self.routing_processing = nn.Linear(self.routing_processing.in_features,
                                            len(self.blocks), bias=False)

    def add_block(self):
        b = nn.Conv2d(in_channels=self.input_channels,
                      out_channels=self.output_channels,
                      kernel_size=3,
                      stride=1)

        for i in count():
            i = str(i)
            if i not in self.blocks:
                self.blocks[i] = b
                return i

    @torch.no_grad()
    def freeze_logits(self, dataset, strategy, task, top_k=1):
        top_k = 1

        device = next(self.parameters()).device

        selection = self.similarity_statistics
        selection = [p[t == task] for p, t in selection]
        selection = np.concatenate(selection, 0)

        topk_rows = np.argpartition(selection, -top_k)[:, -top_k:]
        indexes = next(zip(*Counter(topk_rows.ravel()).most_common(top_k)))

        indexes = torch.tensor(indexes, device=device)

        self.register_buffer(f'idx_{task}', indexes)

        # if self.dynamic_expansion:
        #     self.clear_blocks()

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
            # if self.freeze_embeddings:
            #     self.blocks_embeddings[i].register_hook(hook)

            for p in self.blocks[i].parameters():
                p.register_hook(hook)

            # for p in self.routing_processing.parameters():
            #     p.register_hook(hook)

            # set_requires_grad(self.blocks[i], False)

        self.similarity_statistics = []

    def _get_indexes(self, task):
        return getattr(self, f'idx_{task}', None)

    def get_task_blocks(self, task):
        return getattr(self, f'global_idx_{task}', None)

    def process_routing(self, r, task):
        r = self.routing_processing(r)
        return r

    def get_routing_weights(self, x):

        # blocks_embeddings = torch.cat(list(self.blocks_embeddings.values()), 0)
        # similarity = calculate_similarity(x, blocks_embeddings)

        if self.training:
            # x = (x + torch.randn_like(x) * 1) / 1
            # weights = torch.softmax(x, -1)
            weights = nn.functional.gumbel_softmax(x, hard=True, tau=0.5)
            # s = torch.nn.functional.normalize(s, 2, -1)
            # weights = self.sampler(x)
        else:
            s = x
            idx = torch.topk(x, 1, -1).indices
            weights = torch.zeros_like(x)
            weights.scatter_(-1, idx, 1.0)

        # s = s / torch.norm(s, p=2, dim=-1, keepdim=True)

        # weights = torch.softmax(s, -1)
        # weights = self.sampler(weights)

        return weights, x

    def forward(self, x, task, routing_vector, prev_routing=None,
                tau=None, **kwargs):

        self.current_input = x

        # routing_vector = self.process_routing(routing_vector, task)
        routing_vector = self.routing_model(x)
        routing_vector = self.process_routing(routing_vector, None)

        if prev_routing is not None:
            routing_vector += prev_routing

        self.current_routing = routing_vector

        if len(self.blocks) == 1:
            return [b(x) for b in self.blocks.values()][0]

        distribution, similarity = self.get_routing_weights(routing_vector)

        self.last_distribution = (distribution, similarity)
        self.similarity_statistics.append((similarity.detach().cpu().numpy(),
                                           task.cpu().numpy()))
        stacked_outputs = torch.stack([b(x) for b in self.blocks.values()], -1)

        distribution = distribution.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        stacked_outputs = stacked_outputs * distribution

        d = distribution.sum(-1)
        os = stacked_outputs.sum(-1) / d

        return os


class DynamicMoERoutingLayerCE1(AbsDynamicLayer, DynamicModule):
    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 input_routing_size: int,
                 routing_model: nn.Module,
                 initial_blocks: int = 10,
                 freeze_embeddings=False,
                 dynamic_entity_init=False,
                 dynamic_expansion=False,
                 **kwargs):

        super().__init__()

        self.current_output = None
        self.sampler = SIMPLESampler(1)

        self.current_routing = None
        self.last_distribution = None
        self.current_input = None
        self.embeddings_initializer = None

        self.freeze_embeddings = freeze_embeddings
        self.dynamic_entity_init = dynamic_entity_init
        self.dynamic_expansion = dynamic_expansion

        self.similarity_statistics = []

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.embedding_size = input_routing_size

        self.blocks = nn.ModuleDict()
        self.blocks_embeddings = nn.ParameterDict()

        self.routing_processing = nn.Linear(input_routing_size, initial_blocks,
                                            bias=False)

        self.routing_model = routing_model

        for _ in range(initial_blocks):
            self.add_block()

    def clear_blocks(self):

        indexes = []
        for n, m in self.named_buffers():
            if n.startswith('idx'):
                indexes.extend(map(str, m.tolist()))

        indexes = set(indexes)

        for i in list(self.blocks.keys()):
            if i not in indexes:
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
        tid = experience.current_experience

        if not self.dynamic_expansion:
            return

        self.similarity_statistics = []

        for _ in range(5):
            i = self.add_block()

        self.routing_processing = nn.Linear(self.routing_processing.in_features,
                                            len(self.blocks), bias=False)

    def add_block(self):
        b = nn.Conv2d(in_channels=self.input_channels,
                      out_channels=self.output_channels,
                      kernel_size=3,
                      stride=1)

        for i in count():
            i = str(i)
            if i not in self.blocks:
                self.blocks[i] = b
                return i

    @torch.no_grad()
    def freeze_logits(self, dataset, strategy, task, top_k=1):
        top_k = 1

        device = next(self.parameters()).device

        selection = self.similarity_statistics
        selection = [p[t == task] for p, t in selection]
        selection = np.concatenate(selection, 0)

        topk_rows = np.argpartition(selection, -top_k)[:, -top_k:]
        indexes = next(zip(*Counter(topk_rows.ravel()).most_common(top_k)))

        indexes = torch.tensor(indexes, device=device)

        self.register_buffer(f'idx_{task}', indexes)

        # if self.dynamic_expansion:
        #     self.clear_blocks()

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
            # if self.freeze_embeddings:
            #     self.blocks_embeddings[i].register_hook(hook)

            for p in self.blocks[i].parameters():
                p.register_hook(hook)

            # for p in self.routing_processing.parameters():
            #     p.register_hook(hook)

            # set_requires_grad(self.blocks[i], False)

        self.similarity_statistics = []

    def _get_indexes(self, task):
        return getattr(self, f'idx_{task}', None)

    def get_task_blocks(self, task):
        return getattr(self, f'global_idx_{task}', None)

    def process_routing(self, r, task):
        r = self.routing_processing(r)
        return r

    def get_routing_weights(self, x):

        # blocks_embeddings = torch.cat(list(self.blocks_embeddings.values()), 0)
        # similarity = calculate_similarity(x, blocks_embeddings)

        if self.training:
            # x = (x + torch.randn_like(x) * 1) / 1
            weights = torch.softmax(x, -1)
            # weights = nn.functional.gumbel_softmax(x, hard=True, tau=0.5)
            # s = torch.nn.functional.normalize(s, 2, -1)
            # weights = self.sampler(x)
        else:
            s = x
            idx = torch.topk(x, 1, -1).indices
            weights = torch.zeros_like(x)
            weights.scatter_(-1, idx, 1.0)

        # s = s / torch.norm(s, p=2, dim=-1, keepdim=True)

        # weights = torch.softmax(s, -1)
        # weights = self.sampler(weights)

        return weights, x

    def forward(self, x, task, routing_vector, prev_routing=None,
                tau=None, **kwargs):

        self.current_input = x

        # routing_vector = self.process_routing(routing_vector, task)
        routing_vector = self.routing_model(x)
        routing_vector = self.process_routing(routing_vector, None)

        if prev_routing is not None:
            routing_vector += prev_routing

        self.current_routing = routing_vector

        if len(self.blocks) == 1:
            return [b(x) for b in self.blocks.values()][0]

        idx = self._get_indexes(task)
        if idx is not None and self.training:
            os = self.blocks[idx](x)
            others = [j(x)
                      for i, j in self.blocks.items() if i not in idx]
            others = torch.stack(others, 0)
            self.last_distribution = (None, None)
        else:
            distribution, similarity = self.get_routing_weights(routing_vector)

            self.last_distribution = (distribution, similarity)
            self.similarity_statistics.append((similarity.detach().cpu().numpy(),
                                               task.cpu().numpy()))
            stacked_outputs = torch.stack([b(x) for b in self.blocks.values()], -1)

            distribution = distribution.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            stacked_outputs = stacked_outputs * distribution

            d = distribution.sum(-1)
            os = stacked_outputs.sum(-1) / d

            others = None

        self.current_output = (os, others)

        return os


class RoutingLayer(AbsDynamicLayer, DynamicModule):
    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 # input_routing_size: int,
                 # freeze_embeddings=False,
                 # dynamic_entity_init=False,
                 # dynamic_expansion=False,
                 **kwargs):

        super().__init__()

        self.sampler = SIMPLESampler(1)

        self.current_routing = None
        self.last_distribution = None
        self.current_input = None
        self.embeddings_initializer = None

        # self.freeze_embeddings = freeze_embeddings
        # self.dynamic_entity_init = dynamic_entity_init
        # self.dynamic_expansion = dynamic_expansion

        self.similarity_statistics = []

        self.input_channels = input_channels
        self.output_channels = output_channels
        # self.embedding_size = input_routing_size

        self.blocks = nn.ModuleDict()
        # self.blocks_embeddings = nn.ParameterDict()

        # self.routing_processing = nn.Linear(input_routing_size, 128)

        self._paths = dict()
        self._task_paths = dict()

        for i in range(10):
            b = nn.Conv2d(in_channels=self.input_channels,
                          out_channels=self.output_channels,
                          kernel_size=3,
                          stride=1)

            i = str(i)
            self.blocks[i] = b

    def add_path(self, block_id):
        if str(block_id) not in self.blocks:
            raise ValueError(f'The block id {block_id} '
                             f'is not present in the blocks list.')

        self._paths[len(self._paths)] = block_id

    def assign_path_to_task(self, path_id, task_id):
        if task_id in self._task_paths.keys():
            warnings.warn(f'The task id {task_id} '
                          f'was already assigned to a path '
                          f'{self._task_paths[task_id]}')

        if path_id not in self._paths.keys():
            raise ValueError(f'The path id {path_id} '
                             f'is not present in the paths list.')

        self._task_paths[task_id] = self._paths[path_id]
        del self._paths[path_id]

    def get_unassigned_paths(self):
        return list(self._paths.keys())

    @property
    def paths(self):
        return self._paths

    @property
    def tasks_path(self):
        return self._task_paths

    def clear_blocks(self):
        indexes = []
        for n, m in self.named_buffers():
            if n.startswith('idx'):
                indexes.extend(map(str, m.tolist()))

        indexes = set(indexes)

        for i in list(self.blocks.keys()):
            if i not in indexes:
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

    # def train_adaptation(self, experience):
    #     tid = experience.current_experience
    #
    #     if not self.freeze_embeddings:
    #         for t in range(tid):
    #             idxs = self._get_indexes(t).tolist()
    #             for i in idxs:
    #                 i = str(i)
    #                 if self.embeddings_initializer is not None:
    #                     e = self.embeddings_initializer.rsample([1])
    #                 else:
    #                     e = torch.normal(0, 0.1, [1, 128])
    #
    #                 self.blocks_embeddings[i] = nn.Parameter(e)
    #
    #     if not self.dynamic_expansion and tid > 0:
    #         return
    #
    #     self.similarity_statistics = []
    #
    #     for _ in range(10):
    #         i = self.add_block()

    # def add_block(self):
    #     b = nn.Conv2d(in_channels=self.input_channels,
    #                   out_channels=self.output_channels,
    #                   kernel_size=3,
    #                   stride=1)
    #
    #     for i in count():
    #         i = str(i)
    #         if i not in self.blocks:
    #             self.blocks[i] = b
    #             if self.embeddings_initializer is not None:
    #                 e = self.embeddings_initializer.rsample([1])
    #             else:
    #                 e = torch.normal(0, 0.1, [1, 128])
    #             self.blocks_embeddings[i] = nn.Parameter(e)
    #             return i

    @torch.no_grad()
    def freeze_logits(self, dataset, strategy, task, top_k=1):
        top_k = 1

        device = next(self.parameters()).device

        selection = self.similarity_statistics
        selection = [p[t == task] for p, t in selection]
        selection = np.concatenate(selection, 0)

        topk_rows = np.argpartition(selection, -top_k)[:, -top_k:]
        indexes = next(zip(*Counter(topk_rows.ravel()).most_common(top_k)))

        indexes = torch.tensor(indexes, device=device)

        self.register_buffer(f'idx_{task}', indexes)

        if self.dynamic_entity_init:
            ee = [b for b in self.blocks_embeddings.values()]
            ee = torch.cat(ee, 0)

            mn = ee.mean(0).detach()
            cov = torch.cov(ee.T) + torch.eye(len(mn),
                                              device=mn.device).detach()

            distribution = MultivariateNormal(mn, cov)
            self.embeddings_initializer = distribution

        if self.dynamic_expansion:
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

        for p in self.blocks[str(self._task_paths[task])].parameters():
            p.register_hook(hook)

        return
        for i in self._task_paths[task]:
            i = str(i)
            if self.freeze_embeddings:
                self.blocks_embeddings[i].register_hook(hook)

            for p in self.blocks[i].parameters():
                p.register_hook(hook)

            # for p in self.routing_processing.parameters():
            #     p.register_hook(hook)

            # set_requires_grad(self.blocks[i], False)

        self.similarity_statistics = []

    def _get_indexes(self, task):
        return getattr(self, f'idx_{task}', None)

    def get_task_blocks(self, task):
        return getattr(self, f'global_idx_{task}', None)

    def process_routing(self, r, task):
        r = self.routing_processing(r)
        return r

    # def get_routing_weights(self, x):
    #
    #     blocks_embeddings = torch.cat(list(self.blocks_embeddings.values()), 0)
    #
    #     similarity = calculate_similarity(x, blocks_embeddings)
    #
    #     if self.training:
    #         s = (similarity + torch.randn_like(similarity) * 0.2) / 1
    #         weights = torch.softmax(similarity, -1)
    #         # s = torch.nn.functional.normalize(s, 2, -1)
    #         # weights = self.sampler(s)
    #     else:
    #         idx = torch.topk(similarity, 1, -1).indices
    #         weights = torch.zeros_like(similarity)
    #         weights.scatter_(-1, idx, 1.0)
    #
    #     # s = s / torch.norm(s, p=2, dim=-1, keepdim=True)
    #
    #     # weights = torch.softmax(s, -1)
    #     # weights = self.sampler(weights)
    #
    #     return weights, similarity
    #
    # def forward(self, x, task, routing_vector, prev_routing=None,
    #             tau=None, **kwargs):
    #
    #     self.current_input = x
    #
    #     routing_vector = self.process_routing(routing_vector, task)
    #     if prev_routing is not None:
    #         routing_vector += prev_routing
    #
    #     self.current_routing = routing_vector
    #
    #     if len(self.blocks) == 1:
    #         return [b(x) for b in self.blocks.values()][0]
    #
    #     distribution, similarity = self.get_routing_weights(routing_vector)
    #
    #     self.last_distribution = (distribution, similarity)
    #     self.similarity_statistics.append((similarity.detach().cpu().numpy(),
    #                                        task.cpu().numpy()))
    #     stacked_outputs = torch.stack([b(x) for b in self.blocks.values()], -1)
    #
    #     distribution = distribution.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    #     stacked_outputs = stacked_outputs * distribution
    #
    #     d = distribution.sum(-1)
    #     os = stacked_outputs.sum(-1) / d
    #
    #     return os

    def forward(self, x, task_id=None, path_id=None, **kwargs):

        if task_id is None and path_id is None:
            raise ValueError('Both task_id and path_id are set to None.')

        if task_id is not None and path_id is not None:
            raise ValueError('Both task_id and path_id are not None. '
                             'Select just one.')

        if task_id is None:
            if path_id == 'random':
                i = np.random.choice(np.asarray(list(self._paths.keys())), 1)[0]
            elif isinstance(path_id, (np.integer, int)):
                i = list(self.paths.keys())[path_id]
                # i = path_id
            else:
                raise ValueError('parameter path_id must be an integer '
                                 'or "random".')

            block_id = str(self._paths[i])
        else:
            block_id = str(self._task_paths[task_id])

        f = self.blocks[block_id](x)

        return f
