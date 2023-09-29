from builtins import enumerate
from copy import deepcopy
from itertools import chain
from typing import Optional, Sequence, Iterable, Union, Tuple, Any

from avalanche.training import ExperienceBalancedBuffer, ClassBalancedBuffer, \
    Replay
from avalanche.training.supervised.der import DER
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
# import distinctipy

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
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader, \
    TaskBalancedDataLoader
from avalanche.core import SupervisedPlugin
from avalanche.evaluation.metrics import accuracy_metrics, bwt_metrics
from avalanche.logging import TextLogger
from avalanche.models import MultiTaskModule, IncrementalClassifier, \
    avalanche_forward, MultiHeadClassifier, DynamicModule
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates import SupervisedTemplate

from avalanche.training.utils import trigger_plugins, cycle
from sympy.vector import curl
from torch import cosine_similarity, Tensor
from torch.nn import CrossEntropyLoss
from torch.nn.functional import binary_cross_entropy_with_logits, \
    binary_cross_entropy
from torch.nn.modules.loss import _Loss
from torch.optim import Adam, Optimizer, SGD
from collections import defaultdict

from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms import transforms

from layers import AbsDynamicLayer, DynamicMoERoutingLayerCE, \
    DynamicMoERoutingLayerCE1, RoutingLayer, BlockRoutingLayer
from utils import calculate_distance


def calculate_similarity(x, y, distance: str = None, sigma=1):
    # if distance is None:
    #     distance = self.similarity
    # return cosine_similarity(x, y, -1)
    if distance is None:
        distance = 'cosine'

    # n = x.size(0)
    # m = y.size(0)
    # d = x.size(1)
    # if d != y.size(1):
    #     raise Exception

    # a = x.unsqueeze(1).expand(n, m, d)
    # b = y.unsqueeze(0).expand(n, m, d)

    if distance == 'euclidean':
        similarity = -torch.pow(x - y, 2).sum(2).sqrt()
    elif distance == 'rbf':
        similarity = -torch.pow(x - y, 2).sum(2).sqrt()
        similarity = similarity / (2 * sigma ** 2)
        similarity = torch.exp(similarity)
    elif distance == 'cosine':
        similarity = cosine_similarity(x, y, -1)
    else:
        assert False

    return similarity


class CosineLinearLayer(nn.Linear):
    def __init__(self, in_features: int):
        super().__init__(in_features, out_features=1, bias=False)

    def forward(self, input: Tensor) -> Tensor:
        return 1 / torch.norm(input[:, None] - self.weight[None, :], 2, -1)
        cos = torch.cosine_similarity(input[:, None], self.weight[None, :], -1)
        return cos


class ConcatLinearLayer(nn.Module):
    def __init__(self, input_dim, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((1, embedding_dim)))
        self.linear = nn.Linear(input_dim + embedding_dim, 1)

        nn.init.normal_(self.weight)

    def forward(self, input: Tensor) -> Tensor:
        e = torch.cat((input, self.weight.expand(len(input), -1)), -1)
        return self.linear(e)
        return torch.cosine_similarity(input[:, None], self.weight[None, :],
                                       -1)

class LogitsDataset:
    def __init__(self, base_datasets, logits, features=None, classes=None):
        self.base_dataset = base_datasets
        self.logits = logits
        self.features = features
        self.current_classes = classes

    def __len__(self):
        return len(self.logits)

    def __getitem__(self, item):
        if self.features is None:
            return *self.base_dataset[item], self.logits[item]

        return *self.base_dataset[item], self.logits[item], self.features[item]

    def train(self):
        self.base_dataset = self.base_dataset.train()
        return self

    def eval(self):
        self.base_dataset = self.base_dataset.eval()
        return self

    def subset(self, indexes):
        self.base_dataset = self.base_dataset.subset(indexes)
        self.logits = self.logits[indexes]

        if self.features is not None:
            self.features = self.features[indexes]

        return self


class CentroidsMatching(SupervisedPlugin):
    def __init__(self,
                 sit=True,
                 top_k=1,
                 per_sample_routing_reg=False,
                 centroids_merging_strategy=None,
                 **kwargs):

        super().__init__()

        self.batches_counter = 0
        self.centroids = None
        self.past_dataset = dict()
        self.per_class_memory = dict()
        self.support_sets = dict()

        self.beta = 0.8  # classification loss of past samples weight
        self.alpha = 1  # distance regularization weight
        self.gamma = 0  # future regularizaton loss
        self.theta = 0  # future regularizaton loss

        self.past_task_update_epochs = -1
        self.current_task_update_epochs = 2
        self.future_update_epochs = 5
        self.update_every_n_batches = 100
        self.av = None
        self.distributions = {}
        self.memory_size = 500

        self.past_model = None

        self.tasks_nclasses = {}

    @torch.no_grad()
    def before_training_exp(self, strategy: 'Trainer',
                            **kwargs):
        return
        classes = strategy.experience.classes_in_this_experience
        self.tasks_nclasses[strategy.experience.task_label] = classes

        # return
        tid = strategy.experience.current_experience

        if tid > 0:
            with torch.no_grad():
                strategy.model.eval()
                for v, d in strategy.past_dataset.items():
                    logits = []
                    for x, y, _, l, e in DataLoader(d.eval(),
                                                    batch_size=strategy.train_mb_size):
                        _l, _, _ = strategy.model.eval()(x.to(strategy.device))
                        _l = _l.cpu()
                        # _l = strategy.model.eval()(x.to(strategy.device)).cpu()
                        l = torch.cat((l, _l[:, l.shape[-1]:]), -1)
                        logits.append(l)

                    logits = torch.cat(logits, 0)
                    d.mb_features = logits

            strategy.model.train()

    @torch.no_grad()
    def after_training_epoch(
            self, strategy, *args, **kwargs
    ):
        return
        tid = strategy.experience.current_experience
        current_classes = strategy.experience.classes_in_this_experience
        strategy.model.eval()
        dataset = strategy.experience.dataset.eval()

        all_features = defaultdict(list)
        past_centroids = strategy.model.centroids

        classes_so_far = len(strategy.experience.classes_seen_so_far)

        with torch.no_grad():
            for i, (x, y, t) in enumerate(DataLoader(dataset, batch_size=1)):
                x = x.to(device)
                y = y.item()

                _, f, _ = strategy.model(x)
                f = f[0:1, y]
                f = f.cpu()

                all_features[y].append(f)

            if past_centroids is None:
                centroids = []

                for k, v in all_features.items():
                    c = torch.cat(v, 0).mean(0, keepdim=True)
                    centroids.append(c)

                centroids = torch.cat(centroids, 0)
            else:
                centroids = past_centroids

                if classes_so_far > len(centroids):
                    a, b = centroids.shape
                    centroids = torch.cat(
                        (centroids, torch.zeros((classes_so_far - a, b),
                                                device=centroids.device)))

                for k, v in all_features.items():
                    c = torch.cat(v, 0).mean(0, keepdim=True)
                    c = c.to(strategy.device)
                    # if k > len(centroids):
                    #     centroids = torch.cat((centroids, c))
                    # else:
                    centroids[k] = c

            strategy.model.centroids = centroids.to(strategy.device)

            # n_values = sum(classes_count.values())
            # classes_count = {k: v / n_values
            #                  for k, v in classes_count.items()}
            #
            # all_logits = np.concatenate(all_logits)
            # features = np.concatenate(features)
            # ys = np.asarray(ys)
            #
            # for y in np.unique(ys):
            #     mask = ys == y
            #     f = features[mask]
            #     indexes = np.argwhere(mask).reshape(-1)
            #
            #     km = KMeans(n_clusters=4).fit(f)
            #     # km = AffinityPropagation().fit(f)
            #     centroids = km.cluster_centers_
            #     centroids = torch.tensor(centroids)
            #
            #     f = torch.tensor(f)
            #     distances = calculate_distance(f, centroids)
            #     closest_one = distances.argmin(-1).numpy()
            #
            #     unique_centroids = np.unique(closest_one)
            #     sample_per_centroid = int(
            #         samples_to_save * classes_count[y]) // len(unique_centroids)
            #
            #     for i in unique_centroids:
            #         mask = closest_one == i
            #         _indexes = indexes[mask]
            #
            #         selected = np.random.choice(_indexes,
            #                                     min(sample_per_centroid,
            #                                         len(_indexes)),
            #                                     False)
            #
            #         selected_logits.append(all_logits[selected])
            #         selected_fetures.append(features[selected])
            #         selected_indexes.extend(selected.tolist())

            # for v, features in all_features.items():
            #     features = np.concatenate(features, 0)
            #
            #     km = KMeans(n_clusters=4).fit(features)
            #     centroids = km.cluster_centers_
            #
            #     centroids = torch.tensor(centroids)
            #
            #     features = torch.tensor(features)
            #     distances = calculate_distance(features, centroids)
            #     closest_one = distances.argmin(-1).numpy()
            #
            #     unique_centroids = np.unique(closest_one)
            #     sample_per_centroid = int(
            #         samples_to_save * classes_count[v]) // len(unique_centroids)
            #
            #     for i in unique_centroids:
            #         mask = closest_one == i
            #         _indexes = np.asarray(indexes[v])[mask]
            #
            #         selected = np.random.choice(_indexes, sample_per_centroid,
            #                                     False)
            #
            #         selected_logits.append(all_logits[selected])
            #         selected_indexes.extend(selected.tolist())

        strategy.model.train()

    @torch.no_grad()
    def after_training_exp(self, strategy, **kwargs):
        return
        tid = strategy.experience.current_experience
        current_classes = strategy.experience.classes_in_this_experience
        strategy.model.eval()

        # dataset = strategy.experience.dataset
        # dataset_idx = np.arange(len(dataset))
        # np.random.shuffle(dataset_idx)
        #
        # idx_to_get = dataset_idx[:self.patterns_per_experience]
        # memory = dataset.train().subset(idx_to_get)
        # self.past_dataset[tid] = memory

        # if tid > 0:
        # self.past_model = clone_module(strategy.model)
        # # a = self.past_model.train()(strategy.mb_x, task_labelels=None)
        # self.past_model.adapt = False
        # self.past_model.eval()

        samples_to_save = self.memory_size // (tid + 1)
        # samples_to_save = self.memory_size
        # if self.centroids is None:
        #     self.centroids = strategy.model.get_centroids()
        # else:
        #     c = strategy.model.get_centroids()[current_classes]
        #     self.centroids = torch.cat((self.centroids, c), 0)

        # if tid > 0:
        #     for k, d in self.past_dataset.items():
        #         indexes = np.arange(len(d))
        #         selected = np.random.choice(indexes, samples_to_save, False)
        #         d = d.train().subset(selected)
        #         self.past_dataset[k] = d

        all_features = defaultdict(list)

        all_logits = []
        features = []
        ys = []

        indexes = defaultdict(list)
        classes_count = defaultdict(int)

        selected_indexes = []
        selected_logits = []
        selected_fetures = []

        strategy.model.eval()
        dataset = strategy.experience.dataset.eval()
        device = strategy.device

        with torch.no_grad():
            for i, (x, y, t) in enumerate(DataLoader(dataset)):
                x = x.to(device)
                y = y.item()

                logits, f, _ = strategy.model(x)
                f = f[0:1, y]
                f = f.cpu().numpy()

                all_features[y].append(f[0, y])

                features.append(f)
                ys.append(y)

                all_logits.append(logits.cpu().numpy())
                indexes[y].append(i)
                classes_count[y] += 1

            n_values = sum(classes_count.values())
            classes_count = {k: v / n_values
                             for k, v in classes_count.items()}

            all_logits = np.concatenate(all_logits)
            features = np.concatenate(features)
            ys = np.asarray(ys)

            for y in np.unique(ys):
                mask = ys == y
                f = features[mask]
                indexes = np.argwhere(mask).reshape(-1)

                km = KMeans(n_clusters=4).fit(f)
                # km = AffinityPropagation().fit(f)
                centroids = km.cluster_centers_
                centroids = torch.tensor(centroids)

                f = torch.tensor(f)
                distances = calculate_distance(f, centroids)
                closest_one = distances.argmin(-1).numpy()

                unique_centroids = np.unique(closest_one)
                sample_per_centroid = int(
                    samples_to_save * classes_count[y]) // len(unique_centroids)

                for i in unique_centroids:
                    mask = closest_one == i
                    _indexes = indexes[mask]

                    selected = np.random.choice(_indexes,
                                                min(sample_per_centroid,
                                                    len(_indexes)),
                                                False)

                    selected_logits.append(all_logits[selected])
                    selected_fetures.append(features[selected])
                    selected_indexes.extend(selected.tolist())

            # for v, features in all_features.items():
            #     features = np.concatenate(features, 0)
            #
            #     km = KMeans(n_clusters=4).fit(features)
            #     centroids = km.cluster_centers_
            #
            #     centroids = torch.tensor(centroids)
            #
            #     features = torch.tensor(features)
            #     distances = calculate_distance(features, centroids)
            #     closest_one = distances.argmin(-1).numpy()
            #
            #     unique_centroids = np.unique(closest_one)
            #     sample_per_centroid = int(
            #         samples_to_save * classes_count[v]) // len(unique_centroids)
            #
            #     for i in unique_centroids:
            #         mask = closest_one == i
            #         _indexes = np.asarray(indexes[v])[mask]
            #
            #         selected = np.random.choice(_indexes, sample_per_centroid,
            #                                     False)
            #
            #         selected_logits.append(all_logits[selected])
            #         selected_indexes.extend(selected.tolist())

        # classes = [current_classes] * len(selected_indexes)
        task_memory = dataset.train().subset(selected_indexes)
        logits = torch.cat([torch.tensor(l) for l in selected_logits])
        features = torch.cat([torch.tensor(f) for f in selected_fetures])

        ld = LogitsDataset(task_memory, logits, features, current_classes)

        strategy.model.train()

        self.past_dataset[tid] = ld

        # return
        #
        # def hook(grad_input):
        #     return torch.zeros_like(grad_input)
        #
        # for p, c in strategy.model.paths_per_class[tid]:
        #     p = iter(p)
        #     for module in strategy.model.modules():
        #         if isinstance(module, (AbsDynamicLayer)):
        #             module.freeze_block(next(p))
        #
        #     strategy.model.classifiers[str(c)].register_hook(hook)
        #
        # return
        #
        # for module in strategy.model.modules():
        #     if isinstance(module, (AbsDynamicLayer)):
        #         module.freeze_blocks(tid,
        #                              top_k=self.top_k)

    def before_backward(self, strategy, *args, **kwargs):
        return
        tid = strategy.experience.current_experience
        current_classes = len(self.support_sets)

        # logits = strategy.mb_output
        # mx_current_classes = logits[range(len(logits)), strategy.mb_y]
        # # mx_current_classes = torch.max(logits[:, :current_classes], -1)[0]
        # # mx_future_classes = torch.max(logits[:, current_classes:], -1)[0]
        # mx_future_classes = logits[:, current_classes:].mean(-1)
        #
        # # loss = -mx_current_classes + mx_future_classes + 0.5
        # loss = mx_current_classes - mx_future_classes - 1
        # zeros = torch.zeros_like(loss)
        #
        # loss = torch.maximum(zeros, loss)
        # d = (loss > 0).sum()
        # if d == 0:
        #     d = 1
        # loss = loss.sum() / d
        #
        # strategy.loss += loss * 1=

        y = strategy.mb_y
        x = strategy.mb_x
        logits = strategy.mb_output

        if tid > 0:
            dists = torch.zeros(len(x), device=device)
            ce = torch.zeros(len(x), device=device)
            cos_dist = torch.zeros(len(x), device=device)

            ot_ys = []
            past_task_logits = []

            samples_per_task = len(x) // len(self.past_dataset)
            rest = len(x) % len(self.past_dataset)
            if rest > 0:
                to_add = np.random.choice(
                    np.asarray(list(self.past_dataset.keys())))
            else:
                to_add = -1

            offset = 0

            for i, d in self.past_dataset.items():
                if i == to_add:
                    bs = samples_per_task + rest
                else:
                    bs = samples_per_task

                ot_x, ot_y, _, ot_logits, of_features = next(iter(DataLoader(d,
                                                                             batch_size=bs,
                                                                             shuffle=True)))
                ot_y = ot_y.to(device)
                ot_x = ot_x.to(strategy.device)
                of_features = of_features.to(strategy.device)

                # ot_ys.append(ot_y)
                ot_logits = ot_logits.to(strategy.device)

                # x = torch.cat((x, ot_x), 0)
                # y = torch.cat((y, ot_y), 0)

                l = strategy.model(ot_x)
                past_task_logits.append(l)

                if self.alpha > 0:
                    # p1 = torch.log_softmax(l[:, :ot_logits.shape[-1]], -1)
                    # p2 = torch.softmax(ot_logits, -1)
                    dist = nn.functional.mse_loss(l[:, :ot_logits.shape[-1]],
                                                  ot_logits,
                                                  reduction='none').mean(-1)
                    # dist = 1 - nn.functional.cosine_similarity(l[:, :ot_logits.shape[-1]],
                    #                               ot_logits, -1).mean(-1)
                    # dist = nn.functional.kl_div(p1, p2, reduction='none')
                    dists[offset: offset + bs] = dist

                # _ce = centroids_loss(l[:, :ot_logits.shape[-1]],
                #                      ot_y, 'none')
                if self.beta > 0:
                    _ce = centroids_loss(l, ot_y, 'none')
                    ce[offset: offset + bs] = _ce

                features = strategy.model.current_features
                # with torch.no_grad():
                #     self.past_model.eval()
                #     past_logits = self.past_model(ot_x)
                #     past_features = self.past_model.current_features
                #     n_classes = past_logits.shape[1]
                # cd = cosine_similarity(features[:, :n_classes],
                #                        past_features, -1)
                # cd = cosine_similarity(features[range(len(features)), ot_y],
                #                        past_features[range(len(features)), ot_y], -1)
                # cd = (1 - cd).mean(-1)
                if self.theta > 0:
                    cd = cosine_similarity(features[range(len(features)), ot_y],
                                           of_features, -1)
                    cd = (1 - cd).mean(-1)

                    cos_dist[offset: offset + bs] = cd

                offset += bs

            dist = dists.mean() * self.alpha
            ce = ce.mean() * self.beta
            cos_dist = cos_dist.mean() * self.theta

            # ot_y = torch.cat(ot_ys, 0)
            # past_task_logits = torch.cat(past_task_logits, 0)

            # dist = torch.tensor(0, device=strategy.device)
            # ot_x, ot_y, ot_t = next(
            #     iter(DataLoader(self.av,
            #                     batch_size=strategy.train_mb_size,
            #                     shuffle=True)))
            #
            # ot_x = ot_x.to(strategy.device)
            # ot_y = ot_y.to(strategy.device)
            #
            # with torch.no_grad():
            #     self.past_model.eval()
            #     past_logits = self.past_model(ot_x)
            #     n_centroids = past_logits.shape[1]
            #
            # current_logits = strategy.model(ot_x)
            #
            # # current_logits = current_logits[range(len(ot_x)), ot_y]
            # # past_logits = past_logits[range(len(ot_x)), ot_y]
            #
            # # current_logits = torch.log_softmax(current_logits, -1)
            #
            # # dist = (1 - nn.functional.cosine_similarity(current_logits[:, :n_centroids],
            # #                                             past_logits, -1)).mean(-1) * self.alpha
            #
            # # dist = nn.functional.mse_loss(current_logits[:, :2],
            # #                               past_logits[:, :2]) * self.alpha
            # # dist = (1 - dist).mean() * 10
            # # p1 = torch.softmax(past_logits, -1)
            # dist = nn.functional.kl_div(
            #     torch.log_softmax(current_logits[:, :current_classes - 2], -1),
            #     torch.softmax(past_logits[:, :current_classes - 2],
            #                   -1)) * self.alpha
            # dist = nn.functional.mse_loss(current_logits, past_logits) * self.alpha
            # past_features = self.past_model.logits(ot_x, task_labelels=None)
            # # past_features = self.past_model.current_features
            # n_classes = len(past_features)
            # n_centroids = past_features.shape[1]
            # # past_features = torch.cat(past_features, -1)
            #
            # current_features = strategy.model.logits(ot_x, task_labelels=None)
            # current_features = current_features[:, :n_centroids]
            #
            # # current_features = torch.cat(current_features, -1)
            # #
            # current_features = current_features[range(len(ot_y)), ot_y]
            # past_features = past_features[range(len(ot_y)), ot_y]
            #
            # sim = nn.functional.cosine_similarity(current_features, past_features,
            #                                       -1)
            # dist = (1 - sim).mean() * 1000

            # ot_x, ot_y, ot_t = next(
            #     iter(TaskBalancedDataLoader(self.storage_policy.buffer,
            #                                 batch_size=strategy.train_mb_size,
            #                                 shuffle=True)))
            #
            # ot_x = ot_x.to(strategy.device)
            # ot_y = ot_y.to(strategy.device)

            # ot_x, ot_y, ot_t = next(
            #     iter(DataLoader(self.av,
            #                     batch_size=strategy.train_mb_size,
            #                     shuffle=True)))
            # ot_x = ot_x.to(strategy.device)
            # ot_y = ot_y.to(strategy.device)
            #
            # current_logits = strategy.model(ot_x)
            # ce = centroids_loss(current_logits[:, :n_centroids],
            #                     ot_y, 'mean') * self.beta
            # ce = 0
            strategy.loss += ce + dist + cos_dist

            # if tid > 0:
            #     ot_x, ot_y, ot_t = next(
            #         iter(DataLoader(self.av,
            #                         batch_size=strategy.train_mb_size,
            #                         shuffle=True)))
            #     ot_x = ot_x.to(strategy.device)
            #     ot_y = ot_y.to(strategy.device)
            #
            # x = torch.cat((x, ot_x), 0)
            # y = torch.cat((y, ot_y), 0)
            # logits = torch.cat((logits, past_task_logits), 0)
            # logits = 1 / (- logits)

            # all_logits_one_hot = 1 - nn.functional.one_hot(y, logits.shape[-1])

        # strategy.model.update_centroids(x, y, alpha=0.9)

        return
        mx_current_classes = logits[range(len(logits)), y]
        # mx_current_classes = torch.max(logits[:, :current_classes], -1)[0]
        # mx_future_classes = torch.max(logits[:, current_classes:], -1)[0]
        # mx_future_classes = logits[:, current_classes:].mean(-1)
        one_hot = 1 - nn.functional.one_hot(y, logits.shape[-1])
        # mx = (logits * one_hot).sum(-1) / one_hot.sum(-1)
        # mod_logits = logits * one_hot + (1 - one_hot) * - 100
        # mx = torch.max(mod_logits, -1)[0]

        # mx = torch.max(logits, -1)[0]

        # loss = -mx_current_classes + mx_future_classes - 0.3
        # loss = mx - mx_current_classes + 0.2
        loss = (logits - mx_current_classes.unsqueeze(-1) + 0.2) * one_hot
        max_loss = torch.maximum(torch.zeros_like(loss), loss).sum(-1)
        max_loss = max_loss / ((loss > 0).sum(-1) + 1)

        mask = max_loss > 0
        d = mask.sum()
        if d > 0:
            loss = (max_loss.sum() / d) * self.gamma

            # d = (loss > 0).sum()
            # if d == 0:
            #     d = 1
            # loss = (loss.sum() / d) * self.gamma

            strategy.loss += loss

        # logits_current = logits[:, :current_classes]
        # min_logits_present = torch.min(logits_current, -1).values[:, None]
        #
        # logits_future = logits[:, current_classes:]
        #
        # # future_wrong_classes_logits = logits * all_logits_one_hot
        # future_mse = nn.functional.mse_loss(logits_future, min_logits_present, reduction='mean')
        # # future_mse = future_mse * all_logits_one_hot
        #
        # current_mse = 0
        # if tid > 0:
        #     so_far_one_hot = 1 - nn.functional.one_hot(y, current_classes)
        #     current_mse = nn.functional.mse_loss(logits_current, min_logits_present, reduction='none')
        #     current_mse = current_mse * so_far_one_hot
        #     current_mse = current_mse.sum(-1) / so_far_one_hot.sum(-1)
        #     current_mse = current_mse.mean()
        #
        # strategy.loss += (current_mse + future_mse) * self.gamma
        # mean_logits_so_far = logits_future.sum(-1) / so_far_one_hot.sum(-1)
        #
        # future_wrong_classes_logits = logits * all_logits_one_hot
        # future_wrong_classes_logits = future_wrong_classes_logits.sum(-1) / all_logits_one_hot.sum(-1)
        #
        # # mx_future_classes = logits[:, current_classes:].mean(-1)
        #
        # future_loss = nn.functional.mse_loss(future_wrong_classes_logits, mean_logits_so_far) * self.gamma
        # strategy.loss += future_loss
        #
        # past_loss = 0
        # if tid > 0:
        #     past_wrong_classes_logits = logits_future.sum(-1) / so_far_one_hot.sum(-1)
        #     past_loss = nn.functional.mse_loss(past_wrong_classes_logits, mean_logits_so_far) * self.gamma
        #     strategy.loss += past_loss

        return

        # strategy.loss += dist

        # all_dists = 0
        # for p, c in zip(past_features, current_features):
        #     sim = nn.functional.cosine_similarity(p, c, -1)
        #     dist = 1 - sim
        #     all_dists += dist.mean()
        #
        # all_dists = all_dists / len(past_features)
        # strategy.loss += all_dists * 10
        # ot_x, ot_y, ot_t = next(
        #     iter(TaskBalancedDataLoader(self.storage_policy.buffer,
        #                                 batch_size=strategy.train_mb_size,
        #                                 shuffle=True)))
        #
        # ot_x = ot_x.to(strategy.device)
        # ot_y = ot_y.to(strategy.device)

        # ce = nn.functional.cross_entropy(strategy.model(ot_x,
        #                                                 task_labelels=None)[:,
        #                                  :past_logits.shape[-1]],
        #                                  ot_y)

        ce = centroids_loss(strategy.model(ot_x, task_labelels=None),
                            ot_y, 'mean') * 1
        strategy.loss += ce * dist

        # past_logits = torch.softmax(past_logits, -1)
        # # past_logits = nn.functional.pad(past_logits, (0, current_logits.shape[-1] - past_logits.shape[-1]))
        # current_logits = torch.log_softmax(current_logits, -1)
        # # current_logits = current_logits[:, :past_logits.shape[-1]]
        #
        # kl = nn.functional.kl_div(current_logits[:,:past_logits.shape[-1]],
        #                           past_logits)
        # strategy.loss += kl * 1

        return

        past_logits = torch.softmax(past_logits, -1)
        # past_logits = nn.functional.pad(past_logits, (0, current_logits.shape[-1] - past_logits.shape[-1]))
        current_logits = torch.log_softmax(current_logits, -1)
        # current_logits = current_logits[:, :past_logits.shape[-1]]

        loss = nn.functional.kl_div(current_logits[:, :past_logits.shape[-1]],
                                    past_logits,
                                    reduction='mean')

        strategy.loss += loss * 100
        return
        current_logits = current_logits[:, :past_logits.shape[-1]]

        dist = nn.functional.mse_loss(current_logits,
                                      past_logits)
        # dist = 1 - sim

        strategy.loss += dist.mean() * 10

        return
        # losses = torch.zeros(len(ot_y), device=strategy.device)
        #
        # for t in torch.unique(ot_t):
        #     t = t.item()
        #     mask = ot_t == t
        #     classes = self.tasks_nclasses[t]
        #
        #     past_logits = self.past_model(ot_x[mask], t, cumulative=False)[:,classes]
        #     logits = strategy.model(ot_x[mask], t, cumulative=False)[:, classes]
        #
        #     sim = nn.functional.cosine_similarity(logits,
        #                                           past_logits)
        #     dist = 1 - sim
        #
        #     losses[mask] = dist
        #
        # loss = losses.mean() * 1

        loss = 0

        for t in range(tid):
            classes = self.tasks_nclasses[t]
            past_logits = self.past_model(x, t, cumulative=False)[:, classes]
            logits = strategy.model(x, t, cumulative=False)[:, classes]

            dist = nn.functional.cosine_similarity(logits, past_logits)
            dist = 1 - dist

            loss += dist

        loss = (loss.mean() / tid) * 10

        strategy.loss += loss

        ce = nn.functional.cross_entropy(strategy.model(ot_x, tid - 1),
                                         ot_y) * 1
        strategy.loss += ce

        return

        # mask = tids != tid
        # x = x[mask]
        # tids = tids[mask]
        with torch.no_grad():
            # past_logits = self.past_model(ot_x, ot_t)
            past_logits = self.past_model(ot_x, None)

        # current_logits = strategy.model(ot_x, ot_t)
        current_logits = strategy.model(ot_x, None)

        # current_logits = strategy.model(ot_x, ot_t)
        #
        # loss = nn.functional.mse_loss(current_logits,
        #                               past_logits)
        # loss = nn.functional.cosine_similarity(current_logits[:, :-2],
        #                                        past_logits, -1).mean()

        past_logits = torch.softmax(past_logits, -1)
        # past_logits = nn.functional.pad(past_logits, (0, current_logits.shape[-1] - past_logits.shape[-1]))
        current_logits = torch.log_softmax(current_logits, -1)
        # current_logits = current_logits[:, :past_logits.shape[-1]]

        loss = nn.functional.kl_div(current_logits[:, :past_logits.shape[-1]],
                                    past_logits,
                                    reduction='mean')

        # loss = nn.functional.cross_entropy(current_logits[:, :past_logits.shape[-1]],
        #                                                       ot_y)

        # loss = nn.functional.kl_div(torch.log_softmax(current_logits, -1),
        # nn.functional.one_hot(ot_y, current_logits.shape[-1]).float())
        # current_logits = nn.functional.normalize(current_logits, 2, -1)
        # past_feats = nn.functional.normalize(past_feats, 2, -1)
        # loss = nn.functional.mse_loss(current_logits, past_feats).mean()
        # loss = 1 / (loss + 1)
        # loss = 1 - loss
        # loss = loss.mean()

        strategy.loss += loss * 0

        strategy.loss += nn.functional.cross_entropy(current_logits, ot_y) * 100

        # ot_x, ot_y, ot_t = next(
        #     iter(TaskBalancedDataLoader(self.storage_policy.buffer,
        #                                 batch_size=strategy.train_mb_size,
        #                                 shuffle=True)))
        #
        # ot_x = ot_x.to(strategy.device)
        #
        # with torch.no_grad():
        #     past_logits = self.past_model(ot_x, None)
        #
        # # current_logits = strategy.model(ot_x, ot_t)
        # current_logits = strategy.model(ot_x, None)

        past_features = self.past_model.current_features
        current_features = strategy.model.current_features

        sim = nn.functional.cosine_similarity(current_features,
                                              past_features)
        dist = (1 - sim).mean()

        strategy.loss += dist * 10


class Trainer(SupervisedTemplate):
    def __init__(self, model,
                 optimizer: Optimizer, criterion,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 eval_mb_size: int = None, device=None,
                 plugins=None,
                 batch_size_mem=None,
                 evaluator: EvaluationPlugin = default_evaluator, eval_every=-1,
                 ):

        self.double_sampling = False
        self.tasks_nclasses = dict()

        self.cl_w = 0

        # Past task loss weight
        self.alpha = 1

        self.past_task_reg = 1
        self.past_margin = 1

        # Past task features weight
        self.delta = 0

        # Past task logits weight
        self.gamma = 1
        self.logit_regularization = 'mse'
        self.tau = 1

        self.memory_size = 2500
        rp = CentroidsMatching()

        if batch_size_mem is None:
            self.batch_size_mem = train_mb_size
        else:
            self.batch_size_mem = batch_size_mem

        self.is_finetuning = False
        self.mb_future_features = None
        self.ot_logits = None

        self.past_dataset = {}
        self.current_dataset = {}
        self.past_centroids = {}

        self.batch_features = None
        self.replay_loader = None
        self.past_model = None

        self.base_plugin = rp

        self.main_pi = rp

        if plugins is None:
            plugins = [rp]
        else:
            plugins.append(rp)

        self.dev_split_size = 100

        self._use_logits = False

        if not self._use_logits:
            self._before_forward_f = self._no_logits_before_forward
            self._after_training_exp_f = self._no_logits_after_training_exp
        else:
            self._before_forward_f = self._logits_before_forward
            self._after_training_exp_f = self._logits_after_training_exp

        super().__init__(
            model=model, optimizer=optimizer, criterion=None,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device,
            plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)

    def _after_forward(self, **kwargs):
        self.mb_output, self.mb_features, self.mb_future_logits, self.mb_future_features = self.mb_output
        super()._after_forward(**kwargs)

    def _after_eval_forward(self, **kwargs):
        self.mb_output, self.mb_features, self.mb_future_logits, self.mb_future_features = self.mb_output
        super()._after_forward(**kwargs)

    def _after_training_iteration(self, **kwargs):
        self.ot_logits = None
        super()._after_training_iteration(**kwargs)

    def _after_training_exp(self, **kwargs):
        self._after_training_exp_f(**kwargs)

        # centroids = self.model.centroids
        # paths = self.model.paths_per_class
        #
        # for _, v in paths.values():
        #     v = str(v)
        #     if v not in self.past_centroids:
        #         c = deepcopy(centroids[str(v)].data)
        #         self.past_centroids[str(v)] = c

        super()._after_training_exp(**kwargs)

        # tid = self.experience.current_experience
        # current_classes = self.experience.classes_in_this_experience
        # self.model.eval()
        #
        # # dataset = strategy.experience.dataset
        # # dataset_idx = np.arange(len(dataset))
        # # np.random.shuffle(dataset_idx)
        # #
        # # idx_to_get = dataset_idx[:self.patterns_per_experience]
        # # memory = dataset.train().subset(idx_to_get)
        # # self.past_dataset[tid] = memory
        #
        # # if tid > 0:
        # # self.past_model = clone_module(strategy.model)
        # # # a = self.past_model.train()(strategy.mb_x, task_labelels=None)
        # # self.past_model.adapt = False
        # # self.past_model.eval()
        #
        # samples_to_save = self.memory_size // len(self.experience.classes_seen_so_far)
        # # samples_to_save = self.memory_size
        # # if self.centroids is None:
        # #     self.centroids = self.model.get_centroids()
        # # else:
        # #     c = strategy.model.get_centroids()[current_classes]
        # #     self.centroids = torch.cat((self.centroids, c), 0)
        # # classes = strategy.experience.classes_in_this_experience
        # # self.tasks_nclasses[strategy.experience.task_label] = classes
        # #
        # # # return
        # # tid = strategy.experience.current_experience
        # #
        # # if tid > 0:
        # #     with torch.no_grad():
        # #         strategy.model.eval()
        # #         for v, d in strategy.past_dataset.items():
        # #             logits = []
        # #             for x, y, _, l, e in DataLoader(d.eval(),
        # #                                             batch_size=strategy.train_mb_size):
        # #                 _l, _, _ = strategy.model.eval()(x.to(strategy.device))
        # #                 _l = _l.cpu()
        # #                 # _l = strategy.model.eval()(x.to(strategy.device)).cpu()
        # #                 l = torch.cat((l, _l[:, l.shape[-1]:]), -1)
        # #                 logits.append(l)
        # #
        # #             logits = torch.cat(logits, 0)
        # #             d.logits = logits
        # #
        # #     strategy.model.train()
        #
        # with torch.no_grad():
        #     # if tid > 0:
        #     for k, d in self.past_dataset.items():
        #         indexes = np.arange(len(d))
        #
        #         if len(indexes) > samples_to_save:
        #             selected = np.random.choice(indexes, samples_to_save, False)
        #             d = d.train().subset(selected)
        #             self.past_dataset[k] = d
        #
        #             # logits = []
        #             # for x, y, _, l, e in DataLoader(d.eval(), batch_size=self.train_mb_size):
        #             #     _l, _, _ = self.model.eval()(x.to(self.device))
        #             #     _l = _l.cpu()
        #             #     # _l = strategy.model.eval()(x.to(strategy.device)).cpu()
        #             #     l = torch.cat((l, _l[:, l.shape[-1]:]), -1)
        #             #     logits.append(l)
        #             #
        #             # logits = torch.cat(logits, 0)
        #             # d.logits = logits
        #             #
        #             # self.past_dataset[k] = d
        #
        # # all_features = defaultdict(list)
        # #
        # # all_logits = []
        # # features = []
        # # ys = []
        # #
        # # indexes = defaultdict(list)
        # # classes_count = defaultdict(int)
        # #
        # # selected_indexes = []
        # # selected_logits = []
        # # selected_fetures = []
        # #
        # # self.model.eval()
        # # dataset = self.experience.dataset.eval()
        # # device = self.device
        # #
        # # with torch.no_grad():
        # #     for i, (x, y, t) in enumerate(DataLoader(dataset)):
        # #         x = x.to(device)
        # #         y = y.item()
        # #
        # #         logits, f, _ = self.model(x)
        # #         f = f[0:1, y]
        # #         f = f.cpu().numpy()
        # #
        # #         all_features[y].append(f[0, y])
        # #
        # #         features.append(f)
        # #         ys.append(y)
        # #
        # #         all_logits.append(logits.cpu().numpy())
        # #         indexes[y].append(i)
        # #         classes_count[y] += 1
        # #
        # #     n_values = sum(classes_count.values())
        # #     classes_count = {k: v / n_values
        # #                      for k, v in classes_count.items()}
        # #
        # #     all_logits = np.concatenate(all_logits)
        # #     features = np.concatenate(features)
        # #     ys = np.asarray(ys)
        # #
        # #     for y in np.unique(ys):
        # #         mask = ys == y
        # #         f = features[mask]
        # #         indexes = np.argwhere(mask).reshape(-1)
        # #
        # #         km = KMeans(n_clusters=4).fit(f)
        # #         # km = AffinityPropagation().fit(f)
        # #         centroids = km.cluster_centers_
        # #         centroids = torch.tensor(centroids)
        # #
        # #         f = torch.tensor(f)
        # #         distances = calculate_distance(f, centroids)
        # #         closest_one = distances.argmin(-1).numpy()
        # #
        # #         unique_centroids = np.unique(closest_one)
        # #         sample_per_centroid = int(
        # #             samples_to_save * classes_count[y]) // len(unique_centroids)
        # #
        # #         for i in unique_centroids:
        # #             mask = closest_one == i
        # #             _indexes = indexes[mask]
        # #
        # #             selected = np.random.choice(_indexes,
        # #                                         min(sample_per_centroid,
        # #                                             len(_indexes)),
        # #                                         False)
        # #
        # #             selected_logits.append(all_logits[selected])
        # #             selected_fetures.append(features[selected])
        # #             selected_indexes.extend(selected.tolist())
        # #
        # # task_memory = dataset.train().subset(selected_indexes)
        # # logits = torch.cat([torch.tensor(l) for l in selected_logits])
        # # features = torch.cat([torch.tensor(f) for f in selected_fetures])
        # #
        # # ld = LogitsDataset(task_memory, logits, features, current_classes)
        # #
        # # self.model.train()
        # #
        # # self.past_dataset[tid] = ld

    def sample_past_batch(self, batch_size):
        if len(self.past_dataset) == 0:
            return None

        classes = (set(self.experience.classes_seen_so_far) -
                   set(self.experience.classes_in_this_experience))

        if len(classes) == 0:
            return None

        samples_per_task = batch_size // len(classes)
        rest = batch_size % len(classes)

        if rest > 0:
            to_add = np.random.choice(list(classes))
        else:
            to_add = -1

        x, y, t = [], [], []

        for c in classes:
            d = self.past_dataset[c]

            if c == to_add:
                bs = samples_per_task + rest
            else:
                bs = samples_per_task

            ot_x, ot_y, ot_tid = next(iter(DataLoader(d,
                                                      batch_size=bs,
                                                      shuffle=True)))

            x.append(ot_x)
            y.append(ot_y)
            t.append(ot_tid)

        return torch.cat(x, 0), torch.cat(y, 0), torch.cat(t, 0)

    def _no_logits_before_forward(self, **kwargs):

        super()._before_forward(**kwargs)
        if len(self.past_dataset) == 0:
            return None

        bs = len(self.mb_x)
        classes = (set(self.experience.classes_seen_so_far) -
                   set(self.experience.classes_in_this_experience))

        if len(classes) == 0:
            return

        samples_per_task = bs // len(classes)
        rest = bs % len(classes)

        if rest > 0:
            to_add = np.random.choice(list(classes))
        else:
            to_add = -1

        for c in classes:
            d = self.past_dataset[c]

            if c == to_add:
                bs = samples_per_task + rest
            else:
                bs = samples_per_task

            ot_x, ot_y, ot_tid = next(iter(DataLoader(d,
                                                      batch_size=bs,
                                                      shuffle=True)))

            self.mbatch[0] = torch.cat((self.mbatch[0], ot_x.to(self.device)))
            self.mbatch[1] = torch.cat((self.mbatch[1], ot_y.to(self.device)))
            self.mbatch[2] = torch.cat((self.mbatch[2], ot_tid.to(self.device)))

    @torch.no_grad()
    def _no_logits_after_training_exp(self, **kwargs):

        samples_to_save = self.memory_size // len(
            self.experience.classes_seen_so_far)

        for k, d in self.past_dataset.items():
            indexes = np.arange(len(d))

            if len(indexes) > samples_to_save:
                selected = np.random.choice(indexes, samples_to_save, False)
                d = d.train().subset(selected)
                self.past_dataset[k] = d

        dataset = self.experience.dataset
        ys = np.asarray(dataset.targets)

        for y in np.unique(ys):
            indexes = np.argwhere(ys == y).reshape(-1)
            indexes = np.random.choice(indexes, samples_to_save, False)

            self.past_dataset[y] = dataset.train().subset(indexes)

        if self.experience.current_experience > 0:
            self.past_model = deepcopy(self.model)
            self.past_model.eval()

        return

        all_logits = []
        features = []
        ys = []

        indexes = defaultdict(list)
        classes_count = defaultdict(int)
        features = []

        with torch.no_grad():
            for i, (x, y, t) in enumerate(DataLoader(dataset, batch_size=32)):
                x = x.to(device)

                ys.append(y)
                y = y.to(device)

                f = self.model(x)[1]

                features.append(f[range(len(f)), y].cpu())

            ys = torch.cat(ys, 0)
            features = torch.cat(features, 0)

            ys = np.asarray(ys.numpy())
            features = np.asarray(features.numpy())

            classes_count = collections.Counter(ys)
            tot = sum(classes_count.values())
            classes_count = {k: v / tot for k, v in classes_count.most_common()}

            for y in np.unique(ys):
                mask = ys == y
                f = features[mask]
                indexes = np.argwhere(mask).reshape(-1)

                km = KMeans(n_clusters=4).fit(f)
                # km = AffinityPropagation().fit(f)
                centroids = km.cluster_centers_
                centroids = torch.tensor(centroids)

                f = torch.tensor(f)
                distances = calculate_distance(f, centroids)
                closest_one = distances.argmin(-1).numpy()

                unique_centroids = np.unique(closest_one)
                sample_per_centroid = int(samples_to_save * classes_count[y]) // len(unique_centroids)
                sample_per_centroid = samples_to_save // len(unique_centroids)

                selected_indexes = []

                for i in unique_centroids:
                    mask = closest_one == i
                    _indexes = indexes[mask]

                    selected = np.random.choice(_indexes,
                                                min(sample_per_centroid,
                                                    len(_indexes)),
                                                False)

                    selected_indexes.extend(selected.tolist())

                self.past_dataset[y] = dataset.train().subset(selected_indexes)

    def _logits_before_forward(self, **kwargs):

        super()._before_forward(**kwargs)
        if len(self.past_dataset) == 0:
            return None

        bs = len(self.mb_x)
        classes = (set(self.experience.classes_seen_so_far) -
                   set(self.experience.classes_in_this_experience))

        samples_per_task = bs // len(classes)
        rest = bs % len(classes)

        if rest > 0:
            to_add = np.random.choice(list(classes))
        else:
            to_add = -1

        logits = []
        for c in classes:
            d = self.past_dataset[c]

            if c == to_add:
                bs = samples_per_task + rest
            else:
                bs = samples_per_task

            ot_x, ot_y, ot_tid, ot_logits = next(iter(DataLoader(d,
                                                                 batch_size=bs,
                                                                 shuffle=True)))

            self.mbatch[0] = torch.cat((self.mbatch[0], ot_x.to(self.device)))
            self.mbatch[1] = torch.cat((self.mbatch[1], ot_y.to(self.device)))
            self.mbatch[2] = torch.cat((self.mbatch[2], ot_tid.to(self.device)))

            logits.append(ot_logits)

        self.ot_logits = torch.cat(logits, 0).to(self.device)

    @torch.no_grad()
    def _logits_after_training_exp(self, **kwargs):
        is_training = self.model.training
        self.model.eval()

        samples_to_save = self.memory_size // len(
            self.experience.classes_seen_so_far)

        for k, d in self.past_dataset.items():
            indexes = np.arange(len(d))

            if len(indexes) > samples_to_save:
                selected = np.random.choice(indexes, samples_to_save, False)
                d = d.train().subset(selected)

                # all_logits = []
                # for x, y, _, l in DataLoader(d.eval(),
                #                              batch_size=self.train_mb_size):
                #     _l, _, _ = self.model(x.to(self.device))
                #     _l = _l.cpu()
                #     l = torch.cat((l, _l[:, l.shape[-1]:]), -1)
                #     all_logits.append(l)
                #
                # all_logits = torch.cat(all_logits, 0)
                # d.logits = all_logits
                self.past_dataset[k] = d

        dataset = self.experience.dataset
        ys = np.asarray(dataset.targets)

        for y in np.unique(ys):
            indexes = np.argwhere(ys == y).reshape(-1)
            indexes = np.random.choice(indexes, samples_to_save, False)

            past_dataset = dataset.subset(indexes)

            all_logits = []

            for x, _, _ in DataLoader(past_dataset.eval(),
                                      batch_size=self.eval_mb_size):
                x = x.to(self.device)

                logits = self.model(x)[0]
                all_logits.append(logits)

            all_logits = torch.cat(all_logits, 0).cpu()

            self.past_dataset[y] = LogitsDataset(past_dataset, all_logits)

        self.model.train(is_training)

    def _before_forward(self, **kwargs):
        super()._before_forward(**kwargs)
        self._before_forward_f(**kwargs)

    @torch.no_grad()
    def _before_training_exp(self, **kwargs):

        classes = self.experience.classes_in_this_experience
        self.tasks_nclasses[self.experience.task_label] = classes

        if self._use_logits:
            for k, d in self.past_dataset.items():

                all_logits = []
                for x, y, _, l in DataLoader(d,
                                             batch_size=self.train_mb_size):
                    _l = self.model(x.to(self.device))[0]
                    _l = _l.cpu()
                    l = torch.cat((l, _l[:, l.shape[-1]:]), -1)
                    all_logits.append(l)

                all_logits = torch.cat(all_logits, 0)
                d.logits = all_logits
                self.past_dataset[k] = d
        # else:
        #     if self.experience.current_experience > 0:
        #         self.past_model = deepcopy(self.model)
        #         self.past_model.eval()

        super()._before_training_exp(**kwargs)

        if self.experience.current_experience > 0 and not self._use_logits:
            self.past_model = deepcopy(self.model)
            self.past_model.eval()

    def criterion(self):
        if not self.is_training:
            if isinstance(self.model, MoELogits):
                loss_val = nn.functional.cross_entropy(self.mb_output, self.mb_y)
            else:
                log_p_y = torch.log_softmax(self.mb_output, dim=1)
                loss_val = -log_p_y.gather(1, self.mb_y.unsqueeze(-1)).squeeze(
                    -1).mean()

            return loss_val

        past_reg = 0
        future_reg = 0
        
        pred = self.mb_output

        if self.mb_future_logits is not None:
            pred = torch.cat((self.mb_output, self.mb_future_logits), 1)

        if len(self.past_dataset) == 0:
            if isinstance(self.model, MoELogits):
                loss_val = nn.functional.cross_entropy(pred, self.mb_y, label_smoothing=0)
            else:
                log_p_y = torch.log_softmax(pred, dim=1)
                loss_val = -log_p_y.gather(1, self.mb_y.unsqueeze(-1)).squeeze(-1).mean()
        else:
            s = len(pred) // 2
            pred1, pred2 = torch.split(pred, s, 0)
            y1, y2 = torch.split(self.mb_y, s, 0)

            tid = self.experience.current_experience
            # if tid > 0:
            y1 = y1 - min(self.experience.classes_in_this_experience) # min(classes_in_this_experience)

            neg_pred1 = pred1[:, :len(self.experience.previous_classes)]
            pred1 = pred1[:, len(self.experience.previous_classes):]

            mx = neg_pred1.max(-1).values
            mx_current_classes = pred1[range(len(pred1)), y1]

            past_reg = torch.maximum(torch.zeros_like(mx), mx - mx_current_classes + self.past_margin).mean()
            # past_dist = torch.maximum(torch.zeros_like(mx), mx - mx_current_classes + self.past_margin)
            # past_reg = past_dist.sum() / torch.sum(past_dist > 0)

            w = min(self.clock.train_exp_epochs / 5, 1)
            w = 1

            past_reg = past_reg * self.past_task_reg

            if isinstance(self.model, MoELogits):
                loss1 = nn.functional.cross_entropy(pred1, y1, label_smoothing=0)
                loss2 = nn.functional.cross_entropy(pred2, y2, label_smoothing=0)
            else:
                pred1 = torch.log_softmax(pred1, 1)
                pred2 = torch.log_softmax(pred2, 1)
                loss1 = -pred1.gather(1, y1.unsqueeze(-1)).squeeze(-1).mean()
                loss2 = -pred2.gather(1, y2.unsqueeze(-1)).squeeze(-1).mean()

            loss_val = loss1 + self.alpha * loss2

        loss = loss_val + past_reg + future_reg

        if self.is_training:
            if self.past_model is not None:
                bs = len(self.mb_output) // 2

                if self.double_sampling:
                    x, y, _ = self.sample_past_batch(bs)
                    x, y = x.to(self.device), y.to(self.device)
                    curr_logits, curr_features = self.model(x)[:2]
                else:
                    x, y = self.mb_x[bs:], self.mb_y[bs:]
                    curr_logits = self.mb_output[bs:]
                    curr_features = self.mb_features[bs:]

                with torch.no_grad():
                    past_logits, past_features, _, _ = self.past_model(x)

                classes = [self.tasks_nclasses[t.item()] for t in
                           self.mb_task_id]

                # mask = torch.zeros(len(classes), past_logits.shape[1])
                # mask = torch.scatter(mask, 1, torch.tensor(classes), 1)
                # mask = mask.to(self.device)[bs:]

                if self.gamma > 0:
                    # self.internal_distillation
                    if False:
                        all_lr = 0

                        csf = len(self.experience.previous_classes)
                        for cc, pp in zip(self.model.internal_features[:csf],
                                        self.past_model.internal_features[:csf]):

                            for c, p in zip(cc, pp):
                                c = torch.flatten(c, 1)[bs:]
                                p = torch.flatten(p, 1)

                                lr = 1 - nn.functional.cosine_similarity(c, p, -1)
                                all_lr += lr

                        lr = (all_lr / csf).mean()

                    else:
                        # curr_pred = self.mb_output[bs:, :past_pred.shape[1]]

                        # curr_logits = nn.functional.normalize(curr_logits, 2, -1)
                        # past_logits = nn.functional.normalize(past_logits, 2, -1)

                        # lr = nn.functional.mse_loss(curr_logits, past_logits, reduction='mean')

                        # # mse = nn.functional.mse_loss(curr_pred, past_pred)
                        # mse = 1 - nn.functional.cosine_similarity(curr_pred, past_logits, -1).mean()

                        # mse = mse * mask
                        # mse = mse.sum(-1) / mask.sum(-1)
                        # lr = mse.mean()
                        classes = len(self.experience.classes_in_this_experience)

                        if self.logit_regularization == 'kl':
                            curr_logits = torch.log_softmax(curr_logits / self.tau, -1)
                            past_logits = torch.softmax(past_logits / self.tau, -1)

                            lr = nn.functional.kl_div(curr_logits.log(),
                                                       past_logits)

                        elif self.logit_regularization == 'mse':
                            lr = nn.functional.mse_loss(curr_logits, past_logits)
                        elif self.logit_regularization == 'cosine':
                            lr = 1 - nn.functional.cosine_similarity(curr_logits, past_logits, -1)
                            lr = lr.mean()
                        else:
                            assert False

                    loss += lr * self.gamma

                if self.delta > 0:
                    # classes = [self.tasks_nclasses[t.item()] for t in
                    #            self.mb_task_id]
                    #
                    # mask = torch.zeros(len(classes), past_pred.shape[1])
                    # mask = torch.scatter(mask, 1, torch.tensor(classes), 1)
                    # mask = mask.to(self.device)[bs:]

                    # logits = self.mb_features[bs:]
                    # dist = nn.functional.mse_loss(logits, past_features,
                    #                               reduction='mean')
                    # dist = 1 - nn.functional.cosine_similarity(logits, past_features, -1)

                    # logits = self.mb_features[bs:]
                    # dist = nn.functional.mse_loss(logits, past_features,
                    #                               reduction='none').mean(-1)
                    # dist = dist * mask
                    # dist = dist.sum(-1) / mask.sum(-1, keepdim=True)
                    # dist = dist.mean()

                    # y = self.mb_y[bs:]
                    # logits = self.mb_features[bs:]
                    # curr_features = curr_features[range(len(curr_features)), y]
                    # past_features = past_features[range(len(curr_features)), y]
                    classes = len(self.experience.classes_in_this_experience)

                    curr_features = curr_features[:, :-classes]
                    past_features = past_features[:, :-classes]

                    # dist = nn.functional.mse_loss(curr_features, past_features, reduction='none').mean(-1)

                    dist = 1 - nn.functional.cosine_similarity(curr_features, past_features, -1)
                    # dist = nn.functional.cosine_similarity(logits, past_logits, -1)
                    # dist = 1 - dist
                    # dist = dist * mask
                    # dist = dist.sum(-1) / mask.sum(-1)
                    dist = dist.sum(-1).mean()

                    # logits = logits[range(len(logits)), self.mb_y[bs:]]
                    #
                    # past_logits = past_logits[range(len(past_logits)), self.mb_y[bs:]]
                    #
                    # dist = nn.functional.mse_loss(logits, past_logits)
                    # dist = 1 - sim
                    # dist = dist.mean(-1).mean()

                    loss += dist * self.delta

                    # past_logits = nn.functional.normalize(past_logits, 2, -1)
                    # logits = nn.functional.normalize(logits, 2, -1)
                    #
                    # past_matrix = torch.cdist(past_logits, past_logits)
                    #
                    # current_matrix = torch.cdist(logits, logits)
                    #
                    # norm = torch.linalg.matrix_norm(current_matrix - past_matrix)
                    # norm = norm.mean()
                    #
                    # loss += norm * self.delta

            elif self.ot_logits is not None:
                bs = len(self.mb_output) // 2

                past_logits = self.ot_logits

                if self.gamma > 0:
                    curr_logits = self.mb_output[bs:]

                    lr = nn.functional.mse_loss(curr_logits, past_logits, reduction='mean')

                    loss += lr * self.gamma

        return loss


class MoECentroids(MultiTaskModule):
    def __init__(self, cumulative=False):
        super().__init__()

        self.forced_future = 0
        self.current_features = None
        self.centroids = None
        self.use_future = True
        self.cumulative = cumulative

        self.distance = 'cosine'
        self.adapt = True

        self.layers = nn.ModuleList()

        self.layers.append(BlockRoutingLayer(3, 32, project_dim=None, get_average_features=True))
        self.layers.append(BlockRoutingLayer(32, 64, project_dim=None, get_average_features=True))
        self.layers.append(BlockRoutingLayer(64, 128, project_dim=None, get_average_features=True))

        self.mx = nn.Sequential(nn.ReLU())

        if cumulative:
            self.in_features = 32 * 4 + 64 * 4 + 128 * 4
        else:
            self.in_features = 128 * 16

        self.classifiers = nn.ParameterDict()

        self.gates = nn.ModuleDict()
        self.translate = nn.ModuleDict()

        layers_blocks = [len(l.blocks) for l in self.layers]
        paths = []

        self.centroids = nn.ParameterDict()

        orth = torch.randn(100, self.in_features) * 0.01
        # svd = torch.linalg.svd(centroids)
        # orth = svd[0] @ svd[2]

        while len(paths) < 100:
            b = [np.random.randint(0, l) for l in layers_blocks]
            if b not in paths:
                ln = len(paths)
                v = (b, ln)
                self.centroids[str(ln)] = nn.Parameter(orth[ln],
                                                       requires_grad=False)
                paths.append(v)

        self.available_paths = paths
        self.paths_per_class = {}

        # for l in self.layers:
        #     l.freeze_blocks()

    def eval_adaptation(self, experience):
        if len(experience.classes_seen_so_far) > len(self.paths_per_class):
            self.forced_future = len(experience.classes_seen_so_far) - len(
                self.paths_per_class)
        else:
            self.forced_future = 0

    def train_adaptation(self, experience):
        if not self.adapt:
            return
        self.forced_future = 0

        curr_classes = experience.classes_in_this_experience

        selected_paths = np.random.choice(np.arange(len(self.available_paths)),
                                          len(curr_classes),
                                          replace=False)
        paths = [self.available_paths[i] for i in selected_paths]

        for p, v in self.paths_per_class.values():
            self.centroids[str(v)].requires_grad_(False)
            # print(self.centroids[str(v)])

        for c, p in zip(experience.classes_in_this_experience, paths):
            self.available_paths.remove(p)
            self.paths_per_class[c] = p

            _, v = p
            self.centroids[str(v)].requires_grad_(True)

        for b, l in zip(zip(*[p[0] for p in paths]), self.layers):
            l.activate_blocks(b)

    def forward(self,
                x: torch.Tensor,
                task_labels: torch.Tensor = None,
                **kwargs) \
            -> Tuple[
                Tensor, Union[Tensor, Any], Optional[None], Optional[None]]:

        if task_labels is not None:
            if not isinstance(task_labels, int):
                task_labels = torch.unique(task_labels)
                # assert len(task_labels) == 1
                task_labels = task_labels[0]

        base_paths = list(self.paths_per_class.values())
        random_paths = []

        if self.training and self.use_future:
            sampled_paths = np.random.choice(
                np.arange(len(self.available_paths)),
                5, replace=True)
            random_paths = [self.available_paths[p] for p in sampled_paths]

        if self.forced_future > 0:
            sampled_paths = np.random.choice(
                np.arange(len(self.available_paths)),
                self.forced_future, replace=True)
            base_paths += [self.available_paths[p] for p in sampled_paths]

        all_paths = base_paths + random_paths
        logits = self.features(x, unassigned_path_to_use=all_paths)

        random_logits = None
        random_preds = None

        if len(base_paths) > 0:
            if len(random_paths) > 0:
                random_logits = logits[:, -len(random_paths):]
                random_centroids = torch.stack([self.centroids[str(v)]
                                                for _, v in random_paths], 0)

            centroids = torch.stack([self.centroids[str(v)]
                                     for _, v in base_paths], 0)
            logits = logits[:, :len(base_paths)]

        else:
            centroids = torch.stack([self.centroids[str(v)]
                                     for _, v in all_paths], 0)

        if self.distance == 'euclidean':
            preds = - torch.pow(logits - centroids, 2).sum(2).sqrt()
        else:
            preds = torch.cosine_similarity(logits, centroids, -1)

        if random_logits is not None:
            if self.distance == 'euclidean':
                random_preds = - torch.pow(random_logits - random_centroids, 2).sum(
                    2).sqrt()
            else:
                random_preds = torch.cosine_similarity(random_logits,
                                                       random_centroids,
                                                       -1)

        return preds, logits, random_preds, random_logits

    def features1(self, x, *,
                  task_labels=None,
                  unassigned_path_to_use=None, **kwargs):
        # assert (task_labels is None) ^ (path_id is None)
        if unassigned_path_to_use is not None:
            if isinstance(unassigned_path_to_use, tuple):
                unassigned_path_to_use = [unassigned_path_to_use]
            # elif isinstance(non_assigned_paths, int):
            #     non_assigned_paths = [non_assigned_paths]
            elif (isinstance(unassigned_path_to_use, list)
                  and isinstance(unassigned_path_to_use[0], int)):
                pass
            elif unassigned_path_to_use == 'all':
                unassigned_path_to_use = self.available_paths

            to_iter = unassigned_path_to_use
        else:
            to_iter = self.paths_per_class.values()

        logits = []

        for path in to_iter:
            p, v = path
            path = iter(p)
            _x = x
            current_path = 0
            feats = []

            for l in self.layers[:-1]:
                _p = next(path)
                _x, f = l(_x, _p, current_path)
                _x = _x.relu()
                # f = nn.functional.adaptive_avg_pool2d(_x, 2).flatten(1)
                feats.append(f)

                current_path = _p

            _, _x = self.layers[-1](_x, next(path), current_path)
            # _x = nn.functional.adaptive_avg_pool2d(_x, 2).flatten(1)

            feats.append(_x.flatten(1))

            if str(v) in self.translate:
                # l = l + self.translate[str(v)](l)
                _x = (_x * self.gates[str(v)](_x).sigmoid()
                      + self.translate[str(v)](_x))

            _x = torch.cat(feats, -1)

            logits.append(_x)

        return torch.stack(logits, 1)

    def features(self, x, *,
                 unassigned_path_to_use=None, **kwargs):

        if unassigned_path_to_use is not None:
            if isinstance(unassigned_path_to_use, tuple):
                unassigned_path_to_use = [unassigned_path_to_use]
            elif (isinstance(unassigned_path_to_use, list)
                  and isinstance(unassigned_path_to_use[0], int)):
                assert False
            elif unassigned_path_to_use == 'all':
                unassigned_path_to_use = self.available_paths

            to_iter = unassigned_path_to_use
        else:
            to_iter = self.paths_per_class.values()

        feats = []

        paths = list(zip(*[p for p, _ in to_iter]))
        paths_iterable = iter(paths)
        _x = [x] * len(paths[0])

        for l in self.layers[:-1]:
            _p = next(paths_iterable)
            _x, f = l(_x, _p)
            # _x = _x.relu()
            feats.append(f)

        _, f = self.layers[-1](_x, next(paths_iterable))
        feats.append(f)

        if self.cumulative:
            feats = [torch.cat(l, -1 )for l in list(zip(*feats))]
            logits = torch.stack(feats, 1)
        else:
            logits = torch.stack(f, 1)

        return logits


class MoELogits(MultiTaskModule):
    def __init__(self,
                 cumulative=False,
                 freeze_past_tasks=False,
                 freeze_future_logits=True,
                 path_selection_strategy='usage',
                 prediction_mode='task'):

        super().__init__()

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

        assert path_selection_strategy in ['random', 'usage']
        assert prediction_mode in ['class', 'task']

        self.distance = 'cosine'
        self.adapt = True

        self.layers = nn.ModuleList()

        self.layers.append(BlockRoutingLayer(3, 32, project_dim=None, get_average_features=False))
        self.layers.append(BlockRoutingLayer(32, 64, project_dim=None, get_average_features=False))
        self.layers.append(BlockRoutingLayer(64, 128, project_dim=128, get_average_features=False))

        self.mx = nn.Sequential(nn.ReLU())

        if cumulative:
            self.in_features = 32 * 4 + 64 * 4 + 128 * 4
        else:
            self.in_features = 128

        self.classifiers = nn.ParameterDict()

        self.gates = nn.ModuleDict()
        self.translate = nn.ModuleDict()

        layers_blocks = [len(l.blocks) for l in self.layers]
        paths = []

        self.centroids = nn.ModuleDict()
        self.centroids_scaler = nn.ParameterDict()

        while len(paths) < 100:
            b = [np.random.randint(0, l) for l in layers_blocks]
            if b not in paths:
                ln = len(paths)
                v = (b, ln)
                self.centroids[str(ln)] = nn.Sequential(nn.Flatten(1),
                                                        # nn.ReLU(),
                                                        # CosineLinearLayer(self.in_features),
                                                        # ConcatLinearLayer(self.in_features, 100)
                                                        nn.Sequential(nn.ReLU(), nn.Linear(self.in_features, 1)),
                                                        # nn.Sigmoid()
                                                        )
                paths.append(v)

        self.available_paths = paths
        self.associated_paths = {}
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
        to_samples = task_classes if self.prediction_mode == 'logits' else 1

        if self.path_selection_strategy == 'random' or len(self.associated_paths) == 0:
            selected_paths = np.random.choice(np.arange(len(self.available_paths)),
                                              to_samples,
                                              replace=False)
            paths = [self.available_paths[i] for i in selected_paths]

        elif self.path_selection_strategy == 'usage':
            probs = []

            used_blocks = set()
            for c, (p, v) in self.associated_paths.items():
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

            probs = np.asarray(probs) / sum(probs)
            selected_paths = np.random.choice(np.arange(len(self.available_paths)),
                                              to_samples,
                                              replace=False,
                                              p=probs)

            paths = [self.available_paths[i] for i in selected_paths]

        else:
            assert False

        if self.freeze_past_tasks:
            for pt, v in self.associated_paths.values():

                for p in self.centroids[str(v)].parameters():
                    p.requires_grad_(False)

                for b, l in zip(pt, self.layers):
                    l.freeze_block(b)

                # self.centroids_scaler[str(v)] = nn.Parameter(torch.tensor([1.0]))

            # print(self.centroids[str(v)])

        # for b, l in zip(zip(*[p[0] for p in paths]), self.layers):
        #     l.activate_blocks(b)

        z = experience.classes_in_this_experience \
            if self.prediction_mode == 'logits' else experience.task_labels

        for c, p in zip(z, paths):
            self.available_paths.remove(p)
            self.associated_paths[c] = p

            if self.prediction_mode == 'task':
                l = nn.Linear(self.in_features, task_classes)
                self.centroids[str(p[1])] = l

            if self.freeze_past_tasks or self.freeze_future_logits:
                for b, l in zip(p[0], self.layers):
                    l.freeze_block(b, False)

                for p in self.centroids[str(p[1])].parameters():
                    p.requires_grad_(True)

    def forward(self,
                x: torch.Tensor,
                task_labels: torch.Tensor = None,
                **kwargs) \
            -> Tuple[
                Tensor, Union[Tensor, Any], Optional[None], Optional[None]]:

        if task_labels is not None:
            if not isinstance(task_labels, int):
                task_labels = torch.unique(task_labels)
                # assert len(task_labels) == 1
                task_labels = task_labels[0]

        base_paths = list(self.associated_paths.values())
        random_paths = []

        if self.training and self.use_future:
            sampled_paths = np.random.choice(
                np.arange(len(self.available_paths)),
                5, replace=True)
            random_paths = [self.available_paths[p] for p in sampled_paths]

        if self.forced_future > 0:
            sampled_paths = np.random.choice(
                np.arange(len(self.available_paths)),
                self.forced_future, replace=True)
            base_paths += [self.available_paths[p] for p in sampled_paths]

        all_paths = base_paths + random_paths
        features, all_features = self.features(x, unassigned_path_to_use=all_paths)

        # self.internal_features = list(zip(*all_features))

        logits = []
        for i, (_,v) in enumerate(all_paths):
            l = self.centroids[str(v)](features[:, i])

            if self.freeze_past_tasks and str(v) in self.centroids_scaler:
                l = l / self.centroids_scaler[str(v)]
            logits.append(l)

        # logits = logits.
        # logits = [self.centroids[str(v)](l) for l, (_, v) in zip(features, all_paths)]
        logits = torch.cat(logits, 1)

        random_features = None
        random_logits = None

        # if len(base_paths) > 0:
        #     if len(random_paths) > 0:
        #         random_logits = logits[:, -len(random_paths):]
        #         random_features = features[:, -len(random_paths):]
        #
        #     features = features[:, :len(base_paths)]
        #     logits = logits[:, :len(base_paths)]

        if len(base_paths) > 0 and len(random_paths) > 0:
            random_logits = logits[:, -len(random_paths):]
            random_features = features[:, -len(random_paths):]

            features = features[:, :-len(random_paths)]
            logits = logits[:, :-len(random_paths)]

        # else:
        #     centroids = torch.stack([self.centroids[str(v)]
        #                              for _, v in all_paths], 0)
        #
        # if self.distance == 'euclidean':
        #     preds = - torch.pow(features - centroids, 2).sum(2).sqrt()
        # else:
        #     preds = torch.cosine_similarity(features, centroids, -1)
        #
        # if random_logits is not None:
        #     if self.distance == 'euclidean':
        #         random_preds = - torch.pow(random_logits - random_centroids, 2).sum(
        #             2).sqrt()
        #     else:
        #         random_preds = torch.cosine_similarity(random_logits,
        #                                                random_centroids,
        #                                                -1)

        return logits, features, random_logits, random_features

    def features(self, x, *,
                 unassigned_path_to_use=None, **kwargs):

        if unassigned_path_to_use is not None:
            if isinstance(unassigned_path_to_use, tuple):
                unassigned_path_to_use = [unassigned_path_to_use]
            elif (isinstance(unassigned_path_to_use, list)
                  and isinstance(unassigned_path_to_use[0], int)):
                assert False
            elif unassigned_path_to_use == 'all':
                unassigned_path_to_use = self.available_paths

            to_iter = unassigned_path_to_use
        else:
            to_iter = self.associated_paths.values()

        feats = []

        paths = list(zip(*[p for p, _ in to_iter]))
        paths_iterable = iter(paths)
        _x = [x] * len(paths[0])

        for l in self.layers[:-1]:
            _p = next(paths_iterable)
            _x, f = l(_x, _p)
            # _x = _x.relu()
            feats.append(f)

        _x, f = self.layers[-1](_x, next(paths_iterable))
        feats.append(_x)

        if self.cumulative:
            feats = [torch.cat(l, -1 )for l in list(zip(*feats))]
            logits = torch.stack(feats, 1)
        else:
            logits = torch.stack(f, 1)
            # logits = torch.stack(f, 1).relu()

        return logits, feats


class _MoELogits(MultiTaskModule):
    def __init__(self):
        super().__init__()

        self.forced_future = 0
        self.current_features = None
        self.centroids = None
        self.use_future = True

        self.distance = 'cosine'
        self.adapt = True

        self.layers = nn.ModuleList()

        self.layers.append(BlockRoutingLayer(3, 32, project_dim=32))
        self.layers.append(BlockRoutingLayer(32, 64, project_dim=64))
        self.layers.append(BlockRoutingLayer(64, 128, project_dim=128))

        self.mx = nn.Sequential(nn.ReLU())

        in_features = 128 + 32 + 64

        self.classifiers = nn.ParameterDict()

        self.gates = nn.ModuleDict()
        self.translate = nn.ModuleDict()

        layers_blocks = [len(l.blocks) for l in self.layers]
        paths = []

        self.centroids = nn.ModuleDict()

        while len(paths) < 100:
            b = [np.random.randint(0, l) for l in layers_blocks]
            if b not in paths:
                ln = len(paths)
                v = (b, ln)
                self.centroids[str(ln)] = nn.Sequential(nn.Flatten(1),
                                                        nn.ReLU(),
                                                        CosineLinearLayer(in_features))
                paths.append(v)

        self.available_paths = paths
        self.paths_per_class = {}

    def eval_adaptation(self, experience):
        if len(experience.classes_seen_so_far) > len(self.paths_per_class):
            self.forced_future = len(experience.classes_seen_so_far) - len(
                self.paths_per_class)
        else:
            self.forced_future = 0

    def train_adaptation(self, experience):
        if not self.adapt:
            return
        self.forced_future = 0

        curr_classes = experience.classes_in_this_experience

        selected_paths = np.random.choice(np.arange(len(self.available_paths)),
                                          len(curr_classes),
                                          replace=False)
        paths = [self.available_paths[i] for i in selected_paths]

        # for p, v in self.paths_per_class.values():
        #     self.centroids[str(v)].requires_grad_(False)

        for c, p in zip(experience.classes_in_this_experience, paths):
            self.available_paths.remove(p)
            self.paths_per_class[c] = p

            _, v = p
            # self.centroids[str(v)].requires_grad_(True)

    def forward(self,
                x: torch.Tensor,
                task_labels: torch.Tensor = None,
                **kwargs) \
            -> Tuple[
                Tensor, Union[Tensor, Any], Optional[None], Optional[None]]:

        if task_labels is not None:
            if not isinstance(task_labels, int):
                task_labels = torch.unique(task_labels)
                # assert len(task_labels) == 1
                task_labels = task_labels[0]

        base_paths = list(self.paths_per_class.values())
        random_paths = []

        if self.training and self.use_future:
            sampled_paths = np.random.choice(
                np.arange(len(self.available_paths)),
                5, replace=True)
            random_paths = [self.available_paths[p] for p in sampled_paths]

        if self.forced_future > 0:
            sampled_paths = np.random.choice(
                np.arange(len(self.available_paths)),
                self.forced_future, replace=True)
            base_paths += [self.available_paths[p] for p in sampled_paths]

        all_paths = base_paths + random_paths
        logits = self.features(x, unassigned_path_to_use=all_paths)

        preds = [self.centroids[str(v)](l) for l, (_, v) in zip(logits, all_paths)]
        preds = torch.cat(preds, -1)

        logits = torch.stack(logits, 1).flatten(2)

        random_logits = None
        random_preds = None

        if len(base_paths) > 0 and len(random_paths) > 0:
            random_logits = logits[:, -len(random_paths):]
            random_preds = preds[:, -len(random_paths):]
            logits = logits[:, :len(base_paths)]
            preds = preds[:, :len(base_paths)]

        return preds, logits, random_preds, random_logits

    def features(self, x, *,
                 unassigned_path_to_use=None, **kwargs):

        if unassigned_path_to_use is not None:
            if isinstance(unassigned_path_to_use, tuple):
                unassigned_path_to_use = [unassigned_path_to_use]
            elif (isinstance(unassigned_path_to_use, list)
                  and isinstance(unassigned_path_to_use[0], int)):
                assert False
            elif unassigned_path_to_use == 'all':
                unassigned_path_to_use = self.available_paths

            to_iter = unassigned_path_to_use
        else:
            to_iter = self.paths_per_class.values()

        feats = []

        paths = list(zip(*[p for p, _ in to_iter]))
        paths_iterable = iter(paths)
        _x = [x] * len(paths[0])

        for l in self.layers[:-1]:
            _p = next(paths_iterable)
            _x, f = l(_x, _p)
            # _x = _x.relu()
            feats.append(f)

        logits, _x = self.layers[-1](_x, next(paths_iterable))
        feats.append(_x)

        logits = [torch.cat(l, -1 )for l in list(zip(*feats))]

        # logits = [nn.functional.adaptive_avg_pool2d(l, 4).flatten(1)
        #           for l in logits]

        return logits


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


# device = 'cpu'


def train(train_stream, test_stream, theta=1, train_epochs=1,
          fine_tune_epochs=1):
    # complete_model = MoECentroids()

    complete_model = MoELogits()

    criterion = CrossEntropyLoss()

    optimizer = Adam(complete_model.parameters(), lr=0.001)
    # optimizer = SGD(complete_model.parameters(), lr=0.01)

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=False, epoch=True, experience=True,
                         stream=True, trained_experience=True),
        bwt_metrics(experience=True, stream=True),
        loggers=[
            TextLogger(),
            # CustomTextLogger(),
        ],
    )

    # colors = distinctipy.get_colors(len(train_stream) + 1)

    trainer = Trainer(model=complete_model, optimizer=optimizer,
                      train_mb_size=32, eval_every=5,
                      eval_mb_size=32, evaluator=eval_plugin,
                      criterion=criterion, device=device,
                      train_epochs=train_epochs)

    # class Model(DynamicModule):
    #     def __init__(self, *args, **kwargs):
    #         super().__init__(*args, **kwargs)
    #         self.complete_model = nn.Sequential(
    #             nn.Conv2d(3, 32, 3),
    #             nn.ReLU(),
    #             nn.Conv2d(32, 64, 3),
    #             nn.ReLU(),
    #             nn.Conv2d(64, 128, 3),
    #             nn.ReLU(),
    #             nn.Conv2d(128, 256, 3),
    #             nn.ReLU(),
    #             nn.AdaptiveAvgPool2d(4),
    #             nn.Flatten(),
    #             nn.Linear(256 * 16, 100)
    #         )
    #
    #         self.classifier = IncrementalClassifier(256 * 4,
    #                                                 initial_out_features=2,
    #                                                 masking=False)
    #
    #     def forward(self, x, task_labels=None):
    #         x = self.complete_model(x)
    #         return x
    #         # return self.classifier(x, task_labels=task_labels)
    #
    # trainer = Replay(mem_size=2500, model=Model(),
    #               criterion=criterion, optimizer=optimizer,
    #               train_epochs=train_epochs, eval_every=5,
    #               train_mb_size=32, evaluator=eval_plugin,
    #               device=device)

    for current_task_id, tr in enumerate(train_stream):
        trainer.train(tr, eval_streams=[[e] for e in
                                        test_stream[:current_task_id + 1]])
        # trainer.train(tr)
        # trainer.eval(test_stream[:current_task_id + 1])
        # trainer.eval(test_stream[current_task_id])

        continue
        complete_model.eval()

        all_features = []
        all_labels = []

        with torch.no_grad():
            for d in test_stream[:current_task_id + 1]:
                for x, y, t in DataLoader(d.dataset):
                    x = x.to(device)
                    all_labels.append(y.numpy())

                    f = complete_model(x)[1]
                    all_features.append(f.cpu().numpy())

        all_labels = np.concatenate(all_labels, 0)
        all_features = np.concatenate(all_features, 0)

        import sklearn

        for c in range(all_features.shape[1]):
            pca = sklearn.decomposition.PCA(2)
            f = all_features[:, c]
            f = pca.fit_transform(f)
            labels = (c == all_labels).astype(int)
            labels = ['red' if y != c else 'blue' for y in all_labels]

            fig = plt.figure()
            plt.scatter(f[:, 0], f[:, 1], color=labels)
            plt.show()
            plt.close(fig)

        continue
        with torch.no_grad():
            for i, (c, d) in enumerate(
                    trainer.base_plugin.past_dataset.items()):
                d = DataLoader(d, batch_size=32)
                #
                tot = 0
                cor = 0

                # corr = defaultdict(int)
                # tott = defaultdict(int)

                for x, y, t, _ in d:
                    x = x.to(device)
                    y = y.to(device)

                    logits = complete_model(x, None)
                    pred = logits.argmax(-1)

                    cor += (pred == y).sum().item()
                    tot += len(x)

                print(i, tot, cor, cor / tot)

        continue
        for l in complete_model.layers:
            print(l.tasks_path)

        complete_model.eval()

        with torch.no_grad():
            for i, (c, d) in enumerate(
                    trainer.base_plugin.support_sets.items()):
                d = DataLoader(d, batch_size=64)

                tot = 0
                cor = 0

                for x, y, t in d:
                    x = x.to(device)
                    y = y.to(device)

                    logits = complete_model(x, None)
                    pred = logits.argmax(-1)

                    cor += (pred == y).sum().item()
                    tot += len(x)

                print(i, tot, cor, cor / tot)


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

    model = train(train_epochs=2,
                  fine_tune_epochs=0,
                  train_stream=train_tasks,
                  test_stream=test_tasks)

    # print(model.embeddings.weight)
    # with torch.no_grad():
    #     for i in range(len(train_tasks)):
    #         x, y, t = next(iter(DataLoader(test_tasks[i].dataset,
    #                                        batch_size=128,
    #                                        shuffle=True)))
    #         x = x.to(device)
    #
    #         # for _ in range(5):
    #         # for _ in range(5):
    #         pred = model(x, t)
    #         # print(torch.softmax(pred, -1))
    #
    #         for module in model.modules():
    #             if isinstance(module, AbsDynamicLayer):
    #                 print(i, module.last_distribution[0].mean(0),
    #                       module.get_task_blocks(i))
    #                 # print(i, j, module.last_distribution[1].mean(0))
    #                 print()
    #
    #             # for module in model.modules():
    #             #     if isinstance(module, MoERoutingLayer):
    #             #         print(
    #             #             f'Selection dataset {i} using task routing {j}',
    #             #             torch.unique(
    #             #                 module.last_distribution[0],
    #             #                 return_counts=True))
