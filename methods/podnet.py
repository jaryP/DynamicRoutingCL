from copy import deepcopy
from typing import Callable, Optional

import numpy as np
import torch
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.training import ExemplarsBuffer, ClassBalancedBuffer
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates import SupervisedTemplate
from torch import nn, fx
from torch.optim import Optimizer, SGD
from torch.utils.data import DataLoader
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.resnet import ResNet as PytorchResNet

from models import RoutingModel
from models.backbone import ResNet
from models.routing.model import CondensedRoutingModel
from models.utils import FxCombinedModel, bn_track_stats


def default_graph_extractor(model: nn.Module) -> fx.GraphModule:
    if isinstance(model, PytorchResNet):
        return_nodes = {
            'layer1': 'layer1',
            'layer2': 'layer2',
            'layer3': 'layer3',
            'layer4': 'layer4',
            'flatten': 'features',
            # 'fc': 'logits',
        }
    elif isinstance(model, ResNet):
        return_nodes = {
            'layer1': 'layer1',
            'layer2': 'layer2',
            'layer3': 'layer3',
            'flatten': 'features',
            # 'fc': 'logits',
        }
    else:
        raise ValueError('The models accepted are Pytorch ResNet '
                         'and custom Resnet (from models.backbone)')
    body = create_feature_extractor(model, return_nodes=return_nodes)

    return body


class PodNet(SupervisedTemplate):
    def _after_forward(self, **kwargs):
        self.mb_output, self.intermediate_features = self.mb_output

        # self.intermediate_features = {k: v for k, v in features_d.items()
        #                               if 'layer'in k}
        # self.flat_features = features_d['features']

        super()._after_forward(**kwargs)

    # def make_train_dataloader(
    #     self,
    #     num_workers=0,
    #     shuffle=True,
    #     pin_memory=None,
    #     persistent_workers=False,
    #     drop_last=False,
    #     **kwargs
    # ):
    #     """Data loader initialization.
    #
    #     Called at the start of each learning experience after the dataset
    #     adaptation.
    #
    #     :param num_workers: number of thread workers for the data loading.
    #     :param shuffle: True if the data should be shuffled, False otherwise.
    #     :param pin_memory: If True, the data loader will copy Tensors into CUDA
    #         pinned memory before returning them. Defaults to True.
    #     """
    #
    #     assert self.adapted_dataset is not None
    #
    #     torch.utils.data.DataLoader
    #
    #     other_dataloader_args = self._obtain_common_dataloader_parameters(
    #         batch_size=self.train_mb_size,
    #         num_workers=num_workers,
    #         shuffle=shuffle,
    #         pin_memory=pin_memory,
    #         persistent_workers=persistent_workers,
    #         drop_last=drop_last,
    #     )
    #
    #     self.dataloader = DataLoader(
    #         self.adapted_dataset, **other_dataloader_args
    #     )

    def __init__(self,
                 feature_extractor: nn.Module,
                 classifier: nn.Module,
                 optimizer: Optimizer, mem_size: int, train_mb_size: int = 1,
                 train_epochs: int = 1,
                 lambda_c: float = 1,
                 lambda_f: float = 1,
                 features_nodes: dict = None,
                 scheduled_factor: bool = False,
                 storage_policy: Optional["ExemplarsBuffer"] = None,
                 # graph_extractor_f: Callable = default_graph_extractor,
                 eval_mb_size: int = None, device=None, plugins=None,
                 batch_size_mem=None, task_balanced_dataloader: bool = False,
                 criterion=None,
                 evaluator: EvaluationPlugin = default_evaluator,
                 eval_every=-1):

        if storage_policy is not None:  # Use other storage policy
            self.storage_policy = storage_policy
            assert storage_policy.max_size == mem_size
        else:  # Default
            self.storage_policy = ClassBalancedBuffer(
                max_size=mem_size, adaptive_size=True
            )

        self.lambda_c = lambda_c
        self.lambda_f = lambda_f
        self.scheduled_factor = scheduled_factor

        self.memory_size = mem_size
        self.batch_size_mem = batch_size_mem
        self.task_balanced_dataloader = task_balanced_dataloader

        self.intermediate_features = None
        self.flat_features = None

        if batch_size_mem is None:
            self.batch_size_mem = train_mb_size
        else:
            self.batch_size_mem = batch_size_mem

        self.past_model = None
        self.classifier = classifier

        if features_nodes is None:
            if isinstance(feature_extractor, PytorchResNet):
                features_nodes = {
                    'layer1': 'layer1',
                    'layer2': 'layer2',
                    'layer3': 'layer3',
                    'layer4': 'layer4',
                    'flatten': 'features',
                    # 'fc': 'logits',
                }
            elif isinstance(feature_extractor, ResNet):
                features_nodes = {
                    'layer1': 'layer1',
                    'layer2': 'layer2',
                    'layer3': 'layer3',
                    'flatten': 'features',
                    # 'fc': 'logits',
                }
            else:
                raise ValueError('When setting features_nodes=None, the feature extractors '
                                 'accepted are Pytorch ResNet '
                                 'and custom Resnet (from models.backbone)')

        feature_extractor = create_feature_extractor(feature_extractor,
                                                     return_nodes=features_nodes)

        model = FxCombinedModel(feature_extractor, classifier)

        super().__init__(
            model=model, optimizer=optimizer,
            criterion=None,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device,
            plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)

    def _after_eval_forward(self, **kwargs):
        self.mb_output, _ = self.mb_output

        super()._after_forward(**kwargs)

    def _after_training_exp(self, **kwargs):

        self.storage_policy.update(self, **kwargs)
        super()._after_training_exp(**kwargs)

    @torch.no_grad()
    def _before_training_exp(self,
                             num_workers: int = 0,
                             shuffle: bool = True,
                             drop_last: bool = False,
                             **kwargs):

        super()._before_training_exp(**kwargs)

        if len(self.storage_policy.buffer) == 0:
            return

        batch_size = self.train_mb_size

        batch_size_mem = self.batch_size_mem
        if batch_size_mem is None:
            batch_size_mem = batch_size

        assert self.adapted_dataset is not None
        self.dataloader = ReplayDataLoader(
            self.adapted_dataset,
            self.storage_policy.buffer,
            oversample_small_tasks=True,
            batch_size=batch_size,
            batch_size_mem=batch_size_mem,
            task_balanced_dataloader=self.task_balanced_dataloader,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
        )

        if self.model is not None:
            self.past_model = deepcopy(self.model)

            for m in self.past_model.modules():
                if hasattr(m, 'track_running_stats'):
                    m.track_running_stats = False

    def criterion(self):
        loss = nn.functional.cross_entropy(self.mb_output, self.mb_y)

        if self.past_model is not None and self.is_training:
            if self.scheduled_factor:
                current_classes = len(self.experience.classes_in_this_experience)
                seen_classes = len(self.experience.classes_seen_so_far)

                factor = (seen_classes / current_classes) ** 0.5
            else:
                factor = 1

            with torch.no_grad():
                _, past_outputs = self.past_model(self.mb_x)

            pod_spatial_loss = 0
            pod_flat_loss = 0

            for k, v in self.intermediate_features.items():
                pv = past_outputs[k]

                if k != 'features':
                    a_w = v.sum(dim=2).view(v.shape[0], -1)
                    a_h = v.sum(dim=3).view(v.shape[0], -1)

                    b_w = pv.sum(dim=2).view(v.shape[0], -1)
                    b_h = pv.sum(dim=3).view(v.shape[0], -1)

                    a = torch.cat([a_h, a_w], dim=-1)
                    b = torch.cat([b_h, b_w], dim=-1)

                    pod_spatial_loss += torch.mean(torch.frobenius_norm(a - b, dim=-1))

                else:
                    pod_flat_loss = nn.functional.mse_loss(v, pv,
                                                           reduction='mean')

            pod_spatial_loss = (pod_spatial_loss
                                / (len(self.intermediate_features) - 1))

            # pod_flat_loss = nn.functional.mse_loss(self.flat_features,
            #                                        past_outputs['features'],
            #                                        reduction='mean')

            pod_final = factor * (self.lambda_c * pod_spatial_loss
                         + self.lambda_f * pod_flat_loss)

            loss += pod_final

        return loss
