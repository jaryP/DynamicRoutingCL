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


class BlockRoutingLayer(DynamicModule):
    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 project_dim = None,
                 get_average_features = False,
                 # out_project_dim = None,
                 **kwargs):

        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels

        self.blocks = nn.ModuleDict()
        self.projectors = nn.ModuleDict()

        for i in range(10):
            b = nn.Conv2d(in_channels=self.input_channels,
                          out_channels=self.output_channels,
                          kernel_size=3,
                          stride=1)

            i = str(i)
            self.blocks[i] = b

            if (project_dim is not None and project_dim > 0) or get_average_features:
                l = nn.Sequential(nn.AdaptiveAvgPool2d(4),
                                  nn.Flatten(1))
                
                if project_dim is not None and project_dim > 0:
                    l.append(nn.ReLU())
                    l.append(nn.Linear(output_channels * 16, project_dim))
                    bs = project_dim
                else:
                    bs = output_channels * 4

                # l.append(nn.BatchNorm1d(bs))

                self.projectors[i] = l

    def clean_cache(self):
        keys = [k for k, _ in self.named_buffers() if 'cache' in k]
        for k in keys:
            delattr(self, k)

    def activate_blocks(self, block_ids):
        block_id = list(map(str, block_ids))
        for k in self.blocks.keys():
            flag = k in block_id

            for p in self.blocks[k].parameters():
                p.requires_grad_(flag)

            if k in self.projectors:
                for p in self.projectors[k].parameters():
                    p.requires_grad_(flag)

    def freeze_blocks(self):
        for p in self.blocks.parameters():
            p.requires_grad_(False)

        for p in self.projectors.parameters():
            p.requires_grad_(False)

    def activate_block(self, block_ids):
        block_id = str(block_ids)

        for p in self.blocks[block_id].parameters():
            p.requires_grad_(True)

    def freeze_block(self, block_ids, freeze=True):
        block_id = str(block_ids)

        freeze = not freeze
        for p in self.blocks[block_id].parameters():
            p.requires_grad_(freeze)

        if block_id in self.projectors:
            for p in self.projectors[block_id].parameters():
                p.requires_grad_(freeze)

    def forward(self, x, block_id, **kwargs):
        if isinstance(block_id, (list, tuple)):
            ret = []
            ret_l = []

            for _x, _bid in zip(x, block_id):
                f = self.blocks[str(_bid)](_x).relu()
                if len(self.projectors) > 0:
                    l = self.projectors[str(_bid)](f)
                    ret_l.append(l)
                else:
                    ret_l.append(f)

                ret.append(f)

            if len(ret_l) > 0:
                return ret, ret_l
            return ret, None
        else:
            f = self.blocks[str(block_id)](x)
            l = None
            if len(self.projectors) > 0:
                l = self.projectors[str(block_id)](f)

            if l is not None:
                return f, l
            return f

