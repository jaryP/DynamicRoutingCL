import torchvision
from avalanche.models import DynamicModule
from torch import nn as nn


class ProcessRouting(nn.Module):
    model = torchvision.models.resnet18(pretrained=True)

    def forward(self, x):
        return


def set_requires_grad(m, requires_grad):
    for param in m.parameters():
        param.requires_grad_(requires_grad)


class BlockRoutingLayer(DynamicModule):
    def __init__(self,
                 n_blocks: int,
                 factory,
                 **kwargs):

        super().__init__()

        self.blocks = nn.ModuleDict()
        self.projectors = nn.ModuleDict()

        for i in range(n_blocks):
            i = str(i)
            self.blocks[i] = factory()

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
        if not isinstance(block_id, (list, tuple)):
            x = [x]
            block_id = [block_id]

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
