from torch import nn

from models.backbone.resnet import BasicBlock


# TODO: add pooling operations
def get_conv_block(input_channels, output_channels,
                   kernel_size=3,
                   stride=1,
                   padding=1,
                   bias=True,
                   pooling=None,
                   **kwargs):

    return nn.Sequential(nn.Conv2d(in_channels=input_channels,
                                   out_channels=output_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   bias=bias),
                         nn.ReLU())


def get_double_conv_block(input_channels, output_channels):
    return nn.Sequential(get_conv_block(input_channels, output_channels),
                         get_conv_block(output_channels, output_channels),
                         # nn.MaxPool2d(2)
                         )


def get_resnet_block(input_channels, output_channels):
    return BasicBlock(input_channels, output_channels, option='B')
