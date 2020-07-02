import torch.nn as nn
import torch


class ResidualBlock(nn.Module):
    def __init__(self, kernel_size, dilation, block_id):
        super(ResidualBlock, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.block_id = block_id

    def forward_conv(self, conv, x, dilation):
        padding = [(self.kernel_size - 1) * dilation, 0, 0, 0, 0, 0]
        padded = torch.nn.functional.pad(x, padding)
        input_expanded = padded.unsqueeze(dim=2)
        conv.dilation = dilation
        y = conv(input_expanded)
        y = y.squeeze(dim=2)
        return y


class CpResidualBlockCrossLayer(ResidualBlock):
    def __init__(self, dilation, in_channels, out_channels, kernel_size, block_id, prev_blk):
        super(CpResidualBlockCrossLayer, self).__init__(kernel_size, dilation, block_id)
        if prev_blk is None:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel_size),
                                   padding=0, dilation=dilation, bias=True)
        else:
            self.conv = prev_blk.conv

        self.layer_norm1 = None
        self.rel1 = nn.ReLU()
        self.layer_norm2 = None
        self.rel2 = nn.ReLU()

    def forward(self, inputs):
        # pytorch expect BCHW input that is why we permute and add height in dim 2
        inputs_conv = inputs.permute(0, 2, 1)

        y = self.forward_conv(self.conv, inputs_conv, self.dilation)
        if self.layer_norm1 is None:
            self.layer_norm1 = nn.LayerNorm(y.shape[1:], elementwise_affine=False)
        y = self.layer_norm1(y)
        y = self.rel1(y)

        y = self.forward_conv(self.conv, y, 2*self.dilation)

        if self.layer_norm2 is None:
            self.layer_norm2 = nn.LayerNorm(y.shape[1:], elementwise_affine=False)
        y = self.layer_norm2(y)
        y = self.rel2(y)

        y = y.permute(0, 2, 1)
        return inputs + y


class CpResidualBlockAdjacentLayer(CpResidualBlockCrossLayer):
    def __init__(self, dilation, in_channels, out_channels, kernel_size, block_id, prev_blk):
        super(CpResidualBlockCrossLayer, self).__init__(kernel_size, dilation, block_id)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel_size),
                                   padding=0, dilation=dilation, bias=True)

        self.layer_norm1 = None
        self.rel1 = nn.ReLU()

        self.layer_norm2 = None
        self.rel2 = nn.ReLU()

    def forward(self, inputs):
        # pytorch expect BCHW input that is why we permute and add height in dim 2
        inputs_conv = inputs.permute(0, 2, 1)

        y = self.forward_conv(self.conv, inputs_conv, self.dilation)
        if self.layer_norm1 is None:
            self.layer_norm1 = nn.LayerNorm(y.shape[1:], elementwise_affine=False)
        y = self.layer_norm1(y)
        y = self.rel1(y)

        y = self.forward_conv(self.conv, y, 2 * self.dilation)

        if self.layer_norm2 is None:
            self.layer_norm2 = nn.LayerNorm(y.shape[1:], elementwise_affine=False)
        y = self.layer_norm2(y)
        y = self.rel2(y)

        y = y.permute(0, 2, 1)
        return inputs + y


class CpResidualBlockAdjacentBlock(CpResidualBlockCrossLayer):
    def __init__(self, dilation, in_channels, out_channels, kernel_size, block_id, prev_blk):
        super(CpResidualBlockCrossLayer, self).__init__(kernel_size, dilation, block_id)

        if prev_blk is not None and self.is_adjacent_with_previous(prev_blk, block_id):
            self.conv = prev_blk.conv
        else:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel_size),
                                  padding=0, dilation=dilation, bias=True)

        self.layer_norm1 = None
        self.rel1 = nn.ReLU()

        self.layer_norm2 = None
        self.rel2 = nn.ReLU()

    def forward(self, inputs):
        # pytorch expect BCHW input that is why we permute and add height in dim 2
        inputs_conv = inputs.permute(0, 2, 1)

        y = self.forward_conv(self.conv, inputs_conv, self.dilation)
        if self.layer_norm1 is None:
            self.layer_norm1 = nn.LayerNorm(y.shape[1:], elementwise_affine=False)
        y = self.layer_norm1(y)
        y = self.rel1(y)

        y = self.forward_conv(self.conv, y, 2 * self.dilation)

        if self.layer_norm2 is None:
            self.layer_norm2 = nn.LayerNorm(y.shape[1:], elementwise_affine=False)
        y = self.layer_norm2(y)
        y = self.rel2(y)

        y = y.permute(0, 2, 1)
        return inputs + y

    def is_adjacent_with_previous(self, prev_conv, block_id):
        return (block_id // 2) == (prev_conv.block_id // 2)
