import torch.nn as nn
import torch.nn.functional as F
import torch


class DEQCpResidualBlockAdjacentBlock(nn.Module):
    def __init__(self, dilations, in_channels, out_channels, kernel_size, block_id):
        super(DEQCpResidualBlockAdjacentBlock, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilations
        self.block_id = block_id


        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel_size),
                                   padding=0, dilation=dilations[0], bias=True)
        self.rel1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel_size),
                                   padding=0, dilation=2*dilations[0], bias=True)
        self.rel2 = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        bound = 0.01
        self.conv1.weight.data.normal_(0, bound)
        self.conv1.bias.data.normal_(0, bound)
        self.conv2.weight.data.normal_(0, bound)
        self.conv2.bias.data.normal_(0, bound)

    def copy(self, func):
        self.conv1.weight.data = func.conv1.weight.data.clone()
        self.conv1.bias.data = func.conv1.bias.data.clone()
        self.conv2.weight.data = func.conv2.weight.data.clone()
        self.conv2.bias.data = func.conv2.bias.data.clone()

    def forward_conv(self, conv, x, dilation):
        padding = [(self.kernel_size - 1) * dilation, 0, 0, 0, 0, 0]
        padded = torch.nn.functional.pad(x, padding)
        input_expanded = padded.unsqueeze(dim=2)
        conv.dilation = dilation
        y = conv(input_expanded)
        y = y.squeeze(dim=2)
        return y

    # dilations for deq is a tupple eg(1,4) so we have 4 convolutions or 2 residual blocks -> 1,2,4,8
    def forward(self, z1ss, uss, *kwargs):
        # pytorch expect BCHW input that is why we permute and add height in dim 2

        y1 = self.forward_conv(self.conv1, z1ss, self.dilation[0])
        y1 = self.layer_norm(y1)
        y1 = self.rel1(y1)

        y1 = self.forward_conv(self.conv2, y1, 2 * self.dilation[0])
        y1 = self.layer_norm(y1)
        y1 = self.rel2(y1)

        y1 = uss + y1

        y2 = y1
        y2 = self.forward_conv(self.conv1, y2, self.dilation[1])
        y2 = self.layer_norm(y2)
        y2 = self.rel1(y2)

        y2 = self.forward_conv(self.conv2, y2, 2 * self.dilation[1])
        y2 = self.layer_norm(y2)
        y2 = self.rel2(y2)

        return uss + y2

    def layer_norm(self, x, epsilon=1e-8):
        shape = x.shape
        x = x.permute(0, 2, 1)
        mean = x.mean(dim=len(shape) - 1, keepdim=True)
        variance = x.var(dim=len(shape) - 1, keepdim=True)

        x = (x - mean) / torch.sqrt(variance + epsilon)
        y = x.permute(0, 2, 1)
        return y