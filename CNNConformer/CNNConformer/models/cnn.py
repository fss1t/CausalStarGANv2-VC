import math
import torch
import torch.nn as nn
import torch.nn.functional as F

sqrt2 = math.sqrt(2)
alpha = 0.01


class CNN(nn.Module):
    """
    Convolutional 2D subsampling (to 1/4 length)

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution

    Inputs: inputs
        - **inputs** (batch, channel, dim, time): Tensor containing sequence of inputs

    Returns: outputs, output_lengths
        - **outputs** (batch, channel, dim, time): Tensor produced by the convolution
        - **output_lengths** (batch): list of sequence output lengths
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(CNN, self).__init__()
        base_channels = out_channels // 4
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            ResBlk(base_channels, base_channels * 2, downsample=nn.MaxPool2d((2, 2))),
            ResBlk(base_channels * 2, base_channels * 4, downsample=nn.MaxPool2d((2, 2))),
            ResBlk(base_channels * 4, base_channels * 4, downsample=nn.MaxPool2d((2, 1))),
            ResBlk(base_channels * 4, base_channels * 4, downsample=nn.MaxPool2d((2, 1))),
            ResBlk5_3(base_channels * 4, base_channels * 4, downsample=nn.MaxPool2d((2, 1))),
            ResBlk(base_channels * 4, base_channels * 4),
            ResBlk(base_channels * 4, base_channels * 4),
            ConvBlk(base_channels * 4, base_channels * 4)
        )

    def forward(self, inputs):
        outputs = self.sequential(inputs)
        outputs = outputs.squeeze(2).transpose(1, 2)

        return outputs


class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(alpha),
                 normalize=True, downsample=None):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out

        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1, bias=not normalize)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample != None:
            x = self.downsample(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample != None:
            x = self.downsample(x)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / sqrt2


class ResBlk5_3(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(alpha),
                 normalize=True, downsample=None):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out

        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1, bias=not normalize)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample != None:
            x = F.pad(x, (0, 0, 1, 0))
            x = self.downsample(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample != None:
            x = F.pad(x, (0, 0, 1, 0))
            x = self.downsample(x)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / sqrt2


class ConvBlk(nn.Module):
    def __init__(self, dim_in, dim_out, activate=nn.LeakyReLU(alpha),
                 normalize=True):
        super().__init__()
        self.activate = activate
        self.normalize = normalize

        self.conv1 = nn.Conv2d(dim_in, dim_out, (3, 1), (1, 1), bias=not normalize)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_out, affine=True)

    def forward(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.activate(x)
        x = self.conv1(x)
        if self.normalize:
            x = self.norm2(x)
        x = self.activate(x)
        return x
