import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from .utils import init_weights, get_padding

LRELU_SLOPE = 0.1


class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h

        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        self.tails = []
        for d in dilation:
            self.convs1.append(weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=d,
                               padding=0)))
            self.convs2.append(weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=0)))
            self.tails.append((kernel_size - 1) * (d + 1))
        self.tail = sum(self.tails)

        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2, t in zip(self.convs1, self.convs2, self.tails):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x[:, :, t:]
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h

        self.convs = nn.ModuleList()
        self.tails = []
        for d in dilation:
            self.convs.append(weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=d,
                                                 padding=0)))
            self.tails.append((kernel_size - 1) * d)
        self.tail = sum(self.tails)

        self.convs.apply(init_weights)

    def forward(self, x):
        for c, t in zip(self.convs, self.tails):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x[:, :, t:]
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class Generator(torch.nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2
        self.conv_pre = weight_norm(Conv1d(h.num_mels, h.upsample_initial_channel, h.pre_kernel_size, 1, padding=0))
        self.tail = h.pre_kernel_size - 1
        tails = []

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(h.upsample_initial_channel // (2**i),
                                h.upsample_initial_channel // (2**(i + 1)),
                                k, u, padding=(k - u))))
            tails.append(k - u)

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2**(i + 1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))
            tails[i] += self.resblocks[(i + 1) * self.num_kernels - 1].tail

        self.conv_post = weight_norm(Conv1d(ch, 1, h.post_kernel_size, 1, padding=0))
        tails[-1] += h.post_kernel_size - 1

        for i, u in enumerate(h.upsample_rates):
            self.tail = self.tail * u + tails[i]
        tail_cut = h.hop_size - self.tail % h.hop_size
        self.tail += tail_cut

        self.tail_cuts = []
        hop_size_i = h.hop_size
        for u in h.upsample_rates:
            hop_size_i = hop_size_i // u
            tail_cut_i = tail_cut // hop_size_i
            self.tail_cuts.append(tail_cut_i)
            tail_cut = tail_cut - hop_size_i * tail_cut_i

        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            x = x[:, :, self.tail_cuts[i]:]

            tail_max = self.resblocks[(i + 1) * self.num_kernels - 1].tail
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x[:, :, tail_max - self.resblocks[i * self.num_kernels + j].tail:])
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x[:, :, tail_max - self.resblocks[i * self.num_kernels + j].tail:])
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
