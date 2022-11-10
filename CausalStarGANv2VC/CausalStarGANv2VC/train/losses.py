import torch
from torch import nn
import torch.nn.functional as F


class BCEWithLogitHingeLoss:
    def __init__(self, m=10.0):
        self.m = m
        self.mm = -m

    def __call__(self, reality, label):
        if label == 1:
            reality = torch.clamp(reality, max=self.m)
            loss = torch.mean(torch.log1p(torch.exp(-reality)))
        elif label == 0:
            reality = torch.clamp(reality, min=self.mm)
            loss = torch.mean(torch.log1p(torch.exp(reality)))
        else:
            assert 0, "label ha 0 or 1 ni sitekudasai"
        return loss


def f_r1reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg


def f_f0_mean(f0, vuv):
    f0_sum = torch.sum(f0 * vuv, -1, True)
    vuv_sum = torch.sum(vuv, -1, True)
    vuv_sum += vuv_sum == 0  # prevent division by 0

    f0_mean = f0_sum / vuv_sum
    return f0_mean


def f_loss_f0(f0_x, vuv_x, f0_y, vuv_y):
    f0_x_mean = f_f0_mean(f0_x, vuv_x)
    f0_y_mean = f_f0_mean(f0_y, vuv_y)

    vuv_and = vuv_x * vuv_y
    loss = F.smooth_l1_loss((f0_x - f0_x_mean) * vuv_and, (f0_y - f0_y_mean) * vuv_and)
    return loss
