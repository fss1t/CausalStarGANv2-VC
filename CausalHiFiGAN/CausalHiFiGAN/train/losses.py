import torch
import torch.nn.functional as F


def feature_loss(list_fmap_real, list_fmap_fake):
    loss = 0
    for fmap_real, fmap_fake in zip(list_fmap_real, list_fmap_fake):
        for feat_real, feat_fake in zip(fmap_real, fmap_fake):
            loss += torch.mean(torch.abs(feat_real - feat_fake))
    return loss


def discriminator_loss(list_r_real, list_r_fake):
    loss_real = 0
    loss_fake = 0
    for r_real, r_fake in zip(list_r_real, list_r_fake):
        loss_real += torch.mean((1 - r_real)**2)
        loss_fake += torch.mean(r_fake**2)
    return loss_real, loss_fake


def generator_loss(list_r):
    loss = 0
    for r in list_r:
        loss += torch.mean((1 - r)**2)
    return loss
