import torch

from torch import Tensor


def get_barlow_twins_loss(z_a: Tensor, z_b: Tensor, lambda_coeff: float) -> Tensor:
    """Computes the Barlow Twins loss (https://arxiv.org/abs/2103.03230)
    Adapted from: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
    """
    batch_size = z_a.shape[0]

    # N x D, where N is the batch size and D is output dim of the projection head
    z_a_norm = (z_a - torch.mean(z_a, dim=0)) / torch.std(z_a, dim=0)
    z_b_norm = (z_b - torch.mean(z_b, dim=0)) / torch.std(z_b, dim=0)

    cross_corr = torch.matmul(z_a_norm.T, z_b_norm) / batch_size

    on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
    off_diag = off_diagonal_ele(cross_corr).pow_(2).sum()

    return on_diag + lambda_coeff * off_diag


def off_diagonal_ele(x: Tensor) -> Tensor:
    """Returns a flattened view of the off-diagonal elements of a square matrix
    Adapted from: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
    """
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()