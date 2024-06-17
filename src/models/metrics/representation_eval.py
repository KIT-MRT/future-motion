import torch


@torch.no_grad()
def std_of_l2_normalized(z: torch.Tensor) -> torch.Tensor:
    """Calculates the mean of the standard deviation of z along each dimension.

    This measure was used by [0] to determine the level of collapse of the
    learned representations. If the returned number is 0., the outputs z have
    collapsed to a constant vector. "If the output z has a zero-mean isotropic
    Gaussian distribution" [0], the returned number should be close to 1/sqrt(d)
    where d is the dimensionality of the output.

    [0]: https://arxiv.org/abs/2011.10566

    Args:
        z:
            A torch tensor of shape batch_size x dimension.

    Returns:
        The mean of the standard deviation of the l2 normalized tensor z along
        each dimension.

    """

    if len(z.shape) != 2:
        raise ValueError(
            f"Input tensor must have two dimensions but has {len(z.shape)}!"
        )

    z_norm = torch.nn.functional.normalize(z, dim=1)
    return torch.std(z_norm, dim=0).mean()


def get_barlow_twins_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    lambda_coeff: float,
    return_loss_terms: bool = False,
) -> torch.Tensor:
    batch_size = z1.shape[0]

    # N x D, where N is the batch size and D is output dim of projection head
    z1_norm = (z1 - torch.mean(z1, dim=0)) / torch.std(z1, dim=0)
    z2_norm = (z2 - torch.mean(z2, dim=0)) / torch.std(z2, dim=0)

    cross_corr = torch.matmul(z1_norm.T, z2_norm) / batch_size

    on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
    off_diag = off_diagonal_ele(cross_corr).pow_(2).sum()

    loss = on_diag + lambda_coeff * off_diag

    if return_loss_terms:
        return loss, on_diag, off_diag
    else:
        return loss


def off_diagonal_ele(x: torch.Tensor) -> torch.Tensor:
    """Adapted from: https://github.com/facebookresearch/barlowtwins/blob/main/main.py

    Returns:
        a flattened view of the off-diagonal elements of a square matrix
    """
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
