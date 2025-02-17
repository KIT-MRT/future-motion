import torch


def get_masked_indices(
    masking_ratio: float,
    num_tokens: int,
    batch_size: int,
    device: str,
):
    """Gets indices to mask for MAE pre-training (https://arxiv.org/abs/2111.06377)
    Adapted from: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/mae.py
    """
    num_masked = int(masking_ratio * num_tokens)
    rand_indices = torch.rand(batch_size, num_tokens, device=device).argsort(dim=-1)
    masked_indices = rand_indices[:, :num_masked]

    return masked_indices

def get_masked_and_unmasked_indices(
    masking_ratio: float,
    num_tokens: int,
    device: str,
):
    num_masked = int(masking_ratio * num_tokens)
    rand_indices = torch.rand(num_tokens, device=device).argsort(dim=-1)
    masked_indices = rand_indices[:num_masked]
    unmasked_indices = rand_indices[num_masked:]

    return masked_indices, unmasked_indices

def get_masked_and_unmasked_indices_batch(
    masking_ratio: float,
    num_tokens: int,
    batch_size: int,
    device: str,
):
    num_masked = int(masking_ratio * num_tokens)
    rand_indices = torch.rand(batch_size, num_tokens, device=device).argsort(dim=-1)
    masked_indices = rand_indices[:, :num_masked]
    unmasked_incdices = rand_indices[:, num_masked:]

    return masked_indices, unmasked_incdices