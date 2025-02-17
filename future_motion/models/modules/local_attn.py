import torch
import torch.nn.functional as F

from torch import nn
from local_attention import LocalMHA


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


def FeedForward(dim, mult=4, dropout=0.):
    inner_dim = int(dim * mult * 2 / 3)

    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias = False),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias = False)
    )


class LocalEncoder(nn.Module):
    def __init__(self, n_blocks, dim, attn_window, dropout=0.2):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                LocalMHA(dim=dim, window_size=attn_window, dim_head=dim // 8, heads=8, dropout=dropout, prenorm=True),
                FeedForward(dim=dim, dropout=dropout),
            ]) for _ in range(n_blocks)
        ])
    
    def forward(self, x, mask):
        for attn, ff in self.blocks:
            x = attn(x, mask) + x
            x = ff(x) + x

        return x