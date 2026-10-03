"""Multi-scale CNN stem (spec section 5.1, step 1)."""
from __future__ import annotations

import torch
import torch.nn as nn


class Stem(nn.Module):
    """Parallel 1D convolutions with kernels of 3, 9 and 25 points (~4.5, 13.5 and
    37.5 cm^-1), `channels` each, concatenated, then GroupNorm and GELU. The input
    has two channels, the spectrum (zeroed where invalid) and the validity mask,
    so the stem knows which points are real."""

    def __init__(self, channels=32, kernels=(3, 9, 25), groups=8):
        super().__init__()
        self.convs = nn.ModuleList(
            nn.Conv1d(2, channels, k, padding=k // 2) for k in kernels)
        self.out_channels = channels * len(kernels)
        self.norm = nn.GroupNorm(groups, self.out_channels)
        self.act = nn.GELU()

    def forward(self, x, m):
        mf = m.to(x.dtype)
        inp = torch.stack([x * mf, mf], dim=1)                 # (B, 2, N)
        h = torch.cat([conv(inp) for conv in self.convs], dim=1)
        return self.act(self.norm(h))                           # (B, C, N)
