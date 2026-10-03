"""Multi-scale CNN stem (spec section 5.1, step 1)."""
from __future__ import annotations

import torch
import torch.nn as nn


class MaskedGroupNorm(nn.Module):
    """GroupNorm whose mean and (biased) variance are computed over valid points
    only: per sample and group, over all channels of the group and the positions
    where the mask is True. Invalid positions come out exactly 0. The affine
    parameters are named `weight` and `bias`, like nn.GroupNorm's."""

    def __init__(self, groups, channels, eps=1e-5):
        super().__init__()
        assert channels % groups == 0
        self.groups, self.channels, self.eps = groups, channels, eps
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, h, m):
        B, C, N = h.shape
        G = self.groups
        xf = h.float().view(B, G, C // G, N)
        mf = m.to(xf.dtype)[:, None, None, :]                      # (B, 1, 1, N)
        cnt = (mf.sum(dim=(2, 3), keepdim=True) * (C // G)).clamp(min=1.0)
        mean = (xf * mf).sum(dim=(2, 3), keepdim=True) / cnt
        d = (xf - mean) * mf                                       # centred, zero where invalid
        var = (d * d).sum(dim=(2, 3), keepdim=True) / cnt
        out = (d * torch.rsqrt(var + self.eps)).view(B, C, N)
        out = out * self.weight.float()[None, :, None] + self.bias.float()[None, :, None]
        return (out * m.to(out.dtype)[:, None, :]).to(h.dtype)


class Stem(nn.Module):
    """Parallel 1D convolutions with kernels of 3, 9 and 25 points (~4.5, 13.5 and
    37.5 cm^-1), `channels` each, concatenated, then GroupNorm over valid points
    only (masked), then GELU. The input has two channels, the spectrum (zeroed
    where invalid) and the validity mask, so the stem knows which points are real."""

    def __init__(self, channels=32, kernels=(3, 9, 25), groups=8):
        super().__init__()
        self.convs = nn.ModuleList(
            nn.Conv1d(2, channels, k, padding=k // 2) for k in kernels)
        self.out_channels = channels * len(kernels)
        self.norm = MaskedGroupNorm(groups, self.out_channels)
        self.act = nn.GELU()

    def forward(self, x, m):
        mf = m.to(x.dtype)
        inp = torch.stack([x * mf, mf], dim=1)                 # (B, 2, N)
        h = torch.cat([conv(inp) for conv in self.convs], dim=1)
        return self.act(self.norm(h, m))                        # (B, C, N)
