"""Model A's encoder (spec section 5.1).

stem -> 10-point patches (101 tokens of 15 cm^-1) -> linear projection ->
sinusoidal encoding of each patch's centre wavenumber -> pre-LN transformer
with explicit q/k/v/o Linear layers (so LoRA can wrap them in Plan 3) ->
LayerNorm. Invalid tokens (<= 50% valid points) are never attended to.
No BatchNorm anywhere.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.harmonize import GRID_START, GRID_STEP, N_GRID
from .stem import Stem

PATCH = 10
N_TOKENS = math.ceil(N_GRID / PATCH)        # 101


@dataclass(frozen=True)
class EncoderConfig:
    layers: int
    width: int
    heads: int
    ff: int
    dropout: float = 0.1


PRESETS = {
    "tiny": EncoderConfig(layers=4, width=128, heads=4, ff=512),
    "small": EncoderConfig(layers=6, width=256, heads=8, ff=1024),
    "base": EncoderConfig(layers=8, width=384, heads=8, ff=1536),
}


def patch_centres_cm(patch=PATCH, n_tokens=N_TOKENS):
    idx = patch * torch.arange(n_tokens, dtype=torch.float32) + (patch - 1) / 2
    return GRID_START + GRID_STEP * idx


def sinusoidal(centres_cm, dim, patch=PATCH):
    """Standard sinusoidal table over position = centre / patch width (in patch
    units), so tokens from any wavenumber range share one encoding."""
    pos = centres_cm / (GRID_STEP * patch)
    i = torch.arange(dim // 2, dtype=torch.float32)
    freq = torch.exp(-math.log(10000.0) * 2 * i / dim)
    ang = pos[:, None] * freq[None, :]
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)       # (T, dim)


def token_validity(m, patch=PATCH, n_tokens=N_TOKENS):
    pad = n_tokens * patch - m.shape[1]
    mp = F.pad(m.float(), (0, pad))
    return mp.view(m.shape[0], n_tokens, patch).mean(dim=2) > 0.5   # (B, T)


class Attention(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.o = nn.Linear(width, width)

    def forward(self, h, key_valid):
        B, T, D = h.shape
        split = lambda t: t.view(B, T, self.heads, D // self.heads).transpose(1, 2)
        q, k, v = split(self.q(h)), split(self.k(h)), split(self.v(h))
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=key_valid[:, None, None, :],
            dropout_p=self.dropout if self.training else 0.0)
        return self.o(out.transpose(1, 2).reshape(B, T, D))


class MLP(nn.Module):
    def __init__(self, width, ff, dropout):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(width, ff), nn.Linear(ff, width)
        self.act, self.drop = nn.GELU(), nn.Dropout(dropout)

    def forward(self, h):
        return self.fc2(self.drop(self.act(self.fc1(h))))


class Block(nn.Module):
    def __init__(self, width, heads, ff, dropout):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attn = Attention(width, heads, dropout)
        self.mlp = MLP(width, ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, key_valid):
        h = h + self.drop(self.attn(self.ln1(h), key_valid))
        return h + self.drop(self.mlp(self.ln2(h)))


class SpectralEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.stem = Stem()
        self.proj = nn.Linear(self.stem.out_channels * PATCH, cfg.width)
        self.register_buffer("pos", sinusoidal(patch_centres_cm(), cfg.width), persistent=False)
        self.blocks = nn.ModuleList(
            Block(cfg.width, cfg.heads, cfg.ff, cfg.dropout) for _ in range(cfg.layers))
        self.norm = nn.LayerNorm(cfg.width)

    def tokenize(self, x, m):
        """(B, 1001) spectrum + mask -> (B, 101, width) tokens, (B, 101) validity."""
        h = self.stem(x, m)                                      # (B, C, N)
        h = F.pad(h, (0, N_TOKENS * PATCH - h.shape[2]))
        B, C, _ = h.shape
        h = h.view(B, C, N_TOKENS, PATCH).permute(0, 2, 1, 3).reshape(B, N_TOKENS, C * PATCH)
        return self.proj(h) + self.pos, token_validity(m)

    def encode(self, tokens, key_valid):
        h = tokens
        for blk in self.blocks:
            h = blk(h, key_valid)
        return self.norm(h)

    def forward(self, x, m):
        tokens, valid = self.tokenize(x, m)
        return self.encode(tokens, valid), valid

    def embed(self, x, m):
        """Mean over valid tokens -> (B, width): the spectrum's embedding."""
        h, valid = self.forward(x, m)
        w = valid.to(h.dtype)[..., None]
        return (h * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)


def count_parameters(module):
    return sum(p.numel() for p in module.parameters())
