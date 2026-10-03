"""Corruptions that mimic session-to-session nuisances (spec sections 5.2, 6.1).

Batched torch ops on (B, N) spectra on the shared grid with a (B, N) bool
validity mask, so they run on the GPU inside the training step. Inputs are
min-max normalised over valid points (range 1); outputs are re-normalised the
same way and are 0 wherever the returned mask is False.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .harmonize import GRID_STEP


@dataclass(frozen=True)
class CorruptionConfig:
    baseline_max: float = 0.5           # added baseline peak height, x spectrum range
    shift_cm: float = 5.0               # uniform in +-shift_cm
    stretch: float = 0.005              # uniform in +-stretch (fraction of the axis)
    scale: tuple = (0.8, 1.2)
    snr: tuple = (10.0, 100.0)          # noise sd = range / snr
    broaden_sigma: float = 3.0          # max Gaussian sigma in points
    max_spikes: int = 2


PRETRAIN = CorruptionConfig()
MILD = CorruptionConfig(baseline_max=0.0, shift_cm=2.0, stretch=0.0, scale=(0.9, 1.1),
                        snr=(30.0, 100.0), broaden_sigma=0.0, max_spikes=0)
PRESETS = {"pretrain": PRETRAIN, "mild": MILD}


def _uniform(n, lo, hi, gen, device):
    return lo + (hi - lo) * torch.rand(n, generator=gen, device=device)


def minmax_valid(x, m, eps=1e-6):
    big = torch.finfo(x.dtype).max
    lo = torch.where(m, x, torch.full_like(x, big)).amin(dim=1, keepdim=True)
    hi = torch.where(m, x, torch.full_like(x, -big)).amax(dim=1, keepdim=True)
    out = (x - lo) / (hi - lo).clamp_min(eps)
    return torch.where(m, out, torch.zeros_like(out))


def shift_stretch(x, m, shift_cm, stretch):
    """Move features by shift_cm (positive = to higher wavenumber) and stretch the
    axis about its centre. Linear interpolation; a point is valid only if both
    source neighbours were valid."""
    B, N = x.shape
    p = torch.arange(N, device=x.device, dtype=x.dtype).expand(B, N)
    c = (N - 1) / 2
    src = c + (p - c) / (1 + stretch[:, None]) - shift_cm[:, None] / GRID_STEP
    inside = (src >= 0) & (src <= N - 1)
    i0 = src.floor().clamp(0, N - 2).long()
    w = (src - i0).clamp(0, 1)
    x0, x1 = x.gather(1, i0), x.gather(1, i0 + 1)
    m0, m1 = m.gather(1, i0), m.gather(1, i0 + 1)
    out = x0 * (1 - w) + x1 * w
    valid = inside & m0 & m1
    return torch.where(valid, out, torch.zeros_like(out)), valid


def broaden(x, m, sigma, max_sigma):
    """Per-sample Gaussian blur (sigma in points), normalised over valid points so
    invalid zeros do not bleed in. sigma ~ 0 leaves the spectrum unchanged."""
    if max_sigma <= 0:
        return x
    B, N = x.shape
    half = math.ceil(3 * max_sigma)
    k = torch.arange(-half, half + 1, device=x.device, dtype=x.dtype)
    s = sigma.clamp_min(1e-3)[:, None]
    kern = torch.exp(-0.5 * (k[None, :] / s) ** 2)
    kern = kern / kern.sum(dim=1, keepdim=True)
    mf = m.to(x.dtype)
    num = F.conv1d((x * mf)[None], kern[:, None, :], padding=half, groups=B)[0]
    den = F.conv1d(mf[None], kern[:, None, :], padding=half, groups=B)[0]
    return torch.where(m, num / den.clamp_min(1e-6), torch.zeros_like(x))


def smooth_baseline(B, N, gen, device, dtype=torch.float32):
    """Random cubic polynomial plus 0-2 broad Gaussian humps, min-max scaled to [0, 1]."""
    t = torch.linspace(-1, 1, N, device=device, dtype=dtype)
    coef = _uniform((B, 4), -1, 1, gen, device)
    base = sum(coef[:, i:i + 1] * t[None, :] ** i for i in range(4))
    n_humps = torch.randint(0, 3, (B,), generator=gen, device=device)
    for h in range(2):
        on = (n_humps > h).to(dtype)[:, None]
        centre = _uniform((B, 1), -1, 1, gen, device)
        width = _uniform((B, 1), 0.2, 0.6, gen, device)
        amp = _uniform((B, 1), 0.5, 2.0, gen, device)
        base = base + on * amp * torch.exp(-0.5 * ((t[None, :] - centre) / width) ** 2)
    lo, hi = base.amin(1, keepdim=True), base.amax(1, keepdim=True)
    return (base - lo) / (hi - lo).clamp_min(1e-6)


def add_spikes(x, m, max_spikes, gen):
    """0..max_spikes cosmic spikes, 1-2 points wide, at random valid positions."""
    if max_spikes <= 0:
        return x
    B, N = x.shape
    out = x.clone()
    n = torch.randint(0, max_spikes + 1, (B,), generator=gen, device=x.device)
    probs = m.to(x.dtype) + 1e-12
    for j in range(max_spikes):
        pos = torch.multinomial(probs, 1, generator=gen)[:, 0]
        height = _uniform(B, 0.2, 1.0, gen, x.device) * (n > j)
        wide = torch.rand(B, generator=gen, device=x.device) < 0.5
        rows = torch.arange(B, device=x.device)
        out[rows, pos] += height
        nxt = (pos + 1).clamp(max=N - 1)
        out[rows, nxt] += height * wide * m[rows, nxt]
    return out


def corrupt(x, m, cfg=PRETRAIN, generator=None):
    """Apply cfg's corruptions; returns (x_corrupt, m_corrupt)."""
    B, N = x.shape
    dev = x.device
    g = generator
    x = x.float()
    shift = _uniform(B, -cfg.shift_cm, cfg.shift_cm, g, dev)
    stretch = _uniform(B, -cfg.stretch, cfg.stretch, g, dev)
    x, m = shift_stretch(x, m, shift, stretch)
    x = broaden(x, m, _uniform(B, 0.0, cfg.broaden_sigma, g, dev), cfg.broaden_sigma)
    x = x * _uniform((B, 1), cfg.scale[0], cfg.scale[1], g, dev)
    if cfg.baseline_max > 0:
        amp = _uniform((B, 1), 0.0, cfg.baseline_max, g, dev)
        x = x + amp * smooth_baseline(B, N, g, dev)
    snr = _uniform((B, 1), cfg.snr[0], cfg.snr[1], g, dev)
    x = x + torch.randn(B, N, generator=g, device=dev) / snr
    x = add_spikes(x, m, cfg.max_spikes, g)
    return minmax_valid(x, m), m
