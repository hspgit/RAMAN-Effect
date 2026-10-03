"""Contiguous block masking of valid tokens (spec section 5.2, step 2).

Vectorised: no Python loop over samples, so it runs on the GPU inside the
training step (a per-sample loop cost ~400 ms per batch of 512).
"""
from __future__ import annotations

import torch


def block_mask(valid, ratio=0.5, min_block=2, max_block=6, generator=None):
    """Hide round(ratio * n_valid) valid tokens per sample, in runs of
    min_block..max_block consecutive valid tokens (adjacent runs may merge).
    At least one valid token always stays visible. `generator` must live on
    valid's device. Returns a (B, T) bool tensor."""
    B, T = valid.shape
    dev = valid.device
    n = valid.sum(dim=1, keepdim=True)                                   # (B, 1)
    target = torch.minimum(torch.round(ratio * n), (n - 1).clamp_min(0))
    rank = valid.long().cumsum(dim=1) - 1                                # rank among valid tokens
    # Cut the valid-rank axis into random blocks of min..max tokens.
    lengths = torch.randint(min_block, max_block + 1, (B, T), generator=generator, device=dev)
    ends = lengths.cumsum(dim=1)
    starts = ends - lengths
    eff = (torch.minimum(ends, n) - starts).clamp_min(0)                 # tokens per block
    # Hide whole blocks in random order until the target is met (last one partly).
    order = torch.argsort(torch.rand(B, T, generator=generator, device=dev), dim=1)
    eff_sorted = eff.gather(1, order)
    before = eff_sorted.cumsum(dim=1) - eff_sorted
    take_sorted = torch.minimum((target - before).clamp_min(0), eff_sorted)
    take = torch.zeros_like(take_sorted).scatter(1, order, take_sorted)  # per block id
    r = torch.arange(T, device=dev).expand(B, T).contiguous()
    block = torch.searchsorted(ends, r, right=True).clamp(max=T - 1)     # block of each rank
    hidden_rank = (r - starts.gather(1, block)) < take.gather(1, block)
    return valid & hidden_rank.gather(1, rank.clamp_min(0))


def visible_order(valid, hidden):
    """Indices that gather each sample's visible tokens first.
    Returns keep (B, K) and keep_valid (B, K), K = max visible count."""
    visible = valid & ~hidden
    order = torch.argsort((~visible).to(torch.int8), dim=1, stable=True)
    k = int(visible.sum(dim=1).max())
    keep = order[:, :k]
    return keep, visible.gather(1, keep)
