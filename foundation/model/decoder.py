"""MAE decoder and reconstruction loss (spec section 5.2, steps 3-4).

The decoder is used only during pretraining and is discarded afterwards.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import N_TOKENS, PATCH, Block, patch_centres_cm, sinusoidal

TARGET_EPS = 1e-3   # variance floor for per-patch normalisation: a near-flat
                    # patch is not blown up into unit-variance noise


class MAEDecoder(nn.Module):
    def __init__(self, enc_width, width=128, layers=2, heads=4, ff=512):
        super().__init__()
        self.embed = nn.Linear(enc_width, width)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, width))
        nn.init.normal_(self.mask_token, std=0.02)
        self.register_buffer("pos", sinusoidal(patch_centres_cm(), width), persistent=False)
        self.blocks = nn.ModuleList(Block(width, heads, ff, 0.0) for _ in range(layers))
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, PATCH)

    def forward(self, h_vis, keep, keep_valid, valid):
        """h_vis (B, K, enc_width) for the tokens in keep (B, K) -> (B, T, PATCH)."""
        B = h_vis.shape[0]
        mask_tok = self.mask_token.to(h_vis.dtype)
        src = torch.where(keep_valid[..., None], self.embed(h_vis).to(h_vis.dtype), mask_tok)
        z = mask_tok.expand(B, N_TOKENS, -1).clone()
        z = z.scatter(1, keep[..., None].expand(-1, -1, z.shape[2]), src)
        z = z + self.pos
        for blk in self.blocks:
            z = blk(z, valid)
        return self.head(self.norm(z))


def patchify(x, patch=PATCH, n_tokens=N_TOKENS):
    pad = n_tokens * patch - x.shape[1]
    return F.pad(x, (0, pad)).view(x.shape[0], n_tokens, patch)


def mae_loss(pred, clean, clean_mask, hidden, deriv_weight=0.5):
    """Squared error on hidden tokens' valid points against the per-patch
    normalised clean target, plus deriv_weight x squared error on the first
    derivative (so peaks count more than smooth background)."""
    pred = pred.float()
    tgt = patchify(clean.float())
    pm = patchify(clean_mask.float()) > 0
    cnt = pm.sum(dim=-1, keepdim=True).clamp_min(1)
    mean = (tgt * pm).sum(-1, keepdim=True) / cnt
    var = (((tgt - mean) ** 2) * pm).sum(-1, keepdim=True) / cnt
    tgt_n = (tgt - mean) / torch.sqrt(var + TARGET_EPS)
    w = pm & hidden[..., None]
    mse = (((pred - tgt_n) ** 2) * w).sum() / w.sum().clamp_min(1)
    wd = w[..., 1:] & w[..., :-1]
    dmse = (((pred.diff(dim=-1) - tgt_n.diff(dim=-1)) ** 2) * wd).sum() / wd.sum().clamp_min(1)
    loss = mse + deriv_weight * dmse
    mse, dmse = mse.detach(), dmse.detach()
    return loss, {"mse": mse.item(), "dmse": dmse.item()}
