"""Keep reference spectra out of a candidate set (spec section 4.2).

Used twice: our Bacteria-ID spectra vs the corpus (leakage guard), and probe
spectra vs pretrain spectra (probe isolation, build_corpus).
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch


def spectrum_hash(spec, mask) -> str:
    mask = np.asarray(mask, dtype=bool)
    h = hashlib.sha1()
    h.update(np.packbits(mask).tobytes())
    h.update(np.round(np.asarray(spec, dtype=np.float64)[mask], 4).astype(np.float32).tobytes())
    return h.hexdigest()


class LeakageGuard:
    """Flags candidates that are an exact hash match or a cosine near-duplicate
    (over the points valid in both) of any reference spectrum.

    A near-duplicate only counts if the candidate covers at least `min_overlap`
    of the reference spectrum's valid points, so a tiny cropped window cannot
    trigger a 0.999 match by chance.
    """

    def __init__(self, ref_X, ref_M, cos_threshold=0.999, min_overlap=0.5,
                 device=None, batch_size=1024):
        ref_M = np.asarray(ref_M, dtype=bool)
        self.hashes = {spectrum_hash(x, m) for x, m in zip(ref_X, ref_M)}
        self.cos_threshold = cos_threshold
        self.min_overlap = min_overlap
        self.batch_size = batch_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        O = np.where(ref_M, ref_X, 0.0).astype(np.float32)
        self.O = torch.from_numpy(O).to(self.device)
        self.O2 = self.O * self.O
        self.Om = torch.from_numpy(ref_M.astype(np.float32)).to(self.device)
        self.O_count = self.Om.sum(dim=1).clamp_min(1.0)

    @torch.no_grad()
    def find(self, X, M) -> np.ndarray:
        X = np.asarray(X)
        M = np.asarray(M, dtype=bool)
        leaked = np.array([spectrum_hash(x, m) in self.hashes for x, m in zip(X, M)],
                          dtype=bool).reshape(len(X))
        for s in range(0, len(X), self.batch_size):
            e = s + self.batch_size
            C = torch.from_numpy(np.where(M[s:e], X[s:e], 0.0).astype(np.float32)).to(self.device)
            Cm = torch.from_numpy(M[s:e].astype(np.float32)).to(self.device)
            dot = C @ self.O.T                 # both are zero outside their own masks
            c2 = (C * C) @ self.Om.T           # |candidate|^2 over the intersection
            o2 = Cm @ self.O2.T                # |reference|^2 over the intersection
            cos = dot / torch.sqrt(c2 * o2).clamp_min(1e-12)
            overlap = (Cm @ self.Om.T) / self.O_count
            cos = torch.where(overlap >= self.min_overlap, cos, torch.zeros_like(cos))
            leaked[s:e] |= (cos.max(dim=1).values >= self.cos_threshold).cpu().numpy()
        return leaked
