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
    """Flags candidates that are an exact hash match, or a cosine or Pearson
    near-duplicate (over the points valid in both) of any reference spectrum.

    Pearson correlation catches copies that cosine misses: re-gridded or
    cropped-and-renormalised spectra carry a different offset/scale, which
    lowers their cosine but not their correlation.

    A near-duplicate only counts if the candidate covers at least `min_overlap`
    of the reference spectrum's valid points, so a tiny cropped window cannot
    trigger a 0.999 match by chance.
    """

    def __init__(self, ref_X, ref_M, cos_threshold=0.999, corr_threshold=0.999,
                 min_overlap=0.5, device=None, batch_size=1024):
        ref_M = np.asarray(ref_M, dtype=bool)
        self.hashes = {spectrum_hash(x, m) for x, m in zip(ref_X, ref_M)}
        self.cos_threshold = cos_threshold
        self.corr_threshold = corr_threshold
        self.min_overlap = min_overlap
        self.batch_size = batch_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        O = np.where(ref_M, ref_X, 0.0).astype(np.float32)
        self.O = torch.from_numpy(O).to(self.device)
        self.O2 = self.O * self.O
        self.Om = torch.from_numpy(ref_M.astype(np.float32)).to(self.device)
        self.O_count = self.Om.sum(dim=1).clamp_min(1.0)

    @torch.no_grad()
    def scores(self, X, M):
        """Best cosine and best Pearson correlation of each candidate against
        any qualifying reference (overlap >= min_overlap); 0 if none qualifies."""
        X = np.asarray(X)
        M = np.asarray(M, dtype=bool)
        max_cos = np.zeros(len(X), dtype=np.float32)
        max_corr = np.zeros(len(X), dtype=np.float32)
        for s in range(0, len(X), self.batch_size):
            e = s + self.batch_size
            C = torch.from_numpy(np.where(M[s:e], X[s:e], 0.0).astype(np.float32)).to(self.device)
            Cm = torch.from_numpy(M[s:e].astype(np.float32)).to(self.device)
            n = Cm @ self.Om.T                 # points valid in both
            sc = C @ self.Om.T                 # sum of candidate over the intersection
            so = Cm @ self.O.T                 # sum of reference over the intersection
            dot = C @ self.O.T                 # both are zero outside their own masks
            c2 = (C * C) @ self.Om.T           # |candidate|^2 over the intersection
            o2 = Cm @ self.O2.T                # |reference|^2 over the intersection
            ok = (n / self.O_count) >= self.min_overlap
            cos = dot / torch.sqrt(c2 * o2).clamp_min(1e-12)
            n = n.clamp_min(1.0)
            var_c = c2 - sc * sc / n
            var_o = o2 - so * so / n
            corr = (dot - sc * so / n) / (var_c.clamp_min(1e-12) * var_o.clamp_min(1e-12)).sqrt()
            corr = torch.where((var_c > 0) & (var_o > 0), corr, torch.zeros_like(corr))
            zero = torch.zeros_like(cos)
            max_cos[s:e] = torch.where(ok, cos, zero).max(dim=1).values.cpu().numpy()
            max_corr[s:e] = torch.where(ok, corr, zero).max(dim=1).values.cpu().numpy()
        return max_cos, max_corr

    def flag(self, X, M, max_cos, max_corr) -> np.ndarray:
        """Leak decision from precomputed scores (lets callers reuse them)."""
        X = np.asarray(X)
        M = np.asarray(M, dtype=bool)
        exact = np.array([spectrum_hash(x, m) in self.hashes for x, m in zip(X, M)],
                         dtype=bool).reshape(len(X))
        return exact | (max_cos >= self.cos_threshold) | (max_corr >= self.corr_threshold)

    def find(self, X, M) -> np.ndarray:
        return self.flag(X, M, *self.scores(X, M))
