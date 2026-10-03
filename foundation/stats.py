"""Paired statistics and the pre-registered questions (spec section 8).

Correctness arrays are (seeds, test samples) bool, rows paired by seed.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest

N_BOOT = 10_000
Q3_MARGIN = 0.01          # 1.0 accuracy point
Q3_MAX_FRACTION = 0.05


def mcnemar(correct_a, correct_b):
    """Exact two-sided McNemar test on one paired seed. b: only A right,
    c: only B right."""
    a, b_ = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    if a.shape != b_.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b_.shape}")
    b = int((a & ~b_).sum())
    c = int((~a & b_).sum())
    p = 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue)
    return {"b": b, "c": c, "p": p}


def bootstrap_diff(correct_a, correct_b, n_boot=N_BOOT, seed=0, chunk=500):
    """Accuracy difference A - B: point estimate (mean over seeds) and a 95%
    percentile CI from resampling seeds and test samples together."""
    a = np.asarray(correct_a, bool)
    b = np.asarray(correct_b, bool)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError(f"need equal (seeds, samples) arrays, got {a.shape} and {b.shape}")
    d = a.astype(np.float64) - b.astype(np.float64)              # (S, N)
    S, N = d.shape
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for s in range(0, n_boot, chunk):
        k = min(chunk, n_boot - s)
        w = rng.multinomial(N, np.full(N, 1.0 / N), size=k)       # (k, N) sample counts
        per_seed = w @ d.T / N                                    # (k, S)
        pick = rng.integers(0, S, size=(k, S))
        out[s:s + k] = np.take_along_axis(per_seed, pick, axis=1).mean(axis=1)
    lo, hi = np.percentile(out, [2.5, 97.5])
    return {"diff": float(d.mean()), "lo": float(lo), "hi": float(hi)}


def mean_sd(values):
    v = np.asarray(values, float)
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0


def verdict(ci):
    if ci["lo"] > 0:
        return "A helps"
    if ci["hi"] < 0:
        return "A hurts"
    return "no clear difference"


def q1(diffs):
    """diffs[task][(arm, 'S')] -> bootstrap dict. Success = mean > 0 and CI
    excluding 0 (spec 8.3)."""
    return {task: {f"{a} vs {b}": {**ci, "verdict": verdict(ci),
                                   "success": ci["diff"] > 0 and ci["lo"] > 0}
                   for (a, b), ci in pairs.items() if b == "S"}
            for task, pairs in diffs.items()}


def q3(mean_acc, fraction, candidates=("B2", "LP")):
    """mean_acc[task][arm], fraction[task][arm]. A candidate passes if, on every
    task, its mean accuracy is within Q3_MARGIN of that task's best arm and it
    trains <= Q3_MAX_FRACTION of its parameters."""
    res = {}
    for c in candidates:
        per_task = {}
        for task, accs in mean_acc.items():
            best = max(accs.values())
            per_task[task] = {"gap": best - accs[c], "fraction": fraction[task][c],
                              "ok": accs[c] >= best - Q3_MARGIN - 1e-12
                              and fraction[task][c] <= Q3_MAX_FRACTION}
        res[c] = {"tasks": per_task, "pass": all(t["ok"] for t in per_task.values())}
    return {"candidates": res, "pass": any(r["pass"] for r in res.values())}
