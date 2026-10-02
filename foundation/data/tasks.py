"""Our Bacteria-ID tasks on the common grid (spec section 3).

Reads data/*.npy read-only. Model A's corpus build uses load_ours() only to
know what to KEEP OUT (leakage guard); nothing here feeds pretraining.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .harmonize import HARMONIZE_VERSION, harmonize_many

OUR_NAMES = ("reference", "finetune", "test", "2018clinical", "2019clinical")
SPLIT_SEED = 0       # fixed: every arm and every seed sees the same validation split
VAL_FRACTION = 0.1


@dataclass
class Split:
    X: np.ndarray    # (n, 1001) float32
    M: np.ndarray    # (n, 1001) bool
    y: np.ndarray    # (n,) int64 class index


@dataclass
class Task:
    name: str
    classes: np.ndarray            # raw label value for each class index
    splits: dict


def _cache_key(data_dir: Path) -> str:
    sizes = [(n, (data_dir / f"y_{n}.npy").stat().st_size) for n in OUR_NAMES]
    return hashlib.sha1(repr(sizes).encode()).hexdigest()[:8]


def load_ours(data_dir="data", cache_dir="foundation/data/cache"):
    """Harmonise every Bacteria-ID file once. Returns {name: (X, M, y_raw)}."""
    data_dir = Path(data_dir)
    cache = Path(cache_dir) / f"ours_h{HARMONIZE_VERSION}_{_cache_key(data_dir)}.npz"
    if cache.exists():
        z = np.load(cache)
        return {n: (z[f"{n}_X"], z[f"{n}_M"], z[f"{n}_y"]) for n in OUR_NAMES}
    axis = np.load(data_dir / "wavenumbers.npy")
    out = {}
    for n in OUR_NAMES:
        X = np.load(data_dir / f"X_{n}.npy")
        y = np.load(data_dir / f"y_{n}.npy")
        S, M = harmonize_many(axis, X)
        out[n] = (S, M, y)
    cache.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for n, (S, M, y) in out.items():
        arrays[f"{n}_X"], arrays[f"{n}_M"], arrays[f"{n}_y"] = S, M, y
    np.savez(cache, **arrays)
    return out


def stratified_split(y, val_fraction=VAL_FRACTION, seed=SPLIT_SEED):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    val = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        val.append(idx[: int(round(val_fraction * idx.size))])
    val = np.sort(np.concatenate(val))
    train = np.setdiff1d(np.arange(y.size), val)
    return train, val


def _encode(y_raw, classes):
    lookup = {float(c): i for i, c in enumerate(classes)}
    try:
        return np.array([lookup[float(v)] for v in y_raw], dtype=np.int64)
    except KeyError as e:
        raise ValueError(f"label {e.args[0]} not in task classes {classes.tolist()}") from None


def _split(X, M, y, idx=None):
    if idx is None:
        return Split(X, M, y)
    return Split(X[idx], M[idx], y[idx])


def load_t1(data_dir="data", cache_dir="foundation/data/cache", ours=None) -> Task:
    ours = ours or load_ours(data_dir, cache_dir)
    Xr, Mr, yr_raw = ours["reference"]
    classes = np.unique(yr_raw)
    yr = _encode(yr_raw, classes)
    Xf, Mf, yf_raw = ours["finetune"]
    yf = _encode(yf_raw, classes)
    Xt, Mt, yt_raw = ours["test"]
    yt = _encode(yt_raw, classes)
    rtr, rva = stratified_split(yr)
    ftr, fva = stratified_split(yf)
    return Task("T1_isolates", classes, {
        "ref_train": _split(Xr, Mr, yr, rtr), "ref_val": _split(Xr, Mr, yr, rva),
        "ft_train": _split(Xf, Mf, yf, ftr), "ft_val": _split(Xf, Mf, yf, fva),
        "test": _split(Xt, Mt, yt),
    })


def load_t2(data_dir="data", cache_dir="foundation/data/cache", ours=None) -> Task:
    ours = ours or load_ours(data_dir, cache_dir)
    Xa, Ma, ya_raw = ours["2018clinical"]
    classes = np.unique(ya_raw)
    ya = _encode(ya_raw, classes)
    Xb, Mb, yb_raw = ours["2019clinical"]
    yb = _encode(yb_raw, classes)
    tr, va = stratified_split(ya)
    return Task("T2_clinical", classes, {
        "train": _split(Xa, Ma, ya, tr), "val": _split(Xa, Ma, ya, va),
        "test": _split(Xb, Mb, yb),
    })
