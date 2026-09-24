"""
Content-addressed cache for preprocessed spectra.

Why: RamanDataset runs the full ramanspy pipeline eagerly in __init__ over all
60,000 reference spectra at every script start, and again for the finetune and
test sets. No caching. That cost dominates iteration time.

The footgun avoided: a cache keyed on the input file alone goes stale the moment
a preprocessing parameter changes, after which you silently train on arrays
produced by code you no longer have. So the key includes a declarative
description of the pipeline, and dataset.py builds the real ramanspy pipeline
from that same description. Description and code cannot diverge.

Side benefit for Stage 1: an ablation that drops ASPLS gets a different key
automatically and cannot collide with the baseline.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Callable, Dict

import numpy as np


def file_fingerprint(path: str, full_hash: bool = False) -> Dict[str, Any]:
    """Identify an input file.

    full_hash=False (default) uses size + mtime_ns, which is instant.
    full_hash=True reads the file (about a second for the 480 MB reference
    array) and is what you want if the .npy files might be replaced in place
    with same-size content, or if you move a cache between machines where
    mtime is not preserved.
    """
    st = os.stat(path)
    fp: Dict[str, Any] = {"name": os.path.basename(path), "size": st.st_size}
    if full_hash:
        h = hashlib.blake2b(digest_size=16)
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        fp["blake2b"] = h.hexdigest()
    else:
        fp["mtime_ns"] = st.st_mtime_ns
    return fp


def cache_key(parts: Dict[str, Any]) -> str:
    blob = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.blake2b(blob, digest_size=16).hexdigest()


def load_or_compute(
    cache_dir: str,
    key_parts: Dict[str, Any],
    compute: Callable[[], np.ndarray],
    verbose: bool = True,
) -> np.ndarray:
    """Return the cached array for key_parts, computing and storing it if absent.

    Writes <key>.npy plus <key>.json so a confusing cache entry can be
    inspected rather than guessed at. Writes to .tmp then renames, so an
    interrupted run cannot leave a truncated .npy that later loads as garbage.
    """
    os.makedirs(cache_dir, exist_ok=True)
    key = cache_key(key_parts)
    arr_path = os.path.join(cache_dir, f"{key}.npy")
    meta_path = os.path.join(cache_dir, f"{key}.json")

    if os.path.exists(arr_path):
        if verbose:
            print(f"[cache] hit {key[:8]} ({os.path.basename(arr_path)})")
        return np.load(arr_path)

    if verbose:
        print(f"[cache] miss {key[:8]}, running preprocessing (slow path)")
    arr = compute()
    tmp = arr_path + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, arr_path)
    with open(meta_path, "w") as fh:
        json.dump(
            {
                "key": key,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "key_parts": key_parts,
            },
            fh,
            indent=2,
            sort_keys=True,
            default=str,
        )
    if verbose:
        print(f"[cache] stored {key[:8]} shape={arr.shape}")
    return arr


def purge(cache_dir: str, dry_run: bool = True) -> int:
    """Delete every cache entry. Run this whenever you are unsure."""
    if not os.path.isdir(cache_dir):
        print(f"[cache] no such dir {cache_dir}")
        return 0
    n = 0
    for f in os.listdir(cache_dir):
        if f.endswith((".npy", ".json")):
            n += 1
            if not dry_run:
                os.remove(os.path.join(cache_dir, f))
    print(f"[cache] {'would remove' if dry_run else 'removed'} {n} files in {cache_dir}")
    return n


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Inspect or purge the preprocessing cache")
    p.add_argument("--cache-dir", default=os.path.join("data", ".cache"))
    p.add_argument("--purge", action="store_true", help="actually delete (default is dry run)")
    a = p.parse_args()
    purge(a.cache_dir, dry_run=not a.purge)
