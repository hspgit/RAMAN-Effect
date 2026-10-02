"""Build the pretraining corpus for model A (spec section 4).

    python -m foundation.data.build_corpus --config foundation/configs/corpus_v1.yaml

Pipeline: per source -> harmonise -> drop unusable -> leakage guard against
ALL our Bacteria-ID spectra -> per-sample cap; then corpus-wide exact
de-duplication -> probe/pretrain split by sample -> probe isolation (drop
pretrain near-duplicates of probe spectra) -> sqrt(n) sampling weights ->
write artefacts + manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .harmonize import (GRID_START, GRID_STEP, GRID_STOP, HARMONIZE_VERSION, N_GRID,
                        harmonize_many, is_usable)
from .leakage import LeakageGuard, spectrum_hash
from .sources import REGISTRY, ManualDownloadRequired
from .tasks import load_ours


def cap_per_sample(sample_ids, max_per_sample, rng):
    sample_ids = np.asarray(sample_ids, dtype=object)
    keep = np.ones(sample_ids.size, dtype=bool)
    for sid in np.unique(sample_ids):
        rows = np.flatnonzero(sample_ids == sid)
        if rows.size > max_per_sample:
            drop = rng.choice(rows, size=rows.size - max_per_sample, replace=False)
            keep[drop] = False
    return keep


def assign_splits(source, label, sample_id, probe_sources, probe_fraction,
                  probe_test_fraction, min_class_spectra, seed):
    """Probe slice per (source, class), split by sample group. Classes with
    fewer than 3 groups, or too few spectra, stay in pretrain."""
    split = np.full(len(source), "pretrain", dtype=object)
    rng = np.random.default_rng(seed)
    labelled = np.array([v is not None for v in label], dtype=bool)
    for src in sorted(probe_sources):
        in_src = (source == src) & labelled
        for cls in sorted(set(label[in_src])):
            rows = np.flatnonzero(in_src & (label == cls))
            groups = np.array(sorted(set(sample_id[rows])), dtype=object)
            if groups.size < 3:
                continue
            rng.shuffle(groups)
            n_probe = max(2, int(round(probe_fraction * groups.size)))
            probe_groups = groups[:n_probe]
            n_test = max(1, int(round(probe_test_fraction * n_probe)))
            test_g = list(probe_groups[:n_test])
            train_g = list(probe_groups[n_test:])
            te = rows[np.isin(sample_id[rows], test_g)]
            tr = rows[np.isin(sample_id[rows], train_g)]
            if tr.size + te.size < min_class_spectra:
                continue
            split[tr] = "probe_train"
            split[te] = "probe_test"
    return split


def sampling_weights(weight_group, split):
    """Pretrain sampling weights: each weight group (default: the source;
    related sources can share one) gets total weight proportional to sqrt(n)."""
    w = np.zeros(len(weight_group), dtype=np.float64)
    pre = split == "pretrain"
    for grp in np.unique(weight_group[pre]):
        rows = pre & (weight_group == grp)
        w[rows] = rows.sum() ** -0.5            # per-group total = sqrt(n)
    total = w.sum()
    return w / total if total > 0 else w


SIM_BINS = (("0.99", 0.99, 0.995), ("0.995", 0.995, 0.999), ("0.999", 0.999, np.inf))


def similarity_hist(scores):
    """Counts of candidates by their best similarity to any Bacteria-ID
    spectrum, in the near-miss bands below and at the leakage threshold."""
    scores = np.asarray(scores)
    return {k: int(((scores >= lo) & (scores < hi)).sum()) for k, lo, hi in SIM_BINS}


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def raw_fingerprint(directory):
    """sha256 over (relative path, size) of every downloaded file, so a manifest
    pins exactly which raw files built the corpus without hashing gigabytes."""
    directory = Path(directory)
    if not directory.exists():
        return None
    h = hashlib.sha256()
    for p in sorted(q for q in directory.rglob("*") if q.is_file()):
        h.update(f"{p.relative_to(directory).as_posix()}\t{p.stat().st_size}\n".encode())
    return h.hexdigest()


def _native_range(axes):
    if isinstance(axes, np.ndarray) and axes.ndim == 1:
        return float(np.nanmin(axes)), float(np.nanmax(axes))
    lo = min(float(np.nanmin(a)) for a in axes)
    hi = max(float(np.nanmax(a)) for a in axes)
    return lo, hi


def _new_stats(src):
    return {"status": "ok", "domain": src.domain, "licence": src.licence, "url": src.url,
            "label_kind": src.probe_label_kind, "n_raw": 0, "n_unusable": 0,
            "n_leaked": 0, "n_capped": 0, "native_min": None, "native_max": None,
            "lasers": set(), "max_similarity_hist": {k: 0 for k, _, _ in SIM_BINS}}


def _ingest_source(src, entry, raw, guard, rng, st, S_parts, M_parts, meta_parts):
    """Harmonise one source, apply the leakage guard and per-sample cap, and
    append its kept rows to the part lists."""
    st["raw_fingerprint"] = raw_fingerprint(raw)
    for batch in src.iter_batches(raw):
        S, M = harmonize_many(batch.axes, batch.spectra)
        lo, hi = _native_range(batch.axes)
        st["native_min"] = lo if st["native_min"] is None else min(lo, st["native_min"])
        st["native_max"] = hi if st["native_max"] is None else max(hi, st["native_max"])
        st["lasers"].update(float(v) for v in np.unique(batch.laser_nm) if np.isfinite(v))
        st["n_raw"] += len(S)
        keep = np.array([is_usable(s, m) for s, m in zip(S, M)], dtype=bool)
        st["n_unusable"] += int((~keep).sum())
        idx = np.flatnonzero(keep)
        max_cos, max_corr = guard.scores(S[idx], M[idx])
        leaked = guard.flag(S[idx], M[idx], max_cos, max_corr)
        for k, v in similarity_hist(np.maximum(max_cos, max_corr)).items():
            st["max_similarity_hist"][k] += v
        keep[idx[leaked]] = False
        st["n_leaked"] += int(leaked.sum())
        sample_id = np.asarray(batch.sample_id, dtype=object)
        if entry.get("max_per_sample"):
            idx = np.flatnonzero(keep)
            capped = cap_per_sample(sample_id[idx], entry["max_per_sample"], rng)
            keep[idx[~capped]] = False
            st["n_capped"] += int((~capped).sum())
        S_parts.append(S[keep].astype(np.float16))
        M_parts.append(M[keep])
        meta_parts.append(pd.DataFrame({
            "source": src.name, "domain": src.domain,
            "laser_nm": batch.laser_nm[keep], "sample_id": sample_id[keep],
            "label": np.asarray(batch.label, dtype=object)[keep],
            "weight_group": entry.get("weight_group", src.name),
        }))


def build(cfg: dict) -> dict:
    out_dir, raw_dir = Path(cfg["out_dir"]), Path(cfg["raw_dir"])
    rng = np.random.default_rng(cfg["seed"])
    device = cfg.get("device")

    ours = load_ours(cfg["our_data_dir"], cfg.get("cache_dir", "foundation/data/cache"))
    our_X = np.concatenate([v[0] for v in ours.values()])
    our_M = np.concatenate([v[1] for v in ours.values()])
    guard = LeakageGuard(our_X, our_M, cos_threshold=cfg["leakage_cos_threshold"],
                         corr_threshold=cfg.get("leakage_corr_threshold", 0.999),
                         device=device)

    S_parts, M_parts, meta_parts, stats = [], [], [], {}
    for entry in cfg["sources"]:
        src = REGISTRY[entry["name"]]
        stats[src.name] = st = _new_stats(src)
        n_parts = len(S_parts)
        try:
            src.download(raw_dir / src.name)
            _ingest_source(src, entry, raw_dir / src.name, guard, rng, st,
                           S_parts, M_parts, meta_parts)
        except ManualDownloadRequired as e:
            if not entry.get("optional"):
                raise
            st["status"] = f"skipped: {e}"
            print(f"[corpus] {src.name}: {st['status']}")
        except Exception as e:
            if not entry.get("optional"):
                raise
            # drop everything this source already contributed, so a half-read
            # source never reaches the corpus
            del S_parts[n_parts:], M_parts[n_parts:], meta_parts[n_parts:]
            stats[src.name] = st = _new_stats(src)
            st["status"] = f"failed: {type(e).__name__}: {e}"
            print(f"[corpus] {src.name}: {st['status']}")

    if not S_parts:
        raise RuntimeError("no source produced any spectra; nothing to build")
    S = np.concatenate(S_parts)
    M = np.concatenate(M_parts)
    meta = pd.concat(meta_parts, ignore_index=True)

    # corpus-wide exact de-duplication (keep first occurrence)
    hashes = np.array([spectrum_hash(s.astype(np.float32), m) for s, m in zip(S, M)])
    _, first = np.unique(hashes, return_index=True)
    unique = np.zeros(len(S), dtype=bool)
    unique[first] = True
    for name in stats:
        stats[name]["n_duplicate"] = int(((meta["source"] == name).to_numpy() & ~unique).sum())
    S, M, meta = S[unique], M[unique], meta[unique].reset_index(drop=True)

    source = meta["source"].to_numpy(dtype=object)
    label = meta["label"].to_numpy(dtype=object)
    sample_id = meta["sample_id"].to_numpy(dtype=object)
    probe_sources = {e["name"] for e in cfg["sources"] if e.get("probe")}
    split = assign_splits(source, label, sample_id, probe_sources, cfg["probe_fraction"],
                          cfg["probe_test_fraction"], cfg["min_probe_class_spectra"],
                          cfg["seed"])

    # probe isolation: no pretrain spectrum may near-duplicate a probe spectrum
    probe_rows = np.flatnonzero(split != "pretrain")
    pre_rows = np.flatnonzero(split == "pretrain")
    drop = np.zeros(len(S), dtype=bool)
    if probe_rows.size and pre_rows.size:
        pguard = LeakageGuard(S[probe_rows].astype(np.float32), M[probe_rows],
                              cos_threshold=cfg["leakage_cos_threshold"],
                              corr_threshold=cfg.get("leakage_corr_threshold", 0.999),
                              device=device)
        drop[pre_rows[pguard.find(S[pre_rows], M[pre_rows])]] = True
    for name in stats:
        stats[name]["n_probe_isolation"] = int(((source == name) & drop).sum())
    S, M, meta, split = S[~drop], M[~drop], meta[~drop].reset_index(drop=True), split[~drop]
    source = meta["source"].to_numpy(dtype=object)

    meta["split"] = split
    meta["sample_weight"] = sampling_weights(meta["weight_group"].to_numpy(dtype=object), split)
    for name, st in stats.items():
        rows = source == name
        st["lasers"] = sorted(st["lasers"])
        for s in ("pretrain", "probe_train", "probe_test"):
            st[f"n_{s}"] = int((rows & (split == s)).sum())

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "spectra.npy", S)
    np.save(out_dir / "mask.npy", M)
    meta.to_parquet(out_dir / "meta.parquet", index=False)
    splits = {s: np.flatnonzero(split == s).tolist()
              for s in ("pretrain", "probe_train", "probe_test")}
    (out_dir / "splits.json").write_text(json.dumps(splits))
    manifest = {
        "version": cfg["version"], "git_sha": _git_sha(),
        "harmonize_version": HARMONIZE_VERSION,
        "grid": {"start": GRID_START, "stop": GRID_STOP, "step": GRID_STEP, "n": N_GRID},
        "config": cfg, "sources": stats,
        "totals": {s: len(v) for s, v in splits.items()},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    manifest = build(cfg)
    print(json.dumps(manifest["totals"], indent=2))
    print(f"wrote {cfg['out_dir']}")


if __name__ == "__main__":
    main()
