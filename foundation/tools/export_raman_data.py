"""Export whitelisted raman_data datasets to plain .npz (plan deviation D5).

Runs in its OWN venv so raman_data's dependencies never touch the main env:
    python -m venv foundation/.venv-raman-data
    foundation/.venv-raman-data/Scripts/python.exe -m pip install "git+https://github.com/ml-lab-htw/raman_data@c92289e" pyyaml
    foundation/.venv-raman-data/Scripts/python.exe foundation/tools/export_raman_data.py
Writes foundation/data/raw/raman_data/<id>.npz and export_report.json.
Standalone on purpose: imports nothing from the foundation package.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml


def to_labels(ds):
    task = str(getattr(ds.info, "task_type", ""))
    targets = getattr(ds, "targets", None)
    if targets is None or "Classification" not in task:
        return None
    t = np.asarray(targets)
    names = getattr(ds, "target_names", None)
    if names is not None and len(names) and np.issubdtype(t.dtype, np.integer):
        return np.array([str(names[int(v)]) for v in t])
    return np.array([str(v) for v in t])


def export_one(ds_id, ds, out):
    """Export a single dataset, return report entry dict. Testable without raman_data."""
    labels = to_labels(ds)
    groups = getattr(ds, "group_ids", None)

    # Build complete report entry BEFORE writing anything
    entry = {
        "status": "ok",
        "n": int(len(ds.spectra)),
        "licence": str(getattr(ds.info, "license", "")),
        "description": ds.info.metadata.get("description", "")
    }

    # Write to temporary file atomically (pass file handle to avoid .npz suffix)
    out = Path(out)
    tmp_path = out / f"{ds_id}.npz.tmp"
    final_path = out / f"{ds_id}.npz"

    with open(tmp_path, "wb") as tmp_file:
        np.savez_compressed(
            tmp_file,
            spectra=np.asarray(ds.spectra, dtype=np.float32),
            axis=np.asarray(ds.raman_shifts, dtype=np.float64),
            labels=labels if labels is not None else np.array([], dtype=str),
            group_ids=(np.asarray(groups).astype(str) if groups is not None
                       else np.array([], dtype=str)))

    os.replace(str(tmp_path), str(final_path))
    return entry


def discard(ds_id, out):
    """Delete stale .npz and .npz.tmp files after failed export."""
    out = Path(out)
    for suffix in [".npz", ".npz.tmp"]:
        try:
            (out / f"{ds_id}{suffix}").unlink()
        except FileNotFoundError:
            pass


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")   # Windows consoles choke on odd dataset names
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--whitelist", default="foundation/configs/raman_data_whitelist.yaml")
    ap.add_argument("--out", default="foundation/data/raw/raman_data")
    ap.add_argument("--cache-dir", default="foundation/data/raw/raman_data/_cache")
    args = ap.parse_args(argv)
    from raman_data import raman_data

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}

    for ds_id in yaml.safe_load(Path(args.whitelist).read_text()):
        try:
            ds = raman_data(ds_id, cache_dir=args.cache_dir)
            entry = export_one(ds_id, ds, out)
            report[ds_id] = entry
        except Exception as e:  # keep going; the report says what failed
            report[ds_id] = {"status": f"failed: {type(e).__name__}: {e}"}
            discard(ds_id, out)

        print(f"{ds_id}: {report[ds_id]['status']}")
        # Write report after every dataset for consistency
        (out / "export_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
