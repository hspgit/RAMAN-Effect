"""
Compare finished runs. This is how you verify Stage 0 and how you read a
seed-noise floor.

Usage:
    python tools/compare_runs.py runs/*det-a* runs/*det-b*      # determinism
    python tools/compare_runs.py --tag noise                    # noise floor
    python tools/compare_runs.py --all
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys


def load(run_dir):
    mpath = os.path.join(run_dir, "metrics.json")
    apath = os.path.join(run_dir, "manifest.json")
    if not os.path.exists(mpath):
        return None
    with open(mpath) as fh:
        metrics = json.load(fh)
    manifest = {}
    if os.path.exists(apath):
        with open(apath) as fh:
            manifest = json.load(fh)
    return {"dir": run_dir, "metrics": metrics, "manifest": manifest}


def row(r):
    t = r["metrics"].get("test") or {}
    a = r["manifest"].get("args", {})
    e = r["manifest"].get("env", {})
    return {
        "run": os.path.basename(r["dir"]),
        "model": a.get("model_type"),
        "seed": a.get("seed"),
        "device": e.get("device"),
        "epochs": a.get("epochs"),
        "ft_epochs": a.get("finetune_epochs"),
        "augment": a.get("augment"),
        "acc": t.get("accuracy"),
        "macro_f1": t.get("macro_f1"),
        "loss": t.get("loss"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="*", help="run directories (globs are fine)")
    p.add_argument("--run-root", default="runs")
    p.add_argument("--tag", default=None, help="select runs whose id contains this tag")
    p.add_argument("--all", action="store_true")
    a = p.parse_args()

    dirs = []
    for d in a.dirs:
        dirs.extend(sorted(glob.glob(d)))
    if a.tag:
        dirs.extend(sorted(glob.glob(os.path.join(a.run_root, f"*{a.tag}*"))))
    if a.all or not dirs:
        dirs.extend(sorted(glob.glob(os.path.join(a.run_root, "*"))))
    dirs = [d for d in dict.fromkeys(dirs) if os.path.isdir(d)]

    runs = [r for r in (load(d) for d in dirs) if r]
    if not runs:
        print("no finished runs found")
        return 1

    rows = [row(r) for r in runs]
    hdr = ["run", "model", "seed", "device", "epochs", "ft_epochs", "augment",
           "acc", "macro_f1", "loss"]
    widths = {h: max(len(h), *(len(str(x[h])) for x in rows)) for h in hdr}
    print("  ".join(h.ljust(widths[h]) for h in hdr))
    print("  ".join("-" * widths[h] for h in hdr))
    for x in rows:
        print("  ".join(str(x[h]).ljust(widths[h]) for h in hdr))

    accs = [x["acc"] for x in rows if isinstance(x["acc"], (int, float))]
    if len(accs) >= 2:
        spread = (max(accs) - min(accs)) * 100
        print(f"\nn={len(accs)}  mean={100*statistics.mean(accs):.2f}%  "
              f"spread={spread:.2f} pts", end="")
        if len(accs) >= 3:
            print(f"  stdev={100*statistics.stdev(accs):.2f} pts")
        else:
            print()
        if spread < 0.01:
            print("Identical to 2 decimal places: determinism check PASSES.")
        else:
            print(f"Any future improvement smaller than about {spread:.2f} points "
                  "is inside this noise band.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
