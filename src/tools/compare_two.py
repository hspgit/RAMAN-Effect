"""
Paired comparison between two arbitrary groups of runs.

run_ablation.py and run_phase2.py both pair within a single config tuple, so
neither can answer "is 60 finetune epochs better than 20?" -- that comparison
crosses configs by construction. This does it: pick two groups by tag plus
optional filters, pair them on seed, and run the same statistics.

Usage (from the repo root):
    python src/tools/compare_two.py --a p2_base --a-ft-epochs 20 ^
                                    --b p2_val  --b-ft-epochs 60
    python src/tools/compare_two.py --a p2_base --b p2_val --a-ft-epochs 60 --b-ft-epochs 60
    python src/tools/compare_two.py --list

Caveat that applies to every cross-config comparison: the arms differ in more
than one way. 60 epochs also stretches the CosineAnnealingLR schedule
(T_max = finetune_epochs), so "60 vs 20" is really "60 epochs with a 60-epoch
cosine decay vs 20 with a 20-epoch decay", not just more of the same steps.
If 60 wins you will not know which of the two caused it without a third arm
holding the schedule fixed.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_run_ablation", os.path.join(_HERE, "run_ablation.py"))
_ra = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ra)
mcnemar, paired_ci = _ra.mcnemar, _ra.paired_ci


def load_runs(run_root):
    out = []
    for d in sorted(glob.glob(os.path.join(run_root, "*"))):
        mf, mt = os.path.join(d, "manifest.json"), os.path.join(d, "metrics.json")
        if not (os.path.isdir(d) and os.path.exists(mf) and os.path.exists(mt)):
            continue
        with open(mf) as fh:
            manifest = json.load(fh)
        with open(mt) as fh:
            metrics = json.load(fh)
        test = metrics.get("test")
        if not test or test.get("accuracy") is None:
            continue
        a = manifest.get("args", {})
        out.append({
            "dir": d,
            "tag": a.get("tag"),
            "seed": a.get("seed"),
            "model": a.get("model_type"),
            "pipeline": a.get("pipeline", "legacy_full"),
            "epochs": a.get("epochs"),
            "ft_epochs": a.get("finetune_epochs"),
            "ft_val": a.get("finetune_val_split", 0.0),
            "acc": 100.0 * test["accuracy"],
            "macro_f1": test.get("macro_f1"),
            "loss": test.get("loss"),
            "best_ft_epoch": (metrics.get("finetune_summary") or {}).get("best_epoch"),
            "has_preds": os.path.exists(os.path.join(d, "test_y_pred.npy")),
        })
    return out


def select(runs, tag, model, pipeline, epochs, ft_epochs):
    sel = [r for r in runs if r["tag"] == tag]
    if model:
        sel = [r for r in sel if r["model"] == model]
    if pipeline:
        sel = [r for r in sel if r["pipeline"] == pipeline]
    if epochs is not None:
        sel = [r for r in sel if r["epochs"] == epochs]
    if ft_epochs is not None:
        sel = [r for r in sel if r["ft_epochs"] == ft_epochs]
    by = {}
    for r in sel:
        by.setdefault(r["seed"], []).append(r)
    # most recent wins if a seed was run twice
    return {k: sorted(v, key=lambda x: x["dir"])[-1] for k, v in by.items()}


def describe(group, label):
    if not group:
        return f"{label}: NO RUNS MATCHED"
    r = next(iter(group.values()))
    accs = [g["acc"] for g in group.values()]
    sd = f"{statistics.stdev(accs):.2f}" if len(accs) >= 2 else "n/a"
    beps = [g["best_ft_epoch"] for g in group.values() if g["best_ft_epoch"]]
    extra = f"  best_ft_epochs={sorted(beps)}" if beps else ""
    return (f"{label}: tag={r['tag']} model={r['model']} pipeline={r['pipeline']} "
            f"epochs={r['epochs']} ft={r['ft_epochs']} ft_val={r['ft_val']}\n"
            f"       n={len(accs)} seeds={sorted(group)} "
            f"mean={statistics.mean(accs):.2f}% stdev={sd}{extra}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", help="tag of the baseline group")
    p.add_argument("--b", help="tag of the comparison group")
    p.add_argument("--a-ft-epochs", type=int, default=None)
    p.add_argument("--b-ft-epochs", type=int, default=None)
    p.add_argument("--a-epochs", type=int, default=None)
    p.add_argument("--b-epochs", type=int, default=None)
    p.add_argument("--model", default="legacy_cnn")
    p.add_argument("--pipeline", default="minmax_only")
    p.add_argument("--run-root", default="runs")
    p.add_argument("--list", action="store_true",
                   help="show available (tag, model, epochs, ft_epochs) groups")
    args = p.parse_args()

    runs = load_runs(args.run_root)
    if not runs:
        print(f"no finished runs under {args.run_root}")
        return 1

    if args.list or not (args.a and args.b):
        groups = {}
        for r in runs:
            k = (r["tag"], r["model"], r["pipeline"], r["epochs"], r["ft_epochs"])
            groups.setdefault(k, []).append(r["seed"])
        print(f"{'tag':<18}{'model':<16}{'pipeline':<14}{'ep':>5}{'ft':>5}  seeds")
        print("-" * 72)
        for k in sorted(groups, key=lambda x: [str(v) for v in x]):
            print(f"{str(k[0]):<18}{str(k[1]):<16}{str(k[2]):<14}"
                  f"{str(k[3]):>5}{str(k[4]):>5}  {sorted(groups[k])}")
        if not (args.a and args.b):
            print("\npass --a and --b to compare two of these")
        return 0

    A = select(runs, args.a, args.model, args.pipeline, args.a_epochs, args.a_ft_epochs)
    B = select(runs, args.b, args.model, args.pipeline, args.b_epochs, args.b_ft_epochs)
    print(describe(A, "A"))
    print(describe(B, "B"))
    if not A or not B:
        print("\nNothing to compare. Run with --list to see what exists.")
        return 1

    shared = sorted(set(A) & set(B))
    if len(shared) < 2:
        print(f"\nOnly {len(shared)} shared seed(s); need at least 2 to pair.")
        return 1

    print("\n" + "=" * 70)
    print(f"PAIRED: B ({args.b}) vs A ({args.a})")
    print("=" * 70)
    diffs = []
    for s in shared:
        d = B[s]["acc"] - A[s]["acc"]
        diffs.append(d)
        print(f"  seed {s}: {A[s]['acc']:.2f} -> {B[s]['acc']:.2f}  ({d:+.2f} pts)")

    ci = paired_ci(diffs)
    print(f"\n  mean difference: {ci['mean']:+.2f} pts")
    print(f"  95% CI (t, df={len(diffs)-1}): [{ci['lo']:+.2f}, {ci['hi']:+.2f}]")
    crosses = ci["lo"] <= 0 <= ci["hi"]
    print(f"  interval {'INCLUDES' if crosses else 'EXCLUDES'} zero")

    tb = tc = 0
    for s in shared:
        if not (A[s]["has_preds"] and B[s]["has_preds"]):
            continue
        pa = np.load(os.path.join(A[s]["dir"], "test_y_pred.npy"))
        pb = np.load(os.path.join(B[s]["dir"], "test_y_pred.npy"))
        ta = np.load(os.path.join(A[s]["dir"], "test_y_true.npy"))
        if not np.array_equal(ta, np.load(os.path.join(B[s]["dir"], "test_y_true.npy"))):
            print(f"  seed {s}: test order differs, skipping per-sample test")
            continue
        b, c, chi2, pv = mcnemar(pa == ta, pb == ta)
        tb += b
        tc += c
        print(f"  seed {s} McNemar: A-only-right={b}, B-only-right={c}, "
              f"chi2={chi2:.2f}, p={pv:.4g}")
    if tb + tc:
        chi2 = (abs(tb - tc) - 1) ** 2 / (tb + tc)
        try:
            from scipy.stats import chi2 as chi2_dist
            pv = float(chi2_dist.sf(chi2, 1))
        except Exception:
            pv = float("nan")
        print(f"\n  POOLED McNemar: A-only-right={tb}, B-only-right={tc}, "
              f"chi2={chi2:.2f}, p={pv:.4g}")
        print("  Where the t-interval and McNemar disagree, trust McNemar: the")
        print("  t-interval at df=2 estimates its own spread from three numbers")
        print("  and is narrow whenever those three happen to land close.")
    else:
        print("\n  No saved predictions in both groups; per-sample test skipped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
