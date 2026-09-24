"""
Phase 2 experiment: does instrumenting fine-tuning help?

Four arms, incremental, so each isolates one change:

    p2_base        current behaviour: 20 epochs, no validation, last epoch saved
    p2_val         + stratified 10% validation split, best-val-loss weights kept
    p2_valaug      + the three augmentations during fine-tuning
    p2_valaugwd    + weight decay 0.01 on the finetune optimizer

Each arm is compared against p2_base, paired on seed, with McNemar on the
per-sample predictions. Statistics helpers are imported from run_ablation.py
rather than reimplemented, since they are already tested.

Note on the incremental design: it tells you the cumulative effect of each
addition, not the marginal effect of each in isolation. If p2_valaugwd wins,
you know the stack works; you do not know whether weight decay alone
contributed. That is the right trade at three seeds, because a full factorial
would be eight arms and 24 runs to resolve effects likely under a point.

Usage (from the repo root):
    python src/tools/run_phase2.py --dry-run
    python src/tools/run_phase2.py
    python src/tools/run_phase2.py --analyze-only
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import itertools
import json
import os
import statistics
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

# Reuse the tested statistics from the ablation runner.
_spec = importlib.util.spec_from_file_location(
    "_run_ablation", os.path.join(_HERE, "run_ablation.py"))
_ra = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ra)
mcnemar, paired_ci, run_one = _ra.mcnemar, _ra.paired_ci, _ra.run_one

ARMS = {
    "p2_base": [],
    "p2_val": ["--finetune-val-split", "0.1"],
    "p2_valaug": ["--finetune-val-split", "0.1", "--finetune-augment"],
    "p2_valaugwd": ["--finetune-val-split", "0.1", "--finetune-augment",
                    "--finetune-weight-decay", "0.01"],
}


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
            "acc": 100.0 * test["accuracy"],
            "macro_f1": test.get("macro_f1"),
            "loss": test.get("loss"),
            "best_ft_epoch": (metrics.get("finetune_summary") or {}).get("best_epoch"),
            "has_preds": os.path.exists(os.path.join(d, "test_y_pred.npy")),
        })
    return out


def build_cmd(a, arm, seed):
    cmd = [sys.executable, "main.py",
           "--model-type", a.model_type,
           "--epochs", str(a.epochs),
           "--finetune-epochs", str(a.finetune_epochs),
           "--val-split", str(a.val_split),
           "--batch-size", str(a.batch_size),
           "--pipeline", a.pipeline,
           "--seed", str(seed),
           "--tag", arm,
           "--device", a.device]
    if a.augment:
        cmd.append("--augment")
    if a.diagnostic:
        cmd.append("--diagnostic-test-curve")
    return cmd + ARMS[arm]


def analyse(runs, arms, baseline, a):
    sel = [r for r in runs
           if r["tag"] in arms and r["model"] == a.model_type
           and r["pipeline"] == a.pipeline and r["epochs"] == a.epochs
           and r["ft_epochs"] == a.finetune_epochs and r["seed"] in a.seed_list]
    by = {}
    for r in sel:
        by.setdefault((r["tag"], r["seed"]), []).append(r)
    by = {k: sorted(v, key=lambda x: x["dir"])[-1] for k, v in by.items()}

    print("\n" + "=" * 78)
    print(f"PER-RUN RESULTS  ({a.model_type}, pipeline={a.pipeline})")
    print("=" * 78)
    print(f"{'arm':<14}{'seed':>5}{'acc':>9}{'macro_f1':>10}{'loss':>9}{'best_ep':>9}")
    print("-" * 56)
    for arm in arms:
        for s in a.seed_list:
            r = by.get((arm, s))
            if r is None:
                print(f"{arm:<14}{s:>5}{'MISSING':>9}")
                continue
            print(f"{arm:<14}{s:>5}{r['acc']:>9.2f}{r['macro_f1']:>10.4f}"
                  f"{r['loss']:>9.4f}{str(r['best_ft_epoch']):>9}")

    print("\n" + "=" * 78)
    print("ARM MEANS")
    print("=" * 78)
    for arm in arms:
        accs = [by[(arm, s)]["acc"] for s in a.seed_list if (arm, s) in by]
        if not accs:
            continue
        sd = f"{statistics.stdev(accs):.2f}" if len(accs) >= 2 else "n/a"
        print(f"{arm:<14} n={len(accs)}  mean={statistics.mean(accs):.2f}%  stdev={sd}")

    results = {}
    for arm in arms:
        if arm == baseline:
            continue
        print("\n" + "=" * 78)
        print(f"PAIRED COMPARISON: {arm} vs {baseline}")
        print("=" * 78)
        diffs, pairs = [], []
        for s in a.seed_list:
            x, y = by.get((baseline, s)), by.get((arm, s))
            if x is None or y is None:
                continue
            d = y["acc"] - x["acc"]
            diffs.append(d)
            pairs.append((s, x, y))
            print(f"  seed {s}: {x['acc']:.2f} -> {y['acc']:.2f}  ({d:+.2f} pts)")
        if len(diffs) < 2:
            print("  not enough paired seeds")
            continue

        ci = paired_ci(diffs)
        print(f"\n  mean difference: {ci['mean']:+.2f} pts")
        print(f"  95% CI (t, df={len(diffs)-1}): [{ci['lo']:+.2f}, {ci['hi']:+.2f}]")
        crosses = ci["lo"] <= 0 <= ci["hi"]
        print(f"  interval {'INCLUDES' if crosses else 'EXCLUDES'} zero")

        tb = tc = 0
        for s, x, y in pairs:
            if not (x["has_preds"] and y["has_preds"]):
                continue
            pa = np.load(os.path.join(x["dir"], "test_y_pred.npy"))
            pb = np.load(os.path.join(y["dir"], "test_y_pred.npy"))
            ta = np.load(os.path.join(x["dir"], "test_y_true.npy"))
            if not np.array_equal(ta, np.load(os.path.join(y["dir"], "test_y_true.npy"))):
                print(f"  seed {s}: test order differs, skipping per-sample test")
                continue
            b, c, chi2, p = mcnemar(pa == ta, pb == ta)
            tb += b
            tc += c
            print(f"  seed {s} McNemar: base-only-right={b}, {arm}-only-right={c}, "
                  f"chi2={chi2:.2f}, p={p:.4g}")
        if tb + tc:
            chi2 = (abs(tb - tc) - 1) ** 2 / (tb + tc)
            try:
                from scipy.stats import chi2 as chi2_dist
                p = float(chi2_dist.sf(chi2, 1))
            except Exception:
                p = float("nan")
            print(f"\n  POOLED McNemar: base-only-right={tb}, {arm}-only-right={tc}, "
                  f"chi2={chi2:.2f}, p={p:.4g}")

        results[arm] = {"diffs": diffs, "ci": ci, "mcnemar_b": tb, "mcnemar_c": tc}

    # Stability, which matters as much as the mean for a retraining pipeline.
    print("\n" + "=" * 78)
    print("STABILITY")
    print("=" * 78)
    for arm in arms:
        accs = [by[(arm, s)]["acc"] for s in a.seed_list if (arm, s) in by]
        if len(accs) >= 3:
            print(f"{arm:<14} stdev={statistics.stdev(accs):.2f} pts")
    print("An arm that matches the baseline mean but halves the stdev is still a")
    print("win for a pipeline that has to be retrained on new data.")

    # Where the finetune curve actually peaks.
    eps = [by[(arm, s)]["best_ft_epoch"] for arm in arms for s in a.seed_list
           if (arm, s) in by and by[(arm, s)]["best_ft_epoch"]]
    if eps:
        print(f"\nBest finetune epoch across arms: {sorted(eps)}")
        print(f"(of {a.finetune_epochs} run). Clustering well below the maximum means")
        print("the old last-epoch save was past the peak.")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arms", default=",".join(ARMS),
                   help="comma-separated; first is the baseline")
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--model-type", default="legacy_cnn")
    p.add_argument("--pipeline", default="minmax_only")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--finetune-epochs", type=int, default=20)
    p.add_argument("--val-split", type=float, default=0.33)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--augment", action="store_true", default=True)
    p.add_argument("--no-augment", dest="augment", action="store_false")
    p.add_argument("--diagnostic", action="store_true", default=True,
                   help="record the per-epoch test curve (diagnosis only)")
    p.add_argument("--no-diagnostic", dest="diagnostic", action="store_false")
    p.add_argument("--device", default="auto")
    p.add_argument("--run-root", default="runs")
    p.add_argument("--log-dir", default=os.path.join("results", "phase2"))
    p.add_argument("--reuse", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--analyze-only", action="store_true")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    arms = [s.strip() for s in a.arms.split(",") if s.strip()]
    unknown = [x for x in arms if x not in ARMS]
    if unknown:
        print(f"unknown arm(s) {unknown}; choices: {list(ARMS)}")
        return 2
    a.seed_list = [int(s) for s in a.seeds.split(",") if s.strip()]
    baseline = arms[0]

    existing = load_runs(a.run_root)

    if not a.analyze_only:
        todo = []
        for arm, seed in itertools.product(arms, a.seed_list):
            match = [r for r in existing
                     if r["tag"] == arm and r["seed"] == seed
                     and r["model"] == a.model_type and r["pipeline"] == a.pipeline
                     and r["epochs"] == a.epochs
                     and r["ft_epochs"] == a.finetune_epochs]
            if match and a.reuse:
                print(f"reuse  {arm:<14} seed {seed}  <- {os.path.basename(match[-1]['dir'])}")
                continue
            todo.append((arm, seed))
        print(f"\n{len(todo)} run(s) to execute, "
              f"{len(arms)*len(a.seed_list)-len(todo)} reused.")
        for arm, seed in todo:
            cmd = build_cmd(a, arm, seed)
            if a.dry_run:
                print("  DRY  " + " ".join(cmd))
                continue
            log = os.path.join(a.log_dir, f"{a.model_type}_{arm}_seed{seed}.log")
            rc = run_one(cmd, log)
            if rc != 0:
                print(f"\nRun failed (exit {rc}). Stopping. See {log}")
                return rc
        if a.dry_run:
            return 0
        existing = load_runs(a.run_root)

    res = analyse(existing, arms, baseline, a)
    if a.out and res:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=2, default=str)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
