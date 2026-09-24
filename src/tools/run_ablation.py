"""
Preprocessing ablation: run the arms, then compare them properly.

Design notes, because the statistics are the whole point:

1. PAIRED, not independent. Each arm runs the same seeds, so seed effects
   cancel in the per-seed difference instead of being averaged over. With a
   measured stdev of 1.04 pts, independent means at n=3 resolve about 1.7 pts;
   the paired difference resolves closer to 1.1 pts for the same six runs.

2. PER-SAMPLE, where possible. The test set is fixed at 3000 spectra, and 0.79
   of the 1.04-point stdev is binomial sampling noise from that finite set.
   Comparing accuracies throws that away; comparing the same 3000 predictions
   sample by sample cancels it. McNemar's test on the discordant pairs is
   therefore much more powerful than a t-test on three accuracy numbers.
   Requires Edit 3 of ABLATION_PATCH.md.

3. Honest about n=3. A t-based interval on three differences uses t(2)=4.303
   and is very wide. It is reported anyway, because a wide interval that
   contains zero is a real result ("we cannot distinguish these") and should
   not be dressed up as a null finding.

Usage (from the repo root):
    python src/tools/run_ablation.py --dry-run
    python src/tools/run_ablation.py
    python src/tools/run_ablation.py --arms full,no_aspls,no_despike,none
    python src/tools/run_ablation.py --analyze-only
"""
from __future__ import annotations

import argparse
import glob
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

T_CRIT = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
          8: 2.306, 9: 2.262}  # two-sided 95%, df -> t


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #

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
            continue  # unfinished
        a = manifest.get("args", {})
        out.append({
            "dir": d,
            "arm": a.get("pipeline", "legacy_full"),
            "seed": a.get("seed"),
            "config": (a.get("model_type"), a.get("epochs"), a.get("finetune_epochs"),
                       bool(a.get("augment")), a.get("val_split"), a.get("batch_size"),
                       a.get("mixup_alpha")),
            "acc": 100.0 * test["accuracy"],
            "macro_f1": test.get("macro_f1"),
            "loss": test.get("loss"),
            "has_preds": os.path.exists(os.path.join(d, "test_y_pred.npy")),
        })
    return out


def config_tuple(a):
    return (a.model_type, a.epochs, a.finetune_epochs, bool(a.augment),
            a.val_split, a.batch_size, a.mixup_alpha)


# --------------------------------------------------------------------------- #
# execution
# --------------------------------------------------------------------------- #

def build_cmd(a, arm, seed):
    cmd = [sys.executable, "main.py",
           "--model-type", a.model_type,
           "--epochs", str(a.epochs),
           "--finetune-epochs", str(a.finetune_epochs),
           "--val-split", str(a.val_split),
           "--batch-size", str(a.batch_size),
           "--mixup-alpha", str(a.mixup_alpha),
           "--seed", str(seed),
           "--pipeline", arm,
           "--tag", f"abl-{arm}",
           "--device", a.device]
    if a.augment:
        cmd.append("--augment")
    return cmd


def run_one(cmd, log_path):
    print(f"\n$ {' '.join(cmd)}")
    t0 = time.time()
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=_ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", bufsize=1)
        for line in proc.stdout:
            log.write(line)
            # Only surface the interesting lines; a 100-epoch run prints ~800.
            s = line.rstrip()
            if any(k in s for k in ("set: Average loss", "Run id", "Early stopping",
                                    "Preprocessing variant", "worst-", "[cache]",
                                    "Traceback", "Error")):
                print("  " + s)
        rc = proc.wait()
    mins = (time.time() - t0) / 60
    print(f"  -> exit {rc} in {mins:.1f} min (log: {log_path})")
    return rc


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #

def mcnemar(correct_a, correct_b):
    """McNemar on paired per-sample correctness. Returns (b, c, chi2, p).

    b = A right, B wrong.  c = A wrong, B right.  Concordant pairs carry no
    information about the difference and are correctly ignored.
    Continuity-corrected chi-square, df 1.
    """
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n = b + c
    if n == 0:
        return b, c, 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / n
    try:
        from scipy.stats import chi2 as chi2_dist

        p = float(chi2_dist.sf(chi2, 1))
    except Exception:
        # Normal approximation fallback if scipy is unavailable.
        from math import erfc, sqrt

        p = float(erfc(sqrt(chi2 / 2)))
    return b, c, chi2, p


def paired_ci(diffs):
    n = len(diffs)
    if n < 2:
        return None
    mean = statistics.mean(diffs)
    sd = statistics.stdev(diffs)
    se = sd / (n ** 0.5)
    t = T_CRIT.get(n - 1, 1.96)
    return {"mean": mean, "sd": sd, "se": se, "t_crit": t,
            "lo": mean - t * se, "hi": mean + t * se,
            "t_stat": mean / se if se > 0 else float("inf")}


def analyse(runs, arms, baseline, config, seeds):
    sel = [r for r in runs if r["arm"] in arms and r["config"] == config
           and r["seed"] in seeds]
    by = {}
    for r in sel:
        by.setdefault((r["arm"], r["seed"]), []).append(r)
    # If an arm+seed was run more than once, keep the most recent.
    by = {k: sorted(v, key=lambda x: x["dir"])[-1] for k, v in by.items()}

    print("\n" + "=" * 78)
    print("PER-RUN RESULTS")
    print("=" * 78)
    print(f"{'arm':<14}{'seed':>5}{'acc':>9}{'macro_f1':>10}{'loss':>9}{'preds':>7}")
    print("-" * 54)
    for arm in arms:
        for s in seeds:
            r = by.get((arm, s))
            if r is None:
                print(f"{arm:<14}{s:>5}{'MISSING':>9}")
                continue
            print(f"{arm:<14}{s:>5}{r['acc']:>9.2f}{r['macro_f1']:>10.4f}"
                  f"{r['loss']:>9.4f}{('yes' if r['has_preds'] else 'no'):>7}")

    print("\n" + "=" * 78)
    print("ARM MEANS")
    print("=" * 78)
    for arm in arms:
        accs = [by[(arm, s)]["acc"] for s in seeds if (arm, s) in by]
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
        for s in seeds:
            a, b = by.get((baseline, s)), by.get((arm, s))
            if a is None or b is None:
                continue
            d = b["acc"] - a["acc"]
            diffs.append(d)
            pairs.append((s, a, b, d))
            print(f"  seed {s}: {a['acc']:.2f} -> {b['acc']:.2f}  ({d:+.2f} pts)")

        if len(diffs) < 2:
            print("  not enough paired seeds")
            continue

        ci = paired_ci(diffs)
        print(f"\n  mean difference: {ci['mean']:+.2f} pts")
        print(f"  95% CI (t, df={len(diffs)-1}): [{ci['lo']:+.2f}, {ci['hi']:+.2f}]")
        crosses_zero = ci["lo"] <= 0 <= ci["hi"]
        print(f"  interval {'INCLUDES' if crosses_zero else 'EXCLUDES'} zero")

        # Per-sample paired test, where predictions were saved.
        mc = []
        for s, a, b, _ in pairs:
            if not (a["has_preds"] and b["has_preds"]):
                continue
            ya = np.load(os.path.join(a["dir"], "test_y_pred.npy"))
            yb = np.load(os.path.join(b["dir"], "test_y_pred.npy"))
            ta = np.load(os.path.join(a["dir"], "test_y_true.npy"))
            tb = np.load(os.path.join(b["dir"], "test_y_true.npy"))
            if not np.array_equal(ta, tb):
                print(f"  seed {s}: test label order differs between runs, "
                      "cannot pair per-sample")
                continue
            ca, cb = (ya == ta), (yb == tb)
            bb, cc, chi2, p = mcnemar(ca, cb)
            mc.append({"seed": s, "b": bb, "c": cc, "chi2": chi2, "p": p})
            print(f"  seed {s} McNemar: baseline-only-right={bb}, "
                  f"{arm}-only-right={cc}, chi2={chi2:.2f}, p={p:.4g}")

        if mc:
            tb_, tc_ = sum(m["b"] for m in mc), sum(m["c"] for m in mc)
            n = tb_ + tc_
            chi2 = (abs(tb_ - tc_) - 1) ** 2 / n if n else 0.0
            try:
                from scipy.stats import chi2 as chi2_dist

                p = float(chi2_dist.sf(chi2, 1))
            except Exception:
                p = float("nan")
            print(f"\n  POOLED McNemar over {len(mc)} seeds: "
                  f"baseline-only-right={tb_}, {arm}-only-right={tc_}, "
                  f"chi2={chi2:.2f}, p={p:.4g}")
            print("  (pooling assumes the seeds are exchangeable, which is the "
                  "point of fixing everything but the seed)")
        else:
            print("\n  No saved predictions in both arms, so the per-sample test "
                  "was skipped.\n  Re-run without --reuse to get it; it is the "
                  "stronger test at this sample size.")

        print("\n  READ THIS AS:")
        if not crosses_zero and ci["mean"] > 0:
            print(f"  {arm} is better by {ci['mean']:.2f} pts. Preprocessing was")
            print("  costing accuracy. Revisit the capacity conclusion in the")
            print("  reference doc: the larger models may have been starved of")
            print("  signal rather than over-parameterised.")
        elif not crosses_zero and ci["mean"] < 0:
            print(f"  {arm} is worse by {abs(ci['mean']):.2f} pts. The step is")
            print("  earning its place. Preprocessing is exonerated; move to")
            print("  phase 2 as a mean-raiser.")
        else:
            print("  Cannot distinguish the arms at this sample size. That is a")
            print("  statement about power, not proof of no effect. If |mean| is")
            print("  above about 0.5 pts, more seeds would be worth it; if it is")
            print("  near zero, the step genuinely does not matter and you should")
            print("  drop it for the speed.")
        results[arm] = {"diffs": diffs, "ci": ci, "mcnemar": mc}
    return results


# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arms", default="full,no_aspls",
                   help="comma-separated pipeline variants; first is the baseline")
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--model-type", default="legacy_cnn")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--finetune-epochs", type=int, default=20)
    p.add_argument("--val-split", type=float, default=0.33)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--mixup-alpha", type=float, default=0.0)
    p.add_argument("--augment", action="store_true", default=True)
    p.add_argument("--no-augment", dest="augment", action="store_false")
    p.add_argument("--device", default="auto")
    p.add_argument("--run-root", default="runs")
    p.add_argument("--log-dir", default=os.path.join("results", "ablation"))
    p.add_argument("--reuse", action="store_true",
                   help="reuse matching completed runs. Saves time but loses the "
                        "per-sample McNemar test if those runs lack predictions.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--analyze-only", action="store_true")
    p.add_argument("--out", default=None, help="write results as JSON")
    a = p.parse_args()

    arms = [s.strip() for s in a.arms.split(",") if s.strip()]
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    baseline = arms[0]
    cfg = config_tuple(a)

    existing = load_runs(a.run_root)

    if not a.analyze_only:
        todo = []
        for arm, seed in itertools.product(arms, seeds):
            match = [r for r in existing
                     if r["arm"] == arm and r["seed"] == seed and r["config"] == cfg]
            if match and a.reuse:
                print(f"reuse  {arm:<12} seed {seed}  <- {os.path.basename(match[-1]['dir'])}"
                      f"{'' if match[-1]['has_preds'] else '  (no predictions)'}")
                continue
            todo.append((arm, seed))

        print(f"\n{len(todo)} run(s) to execute, {len(arms)*len(seeds)-len(todo)} reused.")
        for arm, seed in todo:
            cmd = build_cmd(a, arm, seed)
            if a.dry_run:
                print("  DRY  " + " ".join(cmd))
                continue
            log = os.path.join(a.log_dir, f"{a.model_type}_{arm}_seed{seed}.log")
            rc = run_one(cmd, log)
            if rc != 0:
                print(f"\nRun failed (exit {rc}). Stopping so a partial grid is not "
                      f"analysed as if it were complete. See {log}")
                return rc
        if a.dry_run:
            return 0
        existing = load_runs(a.run_root)

    res = analyse(existing, arms, baseline, cfg, seeds)

    if a.out and res:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=2, default=str)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
