"""
Which phase causes the seed variance?

You measured 2.07 points of spread across three identical-config runs. That
number is the final, post-finetune figure, so it cannot say whether the
instability originates in pre-training or in fine-tuning. This script
distinguishes them without training anything: every run already saved both
checkpoints, so both can be evaluated on the test set.

    reference.pth  -> test accuracy with NO fine-tuning
    finetuned.pth  -> test accuracy after fine-tuning (should match the
                      number already in metrics.json, which is a free
                      correctness check on this script)

Read the result like this:

  stdev(reference) large, stdev(finetuned) similar
      -> phase 1 dominates. Fine-tuning inherits the variance rather than
         creating it. Fixing phase 2 selection will NOT shrink the floor much;
         fix pre-training stability instead (init, LR schedule, longer
         warmup, averaging weights over the last epochs).

  stdev(reference) small, stdev(finetuned) large
      -> phase 2 dominates. Twenty unregularised epochs on 100 samples per
         class with no validation is the random walk. Best-epoch selection
         and regularisation there should shrink the floor.

  both large
      -> two independent problems; fix phase 1 first, since phase 2 inherits it.

Usage (from the repo root):
    python src/tools/check_variance_source.py --tag noise100
    python src/tools/check_variance_source.py --tag noise100 --device cuda
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys

# Running a script inside src/tools/ puts that directory on sys.path, not the
# repo root, so 'from src.dataset import ...' would fail. Fix it explicitly.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.dataset import RamanDataset  # noqa: E402
from src.engine import evaluate  # noqa: E402
from src.repro import pick_device  # noqa: E402


def build_model(model_type, num_classes, device):
    if model_type == "resnet":
        from src.model import Raman1DCNN

        return Raman1DCNN(num_classes=num_classes).to(device)
    if model_type == "legacy_cnn":
        from legacy_baseline.model import LegacyCNN

        return LegacyCNN(num_classes=num_classes).to(device)
    if model_type == "transformer":
        from src.model import Transformer1D

        return Transformer1D(num_classes=num_classes).to(device)
    if model_type == "multiscale_cnn":
        from src.model import MultiscaleCNN

        return MultiscaleCNN(num_classes=num_classes).to(device)
    if model_type == "fusion":
        from src.model import FusionNet

        return FusionNet(num_classes=num_classes).to(device)
    raise ValueError(f"unknown model_type {model_type!r}")


def reference_label_mapping(data_dir):
    """Rebuild the label mapping without loading or preprocessing X_reference.

    RamanDataset builds the mapping from sorted(unique(y)). Replicating that
    from y_reference.npy alone avoids loading a 480 MB array (and its 60k-row
    cached counterpart) just to obtain 30 dictionary keys.
    """
    y = np.load(os.path.join(data_dir, "y_reference.npy"))
    return {label: idx for idx, label in enumerate(sorted(np.unique(y)))}


def spread_stats(values):
    if len(values) < 2:
        return {"n": len(values), "mean": values[0] if values else None,
                "stdev": None, "spread": None}
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "stdev": statistics.stdev(values) if len(values) >= 3 else None,
        "spread": max(values) - min(values),
    }


def fmt(stats, label):
    if stats["n"] < 2:
        return f"{label}: n={stats['n']}, cannot measure spread"
    sd = f"{stats['stdev']:.2f}" if stats["stdev"] is not None else "n/a"
    return (f"{label}: n={stats['n']}  mean={stats['mean']:.2f}%  "
            f"spread={stats['spread']:.2f} pts  stdev={sd} pts")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", default="runs")
    p.add_argument("--tag", default="noise100",
                   help="only runs whose id contains this string")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="auto",
                   choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--out", default=None,
                   help="write the full result as JSON here")
    a = p.parse_args()

    device = pick_device(a.device)
    print(f"Using device: {device}")

    run_dirs = sorted(d for d in glob.glob(os.path.join(a.run_root, f"*{a.tag}*"))
                      if os.path.isdir(d))
    if not run_dirs:
        print(f"no runs matching *{a.tag}* under {a.run_root}")
        return 1

    # Load the test set once. Uses the cache, so this is fast.
    mapping = reference_label_mapping(a.data_dir)
    num_classes = len(mapping)
    test_ds = RamanDataset(os.path.join(a.data_dir, "X_test.npy"),
                           os.path.join(a.data_dir, "y_test.npy"),
                           label_mapping=mapping)
    test_loader = DataLoader(test_ds, batch_size=a.batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    print(f"Test set: {len(test_ds)} samples, {num_classes} classes\n")

    rows = []
    for d in run_dirs:
        mpath = os.path.join(d, "manifest.json")
        if not os.path.exists(mpath):
            print(f"skip {os.path.basename(d)}: no manifest")
            continue
        with open(mpath) as fh:
            manifest = json.load(fh)
        args_rec = manifest.get("args", {})
        model_type = args_rec.get("model_type", "legacy_cnn")
        seed = args_rec.get("seed")

        recorded = None
        mj = os.path.join(d, "metrics.json")
        if os.path.exists(mj):
            with open(mj) as fh:
                recorded = (json.load(fh).get("test") or {}).get("accuracy")

        row = {"run": os.path.basename(d), "seed": seed, "model": model_type,
               "recorded_ft_acc": None if recorded is None else 100 * recorded}

        for stage, fname in (("reference", "reference.pth"),
                             ("finetuned", "finetuned.pth")):
            ckpt = os.path.join(d, fname)
            if not os.path.exists(ckpt):
                print(f"  {os.path.basename(d)} [{stage}]: checkpoint missing")
                continue
            model = build_model(model_type, num_classes, device)
            model.load_state_dict(torch.load(ckpt, map_location=device,
                                             weights_only=True))
            print(f"=== {os.path.basename(d)} seed={seed} [{stage}] ===")
            _, acc, stats = evaluate(model, device, test_loader, criterion,
                                     phase=f"{stage}", num_classes=num_classes,
                                     return_stats=True)
            row[f"{stage}_acc"] = acc
            row[f"{stage}_loss"] = stats["loss"]
            row[f"{stage}_macro_f1"] = stats["macro_f1"]
            row[f"{stage}_recall"] = stats["per_class_recall"]

        if "reference_acc" in row and "finetuned_acc" in row:
            row["finetune_gain"] = row["finetuned_acc"] - row["reference_acc"]
        rows.append(row)

    if not rows:
        print("nothing evaluated")
        return 1

    # --- table ---
    print("\n" + "=" * 78)
    print("PER-RUN RESULTS")
    print("=" * 78)
    hdr = f"{'seed':>4}  {'ref acc':>8}  {'ft acc':>8}  {'gain':>7}  {'recorded':>9}  {'match':>5}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        ref = r.get("reference_acc")
        ft = r.get("finetuned_acc")
        rec = r.get("recorded_ft_acc")
        match = ""
        if ft is not None and rec is not None:
            match = "ok" if abs(ft - rec) < 0.01 else "DIFF"
        print(f"{str(r['seed']):>4}  "
              f"{('%.2f' % ref) if ref is not None else '-':>8}  "
              f"{('%.2f' % ft) if ft is not None else '-':>8}  "
              f"{('%+.2f' % r['finetune_gain']) if 'finetune_gain' in r else '-':>7}  "
              f"{('%.2f' % rec) if rec is not None else '-':>9}  "
              f"{match:>5}")

    ref_accs = [r["reference_acc"] for r in rows if "reference_acc" in r]
    ft_accs = [r["finetuned_acc"] for r in rows if "finetuned_acc" in r]
    gains = [r["finetune_gain"] for r in rows if "finetune_gain" in r]

    print("\n" + "=" * 78)
    print("VARIANCE BY STAGE")
    print("=" * 78)
    ref_s, ft_s, gain_s = spread_stats(ref_accs), spread_stats(ft_accs), spread_stats(gains)
    print(fmt(ref_s, "reference checkpoint (phase 1 only)"))
    print(fmt(ft_s, "finetuned checkpoint (phase 1 + 2) "))
    print(fmt(gain_s, "fine-tuning gain                  "))

    # Binomial sampling floor: even a perfectly stable model scores differently
    # on a finite test set only if the predictions change, so this is a lower
    # bound on what any single-run comparison can resolve.
    if ft_s["mean"] is not None:
        p_hat = ft_s["mean"] / 100.0
        se = (p_hat * (1 - p_hat) / len(test_ds)) ** 0.5 * 100
        print(f"\nBinomial standard error at n={len(test_ds)}, p={p_hat:.3f}: "
              f"{se:.2f} pts (irreducible sampling floor)")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if ref_s["spread"] is None or ft_s["spread"] is None:
        print("Need at least 2 runs with both checkpoints.")
    else:
        rs, fs = ref_s["spread"], ft_s["spread"]
        ratio = fs / rs if rs > 0 else float("inf")
        print(f"spread(finetuned) / spread(reference) = {ratio:.2f}")
        if ratio > 1.5:
            print("-> PHASE 2 AMPLIFIES the variance. Fine-tuning is adding")
            print("   instability on top of whatever phase 1 hands it.")
            print("   Best-epoch selection and regularisation in phase 2 should")
            print("   shrink the noise floor. Do that before any ablation.")
        elif ratio < 0.67:
            print("-> PHASE 2 DAMPENS the variance. Fine-tuning is pulling")
            print("   different pre-trained models toward a common solution.")
            print("   The floor originates in phase 1: look at weight init, the")
            print("   LR schedule, or averaging the last few epochs' weights.")
        else:
            print("-> PHASE 2 PASSES THE VARIANCE THROUGH roughly unchanged.")
            print("   Phase 1 is the origin and phase 2 neither helps nor hurts")
            print("   stability. Fixing phase 2 selection may still raise the")
            print("   mean, but it will not shrink the spread much.")

    # --- per-class stability ---
    recalls = [r["finetuned_recall"] for r in rows if "finetuned_recall" in r]
    if len(recalls) >= 3:
        print("\n" + "=" * 78)
        print("PER-CLASS STABILITY ACROSS SEEDS (finetuned)")
        print("=" * 78)
        arr = np.array([[np.nan if v is None else v for v in rec] for rec in recalls],
                       dtype=float)
        mean_r = np.nanmean(arr, axis=0)
        sd_r = np.nanstd(arr, axis=0, ddof=1)
        order = np.argsort(-sd_r)
        print(f"{'class':>5}  {'mean recall':>11}  {'stdev':>7}   per-seed")
        for c in order[:8]:
            per = "  ".join(f"{arr[i, c]:.2f}" for i in range(arr.shape[0]))
            print(f"{c:>5}  {mean_r[c]:>11.2f}  {sd_r[c]:>7.3f}   {per}")
        print("\nHigh stdev = that class's fate is decided by the seed.")
        print("Low mean + low stdev = consistently broken, a real target.")
        consistent = [int(c) for c in np.argsort(mean_r)[:6] if sd_r[c] < 0.10]
        if consistent:
            print(f"Consistently hard and stable across seeds: {consistent}")

    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"rows": rows, "reference": ref_s, "finetuned": ft_s,
                       "gain": gain_s}, fh, indent=2, default=str)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
