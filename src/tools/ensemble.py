"""
Seed ensemble: average the predictions of checkpoints you already trained.

No training. Loads N finetuned checkpoints from completed runs, evaluates each
on the test set, and combines them three ways:

    logit_mean   mean of raw logits. Sensitive to per-model logit scale, but
                 usually the strongest of the three in practice.
    prob_mean    mean of softmax probabilities. Scale-invariant across models,
                 which matters if one seed produces systematically larger
                 logits than another.
    vote         plurality of argmax predictions, ties broken by summed
                 probability. Weakest in principle, included because it is
                 the one that survives if you only have hard labels.

Why this has a decent prior: ensembling helps in proportion to how much the
members DISAGREE while each being individually good. The script reports
pairwise disagreement and oracle accuracy (the ceiling if you always picked
the right member) so you can see whether there was any headroom to exploit,
rather than just reading the final number.

Honest caveat: three models trained on the same data from the same
architecture with only the seed varied are highly correlated. This is the
weakest form of ensembling. A 1-2 point gain is typical; zero is entirely
possible and would mean the seeds converge to near-identical functions.

The result is also not free in deployment: 3x inference cost and 3x the
checkpoints to ship. For a 40k-parameter model that is nothing, but it should
be stated rather than assumed.

Usage (from the repo root):
    python src/tools/ensemble.py --tag p2_base --ft-epochs 20
    python src/tools/ensemble.py --tag p2_base --ft-epochs 20 --out results/ensemble.json
    python src/tools/ensemble.py --dirs runs/a runs/b runs/c
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import statistics
import sys
from itertools import combinations

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.dataset import PIPELINE_VARIANTS, RamanDataset  # noqa: E402
from src.metrics import confusion, summarize  # noqa: E402
from src.repro import pick_device  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_run_ablation", os.path.join(_HERE, "run_ablation.py"))
_ra = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ra)
mcnemar = _ra.mcnemar


def build_model(model_type, num_classes, device):
    if model_type == "legacy_cnn":
        from legacy_baseline.model import LegacyCNN
        return LegacyCNN(num_classes=num_classes).to(device)
    if model_type == "resnet":
        from src.model import Raman1DCNN
        return Raman1DCNN(num_classes=num_classes).to(device)
    if model_type == "multiscale_cnn":
        from src.model import MultiscaleCNN
        return MultiscaleCNN(num_classes=num_classes).to(device)
    if model_type == "fusion":
        from src.model import FusionNet
        return FusionNet(num_classes=num_classes).to(device)
    if model_type == "transformer":
        from src.model import Transformer1D
        return Transformer1D(num_classes=num_classes).to(device)
    raise ValueError(f"unknown model_type {model_type!r}")


def find_runs(run_root, tag, model_type, pipeline, epochs, ft_epochs, seeds):
    out = []
    for d in sorted(glob.glob(os.path.join(run_root, "*"))):
        mf = os.path.join(d, "manifest.json")
        ck = os.path.join(d, "finetuned.pth")
        if not (os.path.exists(mf) and os.path.exists(ck)):
            continue
        with open(mf) as fh:
            a = json.load(fh).get("args", {})
        if tag and a.get("tag") != tag:
            continue
        if model_type and a.get("model_type") != model_type:
            continue
        if pipeline and a.get("pipeline", "legacy_full") != pipeline:
            continue
        if epochs is not None and a.get("epochs") != epochs:
            continue
        if ft_epochs is not None and a.get("finetune_epochs") != ft_epochs:
            continue
        if seeds and a.get("seed") not in seeds:
            continue
        out.append({"dir": d, "ckpt": ck, "args": a})
    # one per seed, most recent
    by = {}
    for r in out:
        by.setdefault(r["args"].get("seed"), []).append(r)
    return [sorted(v, key=lambda x: x["dir"])[-1] for _, v in sorted(by.items())]


@torch.no_grad()
def collect_logits(model, device, loader):
    model.eval()
    chunks, targets = [], []
    for data, target in loader:
        chunks.append(model(data.to(device)).cpu().numpy())
        targets.append(target.numpy())
    return np.concatenate(chunks), np.concatenate(targets)


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="p2_base")
    p.add_argument("--model-type", default="legacy_cnn")
    p.add_argument("--pipeline", default="minmax_only")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--ft-epochs", type=int, default=20)
    p.add_argument("--seeds", default=None, help="comma-separated; default all found")
    p.add_argument("--dirs", nargs="*", default=None,
                   help="explicit run directories, overrides tag matching")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="auto")
    p.add_argument("--run-root", default="runs")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else None

    if a.dirs:
        runs = []
        for d in a.dirs:
            with open(os.path.join(d, "manifest.json")) as fh:
                args_rec = json.load(fh).get("args", {})
            runs.append({"dir": d, "ckpt": os.path.join(d, "finetuned.pth"),
                         "args": args_rec})
    else:
        runs = find_runs(a.run_root, a.tag, a.model_type, a.pipeline,
                         a.epochs, a.ft_epochs, seeds)

    if len(runs) < 2:
        print(f"found {len(runs)} matching run(s) with a finetuned.pth; need >= 2")
        return 1

    device = pick_device(a.device)
    print(f"Using device: {device}")
    print(f"Ensembling {len(runs)} checkpoints:")
    for r in runs:
        print(f"  seed {r['args'].get('seed')}  {os.path.basename(r['dir'])}")

    mapping = {lab: i for i, lab in enumerate(
        sorted(np.unique(np.load(os.path.join(a.data_dir, "y_reference.npy")))))}
    num_classes = len(mapping)
    test_ds = RamanDataset(os.path.join(a.data_dir, "X_test.npy"),
                           os.path.join(a.data_dir, "y_test.npy"),
                           label_mapping=mapping,
                           pipeline_spec=PIPELINE_VARIANTS[a.pipeline])
    loader = DataLoader(test_ds, batch_size=a.batch_size, shuffle=False)
    print(f"Test set: {len(test_ds)} samples, {num_classes} classes\n")

    all_logits, y_true = [], None
    members = []
    for r in runs:
        model = build_model(a.model_type, num_classes, device)
        model.load_state_dict(torch.load(r["ckpt"], map_location=device,
                                         weights_only=True))
        logits, yt = collect_logits(model, device, loader)
        y_true = yt if y_true is None else y_true
        if not np.array_equal(y_true, yt):
            print("label order differs between runs; aborting")
            return 1
        all_logits.append(logits)
        acc = 100.0 * (logits.argmax(1) == yt).mean()
        members.append({"seed": r["args"].get("seed"), "acc": acc,
                        "dir": os.path.basename(r["dir"])})
        print(f"  member seed {r['args'].get('seed')}: {acc:.2f}%")

    L = np.stack(all_logits)          # (M, N, C)
    P = np.stack([softmax(x) for x in all_logits])
    preds = L.argmax(2)               # (M, N)

    # --- how much do the members disagree? ---
    print("\n" + "=" * 62)
    print("DIVERSITY")
    print("=" * 62)
    dis = [100.0 * (preds[i] != preds[j]).mean()
           for i, j in combinations(range(len(runs)), 2)]
    print(f"pairwise disagreement: {['%.2f%%' % d for d in dis]}  "
          f"mean {statistics.mean(dis):.2f}%")
    any_right = (preds == y_true).any(axis=0)
    all_right = (preds == y_true).all(axis=0)
    print(f"oracle (any member right):  {100.0 * any_right.mean():.2f}%")
    print(f"unanimous and right:        {100.0 * all_right.mean():.2f}%")
    print("Oracle is the ceiling: an ensemble cannot exceed it. If oracle is")
    print("barely above the best member, the seeds agree on their mistakes and")
    print("there is nothing for averaging to exploit.")

    # --- combinations ---
    results = {}
    combos = {
        "logit_mean": L.mean(0).argmax(1),
        "prob_mean": P.mean(0).argmax(1),
    }
    votes = np.zeros((len(y_true), num_classes))
    for m in range(len(runs)):
        votes[np.arange(len(y_true)), preds[m]] += 1
    combos["vote"] = (votes + 0.001 * P.mean(0)).argmax(1)

    best_member = max(members, key=lambda m: m["acc"])
    print("\n" + "=" * 62)
    print("ENSEMBLE")
    print("=" * 62)
    mean_member = statistics.mean(m["acc"] for m in members)
    print(f"member mean {mean_member:.2f}%   best member "
          f"{best_member['acc']:.2f}% (seed {best_member['seed']})\n")

    best_pred = all_logits[[m["acc"] for m in members].index(best_member["acc"])].argmax(1)
    for name, pred in combos.items():
        st = summarize(confusion(y_true, pred, num_classes))
        acc = 100.0 * st["accuracy"]
        b, c, chi2, pv = mcnemar(best_pred == y_true, pred == y_true)
        results[name] = {"acc": acc, "macro_f1": st["macro_f1"],
                         "vs_best_member": acc - best_member["acc"],
                         "mcnemar_b": b, "mcnemar_c": c, "p": pv}
        print(f"{name:<12} {acc:>6.2f}%  macro-F1 {st['macro_f1']:.4f}  "
              f"vs best member {acc - best_member['acc']:+.2f} pts  "
              f"(McNemar best-only={b}, ens-only={c}, p={pv:.4g})")

    best_combo = max(results, key=lambda k: results[k]["acc"])
    print(f"\nBest: {best_combo} at {results[best_combo]['acc']:.2f}%, "
          f"{results[best_combo]['vs_best_member']:+.2f} pts over the best single "
          f"model and {results[best_combo]['acc'] - mean_member:+.2f} over the mean.")
    print("Compare against the best single model, not the mean: you would deploy")
    print("one model if you were not ensembling, and you would pick a good one.")
    print("Cost of the ensemble: 3x inference and 3x checkpoints to ship.")

    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"members": members, "diversity": {
                "pairwise_disagreement": dis,
                "oracle": 100.0 * float(any_right.mean()),
                "unanimous_right": 100.0 * float(all_right.mean())},
                "ensembles": results}, fh, indent=2, default=str)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
