"""
"New classes arrive" experiment.

Holds back K of the 30 classes, trains a model that has never seen them, then
introduces them four different ways and measures what each does to the classes
the model already knew.

    retrain   from scratch on all 30. The mandatory baseline. If an
              incremental method cannot beat this on accuracy OR wall-clock,
              it is not worth having. LegacyCNN is 40k params on 60k spectra,
              so retraining is minutes, and "retraining is too expensive" is
              not a premise that holds here.
    naive     expand the head, fine-tune on new-class data only. The obvious
              thing, and the one that should forget badly.
    replay    same, plus N examples per old class mixed in.
    frozen    expand the head, freeze the backbone, train the head only.
              Cannot forget the features; may not learn the new classes well.

Reported per strategy: overall 30-class accuracy, mean recall on the 25 old
classes, mean recall on the 5 new ones, and per-class recall drop against the
same model before the new classes arrived.

Honest limits of this design:

- ONE holdout draw per --holdout-seed. Which 5 classes you hold out matters a
  lot: classes 1 and 5 are near-perfect while 11 and 8 sit at 0.24-0.68, so a
  draw weighted toward easy classes will flatter every method. Run at least
  two holdout seeds before believing an ordering.
- New-class data comes from the finetune split by default (100 per class),
  because that is what "new data arrives" realistically looks like. Using
  --new-source reference gives 2000 per class, which is the easy version.
- The old-class baseline is measured on the 25-class model, whose head has 25
  outputs. After expansion it has 30, so the old classes face 5 extra
  competitors. Some recall loss is therefore expected even from a perfect
  method; that is a property of the task, not a failure of the strategy.

Usage (from the repo root):
    python src/tools/run_incremental.py --dry-run
    python src/tools/run_incremental.py --seeds 1 --holdout-seed 0
    python src/tools/run_incremental.py --seeds 1,2,3 --holdout-seed 0 --out results/incr_h0.json
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.optim as optim  # noqa: E402
from torch.utils.data import ConcatDataset, DataLoader, Subset, random_split  # noqa: E402

from src.dataset import (PIPELINE_VARIANTS, AugmentedDataset,  # noqa: E402
                         RamanDataset)
from src.engine import evaluate, train_epoch  # noqa: E402
from src.incremental import (expand_classifier, extended_mapping,  # noqa: E402
                             forgetting_report, freeze_backbone, group_recall,
                             make_mapping, replay_indices, split_classes)
from src.repro import make_generator, pick_device, seed_worker, set_seed  # noqa: E402
from src.runmeta import RunMeta  # noqa: E402


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-type", default="legacy_cnn")
    p.add_argument("--pipeline", default="minmax_only")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--seeds", default="1")
    p.add_argument("--holdout-seed", type=int, default=0,
                   help="chooses WHICH classes are held out; vary separately "
                        "from --seeds")
    p.add_argument("--n-holdout", type=int, default=5)
    p.add_argument("--holdout", default=None,
                   help="explicit comma-separated label values, overrides the draw")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--finetune-epochs", type=int, default=20)
    p.add_argument("--incr-epochs", type=int, default=20,
                   help="epochs for the incremental strategies")
    p.add_argument("--val-split", type=float, default=0.33)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--finetune-lr-mult", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--augment", action="store_true", default=True)
    p.add_argument("--no-augment", dest="augment", action="store_false")
    p.add_argument("--replay-per-class", type=int, default=20)
    p.add_argument("--new-source", default="finetune",
                   choices=["finetune", "reference"],
                   help="where new-class training data comes from")
    p.add_argument("--strategies", default="retrain,naive,replay,frozen")
    p.add_argument("--device", default="auto")
    p.add_argument("--run-root", default="runs")
    p.add_argument("--out", default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    pipeline_spec = PIPELINE_VARIANTS[args.pipeline]

    paths = {k: (os.path.join(args.data_dir, f"X_{k}.npy"),
                 os.path.join(args.data_dir, f"y_{k}.npy"))
             for k in ("reference", "finetune", "test")}

    all_labels = sorted(np.unique(np.load(paths["reference"][1])))
    if args.holdout:
        new_labels = sorted(float(x) for x in args.holdout.split(","))
        old_labels = [l for l in all_labels if l not in new_labels]
    else:
        old_labels, new_labels = split_classes(all_labels, args.n_holdout,
                                               args.holdout_seed)

    old_map = make_mapping(old_labels)
    full_map = extended_mapping(old_labels, new_labels)
    old_idx = [full_map[l] for l in old_labels]
    new_idx = [full_map[l] for l in new_labels]

    print(f"Held out {len(new_labels)} classes (labels {[int(l) for l in new_labels]}), "
          f"keeping {len(old_labels)}")
    print(f"Old class indices 0..{len(old_labels)-1}, new indices {new_idx}")
    if args.dry_run:
        print("dry run, stopping before any training")
        return 0

    device = pick_device(args.device)
    print(f"Using device: {device}")
    criterion = nn.CrossEntropyLoss()
    ds_kw = dict(pipeline_spec=pipeline_spec, on_unseen="drop")

    # Datasets. Built once; the cache makes repeat construction cheap.
    ref_old = RamanDataset(*paths["reference"], label_mapping=old_map, **ds_kw)
    ft_old = RamanDataset(*paths["finetune"], label_mapping=old_map, **ds_kw)
    test_old = RamanDataset(*paths["test"], label_mapping=old_map, **ds_kw)

    ref_full = RamanDataset(*paths["reference"], label_mapping=full_map, **ds_kw)
    ft_full = RamanDataset(*paths["finetune"], label_mapping=full_map, **ds_kw)
    test_full = RamanDataset(*paths["test"], label_mapping=full_map, **ds_kw)

    n_old, n_all = len(old_labels), len(full_map)
    all_results = []

    for seed in seeds:
        set_seed(seed)
        run = RunMeta(args, device, run_root=args.run_root,
                      tag=f"incr-h{args.holdout_seed}-s{seed}")
        run.log("setup", {"old_labels": [int(x) for x in old_labels],
                          "new_labels": [int(x) for x in new_labels],
                          "old_idx": old_idx, "new_idx": new_idx})
        print(f"\n{'#'*70}\n# seed {seed}   run {run.run_id}\n{'#'*70}")

        def loader(ds, bs, shuffle):
            return DataLoader(ds, batch_size=bs, shuffle=shuffle,
                              generator=make_generator(seed) if shuffle else None)

        def train(model, tr_loader, va_loader, n_epochs, lr, wd, phase, patience):
            opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(n_epochs, 1))
            best, best_w, bad = float("inf"), None, 0
            for ep in range(1, n_epochs + 1):
                train_epoch(model, device, tr_loader, opt, criterion, ep, phase=phase)
                sch.step()
                if va_loader is not None:
                    vl, _ = evaluate(model, device, va_loader, criterion,
                                     phase=f"{phase}-Val",
                                     num_classes=model_out_features(model))
                    if vl < best:
                        best, best_w, bad = vl, copy.deepcopy(model.state_dict()), 0
                    else:
                        bad += 1
                        if patience and bad >= patience:
                            print(f"early stop at epoch {ep}")
                            break
            if best_w is not None:
                model.load_state_dict(best_w)
            return model

        def model_out_features(m):
            last = [mm for mm in m.modules() if isinstance(mm, nn.Linear)][-1]
            return last.out_features

        # ---------- Stage A: the world before the new classes ----------
        t0 = time.time()
        print(f"\n--- Stage A: train on {n_old} known classes ---")
        va_n = int(args.val_split * len(ref_old))
        tr_set, va_set = random_split(ref_old, [len(ref_old) - va_n, va_n],
                                      generator=make_generator(seed))
        model = build_model(args.model_type, n_old, device)
        model = train(model,
                      loader(AugmentedDataset(tr_set, augment=args.augment),
                             args.batch_size, True),
                      loader(AugmentedDataset(va_set, augment=False),
                             args.batch_size * 2, False),
                      args.epochs, args.lr, args.weight_decay, "Pre-train",
                      args.patience)
        model = train(model, loader(ft_old, args.batch_size, True), None,
                      args.finetune_epochs, args.lr * args.finetune_lr_mult, 0.0,
                      "Finetune", 0)

        _, acc_before, st_before = evaluate(model, device,
                                            loader(test_old, args.batch_size * 2, False),
                                            criterion, phase="A: old-only",
                                            num_classes=n_old, return_stats=True)
        base_state = copy.deepcopy(model.state_dict())
        stage_a = {"acc_old_only": acc_before,
                   "per_class_recall": st_before["per_class_recall"],
                   "minutes": (time.time() - t0) / 60}
        run.log("stage_a", stage_a)
        print(f"Stage A: {acc_before:.2f}% on the {n_old} known classes "
              f"({stage_a['minutes']:.1f} min)")

        # new-class training data
        src = ft_full if args.new_source == "finetune" else ref_full
        y_src = np.asarray(src.y)
        new_only = Subset(src, np.where(np.isin(y_src, new_idx))[0].tolist())
        print(f"New-class training pool: {len(new_only)} spectra from {args.new_source}")

        seed_results = {"seed": seed, "stage_a": stage_a, "strategies": {}}

        for strat in strategies:
            print(f"\n--- Stage B [{strat}] ---")
            t1 = time.time()

            if strat == "retrain":
                set_seed(seed)
                m = build_model(args.model_type, n_all, device)
                vn = int(args.val_split * len(ref_full))
                tr2, va2 = random_split(ref_full, [len(ref_full) - vn, vn],
                                        generator=make_generator(seed))
                m = train(m,
                          loader(AugmentedDataset(tr2, augment=args.augment),
                                 args.batch_size, True),
                          loader(AugmentedDataset(va2, augment=False),
                                 args.batch_size * 2, False),
                          args.epochs, args.lr, args.weight_decay,
                          "Retrain", args.patience)
                m = train(m, loader(ft_full, args.batch_size, True), None,
                          args.finetune_epochs, args.lr * args.finetune_lr_mult,
                          0.0, "Retrain-FT", 0)
                info = {"note": "from scratch on all classes"}
            else:
                m = build_model(args.model_type, n_old, device)
                m.load_state_dict(base_state)
                info = expand_classifier(m, n_all, device=device)
                if strat == "frozen":
                    info.update(freeze_backbone(m))
                if strat == "replay":
                    rep = replay_indices(np.asarray(ft_full.y), old_idx,
                                         args.replay_per_class, seed)
                    train_set = ConcatDataset([new_only,
                                               Subset(ft_full, rep.tolist())])
                    info["replay_samples"] = int(len(rep))
                    print(f"replay buffer: {len(rep)} old-class spectra "
                          f"({args.replay_per_class}/class)")
                else:
                    train_set = new_only
                m = train(m, loader(train_set, args.batch_size, True), None,
                          args.incr_epochs, args.lr * args.finetune_lr_mult,
                          0.0, f"Incr-{strat}", 0)

            _, acc_after, st_after = evaluate(
                m, device, loader(test_full, args.batch_size * 2, False),
                criterion, phase=f"B: {strat}", num_classes=n_all,
                return_stats=True)
            rep = forgetting_report(stage_a["per_class_recall"],
                                    st_after["per_class_recall"], old_idx, new_idx)
            rec = {"acc_all": acc_after, "macro_f1": st_after["macro_f1"],
                   "minutes": (time.time() - t1) / 60, "info": info, **rep}
            seed_results["strategies"][strat] = rec
            run.log(f"stage_b_{strat}", rec)

            print(f"[{strat}] 30-class acc {acc_after:.2f}%  "
                  f"old {100*rep['old_recall_after']:.1f}% "
                  f"(was {100*rep['old_recall_before']:.1f}%)  "
                  f"new {100*rep['new_recall_after']:.1f}%  "
                  f"forgetting {100*rep['mean_forgetting']:+.1f} pts  "
                  f"[{rec['minutes']:.1f} min]")

        all_results.append(seed_results)
        run.finish(summary=seed_results)

    # ---------------- summary ----------------
    print("\n" + "=" * 92)
    print(f"SUMMARY  holdout-seed={args.holdout_seed}  "
          f"held-out labels {[int(x) for x in new_labels]}  n_seeds={len(all_results)}")
    print("=" * 92)
    print(f"{'strategy':<10}{'acc(30)':>9}{'old now':>9}{'old was':>9}"
          f"{'forget':>9}{'new':>8}{'min':>7}")
    print("-" * 61)
    for strat in strategies:
        rows = [r["strategies"][strat] for r in all_results if strat in r["strategies"]]
        if not rows:
            continue
        m = lambda k: float(np.mean([r[k] for r in rows]))  # noqa: E731
        print(f"{strat:<10}{m('acc_all'):>9.2f}{100*m('old_recall_after'):>9.1f}"
              f"{100*m('old_recall_before'):>9.1f}{100*m('mean_forgetting'):>+9.1f}"
              f"{100*m('new_recall_after'):>8.1f}{m('minutes'):>7.1f}")

    print("\nHow to read this:")
    print("  forget  = mean per-class recall LOST on the old classes. Some loss is")
    print("            expected even from a perfect method, because the old classes")
    print("            now compete against 5 extra logits.")
    print("  new     = mean recall on the classes that just arrived. A method with")
    print("            low forgetting and low 'new' has simply not learned them.")
    print("  retrain = the bar. An incremental method must beat it on accuracy or")
    print("            on wall-clock, otherwise just retrain.")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"holdout_seed": args.holdout_seed,
                       "new_labels": [int(x) for x in new_labels],
                       "results": all_results}, fh, indent=2, default=str)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
