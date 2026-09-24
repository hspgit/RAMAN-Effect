import argparse
import copy
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from torch.utils.data import Subset

from src.dataset import (PIPELINE_VARIANTS, AugmentedDataset, RamanDataset,
                         stratified_split_indices)
from src.engine import evaluate, train_epoch
from src.model import Raman1DCNN
from src.repro import make_generator, pick_device, seed_worker, set_seed
from src.runmeta import RunMeta


def main():
    parser = argparse.ArgumentParser(description='Deep Learning for Raman Spectra Classification')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=15,
                        help='number of epochs for reference training (default: 15)')
    parser.add_argument('--finetune-epochs', type=int, default=5,
                        help='number of epochs for finetuning (default: 5)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='L2 penalty weight decay (default: 0.01)')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='directory containing the .npy files')
    parser.add_argument('--save-dir', type=str, default='models',
                        help='directory for the mirrored checkpoints')

    parser.add_argument('--val-split', type=float, default=0.0,
                        help='fraction of reference data to use for validation (default: 0.0)')
    parser.add_argument('--augment', action='store_true',
                        help='enable on-the-fly data augmentation during training')

    parser.add_argument('--mixup-alpha', type=float, default=0.0,
                        help='alpha parameter for mixup augmentation (default: 0.0, i.e., disabled)')

    parser.add_argument('--model-type', type=str, default='resnet',
                        choices=['resnet', 'legacy_cnn', 'transformer', 'multiscale_cnn', 'fusion'],
                        help='type of model to use')

    # --- Stage 0 additions: reproducibility, device control, run recording ---
    parser.add_argument('--seed', type=int, default=42,
                        help='seed for python, numpy, torch, splits and loaders (default: 42)')
    parser.add_argument('--strict-determinism', action='store_true',
                        help='request deterministic kernels; slower, use for reportable runs')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'mps', 'cpu'],
                        help="'auto' keeps the old cuda->mps->cpu order; naming a device "
                             "errors instead of silently falling back")
    parser.add_argument('--run-root', type=str, default='runs',
                        help='parent directory for per-run artifact folders')
    parser.add_argument('--tag', type=str, default=None,
                        help="short label folded into the run id, e.g. 'noaspls'")
    parser.add_argument('--num-workers', type=int, default=0,
                        help='DataLoader workers (default: 0, i.e. single-process)')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='preprocessing cache location (default: <data-dir>/.cache)')
    parser.add_argument('--no-cache', action='store_true',
                        help='recompute preprocessing without reading or writing the cache')
    parser.add_argument('--patience', type=int, default=10,
                        help='early-stopping patience on validation loss (default: 10)')
    # Using choices= means a typo fails at argparse rather than silently running the wrong arm.
    parser.add_argument('--pipeline', type=str, default='minmax_only',
                        choices=sorted(PIPELINE_VARIANTS),
                        help='preprocessing variant (default: minmax_only)')

    # --- Phase 2 (fine-tuning) controls. All defaults reproduce the old
    # --- behaviour: no validation, no augmentation, no weight decay,
    # --- last-epoch weights saved.
    parser.add_argument('--finetune-val-split', type=float, default=0.0,
                        help='stratified fraction of the finetune set held for '
                             'validation. 0.0 (default) = no validation, save the '
                             'last epoch, as before. 0.1 gives 10 per class.')
    parser.add_argument('--finetune-augment', action='store_true',
                        help='apply the three augmentations during fine-tuning. '
                             'Never once tried in this repo, despite phase 2 being '
                             'the small, domain-shifted set that most needs it.')
    parser.add_argument('--finetune-weight-decay', type=float, default=0.0,
                        help='L2 penalty for the finetune optimizer. Was silently '
                             '0 because --weight-decay was never passed through.')
    parser.add_argument('--finetune-lr-mult', type=float, default=0.1,
                        help='finetune LR = lr * this (was hardcoded 0.1)')
    parser.add_argument('--finetune-patience', type=int, default=0,
                        help='early-stopping patience on finetune validation loss. '
                             '0 (default) = run all epochs, still keeping the '
                             'best-val weights if a val split exists.')
    parser.add_argument('--diagnostic-test-curve', action='store_true',
                        help='evaluate the TEST set after every finetune epoch and '
                             'record it. DIAGNOSIS ONLY: never select on this, or '
                             'your test number stops being held out.')

    args = parser.parse_args()

    # Seed and device must be settled before anything else touches RNG or CUDA.
    set_seed(args.seed, strict=args.strict_determinism)
    device = pick_device(args.device)
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)
    run = RunMeta(args, device, run_root=args.run_root, tag=args.tag)
    print(f"Run id: {run.run_id}")

    def make_loader(ds, batch_size, shuffle):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker if args.num_workers > 0 else None,
            generator=make_generator(args.seed) if shuffle else None,
            persistent_workers=args.num_workers > 0,
        )

    ds_kwargs = dict(
        cache_dir=args.cache_dir,
        use_cache=not args.no_cache,
        pipeline_spec=PIPELINE_VARIANTS[args.pipeline],
    )
    print(f"Preprocessing variant: {args.pipeline} "
          f"({len(PIPELINE_VARIANTS[args.pipeline])} steps)")

    # 1. Load Reference Data
    print("Loading reference dataset...")
    full_ref_dataset = RamanDataset(os.path.join(args.data_dir, 'X_reference.npy'),
                                    os.path.join(args.data_dir, 'y_reference.npy'),
                                    **ds_kwargs)

    num_classes = len(np.unique(full_ref_dataset.y))
    print(f"Detected {num_classes} classes.")

    if args.val_split > 0:
        val_size = int(args.val_split * len(full_ref_dataset))
        train_size = len(full_ref_dataset) - val_size
        ref_dataset, val_dataset = random_split(
            full_ref_dataset, [train_size, val_size],
            generator=make_generator(args.seed))

        # Apply augmentation only to training data
        train_dataset_wrapped = AugmentedDataset(ref_dataset, augment=args.augment)
        val_dataset_wrapped = AugmentedDataset(val_dataset, augment=False)

        ref_loader = make_loader(train_dataset_wrapped, args.batch_size, True)
        val_loader = make_loader(val_dataset_wrapped, args.batch_size * 2, False)
        print(f"Split reference data: {train_size} train / {val_size} val")
    else:
        train_dataset_wrapped = AugmentedDataset(full_ref_dataset, augment=args.augment)
        ref_loader = make_loader(train_dataset_wrapped, args.batch_size, True)
        val_loader = None

    # 2. Initialize Model
    if args.model_type == 'resnet':
        model = Raman1DCNN(num_classes=num_classes).to(device)
    elif args.model_type == 'legacy_cnn':
        from legacy_baseline.model import LegacyCNN
        model = LegacyCNN(num_classes=num_classes).to(device)
    elif args.model_type == 'transformer':
        from src.model import Transformer1D
        model = Transformer1D(num_classes=num_classes).to(device)
    elif args.model_type == 'multiscale_cnn':
        from src.model import MultiscaleCNN
        model = MultiscaleCNN(num_classes=num_classes).to(device)
    elif args.model_type == 'fusion':
        from src.model import FusionNet
        model = FusionNet(num_classes=num_classes).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    run.log("setup", {"num_classes": num_classes, "param_count": param_count,
                      "pipeline": list(full_ref_dataset.pipeline_spec)})

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 3. Train on Reference Data
    print("\n--- Starting Reference Training ---")
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float('inf')
    patience = args.patience
    patience_counter = 0
    stopped_epoch = args.epochs
    ref_model_path = run.path('reference.pth')

    for epoch in range(1, args.epochs + 1):
        train_stats = train_epoch(model, device, ref_loader, optimizer, criterion, epoch,
                                  phase="Pre-train", mixup_alpha=args.mixup_alpha)
        scheduler.step()

        if val_loader is not None:
            val_loss, val_acc, val_stats = evaluate(model, device, val_loader, criterion,
                                                    phase="Validation",
                                                    num_classes=num_classes,
                                                    return_stats=True)
            run.log_epoch("pretrain", epoch,
                          lr=optimizer.param_groups[0]['lr'],
                          val_loss=val_loss, val_acc=val_acc,
                          val_macro_f1=val_stats['macro_f1'],
                          val_per_class_recall=val_stats['per_class_recall'],
                          **train_stats)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), ref_model_path)
                print(f"--> Best validation loss improved. Saved model to {ref_model_path}")
            else:
                patience_counter += 1
                print(f"--> No improvement in validation loss. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch}! Reverting to best model.")
                stopped_epoch = epoch
                break
        else:
            run.log_epoch("pretrain", epoch,
                          lr=optimizer.param_groups[0]['lr'], **train_stats)

    if val_loader is None:
        torch.save(model.state_dict(), ref_model_path)
        print(f"Reference model saved to {ref_model_path}")
    elif os.path.exists(ref_model_path):
        model.load_state_dict(torch.load(ref_model_path, weights_only=True))
        print("Loaded best reference model for finetuning.")

    # Mirror into --save-dir under a model-typed name. The old shared filename
    # meant consecutive runs of different architectures overwrote each other.
    ref_mirror = os.path.join(args.save_dir, f'{args.model_type}_reference.pth')
    shutil.copyfile(ref_model_path, ref_mirror)

    run.log("pretrain_summary", {"best_val_loss": None if best_val_loss == float('inf')
                                 else best_val_loss,
                                 "stopped_epoch": stopped_epoch,
                                 "early_stopped": stopped_epoch < args.epochs})

    # Defined here rather than in section 5 because the finetune block below
    # needs them for the optional per-epoch diagnostic test curve.
    test_x = os.path.join(args.data_dir, 'X_test.npy')
    test_y = os.path.join(args.data_dir, 'y_test.npy')

    # 4. Finetune Data (if available)
    finetune_x = os.path.join(args.data_dir, 'X_finetune.npy')
    finetune_y = os.path.join(args.data_dir, 'y_finetune.npy')

    if os.path.exists(finetune_x) and os.path.exists(finetune_y) and args.finetune_epochs > 0:
        print("\n--- Starting Finetuning ---")
        finetune_dataset = RamanDataset(finetune_x, finetune_y,
                                        label_mapping=full_ref_dataset.label_mapping,
                                        require_space=full_ref_dataset.label_space,
                                        **ds_kwargs)

        if args.finetune_val_split > 0:
            tr_idx, va_idx = stratified_split_indices(
                finetune_dataset.y, args.finetune_val_split, args.seed)
            ft_train = AugmentedDataset(Subset(finetune_dataset, tr_idx.tolist()),
                                        augment=args.finetune_augment)
            ft_val = AugmentedDataset(Subset(finetune_dataset, va_idx.tolist()),
                                      augment=False)
            finetune_loader = make_loader(ft_train, args.batch_size, True)
            ft_val_loader = make_loader(ft_val, args.batch_size * 2, False)
            print(f"Split finetune data: {len(tr_idx)} train / {len(va_idx)} val "
                  f"({len(va_idx) // num_classes} per class)")
        else:
            finetune_loader = make_loader(
                AugmentedDataset(finetune_dataset, augment=args.finetune_augment),
                args.batch_size, True)
            ft_val_loader = None

        # Optional per-epoch test curve, for diagnosis only.
        diag_loader = None
        if args.diagnostic_test_curve and os.path.exists(test_x) and os.path.exists(test_y):
            diag_loader = make_loader(
                RamanDataset(test_x, test_y,
                             label_mapping=full_ref_dataset.label_mapping,
                             require_space=full_ref_dataset.label_space,
                             **ds_kwargs),
                args.batch_size * 2, False)

        optimizer_ft = optim.Adam(model.parameters(),
                                  lr=args.lr * args.finetune_lr_mult,
                                  weight_decay=args.finetune_weight_decay)
        scheduler_ft = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=args.finetune_epochs)

        best_ft_loss = float('inf')
        best_ft_weights = None
        best_ft_epoch = None
        ft_patience_counter = 0

        for epoch in range(1, args.finetune_epochs + 1):
            ft_stats = train_epoch(model, device, finetune_loader, optimizer_ft, criterion,
                                   epoch, phase="Finetune", mixup_alpha=args.mixup_alpha)
            scheduler_ft.step()
            extra = {}

            if ft_val_loader is not None:
                v_loss, v_acc, v_stats = evaluate(model, device, ft_val_loader, criterion,
                                                  phase="Finetune-Val",
                                                  num_classes=num_classes,
                                                  return_stats=True)
                extra.update(ft_val_loss=v_loss, ft_val_acc=v_acc,
                             ft_val_macro_f1=v_stats['macro_f1'])
                if v_loss < best_ft_loss:
                    best_ft_loss = v_loss
                    # deepcopy, not state_dict() directly: state_dict returns
                    # live tensor references that later epochs mutate in place.
                    best_ft_weights = copy.deepcopy(model.state_dict())
                    best_ft_epoch = epoch
                    ft_patience_counter = 0
                    print(f"--> Finetune val loss improved (epoch {epoch}).")
                else:
                    ft_patience_counter += 1

            if diag_loader is not None:
                d_loss, d_acc, d_stats = evaluate(model, device, diag_loader, criterion,
                                                  phase="DIAGNOSTIC-Test",
                                                  num_classes=num_classes,
                                                  return_stats=True)
                extra.update(diag_test_loss=d_loss, diag_test_acc=d_acc,
                             diag_test_macro_f1=d_stats['macro_f1'])

            run.log_epoch("finetune", epoch,
                          lr=optimizer_ft.param_groups[0]['lr'], **ft_stats, **extra)

            if args.finetune_patience > 0 and ft_patience_counter >= args.finetune_patience:
                print(f"\nFinetune early stopping at epoch {epoch} "
                      f"(best was {best_ft_epoch}).")
                break

        if best_ft_weights is not None:
            model.load_state_dict(best_ft_weights)
            print(f"Restored best finetune weights from epoch {best_ft_epoch} "
                  f"(val loss {best_ft_loss:.4f}).")
        run.log("finetune_summary", {
            "best_epoch": best_ft_epoch,
            "best_val_loss": None if best_ft_loss == float('inf') else best_ft_loss,
            "epochs_run": epoch,
            "val_split": args.finetune_val_split,
            "augment": args.finetune_augment,
            "weight_decay": args.finetune_weight_decay,
        })

        ft_model_path = run.path('finetuned.pth')
        torch.save(model.state_dict(), ft_model_path)
        shutil.copyfile(ft_model_path, os.path.join(args.save_dir,
                                                    f'{args.model_type}_finetuned.pth'))
        print(f"Finetuned model saved to {ft_model_path}")
    else:
        print("\nSkipping finetuning (files not found or finetune-epochs=0).")

    # 5. Evaluate on Test Data (test_x / test_y defined above section 4)
    if os.path.exists(test_x) and os.path.exists(test_y):
        print("\n--- Evaluating on Test Set ---")
        test_dataset = RamanDataset(test_x, test_y,
                                    label_mapping=full_ref_dataset.label_mapping,
                                    require_space=full_ref_dataset.label_space,
                                    **ds_kwargs)
        test_loader = make_loader(test_dataset, args.batch_size * 2, False)
        _, _, test_stats = evaluate(model, device, test_loader, criterion, phase="Test",
                                    num_classes=num_classes, return_stats=True)
        np.save(run.path('test_y_pred.npy'), test_stats.pop('y_pred'))
        np.save(run.path('test_y_true.npy'), test_stats.pop('y_true'))
        run.finish(test=test_stats)
    else:
        print("\nTest data not found, skipping evaluation.")
        run.finish(test=None)


if __name__ == '__main__':
    main()
