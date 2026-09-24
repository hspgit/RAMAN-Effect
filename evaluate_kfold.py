import argparse
import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset, random_split

from src.dataset import AugmentedDataset, RamanDataset
from src.engine import evaluate, train_epoch
from src.repro import make_generator, pick_device, seed_worker, set_seed
from src.runmeta import RunMeta


def main():
    parser = argparse.ArgumentParser(description='5-Fold Finetune Evaluation')
    parser.add_argument('--model-type', type=str, default='legacy_cnn')
    parser.add_argument('--pretrain-epochs', type=int, default=100)
    parser.add_argument('--finetune-epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--strict-determinism', action='store_true')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'mps', 'cpu'])
    parser.add_argument('--run-root', type=str, default='runs')
    parser.add_argument('--tag', type=str, default='kfold')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--no-cache', action='store_true')
    args = parser.parse_args()

    set_seed(args.seed, strict=args.strict_determinism)
    device = pick_device(args.device)
    print(f"Using device: {device}")

    run = RunMeta(args, device, run_root=args.run_root, tag=args.tag)
    print(f"Run id: {run.run_id}")

    def make_loader(ds, batch_size, shuffle):
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker if args.num_workers > 0 else None,
            generator=make_generator(args.seed) if shuffle else None,
            persistent_workers=args.num_workers > 0,
        )

    ds_kwargs = dict(use_cache=not args.no_cache)

    # 1. Load Reference Data for Pre-training
    print("\nLoading reference dataset...")
    full_ref_dataset = RamanDataset(os.path.join(args.data_dir, 'X_reference.npy'),
                                    os.path.join(args.data_dir, 'y_reference.npy'),
                                    **ds_kwargs)
    num_classes = len(np.unique(full_ref_dataset.y))

    val_size = int(0.1 * len(full_ref_dataset))
    train_size = len(full_ref_dataset) - val_size
    ref_train, ref_val = random_split(full_ref_dataset, [train_size, val_size],
                                      generator=make_generator(args.seed))

    ref_train_wrapped = AugmentedDataset(ref_train, augment=args.augment)
    ref_val_wrapped = AugmentedDataset(ref_val, augment=False)

    ref_train_loader = make_loader(ref_train_wrapped, args.batch_size, True)
    ref_val_loader = make_loader(ref_val_wrapped, args.batch_size * 2, False)

    def get_model():
        if args.model_type == 'legacy_cnn':
            from legacy_baseline.model import LegacyCNN
            return LegacyCNN(num_classes=num_classes).to(device)
        else:
            raise ValueError("Only legacy_cnn supported in this quick script")

    model = get_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.pretrain_epochs)

    print("\n--- Phase 1: Pre-training (Single Run) ---")
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    ref_model_path = run.path('kfold_reference.pth')

    for epoch in range(1, args.pretrain_epochs + 1):
        train_stats = train_epoch(model, device, ref_train_loader, optimizer, criterion,
                                  epoch, phase="Pre-train")
        scheduler.step()

        val_loss, val_acc = evaluate(model, device, ref_val_loader, criterion,
                                     phase="Validation", num_classes=num_classes)
        run.log_epoch("pretrain", epoch, val_loss=val_loss, val_acc=val_acc, **train_stats)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), ref_model_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"Pre-training complete. Best model saved to {ref_model_path}")

    # 2. Stratified K-Fold Finetuning
    print("\n--- Phase 2: Stratified K-Fold Finetuning ---")
    finetune_dataset = RamanDataset(os.path.join(args.data_dir, 'X_finetune.npy'),
                                    os.path.join(args.data_dir, 'y_finetune.npy'),
                                    label_mapping=full_ref_dataset.label_mapping,
                                    require_space=full_ref_dataset.label_space,
                                    **ds_kwargs)
    test_dataset = RamanDataset(os.path.join(args.data_dir, 'X_test.npy'),
                                os.path.join(args.data_dir, 'y_test.npy'),
                                label_mapping=full_ref_dataset.label_mapping,
                                require_space=full_ref_dataset.label_space,
                                **ds_kwargs)
    test_loader = make_loader(test_dataset, args.batch_size * 2, False)

    # The original drew five independent random 90/10 splits, which are not
    # folds: they overlap arbitrarily, so the reported spread understates real
    # variance. StratifiedKFold gives disjoint, class-balanced partitions.
    y_ft = np.asarray(finetune_dataset.y)
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    test_accuracies = []
    fold_records = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y_ft)), y_ft), start=1):
        print(f"\n>>> Starting Fold {fold}/{args.n_splits} <<<")
        # Load fresh pre-trained weights
        model = get_model()
        model.load_state_dict(torch.load(ref_model_path, weights_only=True))

        ft_train = Subset(finetune_dataset, tr_idx.tolist())
        ft_val = Subset(finetune_dataset, va_idx.tolist())
        ft_train_loader = make_loader(ft_train, args.batch_size, True)
        ft_val_loader = make_loader(ft_val, args.batch_size * 2, False)

        optimizer_ft = optim.Adam(model.parameters(), lr=args.lr * 0.1)
        scheduler_ft = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft,
                                                            T_max=args.finetune_epochs)

        best_ft_val_loss = float('inf')
        best_ft_weights = None
        best_ft_epoch = None

        for epoch in range(1, args.finetune_epochs + 1):
            train_epoch(model, device, ft_train_loader, optimizer_ft, criterion, epoch,
                        phase=f"Fold {fold} Finetune")
            scheduler_ft.step()
            val_loss, val_acc = evaluate(model, device, ft_val_loader, criterion,
                                         phase=f"Fold {fold} Val", num_classes=num_classes)
            run.log_epoch(f"fold{fold}", epoch, val_loss=val_loss, val_acc=val_acc)

            if val_loss < best_ft_val_loss:
                best_ft_val_loss = val_loss
                # state_dict() returns live tensor references, so the original
                # code's "best" weights were mutated in place by every later
                # epoch and the reload below was a no-op. deepcopy fixes it.
                best_ft_weights = copy.deepcopy(model.state_dict())
                best_ft_epoch = epoch

        # Load best finetuned weights and evaluate on test set
        model.load_state_dict(best_ft_weights)
        print(f"\nEvaluating Fold {fold} on Test Set (best epoch {best_ft_epoch})...")
        _, test_acc, test_stats = evaluate(model, device, test_loader, criterion,
                                           phase=f"Fold {fold} Test",
                                           num_classes=num_classes, return_stats=True)
        test_accuracies.append(test_acc)
        fold_records.append({
            "fold": fold, "best_epoch": best_ft_epoch,
            "best_val_loss": best_ft_val_loss,
            "test_acc": test_acc, "test_macro_f1": test_stats["macro_f1"],
            "test_loss": test_stats["loss"],
            "n_train": len(tr_idx), "n_val": len(va_idx),
        })
        run.log("folds", fold_records)

    # 3. Final Results
    mean_acc = float(np.mean(test_accuracies))
    std_acc = float(np.std(test_accuracies))
    mean_f1 = float(np.mean([r["test_macro_f1"] for r in fold_records]))
    print("\n" + "=" * 40)
    print("FINAL K-FOLD CROSS VALIDATION RESULTS")
    print("=" * 40)
    for i, acc in enumerate(test_accuracies):
        print(f"Fold {i+1} Test Accuracy: {acc:.2f}%")
    print("-" * 40)
    print(f"Average Test Accuracy: {mean_acc:.2f}% +/- {std_acc:.2f}%")
    print(f"Average Test Macro-F1: {mean_f1:.4f}")
    print("NOTE: phase 1 is shared across folds, so this measures finetune")
    print("      variance only, not end-to-end variance.")
    print("=" * 40 + "\n")

    run.finish(kfold={"mean_acc": mean_acc, "std_acc": std_acc,
                      "mean_macro_f1": mean_f1, "folds": fold_records})


if __name__ == '__main__':
    main()
