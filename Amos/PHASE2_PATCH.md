# Phase 2 patch: instrument and regularise fine-tuning

Fine-tuning is worth about 27 accuracy points and is currently the least
supervised part of the pipeline: 20 epochs on 100 samples per class, no
validation, no augmentation, no weight decay, and whatever the last epoch
produced gets saved.

This patch makes each of those a flag. **Every default preserves current
behaviour**, so `p2_base` reproduces what you already measured and the
comparison is clean.

Expect smaller gains than the ASPLS result. You are at 83.89% against a
published ceiling of 86.3%. One to three points is the realistic range, and
part of the value is stability rather than the mean.

---

## Edit 1: `src/dataset.py`, add a stratified split helper

Append at the end of the file:

```python
def stratified_split_indices(y, val_fraction, seed):
    """Class-balanced train/val index split.

    Used for the fine-tune validation split. The finetune set is 100 spectra
    per class, so a random 10% split would give some classes 5 validation
    samples and others 15. Stratifying gives every class exactly 10.

    Caveat worth remembering when reading the numbers: 10 samples per class is
    a small validation set. Per-class recall from it moves in steps of 0.1, and
    overall validation accuracy carries roughly +/-2 points of binomial noise
    at n=300. It is good enough to pick a better epoch than "the last one",
    and not good enough to tune hyperparameters on.
    """
    import numpy as np
    from sklearn.model_selection import StratifiedShuffleSplit

    y = np.asarray(y)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction,
                                 random_state=seed)
    train_idx, val_idx = next(sss.split(np.zeros(len(y)), y))
    return train_idx, val_idx
```

---

## Edit 2: `main.py`, imports

**Find:**
```python
import argparse
import os
import shutil
```

**Replace with:**
```python
import argparse
import copy
import os
import shutil
```

**Find:**
```python
from src.dataset import PIPELINE_VARIANTS, AugmentedDataset, RamanDataset
```

**Replace with:**
```python
from torch.utils.data import Subset

from src.dataset import (PIPELINE_VARIANTS, AugmentedDataset, RamanDataset,
                         stratified_split_indices)
```

---

## Edit 3: `main.py`, new arguments

**Find** the `--pipeline` argument and **insert below it:**

```python
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
```

---

## Edit 4: `main.py`, replace the whole fine-tuning block

**Find** the block beginning `if os.path.exists(finetune_x) and os.path.exists(finetune_y)` and ending with `print(f"Finetuned model saved to {ft_model_path}")`, and **replace with:**

```python
    if os.path.exists(finetune_x) and os.path.exists(finetune_y) and args.finetune_epochs > 0:
        print("\n--- Starting Finetuning ---")
        finetune_dataset = RamanDataset(finetune_x, finetune_y,
                                        label_mapping=full_ref_dataset.label_mapping,
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
```

One ordering dependency: this block references `test_x` and `test_y`, which are
currently defined **after** it under `# 5. Evaluate on Test Data`. Move those two
lines up so they sit just above `# 4. Finetune Data`:

```python
    test_x = os.path.join(args.data_dir, 'X_test.npy')
    test_y = os.path.join(args.data_dir, 'y_test.npy')
```

and delete them from section 5. Without this you get the same `NameError` as
the `ds_kwargs` mistake.

---

## Verify

```powershell
python main.py --model-type legacy_cnn --epochs 1 --finetune-epochs 3 --val-split 0.33 --seed 99 --finetune-val-split 0.1 --finetune-augment --finetune-weight-decay 0.01 --diagnostic-test-curve --tag p2check
```

Four things to see:
- `Split finetune data: 2700 train / 300 val (10 per class)`
- a `Finetune-Val set:` block after each finetune epoch
- a `DIAGNOSTIC-Test set:` block after each finetune epoch
- `Restored best finetune weights from epoch N`

Then confirm nothing regressed with the old flags:

```powershell
python main.py --model-type legacy_cnn --epochs 1 --finetune-epochs 3 --val-split 0.33 --seed 99 --tag p2base
```

No split line, no validation blocks, no restore line. That is the baseline arm
and it must behave exactly as before.

---

## Then run the comparison

```powershell
python src\tools\run_phase2.py --dry-run
python src\tools\run_phase2.py
```
