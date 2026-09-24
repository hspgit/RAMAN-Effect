# Stage 0: make results measurable

Written against your actual files, Python 3.12.10, torch 2.11.0+cu128 (CUDA available),
numpy 2.5.2, ramanspy 0.2.10, scikit-learn 1.9.0, Windows PowerShell.

Stage 0 produces **no better model**. It makes Stage 1 and Stage 2 interpretable.
Two of these changes will make recorded numbers look slightly worse. That is the
fix working, not a regression.

---

## Install

New files, no conflicts:

```
src/repro.py
src/runmeta.py
src/cache.py
src/metrics.py
tools/compare_runs.py
```

Replacements. Back up first:

```powershell
mkdir backup_pre_stage0
copy main.py, evaluate_kfold.py backup_pre_stage0\
copy src\dataset.py, src\engine.py backup_pre_stage0\
```

```
main.py              <- replaces yours
evaluate_kfold.py    <- replaces yours
src/dataset.py       <- replaces yours
src/engine.py        <- replaces yours
```

Add to `requirements.txt` (both already installed, just undeclared):

```
streamlit
plotly
```

Add to `.gitignore`:

```
runs/
data/.cache/
```

`src/model.py`, `legacy_baseline/`, `generate_heatmaps.py` and the notebooks are
untouched. `generate_heatmaps.py` keeps working because it builds its own label
mapping from `y_reference.npy`, which contains all 30 classes, so the new
unseen-label guard never fires there.

---

## Corrections to RESEARCH_REFERENCE.md found while doing this

1. **§0.3 is stale.** You are on `torch 2.11.0+cu128` with `cuda_available=True`.
   The CPU-only wheel problem is already fixed. Tier-1 item #1 is done, and the
   acceptance tests below take minutes, not hours.
2. **`ramanspy.__version__` does not exist** in 0.2.10. `src/repro.py` reads the
   version from `importlib.metadata` instead; `rp.__version__` raises
   `AttributeError`.
3. **PowerShell has no `&&`.** Use `;` to chain. The doc's shell snippets are
   bash.

---

## What changed, and why

### `src/repro.py`, `src/runmeta.py` (new)

`set_seed` seeds python, numpy, torch and CUDA, and disables cuDNN autotuning.
Called as the first statement in both entry points, before any dataset, model or
CUDA tensor exists, because `CUBLAS_WORKSPACE_CONFIG` has no effect after CUDA
initialises.

`pick_device('auto')` reproduces your original cuda to mps to cpu order exactly.
`--device cuda` now **errors** if CUDA is unavailable rather than falling back,
because a silent fallback is how a bad wheel goes unnoticed.

Every run writes `runs/<timestamp>_<model>_<tag>_<hash>/manifest.json` with all
arguments, the git SHA, whether the tree was dirty, the device, and every
relevant library version. This replaces reconstructing batch size by counting
print strides.

### `src/cache.py` + `src/dataset.py`

The four preprocessing steps are now a declarative tuple, `DEFAULT_PIPELINE`,
and `build_pipeline()` constructs the real ramanspy `Pipeline` from that same
tuple. The cache key includes the tuple, so spec and code cannot diverge and a
Stage 1 ablation cannot collide with the baseline's cached array.

First run computes and stores; later runs with the same config load a `.npy`.
Cached arrays stay `(N, 1000)`; `expand_dims` happens after the cache so the
stored file is inspectable.

### `src/dataset.py` unseen-label guard

`on_unseen='error'` is now the default. Previously a label absent from
`label_mapping` was dropped with only a count printed, so feeding in a new class
produced a successful run with a plausible accuracy and the class silently gone.
The error message now names the offending label values and their counts.

This also means pointing the loader at the clinical files against the reference
mapping raises instead of producing a meaningless number. Those files carry
antibiotic-level labels, not isolate labels.

### `src/engine.py`

`evaluate()` now computes macro-F1, macro-recall, per-class recall and precision,
support, and the full confusion matrix. Returns `(loss, acc)` as before;
`return_stats=True` adds the dict. The `Average loss:` line is byte-identical so
`webapp/app.py`'s regex still parses.

`train_epoch()` returns a stats dict instead of `None`. Printed output unchanged.

### `evaluate_kfold.py`

Two fixes. `copy.deepcopy(model.state_dict())` means the "best" weights are
actually kept; previously `state_dict()` handed back live tensor references that
later epochs mutated in place, making the reload a no-op and **the recorded
73.23% a last-epoch number**. And `StratifiedKFold` replaces five independent
random 90/10 draws, which were not folds at all.

Expect the +/- to widen. The old +/-0.11% was measuring overlapping splits.

---

## Behaviour changes to expect

| Change | Effect |
|---|---|
| Explicit generator in `random_split` and loaders | the split and batch order differ from before even at seed 42, because they no longer consume global RNG in the same order. Unavoidable, and harmless since the old numbers were not reproducible under current code anyway. |
| `on_unseen='error'` | anything feeding mismatched labels now raises |
| `deepcopy` fix | `evaluate_kfold` reports best-val, not last-epoch. Could move either way. |
| `StratifiedKFold` | wider, honest error bars |
| Checkpoints | now `runs/<id>/reference.pth`, mirrored to `models/<model_type>_reference.pth`. The shared `models/raman_reference_model.pth` is no longer written. Nothing in the repo reads it. |

---

## Acceptance tests

Stage 0 is done when these pass. Run from the repo root.

### 1. Cache works

```powershell
python main.py --model-type legacy_cnn --epochs 1 --finetune-epochs 1 --val-split 0.33 --seed 7 --tag cache-warm
```

First run prints `[cache] miss` three times, then `[cache] stored`. Run it again:
should print `[cache] hit` three times and start noticeably faster.

### 2. Determinism

```powershell
python main.py --model-type legacy_cnn --epochs 3 --finetune-epochs 2 --val-split 0.33 --augment --seed 7 --tag det-a
python main.py --model-type legacy_cnn --epochs 3 --finetune-epochs 2 --val-split 0.33 --augment --seed 7 --tag det-b
python tools\compare_runs.py --tag det-
```

Both rows must show identical `acc`, `macro_f1` and `loss`. The tool says so
explicitly. If they differ, something is consuming unseeded randomness and
everything downstream is unreliable, so stop and find it.

### 3. Seed-noise floor, the actual point of Stage 0

```powershell
foreach ($s in 1,2,3) { python main.py --model-type legacy_cnn --epochs 3 --finetune-epochs 2 --val-split 0.33 --augment --seed $s --tag noise3 }
python tools\compare_runs.py --tag noise3
```

The printed spread is your detection threshold. Then repeat once at the real
config, which now costs minutes on the 3060:

```powershell
foreach ($s in 1,2,3) { python main.py --model-type legacy_cnn --epochs 100 --finetune-epochs 20 --val-split 0.33 --augment --seed $s --tag noise100 }
python tools\compare_runs.py --tag noise100
```

**If that spread is above about 1.4 points, the augmentation effect in §9.3d of
your reference doc is noise and should stop being cited.** Variance at 3 epochs
is not variance at 100, which is why both runs are needed.

### 4. Unseen labels raise

```powershell
python -c "from src.dataset import RamanDataset; r=RamanDataset('data/X_reference.npy','data/y_reference.npy'); RamanDataset('data/X_2018clinical.npy','data/y_2018clinical.npy',label_mapping=r.label_mapping)"
```

Must raise `ValueError` naming labels `{0.0, 2.0, 3.0, 5.0, 6.0}` with counts of
2000 each. If it prints a warning and continues, the guard did not get applied.

### 5. Per-class metrics appear

Any run's output should now show a `worst-6 recall` line after the accuracy line,
and `runs/<id>/metrics.json` should contain `test.per_class_recall` (30 entries)
and `test.confusion` (30x30).

---

## Capturing a log

```powershell
python main.py --model-type legacy_cnn --epochs 100 --finetune-epochs 20 --val-split 0.33 --augment --seed 1 --tag base 2>&1 | Tee-Object -FilePath results\legacy_cnn_rtx3060_seed1.log
```

The run directory already holds the structured version; the `.log` is only for
`webapp/app.py`.

---

## Not in Stage 0, deliberately

- No architecture changes.
- No training-strategy changes. Fine-tuning still gets no augmentation, no
  weight decay and no validation. Those are Stage 2, and changing them now would
  contaminate the noise-floor measurement.
- No preprocessing behaviour change. `DEFAULT_PIPELINE` is byte-equivalent to
  your original hardcoded pipeline. It is now a command-line-reachable object,
  which is what makes Stage 1 a flag rather than a source edit.

When Stage 0 passes, Stage 1 is:

```powershell
python -c "from src.dataset import RamanDataset" ; # then pass pipeline_spec=() from a small ablation script
```

I will write that script once you confirm these tests pass, since it depends on
the noise floor you measure.
