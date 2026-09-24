# RAMAN-Effect — Codebase Walkthrough

A guided tour of what this repo is, how the code is laid out, how data flows through it,
and what every model does. Written from a read-only exploration of the repo on branch
`hrishikesh` (last commit `fb58ad3 k fold verification and heatmaps`).

---

## 1. What the project is

Two layers live in one repository:

1. **The research/vision layer** — `README.md` (46 KB), `Papers/SERS_ML_Survey.md`, PDFs.
   A literature-review-style write-up about using Surface-Enhanced Raman Spectroscopy
   (SERS) + Wastewater-Based Epidemiology (WBE) + AI for public-health surveillance.
   It is prose, not code, and contains no implementation.

2. **The working ML layer** — everything else. A PyTorch pipeline that classifies
   **1D Raman spectra of bacteria into 30 species**, using the Ho et al. (2019)
   bacteria-ID dataset. This is the part that actually runs.

The ML layer is a *modernization* of an old Jupyter notebook
(`notebooks/NormalizedModelWithBootstrapping.ipynb`) into a modular PyTorch project
with a CLI, five swappable architectures, and a logged experquiment record.

---

## 2. Directory map

```
RAMAN-Effect/
├── main.py                 # THE entrypoint: argparse CLI → pretrain → finetune → test
├── evaluate_kfold.py       # 5-fold finetune robustness check (legacy_cnn only)
├── generate_heatmaps.py    # trains all 5 models briefly → confusion-matrix PNGs
├── convert.py              # utility: docs/*.md → docs-pdf/*.pdf (markdown + weasyprint)
├── test_mixup.py           # smoke test for the mixup path in engine.train_epoch
│
├── src/                    # the library
│   ├── dataset.py          # RamanDataset (load + ramanspy preprocessing), AugmentedDataset
│   ├── model.py            # Raman1DCNN (ResNet), Transformer1D, MultiscaleCNN, FusionNet
│   ├── engine.py           # train_epoch(), evaluate()  ← the only training primitives
│   └── __init__.py
│
├── legacy_baseline/
│   ├── model.py            # LegacyCNN — Keras notebook CNN ported to PyTorch (the champion)
│   └── train.py            # standalone reproduction of the notebook's 67/33 split run
│
├── data/                   # .npy files (gitignored, must be downloaded)
├── docs/                   # 1..10 numbered narrative docs = the project's lab notebook
├── results/                # *.log training transcripts + heatmaps/*.png + loss_curves.png
├── webapp/app.py           # Streamlit dashboard that parses results/*.log into charts
├── presentations/          # Slidev deck (slides.md) + exported PDF
├── tests/                  # ad-hoc scripts (shape checks, ramanspy dataset demo)
├── notebooks/              # the original notebook + extracted_code.py (historical)
├── Papers/, Fellows/       # research writing, Humanitarians AI program docs
└── Amos/                   # (this file)
```

Note: `tests/` are **not** pytest suites in spirit — they are print-based sanity scripts.
Only `tests/test_multiscale.py` has a real `assert`.

---

## 3. The data

Downloaded manually from the bacteria-ID authors' Dropbox into `data/` (gitignored via
`data/*`). Byte sizes confirm the shapes:

| File | Shape | Role |
|---|---|---|
| `X_reference.npy` / `y_reference.npy` | (60000, 1000) | Pre-training. 30 classes x 2000, perfectly balanced |
| `X_finetune.npy` / `y_finetune.npy`   | (3000, 1000)  | Fine-tuning (domain shift adaptation) |
| `X_test.npy` / `y_test.npy`           | (3000, 1000)  | Held-out evaluation |
| `X_2018clinical.npy` / `y_...`        | (10000, 1000) | **Present but never referenced by any code** |
| `X_2019clinical.npy` / `y_...`        | (2500, 1000)  | **Present but never referenced by any code** |
| `wavenumbers.npy`                     | (1000,)       | x-axis: 381.98 → 1792.4 cm⁻¹ |

Each row is one spectrum: 1000 intensity values across the wavenumber axis — the
"chemical fingerprint" of a single bacterial sample. Labels are species IDs 0–29.

The two `*clinical*` splits are an obvious, already-downloaded extension point: they are
the real-world messy data the docs keep referring to, but no script loads them.

---

## 4. Data flow end to end

```
data/X_*.npy (N, 1000) float64
        │
        ▼  src/dataset.py :: RamanDataset.__init__
  ┌──────────────────────────────────────────────────────────────┐
  │ 1. np.load → float32                                         │
  │ 2. ramanspy.preprocessing.Pipeline (if apply_preprocessing)  │
  │      WhitakerHayes()   despike  (cosmic-ray spikes)          │
  │      SavGol(9, 3)      denoise  (Savitzky–Golay)             │
  │      ASPLS()           baseline (auto-fluorescence hump)     │
  │      MinMax()          normalise → [0, 1]                    │
  │ 3. np.expand_dims(axis=1) → (N, 1, 1000)  # C=1 channel      │
  │ 4. label remap: sorted unique labels → 0..C-1                │
  │    (label_mapping passed in from the reference set so         │
  │     finetune/test share identical class indices;              │
  │     unseen labels are dropped with a warning)                 │
  └──────────────────────────────────────────────────────────────┘
        │
        ▼  src/dataset.py :: AugmentedDataset  (train split only, if --augment)
  roll(±5 idx)  +  N(0, 0.01) noise  x  uniform(0.95, 1.05) scale
        │
        ▼  torch DataLoader (batch, shuffle)
        ▼  model(x): (B, 1, 1000) → (B, 30) logits
        ▼  src/engine.py :: train_epoch / evaluate
```

The three augmentations are deliberately *physics-motivated*, which is why they help:
roll ≈ spectrometer miscalibration, noise ≈ sensor/shot noise, scale ≈ laser power and
sample concentration drift.

**Important cost note:** preprocessing runs eagerly in `__init__`, on the full 60,000-sample
array, every time a script starts. There is no cache. That is why loading takes a while.

---

## 5. The training engine (`src/engine.py`)

Tiny and deliberately generic — two functions, no classes, no framework:

- **`train_epoch(model, device, loader, optimizer, criterion, epoch, phase, mixup_alpha=0.0)`**
  Standard loop. If `mixup_alpha > 0`, it does mixup: sample `lam ~ Beta(a, a)`, force
  `lam = max(lam, 1-lam)`, permute the batch, blend inputs `lam*x + (1-lam)*x[perm]`, and
  use the blended loss `lam*CE(out, y_a) + (1-lam)*CE(out, y_b)`. Accuracy is tracked
  against the dominant target only. Prints every 100 batches — those prints are exactly
  what `webapp/app.py` later parses.

- **`evaluate(model, device, loader, criterion, phase)`** → `(loss, acc)`.
  It accumulates `loss * data.size(0)` then divides by dataset length, so the reported
  loss is a true per-sample mean (correct even with a ragged last batch).

Because the engine is architecture-agnostic, adding a model means only touching
`src/model.py` + the dispatch block in `main.py`.

---

## 6. The models

All five take `(B, 1, 1000)` and return `(B, num_classes)` logits (no softmax — the
`nn.CrossEntropyLoss` applies it).

### 6.1 `LegacyCNN` — `legacy_baseline/model.py` — the champion (76.50%)
Port of the Keras notebook model. 6 blocks of
`Conv1d(k=3, padding=1) → ReLU → BatchNorm → AvgPool1d(3, stride=3)`,
channels `1→8→16→32→32→64→128`, then `AdaptiveAvgPool1d(1) → Dropout(0.3) → Linear(128, C)`.

Two details worth noticing:
- The order is Conv → **ReLU → BN** (not the modern Conv → BN → ReLU) because it faithfully
  mirrors how the Keras notebook stacked layers. The comment in the file says so explicitly.
- `AvgPool(3, stride=3)` five times is aggressive downsampling: 1000 → 333 → 111 → 37 → 12 → 4.
  That bottleneck is precisely the structural regularizer the docs credit for its win.

### 6.2 `Raman1DCNN` — a 1D ResNet-18-lite (69.90%)
`Conv1d(1→16, k=7, s=2) → BN → ReLU → MaxPool` stem, then four stages of two
`ResidualBlock1D` each with channels `16→32→64→128` and stride-2 at each new stage;
`AdaptiveAvgPool1d(1) → Linear(128,128) → ReLU → Dropout(0.4) → Linear(128, C)`.
`ResidualBlock1D` is the classic two-conv block with a 1x1-conv+BN shortcut whenever
shape changes. Despite being the "advanced" model, it overfits this dataset.

### 6.3 `Transformer1D` — ViT-style, patched spectra (54.73%)
Reshapes `(B,1,1000)` into 10 patches of 100 points, linearly embeds each to `d_model=64`,
prepends a learned CLS token, adds a learned positional embedding, runs 3
`TransformerEncoderLayer`s (4 heads, FFN 128, dropout 0.1), classifies from the CLS token
via `LayerNorm → Linear`. Note `seq_len=1000` and `patch_size=100` are **hardcoded defaults**
guarded by an assert — this model silently assumes 1000-point spectra.

### 6.4 `MultiscaleCNN` — Inception-style (63.73%)
Stem identical to the ResNet's, then three `MultiscaleBlock1D`s with channels 64→128→256.
Each block runs four parallel branches — bare 1x1, and 1x1→{3, 7, 11} convs — each producing
`out_channels//4` channels, concatenated back together. The motivation: sharp narrow Raman
peaks and broad humps need different receptive fields simultaneously.

### 6.5 `FusionNet` — CNN front-end + Transformer back-end (60.87%; 55.80% with mixup)
CNN compresses 1000 → 50 tokens of width `d_model=64`
(`k7/s2` → 500, `MaxPool(5)` → 100, `k3` conv, `MaxPool(2)` → 50), then permutes to
`(B, 50, 64)`, adds CLS + positional embedding, runs 2 encoder layers (dropout 0.2), and
classifies from CLS. The idea is to give the attention stack the local inductive bias it lacks.
Like `Transformer1D`, the token count (50) is hardcoded to the 1000-point input.

---

## 7. The entrypoint (`main.py`)

One linear script, five steps:

1. **Device pick** — `cuda` → `mps` → `cpu`.
2. **Load reference data**; `num_classes` inferred from `np.unique(y)`.
   Optional `--val-split` uses `random_split` with a fixed seed 42, and wraps the train half
   in `AugmentedDataset(augment=args.augment)` while the val half gets `augment=False`
   (correct — no augmentation leaks into validation).
3. **Model dispatch** on `--model-type` ∈ {`resnet`, `legacy_cnn`, `transformer`,
   `multiscale_cnn`, `fusion`} (imports are lazy, inside the branches).
4. **Phase 1 — pre-train** on reference data: Adam(`--lr` 1e-3, `--weight-decay` 0.01) +
   `CosineAnnealingLR(T_max=epochs)`. If a val split exists, it does best-checkpoint saving
   to `models/raman_reference_model.pth` with early stopping (`patience = 10`, hardcoded),
   then reloads the best weights before finetuning. With no val split, it just saves the
   final epoch.
5. **Phase 2 — finetune** on `X_finetune.npy` with the reference set's `label_mapping`,
   at **lr x 0.1** and its own cosine schedule, saving `models/raman_finetuned_model.pth`.
   Skipped if the files are missing or `--finetune-epochs 0`.
6. **Evaluate** on `X_test.npy` and print loss/accuracy.

Typical invocation, reconstructed from the docs and log filenames:

```bash
python main.py --model-type legacy_cnn --epochs 100 --finetune-epochs 20 \
               --val-split 0.1 --augment
```

Flags: `--batch-size`, `--epochs`, `--finetune-epochs`, `--lr`, `--weight-decay`,
`--data-dir`, `--save-dir`, `--val-split`, `--augment`, `--mixup-alpha`, `--model-type`.

Quirks worth knowing:
- `patience=10` and the checkpoint filenames are not CLI-exposed.
- Checkpoint names don't include the model type, so runs of different architectures
  overwrite each other's `models/raman_reference_model.pth`.
- `--mixup-alpha` is applied in both pre-train and finetune phases.

---

## 8. The supporting scripts

**`evaluate_kfold.py`** — answers "is the 76% real or a lucky split?"
Phase 1 pre-trains `legacy_cnn` once (fixed 10% val, seed 42, early stopping) and saves
`models/kfold_reference_model.pth`. Phase 2 reloads those frozen weights **five times**, each
time re-running `random_split` on the finetune set (90/10, *unseeded*, so genuinely different
folds), finetunes at lr x 0.1, keeps the best-val-loss weights, and evaluates on the test set.
Result in `results/kfold_legacy_cnn.log`: **73.23% ± 0.11%** — remarkably stable.

Two honest caveats: it is not true k-fold (random 90/10 draws can overlap across "folds"),
and `best_ft_weights = model.state_dict()` stores a *reference*, not a deep copy, so the
"best" weights track the live model. It only supports `legacy_cnn` and raises otherwise.

**`generate_heatmaps.py`** — trains all five models for 5 epochs **on the finetune set only**
(no pre-training) and writes `results/heatmaps/confusion_matrix_<model>.png` via
sklearn's `ConfusionMatrixDisplay`. These matrices are diagnostic-only, not the headline numbers.

**`legacy_baseline/train.py`** — apples-to-apples reproduction of the notebook: same
`RamanDataset` preprocessing, 67/33 split (seed 42), batch 128, Adam lr=2e-4 wd=0.01, 10 epochs
(notebook used 200). This produced the 78.70% figure in `docs/4_...` — note that is on a
*validation split of the reference set*, not on `X_test.npy`, which is why it isn't comparable
to the 76.50% headline.

**`webapp/app.py`** — Streamlit + Plotly dashboard. It regex-parses `results/*.log` for
`Epoch: N ... Loss: L ... Acc: A%`, keeps the last value per epoch, and stitches phases
together by detecting when the epoch counter resets (adding an offset) so pre-train and
finetune appear as one continuous x-axis. Multiselect to overlay runs. Run with
`streamlit run webapp/app.py`.

**`convert.py`** — renders `docs/*.md` to styled PDFs in `docs-pdf/` (gitignored).

---

## 9. Results: what the experiments actually found

From `docs/10_EXPERIMENT_SUMMARY.md` (100 pre-train + 20 finetune epochs, test-set accuracy):

| Rank | Model | Config | Test Acc |
|---|---|---|---|
| 1 | **LegacyCNN** | dropout 0.3, augment | **76.50%** |
| 2 | LegacyCNN | no augment | 75.10% |
| 3 | Raman1DCNN (ResNet) | dropout 0.4, augment | 69.90% |
| 4 | Raman1DCNN | no augment | 67.47% |
| 5 | MultiscaleCNN | augment | 63.73% |
| 6 | FusionNet | augment | 60.87% |
| 7 | Transformer1D | no augment | 57.83% |
| 8 | FusionNet | + mixup α=0.2 | 55.80% |
| 9 | Transformer1D | augment | 54.73% |
| — | MultiscaleCNN | weight_decay 0.05 | 34.17% (choked) |

The through-line of the whole project: **capacity hurts here.** A 1000-point spectrum with
60k samples and 30 classes is a small, structurally simple problem. The shallow LegacyCNN's
aggressive pooling is a better prior than residual depth, parallel branches, or attention.
Physics-aware augmentation helps (+1.4 pts); mixup hurts, because linearly blending two
spectra produces a chemically meaningless spectrum.

Also note the consistent train-vs-test gap: LegacyCNN hits ~84% validation on the reference
set but 76.5% on `X_test.npy`. That gap is **domain shift**, not just overfitting — the test
data comes from different measurement conditions, which is the whole reason the finetune
phase exists.

---

## 10. How the pieces relate (dependency view)

```
                    ┌──────────────┐
                    │ src/engine.py│ ← train_epoch, evaluate (no model/data deps)
                    └──────┬───────┘
                           │ used by
   ┌───────────────┬───────┴────────┬──────────────────┬─────────────────┐
   │               │                │                  │                 │
main.py    evaluate_kfold.py  generate_heatmaps.py  legacy_baseline/  test_mixup.py
   │               │                │                 train.py
   └───────┬───────┴────────────────┘                     │
           │ both need                                    │
   ┌───────┴────────┐   ┌──────────────┐   ┌──────────────┴───────┐
   │ src/dataset.py │   │ src/model.py │   │ legacy_baseline/model│
   │  (→ ramanspy)  │   │  4 models    │   │   LegacyCNN          │
   └────────────────┘   └──────────────┘   └──────────────────────┘
           │
      data/*.npy  (gitignored)
           │
   results/*.log ──→ webapp/app.py (Streamlit dashboard)
                └──→ docs/*.md ──→ convert.py ──→ docs-pdf/
```

`src/engine.py` sits at the bottom with zero project dependencies, which is what makes the
model zoo cheap to extend.

---

## 11. Getting it running

```bash
python -m venv venv && venv\Scripts\activate         # Windows
pip install -r requirements.txt                      # note: torch is unpinned
# download the bacteria-ID .npy files into data/
python main.py --model-type legacy_cnn --epochs 100 --finetune-epochs 20 \
               --val-split 0.1 --augment
streamlit run webapp/app.py                          # view the logs
```

`docs/1_SETUP.md` also documents a manual patch to installed `ramanspy`
(`plt.cm.get_cmap()` → `plt.get_cmap()` in `ramanspy/plot/_core.py`) needed only for plotting.
`requirements.txt` looks auto-frozen (it carries transitive deps like `fonttools`, `kiwisolver`)
and is missing `streamlit`, `plotly`, `markdown`, and `weasyprint`, which live in
`webapp/requirements.txt` or nowhere.

The empty `bacteria-ID/` directory is presumably where the original authors' repo was
meant to be cloned.

---

## 12. Where the seams are (if you want to extend it)

- **The clinical splits are unused.** `X_2018clinical` (10k) and `X_2019clinical` (2.5k) are
  already on disk. Evaluating the champion on them is the most obvious next experiment and
  the most faithful to the project's wastewater-surveillance framing.
- **Preprocessing has no cache.** Running the ramanspy pipeline on 60k spectra at every script
  start dominates iteration time. Saving the processed arrays once would pay for itself.
- **Checkpoints collide.** `models/raman_reference_model.pth` is shared across architectures.
- **k-fold isn't quite k-fold.** `evaluate_kfold.py` uses repeated random splits, and only
  supports `legacy_cnn`.
- **Transformer input size is hardcoded.** `Transformer1D` and `FusionNet` assume 1000 points.
- **`tests/` are print scripts.** A real pytest suite over shapes and the label-mapping logic
  would be cheap insurance.
- **No per-class metrics in the main path.** `main.py` reports only loss/accuracy. With 30
  classes and known confusions, per-class precision/recall or a macro-F1 would say more —
  the confusion matrices already exist but are generated by a separate, under-trained script.
