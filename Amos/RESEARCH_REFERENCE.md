# RAMAN-Effect — Exhaustive Research Reference

Complete technical reference for the RAMAN-Effect ML pipeline: every architecture, every CLI
argument, every recorded result, every log and image, and every caveat that affects how you
should interpret the numbers.

**Compiled:** 2026-09-17 · **Branch:** `hrishikesh` · **HEAD:** `fb58ad3` ("k fold verification and heatmaps")
**Companion doc:** `Amos/CODEBASE_WALKTHROUGH.md` (narrative overview — read that first if you're new)

Everything below is either **measured** (I ran it / parsed it) or **inferred** (derived from
log evidence). Inferences are labelled. Where a number in `docs/` disagrees with the logs,
both are shown.

---

## 0. Read this first — five caveats that change how you read every number

These are not nitpicks. Each one invalidates a naive comparison.

### 0.1 The committed logs were produced by a *different version* of `main.py`

`main.py` was last changed in commit `f4e1e29` (2026-09-15) — the same commit that added 11 of
the 18 logs. The diff shows that commit **introduced** per-epoch validation, best-checkpoint
saving, and early stopping. Before it, the code validated **once after all epochs** and saved
the **final-epoch** weights.

You can tell which version produced a log by counting `Validation set:` lines:

| Validation lines | Code version | Logs |
|---|---|---|
| **1** (at the very end) | **OLD** (pre-`f4e1e29`) — no early stopping, saves final epoch | 14 logs, **including every headline result** |
| **N = one per epoch** | **NEW** (current `main.py`) | `early_stopping_test` (27), `fusion_exp1` (55), `fusion_mixup` (69), `kfold_legacy_cnn` (18) |

Corroborating evidence in the old logs: they print `Reference model saved to models/...`, which
in the current code is the `val_loader is None` branch — yet they *also* print
`Split reference data: 40200 train / 19800 val`. Only the old code could print both.

> **Consequence: the 76.50% champion result is not reproducible by running the current
> `main.py` with the same flags.** Today's code would early-stop and reload best-val weights,
> producing a different training trajectory. If you want to reproduce or beat 76.50%, you must
> either check out `1cffee8`-era `main.py`, or re-run the whole grid on current code and treat
> the old numbers as historical only.

### 0.2 All logs came from someone else's Mac

Every log starts with `Using device: mps` (Apple Metal). `legacy_cnn_pipeline.log:1` contains
the literal path `/Users/hrishikesh/Developer/RAMAN-Effect/.venv/lib/python3.13/...`. All were
committed by Hrishikesh Pradhan. Nothing in this repo writes a log file — the only `.log`
reference in the entire codebase is `webapp/app.py:53`, which *reads* them. Logs exist only
because output was manually redirected.

To capture your own (PowerShell):
```powershell
python main.py --model-type legacy_cnn --epochs 100 --finetune-epochs 20 --val-split 0.33 --augment 2>&1 | Tee-Object -FilePath results/legacy_cnn_rtx3060.log
```

### 0.3 Your PyTorch is CPU-only

Measured on this machine: `torch 2.14.0+cpu`, `torch.version.cuda = None`,
`torch.cuda.is_available() = False`. The RTX 3060 (12 GB, driver 591.86, CUDA 13.1) is fine —
the wheel is the problem, because `requirements.txt:26` is a bare `torch` and PyPI's default
Windows wheel ships without CUDA. Fix before running any experiment:

```bash
venv/Scripts/python.exe -m pip uninstall -y torch
venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

### 0.4 Two headline numbers in `docs/` are not test-set numbers

- `docs/4_LEGACY_BASELINE_SUMMARY.md` reports **78.70%**. That is a 33% *validation split of
  the reference set* from `legacy_baseline/train.py`, not `X_test.npy`. It is not comparable to
  the 76.50%.
- `docs/2_DATA_EXPLORATION.md` says the wavenumber range is "381.98 to 1792.4 cm⁻¹". Measured,
  the array is **descending**: `w[0]=1792.40`, `w[1]=1791.20`, `w[-1]=381.98`, step `-1.2`.
  The range is right; the stated direction is backwards. Matters if you index by wavenumber.

### 0.5 The clinical splits do not contain 30 classes

Measured — this is the single most important finding for planning new work. See §3.3.

---

## 1. What the pipeline does

Classifies a **1000-point 1D Raman spectrum** of a bacterial sample into one of **30 species**.
Dataset is the Ho et al. (2019) bacteria-ID set. Training is two-phase: pre-train on clean
reference spectra, then fine-tune on a smaller set measured under different conditions
(domain shift), then evaluate on a held-out test set.

```
data/*.npy → RamanDataset (ramanspy preprocessing) → [AugmentedDataset] → DataLoader
          → model (B,1,1000)→(B,30) → engine.train_epoch / evaluate
```

---

## 2. Repository inventory

### 2.1 Runnable entry points

| Script | Purpose | Model support |
|---|---|---|
| `main.py` | Main CLI: pretrain → finetune → test | all 5 |
| `evaluate_kfold.py` | Pretrain once, then 5× finetune on different splits | `legacy_cnn` only (raises otherwise) |
| `generate_heatmaps.py` | 5-epoch train on finetune set → confusion matrices | all 5, hardcoded |
| `legacy_baseline/train.py` | Reproduce the notebook's 67/33 run | `LegacyCNN` only |
| `test_mixup.py` | Smoke-test the mixup code path on random tensors | n/a |
| `tests/test_model.py` | Shape check for `Raman1DCNN` | n/a |
| `tests/test_multiscale.py` | Shape assert for `MultiscaleCNN` (the only real `assert`) | n/a |
| `tests/test.py` | ramanspy dataset-loading demo → `results/plot.png` | n/a |
| `convert.py` | `docs/*.md` → `docs-pdf/*.pdf` | n/a |
| `webapp/app.py` | Streamlit log dashboard | n/a |

### 2.2 Library modules

| File | Contents |
|---|---|
| `src/dataset.py` | `RamanDataset`, `AugmentedDataset` |
| `src/engine.py` | `train_epoch()`, `evaluate()` |
| `src/model.py` | `ResidualBlock1D`, `Raman1DCNN`, `Transformer1D`, `MultiscaleBlock1D`, `MultiscaleCNN`, `FusionNet` |
| `legacy_baseline/model.py` | `LegacyCNN` |
| `src/__init__.py` | empty |

### 2.3 Dependencies

`requirements.txt` (28 lines, looks pip-frozen — carries transitive deps like `fonttools`,
`kiwisolver`, `narwhals`). Direct ones that matter: `torch` (**unpinned**), `numpy==2.5.2`,
`ramanspy==0.2.10`, `scikit-learn==1.9.0`, `scipy==1.18.1`, `matplotlib==3.11.1`,
`pandas==3.0.5`, `pybaselines==1.2.1`.

**Missing from `requirements.txt`:** `streamlit`, `plotly` (in `webapp/requirements.txt`),
`markdown`, `weasyprint` (used by `convert.py`, declared nowhere).

Known patch: `docs/1_SETUP.md` documents editing installed `ramanspy/plot/_core.py` line ~175,
`plt.cm.get_cmap()` → `plt.get_cmap()`. Only needed for plotting.

Local env measured: Python 3.12.10 in `venv/`, torch 2.14.0+cpu.

---

## 3. The data — exact measured facts

### 3.1 Files

All `float64` on disk, cast to `float32` at load. `data/` is gitignored (`data/*`).

| File | Shape | dtype |
|---|---|---|
| `X_reference.npy` | (60000, 1000) | float64 |
| `y_reference.npy` | (60000,) | float64 |
| `X_finetune.npy` | (3000, 1000) | float64 |
| `y_finetune.npy` | (3000,) | float64 |
| `X_test.npy` | (3000, 1000) | float64 |
| `y_test.npy` | (3000,) | float64 |
| `X_2018clinical.npy` | (10000, 1000) | float64 |
| `y_2018clinical.npy` | (10000,) | float64 |
| `X_2019clinical.npy` | (2500, 1000) | float64 |
| `y_2019clinical.npy` | (2500,) | float64 |
| `wavenumbers.npy` | (1000,) | float64 |

### 3.2 Class distributions (measured)

| Split | n | #classes | label values | per-class count |
|---|---|---|---|---|
| reference | 60000 | 30 | 0–29 | 2000 each (exactly balanced) |
| finetune | 3000 | 30 | 0–29 | 100 each (exactly balanced) |
| test | 3000 | 30 | 0–29 | 100 each (exactly balanced) |
| **2018clinical** | 10000 | **5** | **{0, 2, 3, 5, 6}** | 2000 each |
| **2019clinical** | 2500 | **5** | **{0, 2, 3, 5, 6}** | 500 each |

Labels are stored as floats (`0.0`, `1.0`, …); `RamanDataset` remaps them to int64 `0..C-1`.

### 3.3 ⚠ The clinical splits — read before using them

Both clinical files contain only **5 distinct labels: {0, 2, 3, 5, 6}** — a *sparse, non-
contiguous* subset of the 0–29 range.

What that means for the code as written: if you load them with the reference set's
`label_mapping`, the "unseen labels" filter at `src/dataset.py:61` drops **nothing** (0,2,3,5,6
are all in 0–29), so a run would appear to succeed and produce a plausible-looking accuracy.
But the result is only meaningful **if label ID *k* denotes the same species in the clinical
files as in the reference file.** That correspondence is *not* established anywhere in this
repo — there is no label-name file, and `ramanspy.datasets.bacteria("labels")` (used in
`tests/test.py`) needs network access to fetch the name list.

Two readings, both plausible, with opposite consequences:
- The 5 IDs are the same species subset → evaluation is valid, and it's a genuine
  5-class domain-transfer benchmark.
- The clinical files use their own labelling (e.g. treatment/antibiotic groups) → the numbers
  would be silently meaningless.

**Resolve this before building on it.** The cheapest check: pull the official label list for
the bacteria-ID dataset and confirm what IDs 0, 2, 3, 5, 6 are, and whether the clinical
subsets are described as species or as something else. Until then, treat any clinical-split
result as unverified.

Also note the scale: 10000 + 2500 samples is **4× the combined finetune+test data** currently
in use, and completely untouched by any script.

### 3.4 Wavenumber axis (measured)

```
w[0]   = 1792.40 cm⁻¹
w[1]   = 1791.20 cm⁻¹      step = −1.20 cm⁻¹ (DESCENDING)
w[999] =  381.98 cm⁻¹
```
Span 1410.42 cm⁻¹ over 1000 points. This is the biological fingerprint region.

### 3.5 Input value range (measured)

`X_reference` already lies in **[0.0000, 1.0000]** before any preprocessing — the distributed
arrays are pre-normalised. Sample row 0, first 6 values:
```
[0.35522984, 0.30701063, 0.34318911, 0.33998200, 0.34797164, 0.38699150]
```
`X_2018clinical` is likewise in [0, 1].

> **Research note:** because the data arrives pre-normalised, the pipeline's final `MinMax()`
> step is close to a no-op on the raw array — but it runs *after* ASPLS baseline subtraction,
> which does change the range, so it isn't redundant. Still worth an ablation: the despike and
> baseline steps may be partially undoing normalisation the distributors already applied.

---

## 4. Preprocessing (`src/dataset.py`)

### 4.1 `RamanDataset(X_path, y_path, label_mapping=None, apply_preprocessing=True)`

Executes eagerly in `__init__`:

1. `np.load(X_path).astype(np.float32)`
2. If `apply_preprocessing` (**default True**): load `wavenumbers.npy` from the same directory
   (falls back to `np.arange(L)` if absent), wrap in `rp.SpectralContainer`, and apply:

   | Step | Call | Purpose |
   |---|---|---|
   | 1. Despike | `rp.preprocessing.despike.WhitakerHayes()` | remove cosmic-ray spikes |
   | 2. Denoise | `rp.preprocessing.denoise.SavGol(window_length=9, polyorder=3)` | Savitzky–Golay smoothing |
   | 3. Baseline | `rp.preprocessing.baseline.ASPLS()` | subtract auto-fluorescence hump |
   | 4. Normalise | `rp.preprocessing.normalise.MinMax()` | rescale to [0, 1] |

   All parameters are hardcoded — **not exposed via CLI**. Any preprocessing ablation requires
   editing `src/dataset.py:31-36`.
3. `np.expand_dims(axis=1)` → `(N, 1, 1000)`
4. Label remap. If `label_mapping is None`, builds `{sorted unique label: idx}`. Otherwise uses
   the supplied mapping and **drops** rows whose label isn't in it, printing
   `Warning: Dropped N samples with unseen labels.`
5. `__getitem__` returns `(torch.tensor(X[idx]), torch.tensor(y[idx]))`.

**Performance:** no caching. The full ramanspy pipeline runs on all 60 000 spectra at every
script start, and again on finetune and test sets. This dominates iteration time. Caching the
processed arrays to disk is the highest-value quality-of-life fix in the repo.

### 4.2 `AugmentedDataset(dataset, augment=False)`

Wrapper; when `augment=True`, per `__getitem__` (so re-randomised every epoch):

| # | Transform | Code | Range | Physical analogue |
|---|---|---|---|---|
| 1 | Circular shift | `np.roll(x, shift, axis=-1)` | `randint(-5, 6)` → −5…+5 idx (≤ 6 cm⁻¹) | spectrometer miscalibration |
| 2 | Additive noise | `+ np.random.normal(0, 0.01, shape)` | σ = 0.01 | sensor / shot noise |
| 3 | Multiplicative scale | `* scale` | `uniform(0.95, 1.05)` | laser power, concentration, focus |

Applied in that fixed order, always all three, with no per-transform probability. Ranges are
hardcoded — **not CLI-exposed.**

Note `np.roll` **wraps** — values shifted off one end reappear at the other. At ≤5 points out
of 1000 this is a minor artefact, but it does splice the 381.98 cm⁻¹ end onto the
1792.40 cm⁻¹ end. A padded/cropped shift would be cleaner.

---

## 5. Training engine (`src/engine.py`)

### 5.1 `train_epoch(model, device, train_loader, optimizer, criterion, epoch, phase="Train", mixup_alpha=0.0)`

Returns `None`. Per batch:

```python
if mixup_alpha > 0.0:
    lam = np.random.beta(mixup_alpha, mixup_alpha)
    lam = max(lam, 1 - lam)                      # force lam >= 0.5
    index = torch.randperm(data.size(0)).to(device)
    data = lam * data + (1 - lam) * data[index, :]
    target_a, target_b = target, target[index]
# ...
loss = lam * criterion(output, target_a) + (1 - lam) * criterion(output, target_b)
```

- `lam` is drawn **once per batch**, not per sample.
- Accuracy during mixup is counted against `target_a` (the dominant label) only — so reported
  train accuracy under mixup is optimistic but not meaningless.
- Prints every **100 batches**: `{phase} Epoch: {e} [{seen}/{total} ({pct}%)]\tLoss: {l:.6f}\tAcc: {a:.2f}%`
  The `Loss` is that single batch; `Acc` is the running epoch average. **This print cadence is
  what lets you infer batch size from a log** (see §9.1).

### 5.2 `evaluate(model, device, data_loader, criterion, phase="Test")`

Returns `(test_loss, acc)`. Accumulates `criterion(...).item() * data.size(0)` then divides by
`len(data_loader.dataset)` — a correct per-sample mean even with a ragged final batch. Runs
under `torch.no_grad()` and `model.eval()`. Prints
`\n{phase} set: Average loss: {L:.4f}, Accuracy: {c}/{n} ({acc:.2f}%)\n`.

No per-class metrics, no F1, no top-k. Adding macro-F1 here would propagate to every script.

---

## 6. Model architectures — complete specification

All take `(B, 1, 1000)` → `(B, num_classes)` **logits** (no softmax; `nn.CrossEntropyLoss`
applies it). Parameter counts **measured** at `num_classes=30`.

### 6.1 Summary

| Model | File | Params | fp32 size | Best test acc | Key hyperparameters (hardcoded) |
|---|---|---|---:|---|---|
| **LegacyCNN** | `legacy_baseline/model.py` | **40,446** | 0.16 MB | **76.50%** | dropout 0.3 |
| Raman1DCNN | `src/model.py` | 262,510 | 1.05 MB | 69.90% | dropout 0.4 |
| MultiscaleCNN | `src/model.py` | 193,358 | 0.77 MB | 63.73% | dropout 0.4 |
| FusionNet | `src/model.py` | 75,694 | 0.30 MB | 60.87% | d_model 64, nhead 4, 2 layers, dropout 0.2 |
| Transformer1D | `src/model.py` | 109,726 | 0.44 MB | 57.83% | patch 100, d_model 64, nhead 4, 3 layers, dropout 0.1 |

> **The central empirical result of this project, quantified:** the smallest model — 40 k
> params, **6.5× smaller** than the ResNet — wins by **6.6 points**. Accuracy is *inversely*
> ordered with capacity among the CNNs. Any new architecture proposal should be justified
> against this, and small-model variants should be tried before large ones.

### 6.2 `LegacyCNN` — 40,446 params — the champion

PyTorch port of the Keras CNN from `notebooks/NormalizedModelWithBootstrapping.ipynb`.
Block pattern is `Conv1d(k=3, padding=1) → ReLU → BatchNorm1d → AvgPool1d(3, stride=3)`.

⚠ Order is **Conv → ReLU → BN**, not the modern Conv → BN → ReLU. Deliberate — the file
comment says it mirrors the Keras layer stacking. Changing it is a *different model*; if you
do, log it as an ablation.

Measured layer-by-layer:

| Layer | Op | Shape out |
|---|---|---|
| `conv1` | Conv1d(1→8, k3, p1) | (B, 8, 1000) |
| `bn1` | BatchNorm1d(8) | (B, 8, 1000) |
| `pool1` | AvgPool1d(3, s3) | (B, 8, **333**) |
| `conv2` | Conv1d(8→16, k3, p1) | (B, 16, 333) |
| `bn2` / `pool2` | BN / AvgPool1d(3,s3) | (B, 16, **111**) |
| `conv3` | Conv1d(16→32, k3, p1) | (B, 32, 111) |
| `bn3` / `pool3` | BN / AvgPool1d(3,s3) | (B, 32, **37**) |
| `conv4` | Conv1d(32→32, k3, p1) | (B, 32, 37) |
| `bn4` / `pool4` | BN / AvgPool1d(3,s3) | (B, 32, **12**) |
| `conv5` | Conv1d(32→64, k3, p1) | (B, 64, 12) |
| `bn5` / `pool5` | BN / AvgPool1d(3,s3) | (B, 64, **4**) |
| `conv6` | Conv1d(64→128, k3, p1) | (B, 128, 4) |
| `bn6` | BatchNorm1d(128) | (B, 128, 4) |
| `global_pool` | AdaptiveAvgPool1d(1) | (B, 128, 1) |
| `flatten` + `dropout` | Dropout(**0.3**) | (B, 128) |
| `fc` | Linear(128→30) | (B, 30) |

Sequence length collapses **1000 → 333 → 111 → 37 → 12 → 4 → 1**. That ~250× reduction with
only 40 k params is the structural bottleneck credited for its generalisation.

Note: dropout 0.3 is **hardcoded** at `legacy_baseline/model.py:38`. The "dropout raised to
0.3" experiment in `docs/10` was done by editing this line — it is **not** a CLI flag.
(Commit `f4e1e29` touched this file by exactly 1 line, consistent with that edit.)

### 6.3 `Raman1DCNN` — 262,510 params — 1D ResNet

`ResidualBlock1D(in, out, stride)`: `Conv1d(k3,s=stride,p1,bias=False) → BN → ReLU →
Conv1d(k3,s1,p1,bias=False) → BN`, then `out += shortcut(x)`, then ReLU. `shortcut` is
`nn.Sequential()` (identity) unless `stride != 1 or in != out`, in which case
`Conv1d(k1,s=stride,bias=False) → BN`.

Stem + 4 stages × 2 blocks. Measured shapes:

| Stage | Ops | Shape out |
|---|---|---|
| `conv1`+`bn1`+`relu` | Conv1d(1→16, k7, s2, p3, bias=False) | (B, 16, 500) |
| `pool` | MaxPool1d(k3, s2, p1) | (B, 16, 250) |
| `layer1` | 2 × ResidualBlock1D(16→16, s1) — identity shortcuts | (B, 16, 250) |
| `layer2` | ResidualBlock1D(16→32, **s2**) + ResidualBlock1D(32→32) | (B, 32, 125) |
| `layer3` | ResidualBlock1D(32→64, **s2**) + ResidualBlock1D(64→64) | (B, 64, 63) |
| `layer4` | ResidualBlock1D(64→128, **s2**) + ResidualBlock1D(128→128) | (B, 128, 32) |
| `avgpool` | AdaptiveAvgPool1d(1) → view | (B, 128) |
| `classifier` | Linear(128→128) → ReLU → Dropout(**0.4**) → Linear(128→30) | (B, 30) |

Stages 2–4 use the projection shortcut; stage 1 and the second block of each stage use identity.
Dropout 0.4 hardcoded at `src/model.py:52`.

### 6.4 `MultiscaleCNN` — 193,358 params — Inception-style

`MultiscaleBlock1D(in, out)` splits into `branch_channels = out // 4` per branch, then
concatenates on dim=1:

| Branch | Composition |
|---|---|
| `branch1` | Conv1d(k1) → BN → ReLU |
| `branch2` | Conv1d(k1) → BN → ReLU → Conv1d(**k3**, p1) → BN → ReLU |
| `branch3` | Conv1d(k1) → BN → ReLU → Conv1d(**k7**, p3) → BN → ReLU |
| `branch4` | Conv1d(k1) → BN → ReLU → Conv1d(**k11**, p5) → BN → ReLU |

All convs `bias=False`. Requires `out % 4 == 0`.

| Stage | Ops | Shape out |
|---|---|---|
| `stem` | Conv1d(1→16,k7,s2,p3) → BN → ReLU → MaxPool1d(k3,s2,p1) | (B, 16, 250) |
| `block1` | MultiscaleBlock1D(16→64), 16 ch/branch | (B, 64, 250) |
| `pool1` | MaxPool1d(k2, s2) | (B, 64, 125) |
| `block2` | MultiscaleBlock1D(64→128), 32 ch/branch | (B, 128, 125) |
| `pool2` | MaxPool1d(k2, s2) | (B, 128, **62**) |
| `block3` | MultiscaleBlock1D(128→256), 64 ch/branch | (B, 256, 62) |
| `avgpool` | AdaptiveAvgPool1d(1) | (B, 256) |
| `classifier` | Linear(256→128) → ReLU → Dropout(**0.4**) → Linear(128→30) | (B, 30) |

Receptive-field motivation: sharp narrow Raman peaks (k3) and broad humps (k11) simultaneously.
Note `block3` does **no** downsampling — final length stays 62 until global pooling.

### 6.5 `Transformer1D` — 109,726 params — ViT-style

Signature: `Transformer1D(num_classes=30, seq_len=1000, patch_size=100, d_model=64, nhead=4,
num_layers=3, dim_feedforward=128, dropout=0.1)`.
Guarded by `assert seq_len % patch_size == 0`. **`main.py` instantiates with defaults only** —
none of these are CLI-reachable.

1. `x.view(B, 10, 100)` — 10 non-overlapping patches of 100 points (**no conv stem**, so no
   local inductive bias at all)
2. `patch_embedding = Linear(100 → 64)` → (B, 10, 64)
3. Prepend `cls_token` — `nn.Parameter(torch.randn(1,1,64))` → (B, 11, 64)
4. Add `pos_embedding` — `nn.Parameter(torch.randn(1, 11, 64))`, learned, **additive**
5. `TransformerEncoder(TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128,
   dropout=0.1, activation='relu', batch_first=True), num_layers=3)`
6. `mlp_head = LayerNorm(64) → Linear(64→30)` applied to `x[:, 0, :]` (CLS)

⚠ `seq_len=1000` / `patch_size=100` are hardcoded defaults — this model silently assumes
1000-point input. Both `cls_token` and `pos_embedding` use `torch.randn` (std 1.0) rather than
a scaled init like `0.02 * randn`; at `d_model=64` that's a large-magnitude initialisation and
a plausible contributor to the underfitting. **Cheap experiment: rescale these inits.**

### 6.6 `FusionNet` — 75,694 params — CNN front-end + Transformer back-end

Signature: `FusionNet(num_classes=30, seq_len=1000, d_model=64, nhead=4, num_layers=2,
dim_feedforward=128, dropout=0.2)`. Again defaults-only from `main.py`.

| Step | Op | Shape out |
|---|---|---|
| `cnn.0-2` | Conv1d(1→16, k7, s2, p3, bias=False) → BN → ReLU | (B, 16, 500) |
| `cnn.3` | MaxPool1d(k5, **s5**) | (B, 16, 100) |
| `cnn.4-6` | Conv1d(16→64, k3, s1, p1, bias=False) → BN → ReLU | (B, 64, 100) |
| `cnn.7` | MaxPool1d(k2, s2) | (B, 64, 50) |
| permute | `(B, d_model, 50) → (B, 50, 64)` | (B, 50, 64) |
| +CLS, +pos | `cls_token` (1,1,64), `pos_embedding` (1, **51**, 64) | (B, 51, 64) |
| encoder | 2 × TransformerEncoderLayer(64, 4 heads, FFN 128, dropout **0.2**) | (B, 51, 64) |
| head | LayerNorm(64) → Dropout(0.2) → Linear(64→30) on CLS | (B, 30) |

Total downsample 1000 / (2·5·2) = 50, matching the hardcoded `num_patches = 50`. The extra
`Dropout` in the head (absent in `Transformer1D`) plus 2 (not 3) layers is the deliberate
"keep capacity small" choice noted in the source comment. Same `torch.randn` init concern.

---

## 7. `main.py` — every argument

### 7.1 Complete CLI reference

| Flag | Type | Default | Effect | Notes |
|---|---|---|---|---|
| `--batch-size` | int | `64` | training batch size | eval/val loaders use `batch_size * 2` |
| `--epochs` | int | `15` | pre-training epochs | also sets `CosineAnnealingLR(T_max=epochs)` |
| `--finetune-epochs` | int | `5` | fine-tuning epochs | `0` skips phase 2 entirely |
| `--lr` | float | `0.001` | pre-train LR | finetune uses `lr * 0.1` (hardcoded) |
| `--weight-decay` | float | `0.01` | Adam L2 penalty | **pre-train optimizer only** — see §7.4 |
| `--data-dir` | str | `data` | where the `.npy` files are | |
| `--save-dir` | str | `models` | checkpoint directory | `os.makedirs(exist_ok=True)` |
| `--val-split` | float | `0.0` | fraction of reference held for validation | `0.0` ⇒ no val, no early stopping, no best-checkpointing |
| `--augment` | flag | `False` | enable the 3 augmentations | train split only; val gets `augment=False` |
| `--mixup-alpha` | float | `0.0` | Beta(α,α) mixup strength | `0.0` = off. Applied in **both** phases |
| `--model-type` | str | `resnet` | one of `resnet`, `legacy_cnn`, `transformer`, `multiscale_cnn`, `fusion` | argparse-validated |

`--model-type` → class mapping: `resnet`→`Raman1DCNN`, `legacy_cnn`→`LegacyCNN`,
`transformer`→`Transformer1D`, `multiscale_cnn`→`MultiscaleCNN`, `fusion`→`FusionNet`.

### 7.2 NOT configurable by CLI (must edit source)

| Thing | Location | Value |
|---|---|---|
| Early-stopping patience | `main.py:114` | `10` |
| Finetune LR multiplier | `main.py:154` | `0.1` |
| Reference checkpoint name | `main.py:116` | `raman_reference_model.pth` |
| Finetune checkpoint name | `main.py:161` | `raman_finetuned_model.pth` |
| `random_split` seed | `main.py:74` | `42` |
| Optimizer choice | `main.py:107` | Adam, both phases |
| LR schedule | `main.py:111,155` | CosineAnnealingLR, both phases |
| Loss | `main.py:106` | CrossEntropyLoss |
| All dropout rates | model files | 0.3 / 0.4 / 0.1 / 0.2 |
| All preprocessing params | `src/dataset.py:31-36` | see §4.1 |
| All augmentation ranges | `src/dataset.py:96-105` | see §4.2 |
| Transformer/Fusion hyperparams | `src/model.py:81,219` | defaults only |
| DataLoader `num_workers` | everywhere | unset ⇒ **0** (single-process loading) |

`num_workers=0` is worth changing on your 12-core-ish desktop: with GPU compute restored, data
loading becomes the bottleneck.

### 7.3 Exact control flow

```
1. os.makedirs(save_dir)
2. device: cuda → mps → cpu
3. RamanDataset(X_reference, y_reference)          # preprocessing runs here
   num_classes = len(np.unique(dataset.y))
4. if val_split > 0:
       random_split([N-val, val], seed 42)
       train → AugmentedDataset(augment=args.augment)
       val   → AugmentedDataset(augment=False)
       ref_loader(bs, shuffle=True); val_loader(bs*2, shuffle=False)
   else:
       whole set → AugmentedDataset(augment=args.augment); val_loader = None
5. build model → .to(device)
   criterion = CrossEntropyLoss()
   optimizer = Adam(lr, weight_decay)
   scheduler = CosineAnnealingLR(T_max=epochs)
6. for epoch in 1..epochs:
       train_epoch(..., mixup_alpha)
       scheduler.step()
       if val_loader:
           val_loss, val_acc = evaluate(...)
           if val_loss < best: save ref ckpt; patience=0
           else: patience += 1
           if patience >= 10: break
7. if val_loader is None: save final weights
   elif ckpt exists:      load best  (torch.load weights_only=True)
8. if X_finetune & y_finetune exist and finetune_epochs > 0:
       RamanDataset(..., label_mapping=reference mapping)   # NO AugmentedDataset wrapper
       optimizer_ft = Adam(lr * 0.1)                        # NO weight_decay
       scheduler_ft = CosineAnnealingLR(T_max=finetune_epochs)
       for epoch: train_epoch(phase="Finetune", mixup_alpha)
       save finetuned ckpt
9. if X_test & y_test exist:
       RamanDataset(..., label_mapping=reference mapping); evaluate(phase="Test")
```

### 7.4 Behavioural quirks that matter for experiment design

1. **`--augment` never applies to fine-tuning.** The finetune `DataLoader` is built directly
   from `RamanDataset` at `main.py:151` with no `AugmentedDataset` wrapper. So every "augmented"
   result augmented *only* the pre-training phase. Augmenting the 3000-sample finetune set —
   the small, domain-shifted one that most needs it — is **untested**, one line of code, and an
   obvious first experiment.
2. **`--weight-decay` is silently dropped in phase 2.** `optimizer_ft = optim.Adam(model.parameters(), lr=args.lr * 0.1)`
   — no `weight_decay` argument, so it defaults to 0. The `multiscale_exp2_wd05` result
   (wd=0.05 → 34.17%) therefore only tested high weight decay during *pre-training*.
3. **No early stopping during fine-tuning.** Phase 2 always runs the full `--finetune-epochs`
   and saves the last epoch, with no validation. `evaluate_kfold.py` *does* validate during
   finetuning — a meaningful difference between the two scripts.
4. **Checkpoint filenames don't encode the model type.** Consecutive runs of different
   architectures overwrite each other's `models/raman_reference_model.pth`. Fix before running
   a grid.
5. **`--val-split` changes the training-set size.** `--val-split 0.33` trains on 40 200 rather
   than 60 000 samples. Runs with different val splits are not directly comparable.
6. **`num_classes` is inferred from whatever labels appear in the reference file.** With the
   full reference set that's 30; on a subset it would silently change the head size.
7. **BatchNorm + batch size.** Every CNN here uses BatchNorm1d; the finetune set is 3000
   samples at batch 64 (≈47 batches). Changing `--batch-size` perturbs BN statistics as well as
   optimisation — keep it fixed across comparisons.

---

## 8. Other scripts — full detail

### 8.1 `evaluate_kfold.py`

Args: `--model-type` (default `legacy_cnn`, **only value supported**), `--pretrain-epochs`
(100), `--finetune-epochs` (20), `--batch-size` (64), `--lr` (0.001), `--data-dir` (`data`),
`--augment` (flag).

Hardcoded: val fraction `0.1` (both phases), seed 42 for the *reference* split only, weight
decay `0.01`, patience `10`, checkpoint `models/kfold_reference_model.pth`, folds `range(1, 6)`.

- **Phase 1:** pre-train once with per-epoch validation + early stopping; save best.
- **Phase 2:** 5 iterations; each rebuilds the model, reloads the frozen phase-1 checkpoint,
  re-runs `random_split(finetune, [90%, 10%])` **unseeded**, finetunes at `lr*0.1` with
  per-epoch validation, keeps best-val-loss weights, evaluates on the test set.
- Reports each fold plus `mean ± std` via `np.mean` / `np.std`.

Two defects to be aware of:
- **Not true k-fold.** Repeated *random* 90/10 draws, not disjoint partitions — folds can
  overlap arbitrarily. The ±0.11% spread therefore *understates* real variance, because all
  five folds share the same phase-1 checkpoint and heavily overlapping finetune data. Use
  `sklearn.model_selection.StratifiedKFold` for a defensible number.
- **`best_ft_weights = model.state_dict()` (line 118) stores a reference, not a copy.**
  `state_dict()` returns live tensor references, so subsequent epochs mutate the "best"
  weights in place. `model.load_state_dict(best_ft_weights)` at line 121 is effectively a
  no-op — it reloads the *final* epoch, not the best. Needs
  `copy.deepcopy(model.state_dict())`. **The 73.23% is a last-epoch number, not a best-val one.**

### 8.2 `generate_heatmaps.py`

No CLI args. Hardcoded: `data_dir='data'`, `results_dir='results/heatmaps'`, `epochs = 5`,
batch 64 (train) / 128 (test), `Adam(lr=0.001)` with **no weight decay**, `mixup_alpha=0.0`.

⚠ **Trains on `X_finetune` (3000 samples) for 5 epochs with no pre-training at all.** The
resulting confusion matrices reflect a deliberately under-trained model — they are useful for
*relative* per-class structure, not for accuracy. Builds `label_mapping` itself from
`y_reference.npy` and derives axis labels via `idx_to_label`. Saves at `dpi=300`, figsize 12×10.

### 8.3 `legacy_baseline/train.py`

No CLI. Hardcoded: 67/33 split (`test_size = int(0.33 * N)`) with seed 42, batch 128,
`Adam(lr=2e-4, weight_decay=0.01)`, `epochs = 10` (source comment notes the notebook used 200),
evaluates every 5 epochs plus a final eval. Imports `from model import LegacyCNN` after
`sys.path.append`, so **it must be run from inside `legacy_baseline/`** or the import fails.
Produced the 78.70% figure — a *reference-set validation* number (see §0.4).

### 8.4 `webapp/app.py`

Streamlit + Plotly. `RESULTS_DIR = <repo>/results`. Regex:
```python
r"Epoch:\s*(\d+).*?Loss:\s*([\d\.]+).*?Acc:\s*([\d\.]+)%"
```
Keeps the **last** match per epoch; when the epoch counter *decreases* it adds
`current_phase_offset += last_epoch`, stitching pre-train and finetune into one continuous
x-axis. Multiselect overlays runs (defaults to the first file). Two charts: Loss and
Accuracy (%). Run: `streamlit run webapp/app.py`.

Caveat: the parsed `Loss` is a single batch's loss (see §5.1), so the loss curves are noisier
than true epoch means — and the "Accuracy" is training accuracy, not validation.

---

## 9. Complete experimental record

### 9.1 How the configs were inferred

Logs don't record their own arguments, so the flags below are **inferred** from:
- **Batch size** — from the print stride. Prints occur every 100 batches, so the second
  `Epoch: 1 [X/...]` line gives `X = 100 × batch_size`. `6400` ⇒ **bs 64**; `12800` ⇒ **bs 128**.
- **Val split** — from `Split reference data: A train / B val`. `40200/19800` ⇒ 0.33;
  `54000/6000` ⇒ 0.1.
- **Epochs** — max observed `Pre-train Epoch:` / `Finetune Epoch:` number.
- **Augment / model / dropout** — from the filename and `docs/10_EXPERIMENT_SUMMARY.md`.

### 9.2 Master results table — every log, measured

Sorted by test accuracy. "Final Pre-Acc" = last printed training accuracy of the last
pre-train epoch. "Val" = last `Validation set:` line. **Test = the authoritative number.**

| # | Log file | Model | bs | Pre ep | FT ep | Aug | Final Pre-Acc | Val loss | Val acc | Test loss | **Test acc** | Code ver |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `legacy_cnn_exp2_drop3_ft20.log` | LegacyCNN (drop 0.3) | 64 | 100 | 20 | ✔ | 83.37% | 0.5807 | 82.54% | 0.7769 | **76.50%** 🏆 | OLD |
| 2 | `pure_legacy_cnn.log` | LegacyCNN | 64 | 100 | 20 | ✘ | 90.39% | 0.5252 | 84.11% | 0.8282 | **75.10%** | OLD |
| 3 | `legacy_cnn_100_epochs_augmented.log` | LegacyCNN | 64 | 100 | 5 | ✔ | 84.28% | 0.5567 | 83.15% | 0.8383 | **75.07%** | OLD |
| 4 | `kfold_legacy_cnn.log` | LegacyCNN | 64 | 18 (ES) | 20×5 | ✘ | 75.92% | 0.7781 | 75.47% | 0.8749 | **73.23% ± 0.11%** | NEW |
| 5 | `legacy_cnn_100_epochs.log` | LegacyCNN | 64 | 100 | 5 | ✘ | 91.24% | 0.5272 | 84.24% | 0.8855 | **72.97%** | OLD |
| 6 | `resnet_exp3_ft20.log` | Raman1DCNN (drop 0.4) | 64 | 100 | 20 | ✔ | 78.68% | 0.6902 | 77.61% | 0.9628 | **69.90%** | OLD |
| 7 | `resnet_100_epochs_augmented.log` | Raman1DCNN | 128 | 100 | 5 | ✔ | 88.44% | 0.6271 | 79.80% | 1.0285 | **67.80%** | OLD |
| 8 | `pure_resnet.log` | Raman1DCNN | 64 | 100 | 20 | ✘ | 91.34% | 0.6184 | 80.01% | 1.0343 | **67.47%** | OLD |
| 9 | `resnet_50_epochs_augmented.log` | Raman1DCNN | 128 | 50 | 5 | ✔ | 82.46% | 0.6683 | 78.93% | 1.0604 | **66.90%** | OLD |
| 10 | `multiscale_exp1.log` | MultiscaleCNN | 64 | 100 | 20 | ✔ | 71.11% | 0.9179 | 70.96% | 1.1549 | **63.73%** | OLD |
| 11 | `resnet_50_epochs.log` | Raman1DCNN | 128 | 50 | 5 | ✘ | **99.30%** | 0.7656 | 76.67% | 1.2843 | **61.00%** | OLD |
| 12 | `fusion_exp1.log` | FusionNet | 64 | 55 (ES) | 20 | ✔ | 61.10% | 1.1380 | 62.38% | 1.2126 | **60.87%** | NEW |
| 13 | `pure_transformer.log` | Transformer1D | 64 | 100 | 20 | ✘ | 68.17% | 0.8636 | 71.88% | 1.3351 | **57.83%** | OLD |
| 14 | `fusion_mixup.log` | FusionNet + mixup α0.2 | 64 | 69 (ES) | 20 | ✔ | 52.92% | 1.2968 | 60.34% | 1.4407 | **55.80%** | NEW |
| 15 | `transformer_exp1.log` | Transformer1D | 64 | 100 | 20 | ✔ | 60.63% | 1.0152 | 66.69% | 1.3702 | **54.73%** | OLD |
| 16 | `legacy_cnn_pipeline.log` | LegacyCNN (first run) | 128 | 10 | 5 | ✘ | 75.45% | 0.8671 | 75.10% | 1.5941 | **53.70%** | OLD |
| 17 | `multiscale_exp2_wd05.log` | MultiscaleCNN, **wd 0.05** | 64 | 100 | 20 | ✔ | 28.15% | 2.3805 | 26.84% | 2.2640 | **34.17%** | OLD |
| 18 | `early_stopping_test.log` | Raman1DCNN | 64 | 27 (ES) | **0** | ? | 81.79% | 0.7559 | 76.46% | 3.2026 | **33.37%** | NEW |

`ES` = early-stopped at that epoch. All runs: 30 classes, `--val-split 0.33` except
`kfold_legacy_cnn` (0.1, hardcoded). Test set is always 3000 samples; val is 19800 (or 6000).

### 9.3 Findings that only appear in the raw logs

These are not in `docs/` and are the most useful part of this record.

**(a) Fine-tuning is worth ~43 accuracy points.** `early_stopping_test.log` is the only run
with `--finetune-epochs 0`. Its pre-training was healthy (val 76.46%, val loss 0.7559) yet test
accuracy collapsed to **33.37%** with test loss **3.2026** — a 4.2× loss increase versus
validation. Compare `resnet_exp3_ft20` (same architecture, with finetuning): 69.90%.
→ **The domain gap between reference and test data is enormous, and phase 2 is the single
highest-leverage component in the entire pipeline.** Any new architecture must be evaluated
*with* fine-tuning; without it the ranking is meaningless.

**(b) Train accuracy anti-correlates with test accuracy.** `resnet_50_epochs` reached **99.30%**
training accuracy — the highest of any run — and scored **61.00%** on test, near the bottom of
the CNNs. Meanwhile the champion peaked at 83.37% train. Textbook memorisation. **Do not use
training accuracy to select models here.** Even validation accuracy misleads: `pure_legacy_cnn`
had *higher* val accuracy (84.11%) than the champion (82.54%) but *lower* test accuracy.

**(c) Validation loss ranks models better than validation accuracy.** Across the table, test
accuracy tracks *validation loss* monotonically far better than it tracks validation accuracy.
The champion has the best test loss (0.7769) of any run. → **Select on val loss, or better, on
a held-out slice of the finetune distribution.**

**(d) Augmentation's effect is model-dependent, not universal.**

| Model | No aug | With aug | Δ |
|---|---|---|---|
| LegacyCNN (100/20) | 75.10% | 76.50%¹ | **+1.40** |
| Raman1DCNN (100/20) | 67.47% | 69.90%¹ | **+2.43** |
| Transformer1D (100/20) | 57.83% | 54.73% | **−3.10** |

¹ also changed dropout, so not a clean single-variable ablation.
→ Augmentation helps CNNs, **hurts** the Transformer. Plausibly because the roll augmentation
shifts patch boundaries, and with 100-point non-overlapping patches a ±5 shift changes patch
content without the model having any translation equivariance to absorb it. **Testable:
overlapping or conv-embedded patches should recover this.**

**(e) Mixup degraded FusionNet by 5.07 points** (60.87% → 55.80%), and also depressed training
accuracy (61.10% → 52.92%) — so it acted as a genuine regulariser, just a harmful one. It also
*delayed* early stopping (55 → 69 epochs), consistent with slower convergence rather than
better generalisation.

**(f) High weight decay is catastrophic, not merely suboptimal.** `multiscale_exp2_wd05`
(wd 0.05, 5× default) reached only **28.15%** *training* accuracy — the model never learned at
all. Note that per §7.4.2 this only applied to pre-training. Underfitting, not overfitting.

**(g) Early stopping fires early and may be premature.** In the three NEW-code runs, early
stopping triggered at epochs 27, 55, and 69 with patience 10 on *validation loss*. Yet the
best OLD-code results came from running the full 100 epochs. Since fine-tuning dominates the
final result (finding a), stopping pre-training on reference-set val loss may be optimising
the wrong objective entirely. **Worth testing: patience 20+, or early-stop on a finetune-domain
validation slice instead.**

**(h) Test loss spans 0.78 → 3.20** across runs while accuracy spans 76.5% → 33.4%. Test loss
is the more sensitive discriminator; report both.

### 9.4 Reconciling with `docs/`

`docs/10_EXPERIMENT_SUMMARY.md` matches the logs on all test accuracies. Two additions from my
parse: `docs/8` cites ResNet "100 Ep + Augmentation" as 67.80% (matches log #7, bs 128), and
lists LegacyCNN 100ep train accuracy as 91.24% (matches log #5). No contradictions found in the
test-accuracy column. The discrepancies are only those in §0.4.

---

## 10. Per-class behaviour — confusion matrix analysis

From `results/heatmaps/confusion_matrix_legacy_cnn.png`. ⚠ Generated by `generate_heatmaps.py`,
i.e. **5 epochs on the finetune set with no pre-training** — a much weaker model than the 76.5%
champion. Read as *relative class difficulty*, not absolute performance. Each row totals 100
(100 test samples/class).

**Easy classes** (diagonal ≥ 75): class 1 (**100**), 5 (99), 20 (98), 3 (96), 18 (88), 26 (80),
16 (79), 0 (78), 27 (75).

**Hard classes** (diagonal ≤ 32): class 2 (**12**), 4 (19), 6 (19), 7 (26), 12 (30), 13 (32).

**Strongest confusions** (off-diagonal ≥ 20, `true → predicted`):

| True | → Pred | Count | | True | → Pred | Count |
|---|---|---|---|---|---|---|
| 22 | → 12 | 46 | | 9 | → 4 | 27 |
| 14 | → 3 | 43 | | 17 | → 21 | 25 |
| 15 | → 16 | 34 | | 4 | → 10 | 24 |
| 29 | → 26 | 33 | | 7 | → 12 | 21 |
| 4 | → 3 | 32 | | 13 | → 23 | 21 |
| 9 | → 8 | 27 | | 2 | → 14 | 19 |

**Apparent confusion clusters** (bidirectional or mutually reinforcing):
- **{3, 4, 14}** — 14→3 (43), 4→3 (32), 2→14 (19)
- **{15, 16}** — 15→16 (34) *and* 16→15 (15): symmetric, the classic signature of two
  near-identical classes. If these correspond to a sensitive/resistant strain pair of the same
  species, that is a clinically important and genuinely hard discrimination.
- **{8, 9, 4, 10}** — 9→8 (27), 9→4 (27), 4→10 (24)
- **{6, 7, 12, 22}** — 22→12 (46), 7→12 (21), 7→6 (15)
- **{23, 24, 13, 21}** — 13→23 (21), 13→24 (18), 24→21 (18), 24→23 (17), 17→21 (25)
- **{26, 29}** — 29→26 (33)

→ **Research direction:** confusions are *structured*, not uniform — they cluster, almost
certainly along genus/species-relatedness lines. This motivates (i) hierarchical or two-stage
classification (coarse cluster → fine class within cluster), (ii) reporting macro-F1 and
per-class recall instead of plain accuracy, since a handful of classes carry most of the error,
and (iii) class-weighted loss or targeted augmentation for the hard clusters. **Note:** to
interpret any of this biologically you need the species names for IDs 0–29, which are **not in
this repo** (see §3.3).

I inspected only the `legacy_cnn` matrix in detail; the other four are listed in §11.2 and
comparing cluster structure across architectures is an open, cheap question.

---

## 11. Asset inventory — what to explore and why

### 11.1 Logs (`results/*.log`, 18 files)

| File | Lines | Added | Most useful for |
|---|---|---|---|
| `legacy_cnn_exp2_drop3_ft20.log` | 743 | 2026-09-15 | **The champion run. Your baseline to beat.** |
| `pure_legacy_cnn.log` | 743 | 2026-09-15 | No-augmentation control for the champion |
| `legacy_cnn_100_epochs.log` | 728 | 2026-09-15 | Cleanest overfitting example (91.24% train → 72.97% test) |
| `legacy_cnn_100_epochs_augmented.log` | 728 | 2026-09-15 | Direct augmentation ablation vs. the above |
| `kfold_legacy_cnn.log` | 676 | 2026-09-16 | Variance estimate; 5 fold results + mean±std |
| `resnet_exp3_ft20.log` | 743 | 2026-09-15 | Best ResNet; matched-config comparison to #1 |
| `pure_resnet.log` | 743 | 2026-09-15 | ResNet no-aug control |
| `resnet_100_epochs_augmented.log` | 428 | 2026-09-13 | bs-128 ResNet variant |
| `resnet_50_epochs.log` | 228 | 2026-09-10 | **Extreme memorisation: 99.30% train → 61.00% test** |
| `resnet_50_epochs_augmented.log` | 228 | 2026-09-10 | 50-epoch aug comparison |
| `multiscale_exp1.log` | 743 | 2026-09-15 | Inception architecture baseline |
| `multiscale_exp2_wd05.log` | 743 | 2026-09-15 | **Failure mode: weight decay 0.05 kills learning** |
| `transformer_exp1.log` | 743 | 2026-09-15 | Transformer + aug (worst-of-pair) |
| `pure_transformer.log` | 743 | 2026-09-15 | Transformer no-aug (better!) — the aug-hurts evidence |
| `fusion_exp1.log` | 645 | 2026-09-15 | Best hybrid; NEW code, per-epoch val curve |
| `fusion_mixup.log` | 799 | 2026-09-15 | Mixup ablation; longest log |
| `early_stopping_test.log` | 315 | 2026-09-15 | **Proves fine-tuning is worth ~43 points** |
| `legacy_cnn_pipeline.log` | 70 | 2026-09-10 | Earliest run; 10 epochs; shows the Mac path |

The 11 logs added on 2026-09-15 in one commit form the closest thing to a controlled grid:
`{legacy_cnn, resnet, transformer} × {aug, no-aug}` at 100/20 epochs, plus multiscale and
fusion variants.

### 11.2 Images

| File | Size | Added | Contents / use |
|---|---|---|---|
| `results/heatmaps/confusion_matrix_legacy_cnn.png` | 666 KB | 2026-09-16 | 30×30 CM, champion arch — analysed in §10 |
| `results/heatmaps/confusion_matrix_resnet.png` | 661 KB | 2026-09-16 | Does ResNet confuse the *same* clusters? Open question |
| `results/heatmaps/confusion_matrix_multiscale_cnn.png` | 701 KB | 2026-09-16 | Do multi-scale kernels fix the broad-peak classes? |
| `results/heatmaps/confusion_matrix_transformer.png` | 686 KB | 2026-09-16 | Expect diffuse errors (underfit) |
| `results/heatmaps/confusion_matrix_fusion.png` | 671 KB | 2026-09-16 | CNN+attention error structure |
| `results/loss_curves.png` | 128 KB | 2026-09-13 | Training curves (also copied to `presentations/public/`) |
| `results/plot.png` | 175 KB | 2026-09-09 | Mean spectra per species, stacked — from `tests/test.py`. **The only visualisation of the actual data.** |

All five heatmaps are 3600×3000 @ dpi 300, and all share the §8.2 caveat (5 epochs, no
pre-training).

### 11.3 Docs (`docs/*.md`, 527 lines total)

| File | Lines | Contents |
|---|---|---|
| `1_SETUP.md` | 64 | venv, the ramanspy patch, Dropbox data download |
| `2_DATA_EXPLORATION.md` | 155 | Layman's Raman explanation, dataset stats, plotting recipes |
| `3_PREPROCESSING.md` | 29 | Why each of the 4 pipeline steps exists |
| `4_LEGACY_BASELINE_SUMMARY.md` | 39 | The 78.70% run (⚠ val, not test — §0.4) |
| `5_RESNET_VS_CNN.md` | 42 | Residual/skip-connection rationale (written before the ResNet lost) |
| `6_DATA_AUGMENTATION.md` | 35 | Physical justification for each augmentation |
| `7_TRAINING_STRATEGY.md` | 23 | Adam, weight decay, cosine annealing, two-phase rationale |
| `8_MODEL_COMPARISON.md` | 45 | ResNet vs Legacy tables, "the baseline strikes back" |
| `9_TRAINING_SUMMARY.md` | 51 | Early 10-epoch ResNet run, per-epoch table |
| `10_EXPERIMENT_SUMMARY.md` | 44 | **The master results table with per-model rationale** |

Docs 5 and 9 predate the finding that LegacyCNN wins, and their framing is partly superseded
(doc 9 carries an inline `*Update*` acknowledging this). `docs/10` is the authoritative summary.

### 11.4 Other material

- `presentations/slides.md` — Slidev deck (theme `seriph`) with speaker notes; good narrative
  framing. `slides-export.pdf` is the rendered version. `pnpm` workspace.
- `notebooks/NormalizedModelWithBootstrapping.ipynb` — the original notebook; source of
  `LegacyCNN`. Includes per-class normalisation and **bootstrapping-based data generation**,
  which was **never ported** to the PyTorch pipeline.
- `notebooks/extracted_code.py` (249 lines) — flattened notebook code. Loads `C0_1.csv`,
  `C1_1.csv`, …, `PhSH.csv` — a *completely different, chemical* dataset (~1014 wavenumbers)
  that **is not in this repo**. Uses z-score normalisation (`(x-mean)/sigma`), not MinMax.
  Historical, but documents an untried normalisation choice and a possible second dataset.
- `Papers/SERS_ML_Survey.md` — literature survey; §"ML Approaches" cites comparable work
  (e.g. ConvNet single-cell microbial ID at ~95.64%), useful for positioning 76.5%.
- `Papers/1-s2.0-S0169743923000072-main.pdf`, `Papers/2104.04599.pdf` — source papers.
- `bacteria-ID/` — **empty directory**, presumably intended for the original authors' repo.
- `Fellows/README.md` — Humanitarians AI program requirements (bi-weekly updates etc.).

---

## 12. Reproducibility audit

| Aspect | Status |
|---|---|
| Global seed | ❌ Never set. No `torch.manual_seed`, `np.random.seed`, or `random.seed` anywhere. |
| Split seed | ⚠ Partial. `main.py:74` and `legacy_baseline/train.py` use `manual_seed(42)`; `evaluate_kfold.py`'s **fold** splits are unseeded. |
| Augmentation RNG | ❌ Unseeded `np.random` in `AugmentedDataset`. |
| Mixup RNG | ❌ Unseeded `np.random.beta` / `torch.randperm`. |
| Weight init | ❌ Unseeded — every run starts from different weights. |
| cuDNN determinism | ❌ Not configured. |
| Config capture | ❌ Args are never written to disk or printed. Reconstructed only by inference (§9.1). |
| Log capture | ❌ No logging; manual redirection only (§0.2). |
| Checkpoint naming | ❌ Collides across model types. |
| Env capture | ❌ No version/commit/device stamp in outputs. |
| `torch.load` | ✔ Uses `weights_only=True` (safe). |

**Net effect:** none of the 18 recorded numbers can be exactly reproduced, and small deltas
(the +1.40 augmentation effect, the 73.23% ± 0.11%) are not clearly distinguishable from
seed noise. Before drawing new conclusions, **run each config ≥3 seeds and report mean ± std.**
Fixing seeding + config logging is a small change with outsized value for everything downstream.

---

## 13. Research directions, ranked by (value ÷ effort)

### Tier 1 — cheap, high information

1. **Fix the GPU wheel** (§0.3). Unblocks everything; turns 100-epoch runs from hours into
   minutes. Then set `num_workers > 0` and consider AMP.
2. **Cache preprocessed arrays.** Save the post-ramanspy `(N,1,1000)` float32 to `.npy` once and
   load that. Biggest iteration-speed win after the GPU.
3. **Seed everything + dump args to a JSON/log next to each checkpoint.** Prerequisite for every
   claim below.
4. **Augment during fine-tuning** (§7.4.1) — one line. The finetune set is the small,
   domain-shifted one that most needs regularisation, and this has literally never been tried.
5. **Pass `weight_decay` to the finetune optimizer** (§7.4.2) — one line; currently silently 0.
6. **Fix `evaluate_kfold.py`'s `deepcopy` bug** (§8.1) — the 73.23% is a last-epoch number
   masquerading as best-val.
7. **Report macro-F1 + per-class recall in `evaluate()`.** §10 shows error is concentrated in a
   few classes; accuracy hides this.
8. **Re-run the 2026-09-15 grid on current `main.py`** so the headline numbers are reproducible
   with the code as it exists (§0.1).

### Tier 2 — substantive experiments

9. **Resolve the clinical-split labelling question (§3.3), then evaluate on them.** 12 500
   untouched samples, 4× the current eval data, and the closest thing to the project's stated
   wastewater-surveillance goal. Highest scientific upside in the repo — *conditional on the
   label semantics checking out.*
10. **Attack the domain gap directly.** Finding (a) says fine-tuning is worth 43 points, so the
    gap dominates everything. Try: freezing early layers during finetune; discriminative LRs;
    BN-statistics-only adaptation (AdaBN); test-time augmentation; a domain-adversarial head.
11. **Push the champion smaller.** Accuracy is inversely ordered with capacity (§6.1). Nobody
    has tried a *smaller-than-LegacyCNN* model. Prune channels, drop `conv4`, or try 3–4 blocks.
12. **Hierarchical classification** using the §10 clusters — coarse group, then fine class.
13. **Preprocessing ablation.** Every step is hardcoded and none has been ablated. Given the
    data arrives pre-normalised in [0,1] (§3.5), test: no-ASPLS, no-despike, no-SavGol,
    SavGol window/polyorder sweep, z-score instead of MinMax (as `extracted_code.py` used).
14. **Fix the Transformer's init** (§6.5) — `torch.randn` at std 1.0 for CLS and positional
    embeddings is likely hurting it. Try `0.02 *` scaling, sinusoidal positions, overlapping
    patches, or a conv patch-embedding. Finding (d) suggests patch boundaries interact badly
    with roll augmentation.
15. **Early-stopping criterion** (§9.3g): patience 20+, or stop on finetune-domain val loss
    rather than reference-set val loss.
16. **Ensemble the top CNNs.** Logit averaging over LegacyCNN + ResNet + Multiscale is nearly
    free and the §10 clusters suggest complementary errors.

### Tier 3 — larger / more speculative

17. **Port the notebook's bootstrapping augmentation** (§11.4) — a physics-aware, per-class
    resampling scheme that was never brought into the PyTorch pipeline. Better-motivated than
    mixup, which failed for exactly the reason bootstrapping wouldn't (it preserves real peak
    structure).
18. **Self-supervised pre-training** on reference + the 12 500 clinical spectra (masked-region
    reconstruction / contrastive on augmented views), then fine-tune. Directly targets the
    Transformer's data hunger.
19. **Explainability.** Gradient×input / Grad-CAM over the wavenumber axis, then check whether
    the peaks the model uses match known biochemical assignments — the strongest possible
    validation for a spectroscopy paper, and it needs the descending-axis detail from §3.4.
20. **Calibration + rejection.** For surveillance you need "I don't know". Measure ECE, add
    temperature scaling, and report accuracy-vs-coverage.
21. **1D architectures not yet tried:** dilated/TCN stacks, SE-attention on the LegacyCNN,
    InceptionTime, ROCKET/MiniRocket (a strong, very fast classical baseline for 1D series
    that would contextualise all the deep-learning numbers).

---

## 14. Quick command reference

```bash
# Reproduce the champion config (76.50% under OLD code — see §0.1)
python main.py --model-type legacy_cnn --epochs 100 --finetune-epochs 20 \
               --val-split 0.33 --augment --batch-size 64

# No-augmentation control
python main.py --model-type legacy_cnn --epochs 100 --finetune-epochs 20 --val-split 0.33

# Other architectures
python main.py --model-type resnet        --epochs 100 --finetune-epochs 20 --val-split 0.33 --augment
python main.py --model-type transformer   --epochs 100 --finetune-epochs 20 --val-split 0.33
python main.py --model-type multiscale_cnn --epochs 100 --finetune-epochs 20 --val-split 0.33 --augment
python main.py --model-type fusion        --epochs 100 --finetune-epochs 20 --val-split 0.33 --augment

# Mixup
python main.py --model-type fusion --epochs 100 --finetune-epochs 20 --val-split 0.33 --augment --mixup-alpha 0.2

# The weight-decay failure case
python main.py --model-type multiscale_cnn --epochs 100 --finetune-epochs 20 --val-split 0.33 --augment --weight-decay 0.05

# Variance check (legacy_cnn only)
python evaluate_kfold.py --pretrain-epochs 100 --finetune-epochs 20

# Confusion matrices for all 5 models (5 epochs, finetune set only)
python generate_heatmaps.py

# Notebook baseline reproduction (must run from inside the directory)
cd legacy_baseline && python train.py

# Dashboard
streamlit run webapp/app.py

# Capture a log on Windows
python main.py --model-type legacy_cnn --epochs 100 --finetune-epochs 20 --val-split 0.33 --augment 2>&1 | Tee-Object -FilePath results/legacy_cnn_rtx3060.log
```

Always rename/move `models/*.pth` between runs of different architectures (§7.4.4).

---

## 15. Open questions

1. What species do label IDs 0–29 correspond to? (Not in the repo; needed for §10 and any
   biological interpretation.)
2. Do the clinical splits' labels {0,2,3,5,6} mean the same species as in the reference set?
   (§3.3 — blocks direction #9.)
3. Do the other four confusion matrices show the same cluster structure as `legacy_cnn`?
4. What test accuracy does the champion config reach under the *current* `main.py`?
5. Is the +1.40 augmentation gain larger than seed variance? (§12)
6. Why is `results/loss_curves.png` byte-identical in size to `presentations/public/loss_curves.png`
   — which script generated it? No script in the repo plots loss curves; it was likely made
   ad hoc, meaning it can't currently be regenerated.
7. Where is the CSV dataset (`C0_1.csv`, `PhSH.csv`, …) that `notebooks/extracted_code.py`
   expects, and is it relevant to the SERS/wastewater goal?
