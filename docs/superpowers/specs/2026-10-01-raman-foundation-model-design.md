# Raman Foundation Model (A → B1 / B2) — Design Spec

Date: 2026-10-01
Status: draft, awaiting review

## 1. Purpose

Test whether a general spectral model, pretrained only on other people's Raman
data, is a useful starting point for our Bacteria-ID tasks, and whether one
frozen copy of it can serve more than one task.

- **A** — pretrain a general model on public spontaneous-Raman spectra from
  other sources. **None of our data is used in A**, including for checkpoint
  selection.
- **B1** — adapt A to our data by full fine-tuning.
- **B2** — adapt A to our data with LoRA, backbone frozen.

Goals agreed in brainstorming:

- **(a)** A improves accuracy on our tasks compared with the same architecture
  trained without A.
- **(c)** One frozen A serves both of our tasks through small per-task add-ons.

## 2. Ground rules

1. **New experiment, new code.** Everything lives in a new top-level
   `foundation/` package. It does not import from `src/`, `legacy_baseline/`,
   `main.py` or any existing tool, and no design choice is justified by
   results from the earlier pipeline. The only things read from the existing
   repo are the raw Bacteria-ID `.npy` files in `data/`, read-only.
2. **Our data never reaches A.** "Our data" means every Bacteria-ID file:
   `X/y_reference`, `X/y_finetune`, `X/y_test`, `X/y_2018clinical`,
   `X/y_2019clinical`. It is excluded from the pretraining corpus, from A's
   checkpoint selection, and from all A diagnostics.
3. **The test sets are scored once.** All tuning, size trials and
   reconfiguration use validation data only. `X_test` and `X_2019clinical`
   are evaluated a single time, after every configuration is frozen.
4. **No comparison with the earlier project.** The old models and their
   numbers are not arms, targets or criteria. The final report does not
   mention them unless the user asks for it.

## 3. Our tasks (downstream only)

| Task | Train | Validation | Test | Classes |
|---|---|---|---|---|
| **T1 isolates** | `reference` (60k) then `finetune` (3k) | 10% stratified of reference (stage i); 10% stratified of finetune (stage ii) | `test` (3k) | 30 isolates |
| **T2 clinical** | `2018clinical` (10k) | 10% stratified of 2018 | `2019clinical` (2.5k) | 5 treatment groups, labels {0,2,3,5,6} remapped task-locally to 0–4 |

T1's two-stage split is the dataset authors' design (Ho et al. 2019): a large
reference set, then a small set measured under shifted conditions.

Data facts checked at spec time: the Bacteria-ID axis is 1000 points,
**descending** from 1792.4 to 381.98 cm⁻¹ (step ≈ 1.41 cm⁻¹). The clinical sets
are balanced: 2000 per class in 2018 and 500 per class in 2019.

## 4. Pretraining corpus (A)

### 4.1 Sources

Real spontaneous Raman only, leaning biological.

| Source | Approx. size | Domain | Notes |
|---|---|---|---|
| DeepeR / MDA-MB-231 (Horgan 2021) | ~173k | breast-cancer cells | 500–1800 cm⁻¹; image stacks, so redundant (capped, §4.4); licence not stated |
| Airborne pathogens, single cell (ScienceDB 10.57760/sciencedb.15628) | ~23k | bacteria | 532 nm, 600–1800 cm⁻¹, labelled |
| Zenodo 15279321 single bacterial cells + oils | unknown; counted in the phase 0 audit | bacteria | CC-BY 4.0 |
| `raman_data` package: spontaneous biological sets (e.g. marine pathogens, others) | tens of thousands | bio | MIT package; per-set licences |
| RRUFF (excellent + fair unoriented, LR-Raman) | ~17–25k | minerals | covers 382–500 cm⁻¹ |
| MLROD | ~90–130k (sources disagree) | minerals | CC BY-NC (research use only) |
| Pharma API set (Flanagan 2025, figshare 27931131) | 3.5k | organics, 32 compounds | 785 nm, CC-BY |

**Excluded:**
- Every Bacteria-ID copy: the original files, PB2/PB30, and the `raman_data`
  Bacteria-ID entry.
- All SERS sets.
- All IR data.
- Simulated spectra.

### 4.2 Leakage guard

After resampling (§4.3), drop any corpus spectrum that is either:

- an exact hash match of any spectrum in any of our files, or
- a near-duplicate of one, meaning cosine similarity > 0.999 on the common
  grid.

The number dropped per source is written to the corpus manifest.

### 4.3 Common input format

- **Grid:** 350–1850 cm⁻¹, ascending, step 1.5 cm⁻¹, **1001 points**.
  Bacteria-ID (382–1792) lies inside it. The 2800–3100 cm⁻¹ region is not
  used, because our spectra do not contain it.
- **Resampling:**
  1. Sort each spectrum ascending.
  2. If the source is finer than 1.5 cm⁻¹, Gaussian-smooth it first
     (anti-aliasing).
  3. Interpolate linearly onto the grid.
  4. Never extrapolate.
- **Validity mask:** a boolean array of 1001 points marking the grid points
  covered by the source range. Uncovered points are set to 0 *and* masked.
  The mask is always carried with the spectrum.
- **Preprocessing, identical for every source and for our tasks:**
  1. Remove cosmic spikes (Whitaker–Hayes).
  2. Crop and resample.
  3. Min–max normalise over the valid points only.
- **No baseline correction.** Running one baseline algorithm over very
  different sources introduces its own source-specific artefacts. Baseline
  variation is handled by the pretraining corruption (§5.2). This choice is
  re-checked during the size trials (§7, phase 1).

### 4.4 Balancing

- **DeepeR:** at most 2,000 spectra per image or map.
- **Sampling across sources:** temperature sampling, with source probability
  proportional to √n_source.
- **Metadata per spectrum:** `source`, `laser_nm`, `domain` (bio, mineral or
  chem), `sample_id` (map/sample identifier where available) and `label`
  where available.
- Metadata is **not** a model input in v1. It is used for balancing, splits
  and diagnostics.

### 4.5 Held-out probe slice (A's "is it done?" signal)

A labelled slice of the corpus that pretraining never sees, **split by
sample/map, not by spectrum**:

- airborne-pathogen species;
- RRUFF mineral species with ≥ 5 spectra (top classes);
- the 32 API compounds.

It is about 10–15% of those sources, fixed by seed at corpus-build time.

### 4.6 Artefacts

- `foundation/data/corpus_v1/spectra.npy`: float16, shape (N, 1001).
- `foundation/data/corpus_v1/mask.npy`: bool, shape (N, 1001).
- `foundation/data/corpus_v1/meta.parquet`.
- `foundation/data/corpus_v1/splits.json`: pretrain / probe indices.
- `foundation/data/corpus_v1/manifest.json`: per source, the count before and
  after filtering, wavenumber range, laser, licence, download URL and
  checksum, plus the number of spectra removed by the leakage guard.

The raw downloads and the built corpus are gitignored. The manifest and the
build script are committed.

## 5. Model A

### 5.1 Architecture

Size is a configuration value. Three presets go into the size trials:

| Preset | Layers | Width d | Heads | FF width | Encoder params (≈12·d²·L) |
|---|---|---|---|---|---|
| Tiny | 4 | 128 | 4 | 512 | ~0.8M |
| Small | 6 | 256 | 8 | 1024 | ~4.7M |
| Base | 8 | 384 | 8 | 1536 | ~14M |

Pipeline for an input spectrum x of shape (1001) with its mask m:

1. **Multi-scale CNN stem.** Three parallel 1D convolutions with kernels 3, 9
   and 25 points (≈4.5, 13.5 and 37.5 cm⁻¹), 32 channels each, concatenated
   to 96 channels. GroupNorm and GELU. The input is the spectrum with the
   mask as a second channel, so the stem knows which points are real.
2. **Patching.** Pad to 1010 points. Non-overlapping 10-point patches give
   101 tokens of 15 cm⁻¹ each. A linear projection maps each to width d.
3. **Token validity.** A token is valid if more than 50% of its points are
   valid. Invalid tokens are excluded from attention through a key-padding
   mask.
4. **Position encoding.** Sinusoidal, computed from each patch's
   **centre wavenumber in cm⁻¹**, not its index.
5. **Transformer encoder.**
   - Pre-LayerNorm, dropout 0.1.
   - Attention is implemented with explicit `q`, `k`, `v` and `o` Linear
     layers rather than `nn.MultiheadAttention`, so LoRA can wrap every
     projection.
6. **Pooling.** Mean over valid tokens gives the embedding, of width d.

**No BatchNorm anywhere.** Batches mix very different sources, and the
fine-tuning batches are small. Accepted consequence: AdaBN-style adaptation
is not available.

### 5.2 Pretraining objective

**1. Corrupt the input.** These apply to the encoder input only; the target
is the uncorrupted spectrum. Each transform is drawn independently per sample:

- additive smooth baseline: a random low-order polynomial plus 0–2 broad
  Gaussian humps (fluorescence-like);
- wavenumber shift of ±5 cm⁻¹ and stretch of ±0.5%;
- intensity scaling between 0.8 and 1.2;
- Gaussian noise with a random SNR between 10 and 100;
- peak broadening by Gaussian convolution with σ from 0 to 3 points;
- 0–2 synthetic cosmic spikes.

The corrupted spectrum is then re-normalised (min–max over valid points).

**2. Mask.**
- Hide 50% of the valid tokens, in contiguous blocks of 2–6 tokens.
- The encoder sees only the visible tokens, MAE-style.

**3. Decode.**
- A light decoder (2 layers, width 128) receives the encoder outputs plus
  learned mask tokens with position encodings.
- It predicts the 10 values of every hidden patch.
- The decoder is discarded after pretraining.

**4. Loss.** Computed on hidden patches and valid points only:
- squared error against the per-patch-normalised clean target; plus
- 0.5 × squared error on the first derivative, so peaks count more than
  smooth background.

### 5.3 Training settings

- **Optimiser:** AdamW, β = (0.9, 0.95), weight decay 0.05.
- **Batch and learning rate:** batch size 512, base learning rate 3e-4.
- **Schedule:** 10 warmup epochs, then cosine decay.
- **Precision:** mixed (fp16 autocast with gradient scaling) on the RTX 3060.
- **Checkpoints:** every 10 epochs.
- **No time limits.** Runs are never stopped by elapsed time, so a run that is
  still improving is never cut off. A run ends only for one of two reasons:
  1. it reaches `max_epochs`, set in the config (default 200 for the main
     run, 50 for each size trial); or
  2. early stopping on the probe score fires (§5.4).
- **Resume.** If the probe score is still improving when `max_epochs` is
  reached, the run says so in its final line. It can then be resumed from its
  last checkpoint with a larger `max_epochs`. The optimiser and scheduler
  state are saved with each checkpoint so a resume continues exactly.
- **Progress output:** after every epoch, print a single line showing the
  epoch number out of `max_epochs` and how many are left, for example:

  ```
  [pretrain] epoch 1/200 (199 left) | loss 0.0412 | lr 3.0e-05 | 6m12s elapsed, ~20h ETA
  ```

  Elapsed time and ETA (average epoch time × epochs left) are **for display
  only and never stop a run**. The final line says why the run ended, for
  example `stopped at epoch 87/200: probe score flat for 3 checkpoints` or
  `finished 200/200: probe score still improving, resume with a larger
  max_epochs if needed`. Adaptation runs (§6) use the same format, tagged by
  arm, task, stage and seed, for example
  `[B1 T1 stage-ii seed 3] epoch 4/20 (16 left) | ...`.

### 5.4 Monitoring and stopping (corpus only)

Every 10 epochs, on a frozen checkpoint:

- **Probe score.** Fit k-NN (k = 20, cosine) and logistic-regression probes on
  the embeddings of the probe-slice training part. Score them on its
  held-out part, for each probe task (§4.5). The **probe score** is the mean
  macro-F1 across probe tasks.
- **RankMe** effective rank of the embeddings, to detect collapse.
- **Source-ID probe.** A linear classifier predicting `source` from the
  embedding. Very high accuracy here, while chemistry probes stay flat, means
  A is encoding instrument or dataset quirks.
- **Reconstruction samples.** Saved for 16 fixed probe spectra.

**Stop** when the probe score has not improved for 3 consecutive checkpoints,
or at `max_epochs` (§5.3). Keep the **best-probe** checkpoint, not the last.

**Gate G1.** The best checkpoint's k-NN probe score must beat k-NN on the raw
normalised spectra of the same probe slice. If it does not, stop and
diagnose before any adaptation.

## 6. Adaptation (B1, B2) and arms

### 6.1 Arms

Every arm runs on both tasks.

| Arm | Initialisation | What is trained | Role |
|---|---|---|---|
| **S** | random | everything | Same architecture, no A. The control for "did A help?" |
| **LP** | A | a new linear head only; A frozen | The cheapest reuse; shows what A learned by itself |
| **B1** | A | head first (LP warm-up), then everything | Branch 1 |
| **B2** | A | LoRA adapters + norm parameters + head; A frozen | Branch 2 |

Details:

- **B1:**
  1. Train the head only for 5 epochs. This warm-up happens once, at the
     start of the first stage; for T1, stage ii continues the full
     fine-tuning.
  2. Unfreeze everything.
  3. Use layer-wise learning-rate decay of 0.75 (the stem gets the smallest
     learning rate).
  4. Early-stop on validation loss with patience 5.
- **B2:**
  - Rank r on every `q`, `k`, `v`, `o` and both feed-forward Linear layers,
    with α = 2r and LoRA dropout 0.05.
  - Learning rate 10× B1's base.
  - r is the largest of {8, 4, 2} that keeps the trainable parameter fraction
    at **≤ 5%** for the chosen size. The fraction is reported.
- **S:** the same schedule as B1's unfrozen phase, without the warm-up.
- **Augmentation during adaptation:** the §5.2 corruption family at reduced
  strength (shift ±2 cm⁻¹, scale 0.9–1.1, SNR 30–100), with no masking.
- **T1 protocol:** stage i on the reference set, then stage ii on the
  finetune set. The arm's rule (what is frozen or trained) applies in both
  stages.
- **T2 protocol:** a single stage on 2018clinical.

### 6.2 Hyperparameter fairness

Each arm gets the same tuning budget:

1. Run a learning-rate sweep of 3 values on seed 1.
2. Select on the task's **validation** split (for T1, the stage-ii finetune
   validation split).
3. Fix the selected value for all seeds.

### 6.3 Reuse deliverable (goal c)

One A checkpoint file serves both tasks. Each task stores only its add-on:
the head (LP), or LoRA plus norm parameters plus head (B2).

Reported per task:
- trainable parameters;
- add-on file size;
- accuracy.

## 7. Phases and gates

| Phase | Work | Rough GPU estimate (not a limit) | Gate to continue |
|---|---|---|---|
| **0 Data audit and corpus build** | Download each source; verify counts, ranges, lasers, formats and licences; build corpus_v1 with the leakage guard and probe slice | CPU mostly | **G0:** ≥ 150k usable pretraining spectra after filtering and deduplication. If not, stop and revisit the source list |
| **1 Size trials** | Pretrain Tiny, Small and Base for the **same number of epochs** (`max_epochs` 50, same early-stopping rule) so the sizes are compared on equal data seen. Analyse probe curves, reconstructions, embedding maps coloured by source and domain, RankMe and the source-ID probe. Re-check the no-baseline-correction choice by inspecting reconstructions and source-ID leakage | ~15–20 h | Pick size, mask ratio and corruption strengths on the **probe score only**. Up to one extra round of trials if the first is inconclusive |
| **2 Main pretraining** | Pretrain the chosen configuration up to `max_epochs` 200, resumable if still improving | ~1 day | **G1** (§5.4) |
| **3 Adaptation** | Learning-rate sweeps, then 4 arms × 2 tasks × 5 seeds | ~13 h | All selection on validation only |
| **4 Final evaluation** | Score T1 test and T2 2019 **once**; statistics; report | < 1 h | — |

The estimates exist only to plan the week. No run is stopped by time. If a
run's printed ETA (§5.3) shows the week will be exceeded, we decide then
whether to let it continue.

## 8. Evaluation

### 8.1 Metrics

Per run:
- accuracy, macro-F1, per-class recall and the confusion matrix;
- trainable parameters, add-on size and wall-clock time.

### 8.2 Statistics

- Seeds 1–5, paired across arms by seed.
- **McNemar's test** on per-sample predictions over the same test set, for
  each paired seed.
- **Bootstrap 95% CI** of the accuracy difference, resampling test samples
  and seeds (10,000 resamples).
- Mean ± sd per arm.

### 8.3 Pre-registered questions and success criteria

| # | Comparison | Success means |
|---|---|---|
| **Q1** (goal a) | B1 vs S, and B2 vs S, per task | Mean difference > 0 with the 95% bootstrap CI excluding 0 → "A helps" |
| **Q2** | B1 vs B2, per task | Report the difference with its CI and the trainable-parameter ratio. Descriptive; no pass/fail threshold |
| **Q3** (goal c) | B2 or LP vs the best arm on that task | On **both** tasks, mean accuracy within **1.0 point** of the best arm, with ≤ 5% trainable parameters |

### 8.4 Known limitation

There is one pretraining run per configuration. Variation between pretraining
seeds is not measured and is stated in the report.

### 8.5 Optional, if budget remains

A data-efficiency curve on T1 stage ii: 5, 10, 25, 50 and 100 spectra per
class, for arms S, B1 and B2, with 3 seeds.

## 9. Code layout

```
foundation/
  README.md            how to build the corpus, pretrain, adapt, evaluate
  configs/             yaml: corpus, model presets, pretrain, adapt
  data/
    sources/           one module per source: download + parse to (wavenumbers, intensities, meta)
    harmonize.py       despike, sort, anti-alias, resample, mask, normalise
    build_corpus.py    sources -> leakage guard -> balancing meta -> splits -> corpus_v1
    audit.py           phase 0 report (counts, ranges, licences)
    tasks.py           our Bacteria-ID tasks T1/T2 on the common grid
    corrupt.py         corruption transforms (pretrain + mild adaptation variant)
  model/
    stem.py            multi-scale CNN stem
    encoder.py         patching, wavenumber positional encoding, transformer with explicit q/k/v/o
    decoder.py         MAE decoder (pretraining only)
    lora.py            LoRA wrapper for Linear layers; inject / count / save / load adapters
  pretrain.py          MAE pretraining loop, checkpointing, monitoring
  probe.py             k-NN / logistic probes, RankMe, source-ID probe
  adapt.py             arms S / LP / B1 / B2 on T1 / T2
  evaluate.py          single final test pass, McNemar, bootstrap CIs
  analysis/            reconstruction plots, embedding maps, probe curves
  runs/                per-run outputs (gitignored)
tests/foundation/      pytest: shapes, masking, resampling, LoRA, leakage guard
```

Each run writes a folder containing the config snapshot, git SHA, library
versions, metrics JSON and per-sample predictions.

## 10. Testing

Unit tests (pytest) before any long run:

- **Resampling:** reproduces a known synthetic peak position within 0.5 cm⁻¹,
  handles descending input, and never extrapolates (the mask is false
  outside the source range).
- **Masking:** about 50% of valid tokens are hidden in blocks; invalid tokens
  are never attended to.
- **Model:** forward shapes for each preset; parameter counts match §5.1
  within 10%.
- **LoRA:**
  - zero-initialised adapters leave outputs unchanged;
  - only adapter, norm and head parameters require gradients;
  - the trainable fraction is ≤ 5%;
  - save/load round-trips exactly.
- **Leakage guard:** a Bacteria-ID spectrum planted in a fake source is
  removed.
- **Smoke run:** 50 pretraining steps plus 1 adaptation epoch on a tiny
  subset, end to end.

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Corpus smaller than expected or hard to obtain | Phase 0 audit with gate G0 before any modelling |
| A learns instrument and source quirks instead of chemistry | Source-ID probe and embedding maps; validity masks instead of zero-fill; shared preprocessing |
| Domain mismatch with our bacteria (negative transfer) | Bio-leaning corpus; the S arm detects it; a negative Q1 is a legitimate result |
| Transformer underperforms on Raman | CNN stem, small presets, size trials; the S arm isolates architecture from pretraining |
| Leakage of our data into A | Hash and near-duplicate guard against all our files; probes use corpus data only |
| Overfitting through trial and error | All selection on validation or the probe slice; the test sets are used once |
| Licences (MLROD non-commercial, DeepeR unstated) | Recorded in the manifest; research use only |

## 12. Out of scope (v1)

- VICReg or other invariance losses.
- Laser-wavelength or source tokens as model input.
- Simulated spectra (QM9S etc.).
- SERS and IR data.
- Other-lab downstream tasks.
- Multiple pretraining seeds.
- AdaBN or test-time adaptation.
- Any comparison with the earlier project's models.
