# Foundation Model — Plan 2: Model A, Pretraining and Size Trials

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build model A (a multi-scale CNN stem feeding a wavenumber-aware transformer) and its masked-spectrum pretraining with probes and diagnostics. Then run the three size trials, report them so the user can understand and reconfigure, and run the main pretraining up to gate G1.

**Architecture:**
- **Spectra:** they come from `foundation/data/corpus_v1`, built in Plan 1. They're held in RAM and sampled with the corpus's √n source weights.
- **Each training step, all on the GPU:**
  1. Corrupt the batch.
  2. Tokenise: 101 patches of 15 cm⁻¹.
  3. Hide 50% of the valid tokens in blocks.
  4. Encode the visible tokens.
  5. Decode every token.
  6. Score the reconstruction of the hidden patches of the **clean** spectrum.
- **Every 10 epochs:** frozen-embedding probes on the corpus's held-out probe slice decide the best checkpoint and early stopping. Elapsed time never stops a run.
- **Afterwards:** a report module turns each run into plots and a comparison table.

**Tech Stack:** Python 3.12 repo venv: torch 2.11 with CUDA (RTX 3060, 12 GB), numpy, pandas, scikit-learn 1.9, matplotlib 3.11, pyyaml.

**Spec:** `docs/superpowers/specs/2026-10-01-raman-foundation-model-design.md`. This plan implements §5 (model A, pretraining, monitoring) and phases 1–2 of §7 (size trials, main pretraining, G1). Plan 3 covers adaptation and evaluation (§6, §8), including LoRA.

**Input:** `foundation/data/corpus_v1` from Plan 1 (manifest committed at `4fed5b3`). It has 244,640 spectra: 242,156 pretrain, 1,342 probe_train and 1,142 probe_test. The two probe tasks are `airborne` (5 species) and `rruff` (249 mineral species, about 3 spectra per class per side).

## Global Constraints

- **Separate code:** nothing in `foundation/` imports from `src/`, `legacy_baseline/`, `main.py`, `evaluate_kfold.py` or `generate_heatmaps.py`.
- **Our data stays out of A:** no Bacteria-ID file is read anywhere in this plan. Probes and checkpoint selection use the corpus probe slice only.
- **Grid and tokens:** 1001 points on 350–1850 cm⁻¹; 10-point patches give **101 tokens** of 15 cm⁻¹. A token is valid if **more than 50%** of its points are valid. Invalid tokens are never attended to.
- **Presets** (layers / width / heads / FF):
  - tiny: 4 / 128 / 4 / 512;
  - small: 6 / 256 / 8 / 1024;
  - base: 8 / 384 / 8 / 1536.
  - Dropout is 0.1 and normalisation is pre-LN. Transformer-block parameters must be within 10% of 12·d²·L.
- **Stem:** parallel Conv1d with kernels 3, 9 and 25, 32 channels each, giving 96; then GroupNorm and GELU. The input is 2 channels: the spectrum and its mask. **No BatchNorm anywhere.**
- **Position encoding:** sinusoidal over each patch's **centre wavenumber in cm⁻¹**.
- **Attention:** explicit `q`, `k`, `v`, `o` Linear layers, and an MLP of `fc1` and `fc2` (Plan 3's LoRA wraps these by name).
- **Corruption** (`pretrain` preset), applied to the input only:
  - baseline up to 0.5 × range (cubic plus 0–2 humps);
  - shift ±5 cm⁻¹ and stretch ±0.5%;
  - scale 0.8–1.2;
  - SNR 10–100;
  - broadening σ 0–3 points;
  - 0–2 cosmic spikes, each 1–2 points wide.
  - Then re-normalise min–max over the valid points.
- **Mild corruption** (for Plan 3): shift ±2 cm⁻¹, scale 0.9–1.1, SNR 30–100. Nothing else.
- **Masking:** hide 50% of the valid tokens in contiguous blocks of 2–6 tokens. The encoder sees only the visible tokens. At least one token always stays visible.
- **Decoder:** 2 layers, width 128, predicting 10 values per patch. It is discarded after pretraining.
- **Loss:** on hidden patches and valid points only. It is the squared error against the per-patch-normalised clean target plus 0.5 × the squared error on the first derivative.
- **Optimiser:** AdamW, β = (0.9, 0.95), weight decay 0.05, batch 512, learning rate 3e-4, 10 warmup epochs then cosine decay, fp16 autocast with a GradScaler.
- **Run length:** checkpoint and probe every 10 epochs. `max_epochs` is 50 for the trials and 200 for the main run. A run stops at `max_epochs` or after **3** probe checkpoints without improvement. **Never on time.** The best-probe checkpoint is kept, and runs are resumable.
- **Progress line:** `[pretrain] epoch E/MAX (L left) | loss … | lr … | … elapsed, ~… ETA`. Elapsed time and ETA are display only. The final line says why the run ended.
- **G1:** the best checkpoint's k-NN probe score must beat k-NN on the raw normalised spectra of the same probe slice.
- **Python:** `venv/Scripts/python.exe` from the repo root. Every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Deviations from the spec (decided while building and testing the prototype)

| # | Spec | Decision | Why |
|---|---|---|---|
| P1 | k-NN probe with k = 20 | k = 20 with **distance weighting** (`weights="distance"`) | The RRUFF probe has about 3 training spectra per class. Uniform k = 20 voting would be dominated by other classes |
| P2 | "probe score = mean macro-F1 across probe tasks" | probe score = mean over tasks of (k-NN F1 + logistic F1)/2. The `knn_score` is kept separately, for G1 | The spec names both probes but one number |
| P3 | per-patch-normalised target | The variance floor is `TARGET_EPS = 1e-3` | Otherwise a near-flat patch becomes unit-variance noise to predict |
| P4 | — | Gradient clipping at norm 1.0, and a run aborts after 20 consecutive non-finite losses | Stability under fp16; a NaN run must not spin silently |
| P5 | resume with a larger `max_epochs` | The cosine schedule is re-stretched to the new total from the current step | Keeps resume a one-flag operation |
| P6 | source-ID probe | Uses a fixed source-balanced pretrain sample (≤ 300 per source), also used for RankMe and the embedding map | Uniform sampling would be 54% MLROD |
| P7 | — | `model/lora.py` moves to Plan 3 | Only adaptation uses it |
| P8 | — | Block masking is fully vectorised and runs on the GPU | The per-sample Python loop cost 395 of 597 ms per step. Vectorised, a step takes 115/179/311 ms (tiny/small/base) |

**Measured on this machine against corpus_v1** (473 steps per epoch, 60-step benchmark):

| Preset | Per epoch | 50-epoch trial | 200-epoch main run | Peak GPU memory |
|---|---|---|---|---|
| tiny | 0.9 min | ~0.8 h | ~3.0 h | 1.7 GB |
| small | 1.4 min | ~1.2 h | ~4.7 h | 3.0 GB |
| base | 2.4 min | ~2.0 h | ~8.2 h | 4.9 GB |

**Known risk for G1:** in the benchmark, k-NN on raw normalised spectra already scores **0.625** on the probe slice. RRUFF mineral spectra have sharp, distinctive peaks that raw matching handles well. An untrained encoder scored 0.21–0.24. G1 is a real bar; Task 9 says what to do if it isn't met.

## Review Focus

Each item has a test in its owning task:

1. **A flat or degenerate spectrum through the corruption** (zero range): it must stay finite, never NaN. Task 1, `test_flat_spectrum_stays_finite`.
2. **A spectrum with very few valid tokens, or a mask ratio near 1:** at least one token stays visible, and invalid tokens never influence valid ones. Task 2, `test_invalid_tokens_never_influence_valid_ones`; Task 3, `test_block_mask_always_leaves_a_visible_token` and `test_block_mask_handles_uneven_valid_counts_and_gaps`.
3. **Resuming a run with a larger `max_epochs`** continues at the next epoch, with no repeated or skipped epochs. Task 5, `test_resume_continues_from_the_last_checkpoint`.
4. **A crash while a checkpoint is being written** must never corrupt the previous checkpoint. Task 5, `test_checkpoint_save_is_atomic`.
5. **An fp16 overflow or a NaN loss:** the step is skipped, and a run of them aborts loudly instead of training on garbage. Task 5, `test_nonfinite_losses_abort_instead_of_spinning`.

---

## File Structure

```
foundation/
  data/corrupt.py           Task 1  corruption presets (pretrain, mild) + batched torch transforms
  model/__init__.py         Task 2
  model/stem.py             Task 2  multi-scale CNN stem
  model/encoder.py          Task 2  presets, patching, wavenumber position encoding, transformer, embed()
  model/masking.py          Task 3  vectorised block masking, visible-token ordering
  model/decoder.py          Task 3  MAE decoder, patchify, mae_loss
  probe.py                  Task 4  probe set, embed, k-NN/logistic F1, RankMe, source-ID, raw baseline
  pretrain.py               Task 5  config, Corpus, training loop, checkpoints, resume, CLI
  configs/pretrain_trial_{tiny,small,base}.yaml   Task 5
  analysis/__init__.py      Task 6
  analysis/report.py        Task 6  plots, summary, comparison table, size recommendation, G1
  README.md                 Task 7  (append the pretraining section)
tests/foundation/
  conftest.py               Task 4  fake_corpus fixture shared by Tasks 4-6
  test_corrupt.py  test_encoder.py  test_mae.py  test_probe.py  test_pretrain.py  test_report.py
```

All tests: `venv/Scripts/python.exe -m pytest tests/foundation -v`.

---

### Task 1: Corruption transforms

**Files:**
- Create: `foundation/data/corrupt.py`
- Test: `tests/foundation/test_corrupt.py`

**Interfaces:**
- Consumes: `GRID_STEP` from `foundation/data/harmonize.py` (Plan 1).
- Produces:
  - `CorruptionConfig` (frozen dataclass) and the presets `PRETRAIN`, `MILD` and `PRESETS = {"pretrain", "mild"}`.
  - `minmax_valid(x, m)`.
  - `shift_stretch(x, m, shift_cm: Tensor[B], stretch: Tensor[B]) -> (x, m)`.
  - `broaden(x, m, sigma: Tensor[B], max_sigma)`.
  - `smooth_baseline(B, N, gen, device)` and `add_spikes(x, m, max_spikes, gen)`.
  - **`corrupt(x, m, cfg=PRETRAIN, generator=None) -> (x_corrupt, m_corrupt)`**. x is (B, 1001) float and m is bool, on any device; the generator must live on the same device.

- [ ] **Step 1: Write the failing tests**

`tests/foundation/test_corrupt.py`:
```python
import numpy as np
import torch

from foundation.data.corrupt import (MILD, PRETRAIN, CorruptionConfig, broaden, corrupt,
                                     minmax_valid, shift_stretch)


def _peaks(B=8, N=1001, centre=500, width=4.0):
    p = torch.arange(N, dtype=torch.float32)
    x = torch.exp(-0.5 * ((p - centre) / width) ** 2).expand(B, N).clone()
    m = torch.ones(B, N, dtype=torch.bool)
    m[:, :40] = False                    # like a source that starts above 350 cm^-1
    return minmax_valid(x, m), m


def test_output_is_finite_normalised_and_zero_off_mask():
    x, m = _peaks()
    g = torch.Generator().manual_seed(0)
    y, ym = corrupt(x, m, PRETRAIN, generator=g)
    assert y.shape == x.shape and ym.dtype == torch.bool
    assert torch.isfinite(y).all()
    assert torch.allclose(y[ym].min(), torch.tensor(0.0), atol=1e-6)
    assert y[ym].max() <= 1 + 1e-6
    assert (y[~ym] == 0).all()


def test_flat_spectrum_stays_finite():
    x = torch.zeros(4, 1001)
    m = torch.ones(4, 1001, dtype=torch.bool)
    y, _ = corrupt(x, m, PRETRAIN, generator=torch.Generator().manual_seed(1))
    assert torch.isfinite(y).all()


def test_same_generator_seed_gives_same_corruption():
    x, m = _peaks()
    a, _ = corrupt(x, m, PRETRAIN, generator=torch.Generator().manual_seed(7))
    b, _ = corrupt(x, m, PRETRAIN, generator=torch.Generator().manual_seed(7))
    c, _ = corrupt(x, m, PRETRAIN, generator=torch.Generator().manual_seed(8))
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_shift_moves_peak_to_higher_wavenumber_by_shift_over_step():
    x, m = _peaks(B=1)
    y, ym = shift_stretch(x, m, torch.tensor([4.5]), torch.tensor([0.0]))
    assert int(y[0].argmax()) == 503          # 4.5 cm^-1 / 1.5 cm^-1 = 3 points


def test_mask_moves_with_the_shift_and_never_extrapolates():
    x, m = _peaks(B=2)                         # valid from index 40
    _, ym = shift_stretch(x, m, torch.tensor([4.5, -4.5]), torch.tensor([0.0, 0.0]))
    assert int(ym[0].float().argmax()) == 43   # +3 points
    assert int(ym[1].float().argmax()) == 37   # -3 points
    assert not ym[1, -3:].any()                # nothing to copy from past the end


def test_broaden_widens_band_and_zero_sigma_is_identity():
    x, m = _peaks(B=2)
    y = broaden(x, m, torch.tensor([0.0, 3.0]), 3.0)
    assert torch.allclose(y[0], x[0], atol=1e-5)
    width = lambda v: int((v > 0.5 * v.max()).sum())
    assert width(y[1]) > width(x[1])


def test_mild_preset_stays_close_to_the_input():
    x, m = _peaks(B=16)
    y, ym = corrupt(x, m, MILD, generator=torch.Generator().manual_seed(3))
    a, b = x[ym].numpy(), y[ym].numpy()
    assert np.corrcoef(a, b)[0, 1] > 0.9
    assert MILD.max_spikes == 0 and MILD.baseline_max == 0


def test_works_on_cuda_when_available():
    if not torch.cuda.is_available():
        return
    x, m = _peaks()
    g = torch.Generator(device="cuda").manual_seed(0)
    y, _ = corrupt(x.cuda(), m.cuda(), PRETRAIN, generator=g)
    assert y.is_cuda and torch.isfinite(y).all()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `venv/Scripts/python.exe -m pytest tests/foundation/test_corrupt.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'foundation.data.corrupt'`.

- [ ] **Step 3: Write the implementation**

`foundation/data/corrupt.py`:
```python
"""Corruptions that mimic session-to-session nuisances (spec sections 5.2, 6.1).

Batched torch ops on (B, N) spectra on the shared grid with a (B, N) bool
validity mask, so they run on the GPU inside the training step. Inputs are
min-max normalised over valid points (range 1); outputs are re-normalised the
same way and are 0 wherever the returned mask is False.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .harmonize import GRID_STEP


@dataclass(frozen=True)
class CorruptionConfig:
    baseline_max: float = 0.5           # added baseline peak height, x spectrum range
    shift_cm: float = 5.0               # uniform in +-shift_cm
    stretch: float = 0.005              # uniform in +-stretch (fraction of the axis)
    scale: tuple = (0.8, 1.2)
    snr: tuple = (10.0, 100.0)          # noise sd = range / snr
    broaden_sigma: float = 3.0          # max Gaussian sigma in points
    max_spikes: int = 2


PRETRAIN = CorruptionConfig()
MILD = CorruptionConfig(baseline_max=0.0, shift_cm=2.0, stretch=0.0, scale=(0.9, 1.1),
                        snr=(30.0, 100.0), broaden_sigma=0.0, max_spikes=0)
PRESETS = {"pretrain": PRETRAIN, "mild": MILD}


def _uniform(n, lo, hi, gen, device):
    return lo + (hi - lo) * torch.rand(n, generator=gen, device=device)


def minmax_valid(x, m, eps=1e-6):
    big = torch.finfo(x.dtype).max
    lo = torch.where(m, x, torch.full_like(x, big)).amin(dim=1, keepdim=True)
    hi = torch.where(m, x, torch.full_like(x, -big)).amax(dim=1, keepdim=True)
    out = (x - lo) / (hi - lo).clamp_min(eps)
    return torch.where(m, out, torch.zeros_like(out))


def shift_stretch(x, m, shift_cm, stretch):
    """Move features by shift_cm (positive = to higher wavenumber) and stretch the
    axis about its centre. Linear interpolation; a point is valid only if both
    source neighbours were valid."""
    B, N = x.shape
    p = torch.arange(N, device=x.device, dtype=x.dtype).expand(B, N)
    c = (N - 1) / 2
    src = c + (p - c) / (1 + stretch[:, None]) - shift_cm[:, None] / GRID_STEP
    inside = (src >= 0) & (src <= N - 1)
    i0 = src.floor().clamp(0, N - 2).long()
    w = (src - i0).clamp(0, 1)
    x0, x1 = x.gather(1, i0), x.gather(1, i0 + 1)
    m0, m1 = m.gather(1, i0), m.gather(1, i0 + 1)
    out = x0 * (1 - w) + x1 * w
    valid = inside & m0 & m1
    return torch.where(valid, out, torch.zeros_like(out)), valid


def broaden(x, m, sigma, max_sigma):
    """Per-sample Gaussian blur (sigma in points), normalised over valid points so
    invalid zeros do not bleed in. sigma ~ 0 leaves the spectrum unchanged."""
    if max_sigma <= 0:
        return x
    B, N = x.shape
    half = math.ceil(3 * max_sigma)
    k = torch.arange(-half, half + 1, device=x.device, dtype=x.dtype)
    s = sigma.clamp_min(1e-3)[:, None]
    kern = torch.exp(-0.5 * (k[None, :] / s) ** 2)
    kern = kern / kern.sum(dim=1, keepdim=True)
    mf = m.to(x.dtype)
    num = F.conv1d((x * mf)[None], kern[:, None, :], padding=half, groups=B)[0]
    den = F.conv1d(mf[None], kern[:, None, :], padding=half, groups=B)[0]
    return torch.where(m, num / den.clamp_min(1e-6), torch.zeros_like(x))


def smooth_baseline(B, N, gen, device, dtype=torch.float32):
    """Random cubic polynomial plus 0-2 broad Gaussian humps, min-max scaled to [0, 1]."""
    t = torch.linspace(-1, 1, N, device=device, dtype=dtype)
    coef = _uniform((B, 4), -1, 1, gen, device)
    base = sum(coef[:, i:i + 1] * t[None, :] ** i for i in range(4))
    n_humps = torch.randint(0, 3, (B,), generator=gen, device=device)
    for h in range(2):
        on = (n_humps > h).to(dtype)[:, None]
        centre = _uniform((B, 1), -1, 1, gen, device)
        width = _uniform((B, 1), 0.2, 0.6, gen, device)
        amp = _uniform((B, 1), 0.5, 2.0, gen, device)
        base = base + on * amp * torch.exp(-0.5 * ((t[None, :] - centre) / width) ** 2)
    lo, hi = base.amin(1, keepdim=True), base.amax(1, keepdim=True)
    return (base - lo) / (hi - lo).clamp_min(1e-6)


def add_spikes(x, m, max_spikes, gen):
    """0..max_spikes cosmic spikes, 1-2 points wide, at random valid positions."""
    if max_spikes <= 0:
        return x
    B, N = x.shape
    out = x.clone()
    n = torch.randint(0, max_spikes + 1, (B,), generator=gen, device=x.device)
    probs = m.to(x.dtype) + 1e-12
    for j in range(max_spikes):
        pos = torch.multinomial(probs, 1, generator=gen)[:, 0]
        height = _uniform(B, 0.2, 1.0, gen, x.device) * (n > j)
        wide = torch.rand(B, generator=gen, device=x.device) < 0.5
        rows = torch.arange(B, device=x.device)
        out[rows, pos] += height
        nxt = (pos + 1).clamp(max=N - 1)
        out[rows, nxt] += height * wide * m[rows, nxt]
    return out


def corrupt(x, m, cfg=PRETRAIN, generator=None):
    """Apply cfg's corruptions; returns (x_corrupt, m_corrupt)."""
    B, N = x.shape
    dev = x.device
    g = generator
    x = x.float()
    shift = _uniform(B, -cfg.shift_cm, cfg.shift_cm, g, dev)
    stretch = _uniform(B, -cfg.stretch, cfg.stretch, g, dev)
    x, m = shift_stretch(x, m, shift, stretch)
    x = broaden(x, m, _uniform(B, 0.0, cfg.broaden_sigma, g, dev), cfg.broaden_sigma)
    x = x * _uniform((B, 1), cfg.scale[0], cfg.scale[1], g, dev)
    if cfg.baseline_max > 0:
        amp = _uniform((B, 1), 0.0, cfg.baseline_max, g, dev)
        x = x + amp * smooth_baseline(B, N, g, dev)
    snr = _uniform((B, 1), cfg.snr[0], cfg.snr[1], g, dev)
    x = x + torch.randn(B, N, generator=g, device=dev) / snr
    x = add_spikes(x, m, cfg.max_spikes, g)
    return minmax_valid(x, m), m
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `venv/Scripts/python.exe -m pytest tests/foundation/test_corrupt.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add foundation/data/corrupt.py tests/foundation/test_corrupt.py
git commit -m "foundation: corruption transforms for pretraining (and mild preset for adaptation)"
```

---

### Task 2: Stem and encoder

**Files:**
- Create: `foundation/model/__init__.py` (empty), `foundation/model/stem.py`, `foundation/model/encoder.py`
- Test: `tests/foundation/test_encoder.py`

**Interfaces:**
- Consumes: `GRID_START`, `GRID_STEP` and `N_GRID` from `foundation/data/harmonize.py`.
- Produces:
  - `Stem(channels=32, kernels=(3, 9, 25), groups=8)`, with `out_channels` and `forward(x, m) -> (B, 96, N)`.
  - `PATCH = 10` and `N_TOKENS = 101`; `EncoderConfig(layers, width, heads, ff, dropout=0.1)`; `PRESETS` with keys `tiny`, `small`, `base`.
  - `patch_centres_cm()`, `sinusoidal(centres_cm, dim)` and `token_validity(m) -> (B, 101) bool`.
  - `Attention` (with `.q`, `.k`, `.v`, `.o`), `MLP` (with `.fc1`, `.fc2`) and `Block(width, heads, ff, dropout)`, whose `forward(h, key_valid)` the decoder reuses.
  - `SpectralEncoder(cfg)`:
    - `.cfg`;
    - `tokenize(x, m) -> (tokens (B, 101, width), valid (B, 101))`;
    - `encode(tokens, key_valid) -> (B, K, width)`;
    - `forward(x, m) -> (h, valid)`;
    - `embed(x, m) -> (B, width)`, the mean over valid tokens.
  - `count_parameters(module)`.

- [ ] **Step 1: Write the failing tests**

`tests/foundation/test_encoder.py`:
```python
import pytest
import torch

from foundation.model.encoder import (N_TOKENS, PRESETS, SpectralEncoder, count_parameters,
                                      patch_centres_cm, token_validity)


def _batch(B=3, start=40, stop=900):
    x = torch.rand(B, 1001)
    m = torch.zeros(B, 1001, dtype=torch.bool)
    m[:, start:stop] = True
    return x * m, m


@pytest.mark.parametrize("name", ["tiny", "small", "base"])
def test_forward_shapes_and_block_parameter_counts(name):
    cfg = PRESETS[name]
    enc = SpectralEncoder(cfg).eval()
    x, m = _batch()
    h, valid = enc(x, m)
    assert h.shape == (3, N_TOKENS, cfg.width) and valid.shape == (3, N_TOKENS)
    assert enc.embed(x, m).shape == (3, cfg.width)
    blocks = count_parameters(enc.blocks)
    assert abs(blocks - 12 * cfg.width ** 2 * cfg.layers) / (12 * cfg.width ** 2 * cfg.layers) < 0.10


def test_attention_layers_are_explicit_linears_for_lora():
    enc = SpectralEncoder(PRESETS["tiny"])
    names = {n for n, mod in enc.named_modules() if isinstance(mod, torch.nn.Linear)}
    for part in ("attn.q", "attn.k", "attn.v", "attn.o", "mlp.fc1", "mlp.fc2"):
        assert f"blocks.0.{part}" in names


def test_no_batchnorm_anywhere():
    enc = SpectralEncoder(PRESETS["small"])
    assert not any(isinstance(mod, torch.nn.modules.batchnorm._BatchNorm) for mod in enc.modules())


def test_patch_centres_are_wavenumbers():
    c = patch_centres_cm()
    assert c[0] == pytest.approx(350 + 1.5 * 4.5) and c.shape == (N_TOKENS,)


def test_token_validity_needs_more_than_half_valid_points():
    m = torch.zeros(1, 1001, dtype=torch.bool)
    m[0, :15] = True                              # token 0 full, token 1 half
    v = token_validity(m)
    assert v[0, 0] and not v[0, 1] and not v[0, -1]   # last token has 1 real point


def test_invalid_tokens_never_influence_valid_ones():
    enc = SpectralEncoder(PRESETS["tiny"]).eval()
    tokens = torch.randn(2, N_TOKENS, 128)
    valid = torch.ones(2, N_TOKENS, dtype=torch.bool)
    valid[:, 60:] = False
    a = enc.encode(tokens, valid)
    tokens2 = tokens.clone()
    tokens2[:, 60:] = 100 * torch.randn(2, N_TOKENS - 60, 128)
    b = enc.encode(tokens2, valid)
    assert torch.allclose(a[:, :60], b[:, :60], atol=1e-5)


def test_embed_ignores_invalid_tokens():
    enc = SpectralEncoder(PRESETS["tiny"]).eval()
    x, m = _batch(B=1)
    e1 = enc.embed(x, m)
    x2 = x.clone()
    x2[0, 950:] = 5.0                             # invalid region (mask False)
    assert torch.allclose(e1, enc.embed(x2, m), atol=1e-6)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `venv/Scripts/python.exe -m pytest tests/foundation/test_encoder.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'foundation.model'`.

- [ ] **Step 3: Write the implementation**

`foundation/model/__init__.py`: an empty file.

`foundation/model/stem.py`:
```python
"""Multi-scale CNN stem (spec section 5.1, step 1)."""
from __future__ import annotations

import torch
import torch.nn as nn


class Stem(nn.Module):
    """Parallel 1D convolutions with kernels of 3, 9 and 25 points (~4.5, 13.5 and
    37.5 cm^-1), `channels` each, concatenated, then GroupNorm and GELU. The input
    has two channels, the spectrum (zeroed where invalid) and the validity mask,
    so the stem knows which points are real."""

    def __init__(self, channels=32, kernels=(3, 9, 25), groups=8):
        super().__init__()
        self.convs = nn.ModuleList(
            nn.Conv1d(2, channels, k, padding=k // 2) for k in kernels)
        self.out_channels = channels * len(kernels)
        self.norm = nn.GroupNorm(groups, self.out_channels)
        self.act = nn.GELU()

    def forward(self, x, m):
        mf = m.to(x.dtype)
        inp = torch.stack([x * mf, mf], dim=1)                 # (B, 2, N)
        h = torch.cat([conv(inp) for conv in self.convs], dim=1)
        return self.act(self.norm(h))                           # (B, C, N)
```

`foundation/model/encoder.py`:
```python
"""Model A's encoder (spec section 5.1).

stem -> 10-point patches (101 tokens of 15 cm^-1) -> linear projection ->
sinusoidal encoding of each patch's centre wavenumber -> pre-LN transformer
with explicit q/k/v/o Linear layers (so LoRA can wrap them in Plan 3) ->
LayerNorm. Invalid tokens (<= 50% valid points) are never attended to.
No BatchNorm anywhere.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.harmonize import GRID_START, GRID_STEP, N_GRID
from .stem import Stem

PATCH = 10
N_TOKENS = math.ceil(N_GRID / PATCH)        # 101


@dataclass(frozen=True)
class EncoderConfig:
    layers: int
    width: int
    heads: int
    ff: int
    dropout: float = 0.1


PRESETS = {
    "tiny": EncoderConfig(layers=4, width=128, heads=4, ff=512),
    "small": EncoderConfig(layers=6, width=256, heads=8, ff=1024),
    "base": EncoderConfig(layers=8, width=384, heads=8, ff=1536),
}


def patch_centres_cm(patch=PATCH, n_tokens=N_TOKENS):
    idx = patch * torch.arange(n_tokens, dtype=torch.float32) + (patch - 1) / 2
    return GRID_START + GRID_STEP * idx


def sinusoidal(centres_cm, dim, patch=PATCH):
    """Standard sinusoidal table over position = centre / patch width (in patch
    units), so tokens from any wavenumber range share one encoding."""
    pos = centres_cm / (GRID_STEP * patch)
    i = torch.arange(dim // 2, dtype=torch.float32)
    freq = torch.exp(-math.log(10000.0) * 2 * i / dim)
    ang = pos[:, None] * freq[None, :]
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)       # (T, dim)


def token_validity(m, patch=PATCH, n_tokens=N_TOKENS):
    pad = n_tokens * patch - m.shape[1]
    mp = F.pad(m.float(), (0, pad))
    return mp.view(m.shape[0], n_tokens, patch).mean(dim=2) > 0.5   # (B, T)


class Attention(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.o = nn.Linear(width, width)

    def forward(self, h, key_valid):
        B, T, D = h.shape
        split = lambda t: t.view(B, T, self.heads, D // self.heads).transpose(1, 2)
        q, k, v = split(self.q(h)), split(self.k(h)), split(self.v(h))
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=key_valid[:, None, None, :],
            dropout_p=self.dropout if self.training else 0.0)
        return self.o(out.transpose(1, 2).reshape(B, T, D))


class MLP(nn.Module):
    def __init__(self, width, ff, dropout):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(width, ff), nn.Linear(ff, width)
        self.act, self.drop = nn.GELU(), nn.Dropout(dropout)

    def forward(self, h):
        return self.fc2(self.drop(self.act(self.fc1(h))))


class Block(nn.Module):
    def __init__(self, width, heads, ff, dropout):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attn = Attention(width, heads, dropout)
        self.mlp = MLP(width, ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, key_valid):
        h = h + self.drop(self.attn(self.ln1(h), key_valid))
        return h + self.drop(self.mlp(self.ln2(h)))


class SpectralEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.stem = Stem()
        self.proj = nn.Linear(self.stem.out_channels * PATCH, cfg.width)
        self.register_buffer("pos", sinusoidal(patch_centres_cm(), cfg.width), persistent=False)
        self.blocks = nn.ModuleList(
            Block(cfg.width, cfg.heads, cfg.ff, cfg.dropout) for _ in range(cfg.layers))
        self.norm = nn.LayerNorm(cfg.width)

    def tokenize(self, x, m):
        """(B, 1001) spectrum + mask -> (B, 101, width) tokens, (B, 101) validity."""
        h = self.stem(x, m)                                      # (B, C, N)
        h = F.pad(h, (0, N_TOKENS * PATCH - h.shape[2]))
        B, C, _ = h.shape
        h = h.view(B, C, N_TOKENS, PATCH).permute(0, 2, 1, 3).reshape(B, N_TOKENS, C * PATCH)
        return self.proj(h) + self.pos, token_validity(m)

    def encode(self, tokens, key_valid):
        h = tokens
        for blk in self.blocks:
            h = blk(h, key_valid)
        return self.norm(h)

    def forward(self, x, m):
        tokens, valid = self.tokenize(x, m)
        return self.encode(tokens, valid), valid

    def embed(self, x, m):
        """Mean over valid tokens -> (B, width): the spectrum's embedding."""
        h, valid = self.forward(x, m)
        w = valid.to(h.dtype)[..., None]
        return (h * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)


def count_parameters(module):
    return sum(p.numel() for p in module.parameters())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `venv/Scripts/python.exe -m pytest tests/foundation/test_encoder.py -v`
Expected: 9 passed (the parametrised shape test counts as 3).

- [ ] **Step 5: Commit**

```bash
git add foundation/model/__init__.py foundation/model/stem.py foundation/model/encoder.py tests/foundation/test_encoder.py
git commit -m "foundation: model A encoder (multi-scale stem, wavenumber positions, q/k/v/o attention)"
```

---

### Task 3: Block masking, MAE decoder and loss

**Files:**
- Create: `foundation/model/masking.py`, `foundation/model/decoder.py`
- Test: `tests/foundation/test_mae.py`

**Interfaces:**
- Consumes: `N_TOKENS`, `PATCH`, `Block`, `patch_centres_cm`, `sinusoidal`, `PRESETS` and `SpectralEncoder` (Task 2).
- Produces:
  - `block_mask(valid, ratio=0.5, min_block=2, max_block=6, generator=None) -> hidden (B, T) bool`. It is vectorised; the generator must be on valid's device.
  - `visible_order(valid, hidden) -> (keep (B, K) long, keep_valid (B, K) bool)`.
  - `TARGET_EPS = 1e-3`; `MAEDecoder(enc_width, width=128, layers=2, heads=4, ff=512)` with `forward(h_vis, keep, keep_valid, valid) -> (B, 101, 10)`.
  - `patchify(x) -> (B, 101, 10)`.
  - `mae_loss(pred, clean, clean_mask, hidden, deriv_weight=0.5) -> (loss tensor, {"mse": float, "dmse": float})`.

- [ ] **Step 1: Write the failing tests**

`tests/foundation/test_mae.py`:
```python
import torch

from foundation.model.decoder import MAEDecoder, mae_loss, patchify
from foundation.model.encoder import N_TOKENS, PRESETS, SpectralEncoder
from foundation.model.masking import block_mask, visible_order


def _batch(B=3, start=40, stop=900):
    x = torch.rand(B, 1001)
    m = torch.zeros(B, 1001, dtype=torch.bool)
    m[:, start:stop] = True
    return x * m, m


def test_block_mask_hides_half_of_valid_tokens_in_runs():
    valid = torch.zeros(4, N_TOKENS, dtype=torch.bool)
    valid[:, 5:85] = True                          # 80 valid tokens
    hidden = block_mask(valid, generator=torch.Generator().manual_seed(0))
    assert (hidden & ~valid).sum() == 0
    assert (hidden.sum(dim=1) == 40).all()
    runs = torch.diff(hidden.int(), dim=1).eq(1).sum(dim=1) + hidden[:, 0].int()
    assert (40 / runs.float()).min() >= 2.0        # mean run length >= min_block


def test_block_mask_always_leaves_a_visible_token():
    valid = torch.zeros(2, N_TOKENS, dtype=torch.bool)
    valid[:, 10:12] = True                         # only 2 valid tokens
    hidden = block_mask(valid, ratio=0.99, generator=torch.Generator().manual_seed(0))
    assert ((valid & ~hidden).sum(dim=1) >= 1).all()


def test_visible_order_gathers_visible_tokens_first():
    valid = torch.ones(2, N_TOKENS, dtype=torch.bool)
    hidden = torch.zeros(2, N_TOKENS, dtype=torch.bool)
    hidden[0, :50] = True
    hidden[1, :30] = True
    keep, keep_valid = visible_order(valid, hidden)
    assert keep.shape == (2, 71)
    assert keep_valid[0].sum() == 51 and keep_valid[1].all()
    assert not hidden.gather(1, keep)[keep_valid].any()


def test_decoder_shape_and_hidden_tokens_get_information_from_visible():
    torch.manual_seed(0)
    enc, dec = SpectralEncoder(PRESETS["tiny"]).eval(), MAEDecoder(128).eval()
    x, m = _batch(B=2)
    tokens, valid = enc.tokenize(x, m)
    hidden = block_mask(valid, generator=torch.Generator().manual_seed(1))
    keep, keep_valid = visible_order(valid, hidden)
    h = enc.encode(tokens.gather(1, keep[..., None].expand(-1, -1, 128)), keep_valid)
    pred = dec(h, keep, keep_valid, valid)
    assert pred.shape == (2, N_TOKENS, 10)
    h2 = h + 1.0
    assert not torch.allclose(dec(h2, keep, keep_valid, valid)[hidden], pred[hidden])


def test_mae_loss_is_zero_for_perfect_prediction_and_ignores_visible_tokens():
    x, m = _batch(B=2)
    tgt = patchify(x)
    pm = patchify(m.float()) > 0
    cnt = pm.sum(-1, keepdim=True).clamp_min(1)
    mean = (tgt * pm).sum(-1, keepdim=True) / cnt
    var = (((tgt - mean) ** 2) * pm).sum(-1, keepdim=True) / cnt
    perfect = (tgt - mean) / torch.sqrt(var + 1e-3)
    hidden = torch.zeros(2, N_TOKENS, dtype=torch.bool)
    hidden[:, 20:40] = True
    loss, parts = mae_loss(perfect, x, m, hidden)
    assert float(loss) < 1e-8
    wrong_visible = perfect.clone()
    wrong_visible[:, 50:] += 3.0
    assert float(mae_loss(wrong_visible, x, m, hidden)[0]) < 1e-8


def test_mae_loss_is_finite_on_flat_patches():
    x = torch.zeros(1, 1001)
    m = torch.ones(1, 1001, dtype=torch.bool)
    hidden = torch.ones(1, N_TOKENS, dtype=torch.bool)
    loss, _ = mae_loss(torch.zeros(1, N_TOKENS, 10), x, m, hidden)
    assert torch.isfinite(loss) and float(loss) == 0.0


def test_block_mask_handles_uneven_valid_counts_and_gaps():
    valid = torch.zeros(3, N_TOKENS, dtype=torch.bool)
    valid[0, 0:101] = True
    valid[1, 10:20] = True
    valid[1, 50:60] = True                         # two separate valid regions
    valid[2, 30:33] = True                         # 3 valid tokens
    hidden = block_mask(valid, generator=torch.Generator().manual_seed(4))
    assert (hidden & ~valid).sum() == 0
    assert hidden.sum(dim=1).tolist() == [50, 10, 2]   # round(0.5*n), capped at n-1


def test_block_mask_is_reproducible_with_a_generator():
    valid = torch.ones(5, N_TOKENS, dtype=torch.bool)
    a = block_mask(valid, generator=torch.Generator().manual_seed(9))
    b = block_mask(valid, generator=torch.Generator().manual_seed(9))
    assert torch.equal(a, b)


def test_block_mask_runs_on_cuda_when_available():
    if not torch.cuda.is_available():
        return
    valid = torch.ones(512, N_TOKENS, dtype=torch.bool, device="cuda")
    hidden = block_mask(valid, generator=torch.Generator(device="cuda").manual_seed(0))
    assert hidden.is_cuda and (hidden.sum(dim=1) == 50).all()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `venv/Scripts/python.exe -m pytest tests/foundation/test_mae.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'foundation.model.decoder'`.

- [ ] **Step 3: Write the implementation**

`foundation/model/masking.py`:
```python
"""Contiguous block masking of valid tokens (spec section 5.2, step 2).

Vectorised: no Python loop over samples, so it runs on the GPU inside the
training step (a per-sample loop cost ~400 ms per batch of 512).
"""
from __future__ import annotations

import torch


def block_mask(valid, ratio=0.5, min_block=2, max_block=6, generator=None):
    """Hide round(ratio * n_valid) valid tokens per sample, in runs of
    min_block..max_block consecutive valid tokens (adjacent runs may merge).
    At least one valid token always stays visible. `generator` must live on
    valid's device. Returns a (B, T) bool tensor."""
    B, T = valid.shape
    dev = valid.device
    n = valid.sum(dim=1, keepdim=True)                                   # (B, 1)
    target = torch.minimum(torch.round(ratio * n), (n - 1).clamp_min(0))
    rank = valid.long().cumsum(dim=1) - 1                                # rank among valid tokens
    # Cut the valid-rank axis into random blocks of min..max tokens.
    lengths = torch.randint(min_block, max_block + 1, (B, T), generator=generator, device=dev)
    ends = lengths.cumsum(dim=1)
    starts = ends - lengths
    eff = (torch.minimum(ends, n) - starts).clamp_min(0)                 # tokens per block
    # Hide whole blocks in random order until the target is met (last one partly).
    order = torch.argsort(torch.rand(B, T, generator=generator, device=dev), dim=1)
    eff_sorted = eff.gather(1, order)
    before = eff_sorted.cumsum(dim=1) - eff_sorted
    take_sorted = torch.minimum((target - before).clamp_min(0), eff_sorted)
    take = torch.zeros_like(take_sorted).scatter(1, order, take_sorted)  # per block id
    r = torch.arange(T, device=dev).expand(B, T).contiguous()
    block = torch.searchsorted(ends, r, right=True).clamp(max=T - 1)     # block of each rank
    hidden_rank = (r - starts.gather(1, block)) < take.gather(1, block)
    return valid & hidden_rank.gather(1, rank.clamp_min(0))


def visible_order(valid, hidden):
    """Indices that gather each sample's visible tokens first.
    Returns keep (B, K) and keep_valid (B, K), K = max visible count."""
    visible = valid & ~hidden
    order = torch.argsort((~visible).to(torch.int8), dim=1, stable=True)
    k = int(visible.sum(dim=1).max())
    keep = order[:, :k]
    return keep, visible.gather(1, keep)
```

`foundation/model/decoder.py`:
```python
"""MAE decoder and reconstruction loss (spec section 5.2, steps 3-4).

The decoder is used only during pretraining and is discarded afterwards.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import N_TOKENS, PATCH, Block, patch_centres_cm, sinusoidal

TARGET_EPS = 1e-3   # variance floor for per-patch normalisation: a near-flat
                    # patch is not blown up into unit-variance noise


class MAEDecoder(nn.Module):
    def __init__(self, enc_width, width=128, layers=2, heads=4, ff=512):
        super().__init__()
        self.embed = nn.Linear(enc_width, width)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, width))
        nn.init.normal_(self.mask_token, std=0.02)
        self.register_buffer("pos", sinusoidal(patch_centres_cm(), width), persistent=False)
        self.blocks = nn.ModuleList(Block(width, heads, ff, 0.0) for _ in range(layers))
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, PATCH)

    def forward(self, h_vis, keep, keep_valid, valid):
        """h_vis (B, K, enc_width) for the tokens in keep (B, K) -> (B, T, PATCH)."""
        B = h_vis.shape[0]
        mask_tok = self.mask_token.to(h_vis.dtype)
        src = torch.where(keep_valid[..., None], self.embed(h_vis).to(h_vis.dtype), mask_tok)
        z = mask_tok.expand(B, N_TOKENS, -1).clone()
        z = z.scatter(1, keep[..., None].expand(-1, -1, z.shape[2]), src)
        z = z + self.pos
        for blk in self.blocks:
            z = blk(z, valid)
        return self.head(self.norm(z))


def patchify(x, patch=PATCH, n_tokens=N_TOKENS):
    pad = n_tokens * patch - x.shape[1]
    return F.pad(x, (0, pad)).view(x.shape[0], n_tokens, patch)


def mae_loss(pred, clean, clean_mask, hidden, deriv_weight=0.5):
    """Squared error on hidden tokens' valid points against the per-patch
    normalised clean target, plus deriv_weight x squared error on the first
    derivative (so peaks count more than smooth background)."""
    pred = pred.float()
    tgt = patchify(clean.float())
    pm = patchify(clean_mask.float()) > 0
    cnt = pm.sum(dim=-1, keepdim=True).clamp_min(1)
    mean = (tgt * pm).sum(-1, keepdim=True) / cnt
    var = (((tgt - mean) ** 2) * pm).sum(-1, keepdim=True) / cnt
    tgt_n = (tgt - mean) / torch.sqrt(var + TARGET_EPS)
    w = pm & hidden[..., None]
    mse = (((pred - tgt_n) ** 2) * w).sum() / w.sum().clamp_min(1)
    wd = w[..., 1:] & w[..., :-1]
    dmse = (((pred.diff(dim=-1) - tgt_n.diff(dim=-1)) ** 2) * wd).sum() / wd.sum().clamp_min(1)
    loss = mse + deriv_weight * dmse
    mse, dmse = mse.detach(), dmse.detach()
    return loss, {"mse": mse.item(), "dmse": dmse.item()}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `venv/Scripts/python.exe -m pytest tests/foundation/test_mae.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add foundation/model/masking.py foundation/model/decoder.py tests/foundation/test_mae.py
git commit -m "foundation: vectorised block masking, MAE decoder and reconstruction loss"
```

---

### Task 4: Probes

**Files:**
- Create: `foundation/probe.py`, `tests/foundation/conftest.py`
- Test: `tests/foundation/test_probe.py`

**Interfaces:**
- Consumes: `SpectralEncoder.embed` and `.cfg.width` (Task 2). Corpus arrays and the meta DataFrame have the Plan 1 layout: meta columns `source`, `split` and `label`, with `split` ∈ {`pretrain`, `probe_train`, `probe_test`}.
- Produces:
  - `KNN_K = 20`; `ProbeTask(name, train_idx, test_idx, y_train, y_test)`; `ProbeSet(tasks, sample_idx, sample_source)`.
  - `build_probe_set(meta, seed=0, per_source=300)`.
  - `embed(encoder, X, M, idx, device, batch=1024) -> np.ndarray`.
  - `knn_f1`, `logreg_f1`, `rankme(emb)` and `source_id_bacc(emb, sources, seed=0)` (NaN if fewer than 2 sources).
  - `evaluate(encoder, X, M, probe_set, device) -> {"probe_score", "knn_score", "logreg_score", "per_task", "rankme", "source_id_bacc"}`.
  - `raw_knn_score(X, M, probe_set)`.
  - **Test fixture** `fake_corpus(tmp_path) -> Path`: a tiny corpus_v1 look-alike. It's used by Tasks 4–6.

- [ ] **Step 1: Write the shared fixture and the failing tests**

`tests/foundation/conftest.py`:
```python
"""Shared fixtures for the foundation tests."""
import json

import numpy as np
import pandas as pd
import pytest

from foundation.data.harmonize import GRID


def make_fake_corpus(d, seed=0):
    """A tiny corpus_v1 look-alike: 120 pretrain rows of one source and a
    3-class probe source with 18 probe_train and 18 probe_test rows."""
    rng = np.random.default_rng(seed)
    rows, X, M = [], [], []
    def add(source, split, label, centre):
        y = np.exp(-0.5 * ((GRID - centre) / 8) ** 2) + rng.normal(0, 0.02, GRID.size)
        y = (y - y.min()) / (y.max() - y.min())
        X.append(y.astype(np.float16)); M.append(np.ones(GRID.size, bool))
        rows.append({"source": source, "domain": "bio", "laser_nm": 532.0,
                     "sample_id": f"{source}{len(rows)}", "label": label, "split": split,
                     "weight_group": source, "sample_weight": 0.0})
    for i in range(120):
        add("bulk", "pretrain", None, rng.uniform(500, 1700))
    for split in ("probe_train", "probe_test"):
        for cls, c in enumerate((700, 1100, 1500)):
            for _ in range(6):
                add("probe", split, f"c{cls}", c + rng.normal(0, 3))
    meta = pd.DataFrame(rows)
    meta.loc[meta.split == "pretrain", "sample_weight"] = 1 / 120
    d.mkdir()
    np.save(d / "spectra.npy", np.stack(X)); np.save(d / "mask.npy", np.stack(M))
    meta.to_parquet(d / "meta.parquet", index=False)
    (d / "manifest.json").write_text(json.dumps({"version": "fake"}))
    return d



@pytest.fixture
def fake_corpus(tmp_path):
    return make_fake_corpus(tmp_path / "corpus")
```

`tests/foundation/test_probe.py`:
```python
import numpy as np
import pandas as pd
import pytest
import torch

from foundation.model.encoder import PRESETS, SpectralEncoder
from foundation.probe import (build_probe_set, embed, evaluate, knn_f1, logreg_f1,
                              rankme, raw_knn_score, source_id_bacc)


def _clusters(n_per=10, classes=3, dim=8, seed=0, spread=0.05):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(classes, dim))
    X = np.concatenate([c + spread * rng.normal(size=(n_per, dim)) for c in centres])
    y = np.repeat([f"c{i}" for i in range(classes)], n_per)
    return X, y


def test_knn_and_logreg_separate_clean_clusters():
    X, y = _clusters()
    Xt, yt = _clusters(seed=0)
    assert knn_f1(X, y, Xt, yt) == pytest.approx(1.0)
    assert logreg_f1(X, y, Xt, yt) == pytest.approx(1.0)


def test_knn_works_with_three_examples_per_class():
    # RRUFF-like: many classes, ~3 training spectra each, k=20 > class size
    X, y = _clusters(n_per=3, classes=40)
    Xt, yt = _clusters(n_per=3, classes=40, seed=0)
    assert knn_f1(X, y, Xt, yt) > 0.95


def test_rankme_counts_effective_dimensions():
    rng = np.random.default_rng(0)
    full = rng.normal(size=(500, 16))
    collapsed = np.outer(rng.normal(size=500), rng.normal(size=16))
    assert rankme(full) > 12 and rankme(collapsed) < 1.5


def test_source_id_is_nan_with_a_single_source():
    X, _ = _clusters()
    assert np.isnan(source_id_bacc(X, np.array(["a"] * len(X))))
    Xs, ys = _clusters(n_per=20)
    assert source_id_bacc(Xs, ys) > 0.9


def test_build_probe_set_uses_probe_rows_and_a_balanced_pretrain_sample():
    meta = pd.DataFrame({
        "source": ["big"] * 1000 + ["small"] * 5 + ["p"] * 8,
        "split": ["pretrain"] * 1005 + ["probe_train"] * 4 + ["probe_test"] * 4,
        "label": [None] * 1005 + ["x", "y"] * 4})
    ps = build_probe_set(meta, per_source=300)
    assert [t.name for t in ps.tasks] == ["p"]
    assert len(ps.tasks[0].train_idx) == 4 and len(ps.tasks[0].test_idx) == 4
    counts = pd.Series(ps.sample_source).value_counts().to_dict()
    assert counts == {"big": 300, "small": 5}
    assert (meta.loc[ps.sample_idx, "split"] == "pretrain").all()


def test_evaluate_and_raw_baseline_on_fake_corpus(fake_corpus):
    X = np.load(fake_corpus / "spectra.npy")
    M = np.load(fake_corpus / "mask.npy")
    meta = pd.read_parquet(fake_corpus / "meta.parquet")
    ps = build_probe_set(meta)
    enc = SpectralEncoder(PRESETS["tiny"])
    out = evaluate(enc, X, M, ps, torch.device("cpu"))
    for key in ("probe_score", "knn_score", "logreg_score", "rankme", "source_id_bacc"):
        assert key in out
    assert out["probe_score"] == pytest.approx((out["knn_score"] + out["logreg_score"]) / 2)
    assert raw_knn_score(X, M, ps) > 0.9          # fake classes are easy on raw spectra
    assert embed(enc, X, M, ps.sample_idx, torch.device("cpu")).shape == (len(ps.sample_idx), 128)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `venv/Scripts/python.exe -m pytest tests/foundation/test_probe.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'foundation.probe'`.

- [ ] **Step 3: Write the implementation**

`foundation/probe.py`:
```python
"""Probes on frozen embeddings: the "is A done / good?" signal (spec section 5.4).

Uses the pretraining corpus only (its held-out probe slice and a fixed sample
of pretrain rows); our Bacteria-ID data never appears here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

KNN_K = 20
SOURCE_SAMPLE_PER_SOURCE = 300


@dataclass
class ProbeTask:
    name: str
    train_idx: np.ndarray
    test_idx: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


@dataclass
class ProbeSet:
    tasks: list
    sample_idx: np.ndarray        # fixed pretrain rows for RankMe and the source-ID probe
    sample_source: np.ndarray


def build_probe_set(meta, seed=0, per_source=SOURCE_SAMPLE_PER_SOURCE):
    """One task per probe source; a fixed, source-balanced pretrain sample."""
    split = meta["split"].to_numpy()
    source = meta["source"].to_numpy(dtype=object)
    label = meta["label"].to_numpy(dtype=object)
    tasks = []
    for src in sorted(set(source[split != "pretrain"])):
        tr = np.flatnonzero((source == src) & (split == "probe_train"))
        te = np.flatnonzero((source == src) & (split == "probe_test"))
        tasks.append(ProbeTask(src, tr, te, label[tr].astype(str), label[te].astype(str)))
    rng = np.random.default_rng(seed)
    pre = np.flatnonzero(split == "pretrain")
    picks = []
    for src in sorted(set(source[pre])):
        rows = pre[source[pre] == src]
        picks.append(rng.choice(rows, size=min(per_source, rows.size), replace=False))
    sample = np.sort(np.concatenate(picks))
    return ProbeSet(tasks, sample, source[sample].astype(str))


@torch.no_grad()
def embed(encoder, X, M, idx, device, batch=1024):
    encoder.eval()
    out = []
    for s in range(0, len(idx), batch):
        rows = idx[s:s + batch]
        x = torch.as_tensor(np.asarray(X[rows], dtype=np.float32), device=device)
        m = torch.as_tensor(np.asarray(M[rows]), device=device)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out.append(encoder.embed(x, m).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, encoder.cfg.width), np.float32)


def knn_f1(tr, y_tr, te, y_te, k=KNN_K):
    # Distance weighting keeps k=20 meaningful when a class has only a few
    # training examples (RRUFF: ~3 per mineral species).
    clf = KNeighborsClassifier(n_neighbors=min(k, len(y_tr)), metric="cosine",
                               weights="distance").fit(tr, y_tr)
    return float(f1_score(y_te, clf.predict(te), average="macro",
                          labels=np.unique(y_te), zero_division=0))


def logreg_f1(tr, y_tr, te, y_te):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(tr, y_tr)
    return float(f1_score(y_te, clf.predict(te), average="macro",
                          labels=np.unique(y_te), zero_division=0))


def rankme(emb, eps=1e-7):
    """Effective rank of the embedding matrix (Garrido et al. 2023)."""
    s = np.linalg.svd(np.asarray(emb, dtype=np.float64), compute_uv=False)
    p = s / s.sum() + eps
    return float(np.exp(-(p * np.log(p)).sum()))


def source_id_bacc(emb, sources, seed=0):
    """Balanced accuracy of a linear probe predicting the source. High values
    while the chemistry probes stay flat mean A encodes instrument/dataset quirks."""
    keep = np.isin(sources, [s for s in np.unique(sources) if (sources == s).sum() >= 4])
    if len(np.unique(sources[keep])) < 2:
        return float("nan")
    tr, te, ytr, yte = train_test_split(emb[keep], sources[keep], test_size=0.5,
                                        stratify=sources[keep], random_state=seed)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(tr, ytr)
    return float(balanced_accuracy_score(yte, clf.predict(te)))


def evaluate(encoder, X, M, probe_set, device):
    per_task = {}
    for t in probe_set.tasks:
        tr = embed(encoder, X, M, t.train_idx, device)
        te = embed(encoder, X, M, t.test_idx, device)
        per_task[t.name] = {"knn_f1": knn_f1(tr, t.y_train, te, t.y_test),
                            "logreg_f1": logreg_f1(tr, t.y_train, te, t.y_test)}
    sample = embed(encoder, X, M, probe_set.sample_idx, device)
    knn = float(np.mean([v["knn_f1"] for v in per_task.values()]))
    lr = float(np.mean([v["logreg_f1"] for v in per_task.values()]))
    return {"probe_score": (knn + lr) / 2, "knn_score": knn, "logreg_score": lr,
            "per_task": per_task, "rankme": rankme(sample),
            "source_id_bacc": source_id_bacc(sample, probe_set.sample_source)}


def raw_knn_score(X, M, probe_set):
    """G1 baseline: k-NN on the raw normalised spectra of the same probe slice."""
    scores = []
    for t in probe_set.tasks:
        raw = lambda idx: np.where(M[idx], np.asarray(X[idx], np.float32), 0.0)
        scores.append(knn_f1(raw(t.train_idx), t.y_train, raw(t.test_idx), t.y_test))
    return float(np.mean(scores))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `venv/Scripts/python.exe -m pytest tests/foundation/test_probe.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add foundation/probe.py tests/foundation/conftest.py tests/foundation/test_probe.py
git commit -m "foundation: corpus-only probes (k-NN, logistic, RankMe, source-ID, raw baseline)"
```

---

### Task 5: Pretraining loop, checkpoints and resume

**Files:**
- Create: `foundation/pretrain.py`, `foundation/configs/pretrain_trial_tiny.yaml`, `foundation/configs/pretrain_trial_small.yaml`, `foundation/configs/pretrain_trial_base.yaml`
- Test: `tests/foundation/test_pretrain.py`

**Interfaces:**
- Consumes: `CORRUPTIONS`/`corrupt` (Task 1); `PRESETS`, `SpectralEncoder`, `count_parameters` (Task 2); `block_mask`, `visible_order`, `MAEDecoder`, `mae_loss`, `patchify`, `TARGET_EPS` (Task 3); `build_probe_set`, `evaluate`, `raw_knn_score` (Task 4).
- Produces:
  - `DEFAULTS` and `load_config(path=None, **overrides)`.
  - `Corpus(corpus_dir)` with `.X`, `.M`, `.meta`, `.pretrain_idx`, `.weights` and `.sample(n, generator, device)`.
  - `lr_factor(step, warmup_steps, total_steps)`; `build_models(cfg, device)`; `make_optimizer(cfg, enc, dec)`.
  - `train_step(...)`, `reconstruct(...)` and `_save_atomic(obj, path)`.
  - **`run(cfg, run_dir=None, resume=False, log=print) -> (run_dir, track)`**.
  - The CLI `python -m foundation.pretrain --config CFG | --resume RUN [--max-epochs N] [--device D]`.
  - **On-disk run layout** in `foundation/runs/pretrain/<stamp>_<preset>_<tag>/`, which Task 6 and Plan 3 rely on:
    - `config.yaml` and `manifest.json` (`raw_knn_score`, `encoder_params`, `steps_per_epoch`, git SHA, versions);
    - `metrics.jsonl`, with three kinds of line:
      - per-epoch lines `{"epoch", "loss", "mse", "dmse", "lr", "seconds", "skipped_steps"}`;
      - per-probe lines `{"epoch", "probe": {...}, "improved"}`;
      - a final `{"end": reason, "track": {...}}`;
    - `ckpt_last.pt` (full state, for resume);
    - **`ckpt_best.pt` = `{"encoder": state_dict, "preset": str, "epoch": int, "probe": dict}`**, which Plan 3 loads;
    - `recon_epochNNNN.npz`.

- [ ] **Step 1: Write the failing tests**

`tests/foundation/test_pretrain.py`:
```python
import json
import re

import numpy as np
import pytest

import foundation.pretrain as pt


@pytest.fixture
def cfg(tmp_path, fake_corpus):
    return pt.load_config(None, preset="tiny", corpus_dir=str(fake_corpus), device="cpu",
                          run_root=str(tmp_path / "runs"), batch_size=8, steps_per_epoch=2,
                          max_epochs=2, warmup_epochs=1, probe_every=1, amp=False, n_recon=4)


def test_lr_warms_up_then_decays_to_zero():
    f = [pt.lr_factor(s, 10, 100) for s in range(100)]
    assert f[0] == pytest.approx(0.1) and f[9] == pytest.approx(1.0)
    assert f[10] == pytest.approx(1.0) and f[99] < 0.01
    assert all(a >= b for a, b in zip(f[10:], f[11:]))


def test_run_writes_artefacts_and_progress_lines(cfg):
    lines = []
    run_dir, track = pt.run(cfg, log=lines.append)
    for name in ("manifest.json", "config.yaml", "metrics.jsonl", "ckpt_last.pt",
                 "ckpt_best.pt", "recon_epoch0001.npz", "recon_epoch0002.npz"):
        assert (run_dir / name).exists(), name
    assert any(re.search(r"\[pretrain\] epoch 1/2 \(1 left\) \| loss [\d.]+ \| lr \S+ \| \S+ "
                         r"elapsed, ~\S+ ETA", l) for l in lines)
    assert lines[-1].startswith("[pretrain] finished 2/2")
    assert "raw_knn_score" in json.loads((run_dir / "manifest.json").read_text())
    rec = np.load(run_dir / "recon_epoch0002.npz")
    assert rec["reconstruction"].shape == rec["clean"].shape == (4, 1001)


def test_resume_continues_from_the_last_checkpoint(cfg):
    run_dir, _ = pt.run(cfg, log=lambda s: None)
    lines = []
    cfg3 = pt.load_config(run_dir / "config.yaml", max_epochs=3)
    pt.run(cfg3, run_dir=run_dir, resume=True, log=lines.append)
    epochs = [json.loads(l)["epoch"] for l in (run_dir / "metrics.jsonl").read_text().splitlines()
              if '"loss"' in l]
    assert epochs == [1, 2, 3]
    assert any("epoch 3/3 (0 left)" in l for l in lines)


def test_early_stopping_when_probe_is_flat(cfg, monkeypatch):
    monkeypatch.setattr(pt, "evaluate", lambda *a, **k: {
        "probe_score": 0.5, "knn_score": 0.5, "logreg_score": 0.5, "per_task": {},
        "rankme": 1.0, "source_id_bacc": 0.5})
    cfg.update(max_epochs=6, patience=2)
    lines = []
    _, track = pt.run(cfg, log=lines.append)
    assert lines[-1] == "[pretrain] stopped at epoch 3/6: probe score flat for 2 checkpoints"
    assert track["best_epoch"] == 1


def test_nonfinite_losses_abort_instead_of_spinning(cfg, monkeypatch):
    import torch
    monkeypatch.setattr(pt, "mae_loss", lambda *a, **k: (torch.tensor(float("nan")), {}))
    cfg.update(max_nonfinite_steps=1, steps_per_epoch=3)
    with pytest.raises(RuntimeError, match="non-finite"):
        pt.run(cfg, log=lambda s: None)


def test_checkpoint_save_is_atomic(tmp_path):
    pt._save_atomic({"a": 1}, tmp_path / "ck.pt")
    assert (tmp_path / "ck.pt").exists() and not (tmp_path / "ck.pt.tmp").exists()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `venv/Scripts/python.exe -m pytest tests/foundation/test_pretrain.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'foundation.pretrain'`.

- [ ] **Step 3: Write the implementation and the trial configs**

`foundation/pretrain.py`:
```python
"""Masked-spectrum pretraining of model A (spec sections 5.2-5.4).

    python -m foundation.pretrain --config foundation/configs/pretrain_trial_small.yaml
    python -m foundation.pretrain --resume foundation/runs/pretrain/<run> --max-epochs 300

Runs end only at max_epochs or when the probe score has not improved for
`patience` checkpoints; never on elapsed time. Our Bacteria-ID data is never
loaded here.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from .data.corrupt import PRESETS as CORRUPTIONS
from .data.corrupt import corrupt
from .model.decoder import TARGET_EPS, MAEDecoder, mae_loss, patchify
from .model.encoder import PRESETS, SpectralEncoder, count_parameters
from .model.masking import block_mask, visible_order
from .probe import build_probe_set, evaluate, raw_knn_score

DEFAULTS = {
    "tag": "run", "preset": "small", "corpus_dir": "foundation/data/corpus_v1",
    "run_root": "foundation/runs/pretrain", "seed": 0, "device": "cuda",
    "batch_size": 512, "lr": 3.0e-4, "weight_decay": 0.05, "betas": [0.9, 0.95],
    "warmup_epochs": 10, "max_epochs": 200, "grad_clip": 1.0,
    "mask_ratio": 0.5, "mask_block": [2, 6], "deriv_weight": 0.5,
    "corruption": "pretrain", "probe_every": 10, "patience": 3, "amp": True,
    "n_recon": 16, "steps_per_epoch": None, "max_nonfinite_steps": 20,
}


def load_config(path=None, **overrides):
    cfg = dict(DEFAULTS)
    if path:
        cfg.update(yaml.safe_load(Path(path).read_text()) or {})
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


class Corpus:
    """corpus_v1 artefacts held in RAM; only pretrain rows are ever sampled."""

    def __init__(self, corpus_dir):
        d = Path(corpus_dir)
        self.X = np.load(d / "spectra.npy")                 # float16 (N, 1001)
        self.M = np.load(d / "mask.npy")                    # bool (N, 1001)
        self.meta = pd.read_parquet(d / "meta.parquet")
        split = self.meta["split"].to_numpy()
        self.pretrain_idx = np.flatnonzero(split == "pretrain")
        w = self.meta["sample_weight"].to_numpy()[self.pretrain_idx]
        self.weights = torch.as_tensor(w / w.sum(), dtype=torch.float64)
        self.Xt = torch.from_numpy(self.X)
        self.Mt = torch.from_numpy(self.M)
        self.manifest_version = json.loads((d / "manifest.json").read_text()).get("version")

    def sample(self, n, generator, device):
        pick = torch.multinomial(self.weights, n, replacement=True, generator=generator)
        rows = torch.as_tensor(self.pretrain_idx)[pick]
        return (self.Xt[rows].to(device, non_blocking=True).float(),
                self.Mt[rows].to(device, non_blocking=True))


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _save_atomic(obj, path):
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _fmt_secs(s):
    s = int(s)
    h, m = divmod(s // 60, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s % 60:02d}s"


def lr_factor(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def build_models(cfg, device):
    enc = SpectralEncoder(PRESETS[cfg["preset"]]).to(device)
    dec = MAEDecoder(enc.cfg.width).to(device)
    return enc, dec


def make_optimizer(cfg, enc, dec):
    decay, no_decay = [], []
    for name, p in list(enc.named_parameters()) + list(dec.named_parameters()):
        (decay if p.ndim >= 2 and "mask_token" not in name else no_decay).append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                              {"params": no_decay, "weight_decay": 0.0}],
                             lr=cfg["lr"], betas=tuple(cfg["betas"]))


def train_step(cfg, enc, dec, opt, sched, scaler, corpus, gens, device):
    x, m = corpus.sample(cfg["batch_size"], gens["cpu"], device)
    with torch.no_grad():
        xc, mc = corrupt(x, m, CORRUPTIONS[cfg["corruption"]], generator=gens["dev"])
    enc.train(), dec.train()
    with torch.autocast(device.type, enabled=cfg["amp"] and device.type == "cuda"):
        tokens, valid = enc.tokenize(xc, mc)
        hidden = block_mask(valid, cfg["mask_ratio"], *cfg["mask_block"], generator=gens["dev"])
        keep, keep_valid = visible_order(valid, hidden)
        vis = tokens.gather(1, keep[..., None].expand(-1, -1, tokens.shape[2]))
        pred = dec(enc.encode(vis, keep_valid), keep, keep_valid, valid)
    loss, parts = mae_loss(pred, x, m, hidden, cfg["deriv_weight"])
    opt.zero_grad(set_to_none=True)
    if not torch.isfinite(loss):
        return None
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()), cfg["grad_clip"])
    scaler.step(opt)
    scaler.update()
    sched.step()
    return loss.item(), parts


@torch.no_grad()
def reconstruct(enc, dec, X, M, rows, cfg, device):
    """Fixed corruption + mask on fixed probe spectra -> arrays for plotting."""
    enc.eval(), dec.eval()
    x = torch.as_tensor(np.asarray(X[rows], np.float32), device=device)
    m = torch.as_tensor(np.asarray(M[rows]), device=device)
    g = torch.Generator(device=device).manual_seed(1234)
    xc, mc = corrupt(x, m, CORRUPTIONS[cfg["corruption"]], generator=g)
    tokens, valid = enc.tokenize(xc, mc)
    hidden = block_mask(valid, cfg["mask_ratio"], *cfg["mask_block"], generator=g)
    keep, keep_valid = visible_order(valid, hidden)
    vis = tokens.gather(1, keep[..., None].expand(-1, -1, tokens.shape[2]))
    pred = dec(enc.encode(vis, keep_valid), keep, keep_valid, valid).float()
    tgt, pm = patchify(x), patchify(m.float()) > 0
    cnt = pm.sum(-1, keepdim=True).clamp_min(1)
    mean = (tgt * pm).sum(-1, keepdim=True) / cnt
    sd = torch.sqrt((((tgt - mean) ** 2) * pm).sum(-1, keepdim=True) / cnt + TARGET_EPS)
    recon = (pred * sd + mean).flatten(1)[:, :x.shape[1]]
    cpu = lambda t: t.cpu().numpy()
    return {"clean": cpu(x), "mask": cpu(m), "corrupted": cpu(xc), "corrupted_mask": cpu(mc),
            "hidden_tokens": cpu(hidden), "reconstruction": cpu(recon)}


def _state(cfg, enc, dec, opt, scaler, sched, epoch, track, gens):
    return {"config": cfg, "epoch": epoch, "track": track,
            "encoder": enc.state_dict(), "decoder": dec.state_dict(),
            "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
            "scheduler": sched.state_dict(),
            "gen_cpu": gens["cpu"].get_state(), "gen_dev": gens["dev"].get_state()}


def run(cfg, run_dir=None, resume=False, log=print):
    device = torch.device(cfg["device"] if cfg["device"] != "cuda" or torch.cuda.is_available()
                          else "cpu")
    torch.manual_seed(cfg["seed"])
    corpus = Corpus(cfg["corpus_dir"])
    probe_set = build_probe_set(corpus.meta, seed=cfg["seed"])
    enc, dec = build_models(cfg, device)
    opt = make_optimizer(cfg, enc, dec)
    steps = cfg["steps_per_epoch"] or math.ceil(len(corpus.pretrain_idx) / cfg["batch_size"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(
        s, cfg["warmup_epochs"] * steps, cfg["max_epochs"] * steps))
    scaler = torch.amp.GradScaler(device.type, enabled=cfg["amp"] and device.type == "cuda")
    gens = {"cpu": torch.Generator().manual_seed(cfg["seed"]),
            "dev": torch.Generator(device=device).manual_seed(cfg["seed"] + 1)}
    recon_rows = np.concatenate([t.test_idx[:cfg["n_recon"] // max(1, len(probe_set.tasks))]
                                 for t in probe_set.tasks])[:cfg["n_recon"]]

    if resume:
        ck = torch.load(Path(run_dir) / "ckpt_last.pt", map_location=device, weights_only=True)
        enc.load_state_dict(ck["encoder"]), dec.load_state_dict(ck["decoder"])
        opt.load_state_dict(ck["optimizer"]), scaler.load_state_dict(ck["scaler"])
        sched.load_state_dict(ck["scheduler"])
        gens["cpu"].set_state(ck["gen_cpu"]), gens["dev"].set_state(ck["gen_dev"])
        start, track = ck["epoch"] + 1, ck["track"]
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = Path(cfg["run_root"]) / f"{stamp}_{cfg['preset']}_{cfg['tag']}"
        run_dir.mkdir(parents=True)
        start = 1
        track = {"best_score": None, "best_epoch": None, "bad_checks": 0,
                 "raw_knn_score": raw_knn_score(corpus.X, corpus.M, probe_set)}
        (run_dir / "manifest.json").write_text(json.dumps({
            "git_sha": _git_sha(), "corpus": cfg["corpus_dir"],
            "corpus_version": corpus.manifest_version, "device": str(device),
            "torch": torch.__version__, "python": platform.python_version(),
            "encoder_params": count_parameters(enc), "decoder_params": count_parameters(dec),
            "steps_per_epoch": steps, "raw_knn_score": track["raw_knn_score"]}, indent=2))
    run_dir = Path(run_dir)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    metrics = open(run_dir / "metrics.jsonl", "a", encoding="utf-8")
    log(f"[pretrain] {run_dir} | {cfg['preset']} | {count_parameters(enc):,} encoder params | "
        f"{steps} steps/epoch | epochs {start}..{cfg['max_epochs']}")

    t0, epoch_secs, reason = time.time(), [], None
    nonfinite_streak = 0
    for epoch in range(start, cfg["max_epochs"] + 1):
        te = time.time()
        losses, parts_sum, skipped = [], {"mse": 0.0, "dmse": 0.0}, 0
        for _ in range(steps):
            out = train_step(cfg, enc, dec, opt, sched, scaler, corpus, gens, device)
            if out is None:
                skipped += 1
                nonfinite_streak += 1
                if nonfinite_streak > cfg["max_nonfinite_steps"]:
                    raise RuntimeError(f"{nonfinite_streak} consecutive non-finite losses at "
                                       f"epoch {epoch}; stopping")
                continue
            nonfinite_streak = 0
            losses.append(out[0])
            for k in parts_sum:
                parts_sum[k] += out[1][k]
        epoch_secs.append(time.time() - te)
        n = max(1, len(losses))
        rec = {"epoch": epoch, "loss": float(np.mean(losses)) if losses else float("nan"),
               "mse": parts_sum["mse"] / n, "dmse": parts_sum["dmse"] / n,
               "lr": opt.param_groups[0]["lr"], "seconds": epoch_secs[-1], "skipped_steps": skipped}
        metrics.write(json.dumps(rec) + "\n")
        metrics.flush()
        left = cfg["max_epochs"] - epoch
        eta = np.mean(epoch_secs) * left
        log(f"[pretrain] epoch {epoch}/{cfg['max_epochs']} ({left} left) | loss {rec['loss']:.4f} "
            f"| lr {rec['lr']:.1e} | {_fmt_secs(time.time() - t0)} elapsed, ~{_fmt_secs(eta)} ETA")

        if epoch % cfg["probe_every"] == 0 or epoch == cfg["max_epochs"]:
            probe = evaluate(enc, corpus.X, corpus.M, probe_set, device)
            improved = track["best_score"] is None or probe["probe_score"] > track["best_score"] + 1e-4
            if improved:
                track.update(best_score=probe["probe_score"], best_epoch=epoch, bad_checks=0)
                _save_atomic({"encoder": enc.state_dict(), "preset": cfg["preset"],
                              "epoch": epoch, "probe": probe}, run_dir / "ckpt_best.pt")
            else:
                track["bad_checks"] += 1
            np.savez_compressed(run_dir / f"recon_epoch{epoch:04d}.npz",
                                **reconstruct(enc, dec, corpus.X, corpus.M, recon_rows, cfg, device))
            metrics.write(json.dumps({"epoch": epoch, "probe": probe, "improved": improved}) + "\n")
            metrics.flush()
            _save_atomic(_state(cfg, enc, dec, opt, scaler, sched, epoch, track, gens),
                         run_dir / "ckpt_last.pt")
            log(f"[probe] epoch {epoch} | probe {probe['probe_score']:.4f} "
                f"(knn {probe['knn_score']:.4f}, logreg {probe['logreg_score']:.4f}, "
                f"raw-knn {track['raw_knn_score']:.4f}) | rankme {probe['rankme']:.1f} "
                f"| source-id {probe['source_id_bacc']:.3f} | best {track['best_score']:.4f} "
                f"@ {track['best_epoch']}")
            if track["bad_checks"] >= cfg["patience"]:
                reason = (f"stopped at epoch {epoch}/{cfg['max_epochs']}: probe score flat for "
                          f"{cfg['patience']} checkpoints")
                break
    else:
        if track["bad_checks"] == 0:
            reason = (f"finished {cfg['max_epochs']}/{cfg['max_epochs']}: probe score still "
                      "improving, resume with a larger max_epochs if needed")
        else:
            reason = (f"finished {cfg['max_epochs']}/{cfg['max_epochs']}: best probe score at "
                      f"epoch {track['best_epoch']}")
    metrics.write(json.dumps({"end": reason, "track": track}) + "\n")
    metrics.close()
    log(f"[pretrain] {reason}")
    return run_dir, track


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--resume", help="run directory to continue from its last checkpoint")
    ap.add_argument("--max-epochs", type=int)
    ap.add_argument("--device")
    args = ap.parse_args(argv)
    if args.resume:
        cfg = load_config(Path(args.resume) / "config.yaml", max_epochs=args.max_epochs,
                          device=args.device)
        run(cfg, run_dir=args.resume, resume=True, log=lambda s: print(s, flush=True))
    else:
        if not args.config:
            ap.error("--config or --resume is required")
        cfg = load_config(args.config, max_epochs=args.max_epochs, device=args.device)
        run(cfg, log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
```

`foundation/configs/pretrain_trial_tiny.yaml`:
```yaml
# Phase 1 size trial (spec section 7): every preset gets the same 50 epochs.
tag: trial
preset: tiny
corpus_dir: foundation/data/corpus_v1
run_root: foundation/runs/pretrain
seed: 0
device: cuda
batch_size: 512
lr: 3.0e-4
weight_decay: 0.05
betas: [0.9, 0.95]
warmup_epochs: 10
max_epochs: 50
grad_clip: 1.0
mask_ratio: 0.5
mask_block: [2, 6]
deriv_weight: 0.5
corruption: pretrain
probe_every: 10
patience: 3
amp: true
n_recon: 16
```

`foundation/configs/pretrain_trial_small.yaml`:
```yaml
# Phase 1 size trial (spec section 7): every preset gets the same 50 epochs.
tag: trial
preset: small
corpus_dir: foundation/data/corpus_v1
run_root: foundation/runs/pretrain
seed: 0
device: cuda
batch_size: 512
lr: 3.0e-4
weight_decay: 0.05
betas: [0.9, 0.95]
warmup_epochs: 10
max_epochs: 50
grad_clip: 1.0
mask_ratio: 0.5
mask_block: [2, 6]
deriv_weight: 0.5
corruption: pretrain
probe_every: 10
patience: 3
amp: true
n_recon: 16
```

`foundation/configs/pretrain_trial_base.yaml`:
```yaml
# Phase 1 size trial (spec section 7): every preset gets the same 50 epochs.
tag: trial
preset: base
corpus_dir: foundation/data/corpus_v1
run_root: foundation/runs/pretrain
seed: 0
device: cuda
batch_size: 512
lr: 3.0e-4
weight_decay: 0.05
betas: [0.9, 0.95]
warmup_epochs: 10
max_epochs: 50
grad_clip: 1.0
mask_ratio: 0.5
mask_block: [2, 6]
deriv_weight: 0.5
corruption: pretrain
probe_every: 10
patience: 3
amp: true
n_recon: 16
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `venv/Scripts/python.exe -m pytest tests/foundation/test_pretrain.py -v`
Expected: 6 passed, with no warnings in the output.

- [ ] **Step 5: Commit**

```bash
git add foundation/pretrain.py foundation/configs/pretrain_trial_tiny.yaml foundation/configs/pretrain_trial_small.yaml foundation/configs/pretrain_trial_base.yaml tests/foundation/test_pretrain.py
git commit -m "foundation: masked-spectrum pretraining loop with probes, checkpoints and resume"
```

---

### Task 6: Run report, size comparison and G1

**Files:**
- Create: `foundation/analysis/__init__.py` (empty), `foundation/analysis/report.py`
- Test: `tests/foundation/test_report.py`

**Interfaces:**
- Consumes: the Task 5 run layout and `Corpus`; `PATCH`, `PRESETS`, `SpectralEncoder` (Task 2); `build_probe_set`, `embed` (Task 4); `GRID` (Plan 1).
- Produces:
  - `summarize(run_dir) -> dict`, with `g1_pass = best knn_score > raw_knn_score`.
  - `recommend(summaries)`: the highest best probe score, where a smaller preset wins ties within 0.01.
  - `report_run(run_dir, corpus=None, device=None)`, which writes `analysis/curves.png`, `recon.png`, `embedding.png` and `summary.json`.
  - `table(summaries) -> markdown`.
  - The CLI `python -m foundation.analysis.report RUN [RUN ...] [--out FILE] [--no-embedding]`.

- [ ] **Step 1: Write the failing tests**

`tests/foundation/test_report.py`:
```python
import json

import pytest

import foundation.pretrain as pt
from foundation.analysis import report


@pytest.fixture
def finished_run(tmp_path, fake_corpus):
    corpus = fake_corpus
    cfg = pt.load_config(None, preset="tiny", corpus_dir=str(corpus), device="cpu",
                         run_root=str(tmp_path / "runs"), batch_size=8, steps_per_epoch=2,
                         max_epochs=2, warmup_epochs=1, probe_every=1, amp=False, n_recon=4)
    run_dir, _ = pt.run(cfg, log=lambda s: None)
    return run_dir, pt.Corpus(corpus)


def test_report_writes_plots_and_summary(finished_run):
    import torch
    run_dir, corpus = finished_run
    s = report.report_run(run_dir, corpus, torch.device("cpu"))
    for name in ("curves.png", "recon.png", "embedding.png", "summary.json"):
        assert (run_dir / "analysis" / name).exists(), name
    assert s["preset"] == "tiny" and s["epochs_run"] == 2 and s["best_epoch"] in (1, 2)
    assert s["g1_pass"] == (s["knn_score"] > s["raw_knn_score"])
    assert json.loads((run_dir / "analysis" / "summary.json").read_text())["run"] == run_dir.name


def test_recommend_prefers_smaller_preset_within_tie():
    mk = lambda preset, score: {"preset": preset, "probe_score": score, "run": preset}
    assert report.recommend([mk("tiny", 0.60), mk("small", 0.605), mk("base", 0.70)])["preset"] == "base"
    assert report.recommend([mk("tiny", 0.60), mk("small", 0.605), mk("base", 0.609)])["preset"] == "tiny"
    assert report.recommend([mk("small", None)]) is None


def test_table_has_one_row_per_run(finished_run):
    run_dir, _ = finished_run
    text = report.table([report.summarize(run_dir)])
    assert text.count("\n") == 2 and "tiny" in text
```

- [ ] **Step 2: Run them to verify they fail**

Run: `venv/Scripts/python.exe -m pytest tests/foundation/test_report.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'foundation.analysis'`.

- [ ] **Step 3: Write the implementation**

`foundation/analysis/__init__.py`: an empty file.

`foundation/analysis/report.py`:
```python
"""Understand what a pretraining run learned (spec section 7, phase 1).

    python -m foundation.analysis.report RUN_DIR [RUN_DIR ...] [--out report.md]

For each run, writes RUN_DIR/analysis/ with:
  curves.png      loss, probe scores vs the raw-spectra k-NN baseline, RankMe, source-ID
  recon.png       reconstructions of 4 fixed probe spectra at the best epoch
  embedding.png   2-D PCA of the best checkpoint's embeddings, by domain and by source
  summary.json
Then prints one comparison table, the size recommendation (highest best probe
score; a smaller preset wins ties within 0.01) and gate G1 for every run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from ..data.harmonize import GRID  # noqa: E402
from ..model.encoder import PATCH, PRESETS, SpectralEncoder  # noqa: E402
from ..pretrain import Corpus  # noqa: E402
from ..probe import build_probe_set, embed  # noqa: E402

SIZE_ORDER = ["tiny", "small", "base"]
TIE = 0.01


def read_metrics(run_dir):
    epochs, probes, end = [], [], None
    for line in (Path(run_dir) / "metrics.jsonl").read_text().splitlines():
        rec = json.loads(line)
        if "loss" in rec:
            epochs.append(rec)
        elif "probe" in rec:
            probes.append(rec)
        elif "end" in rec:
            end = rec
    return epochs, probes, end


def summarize(run_dir):
    run_dir = Path(run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    epochs, probes, end = read_metrics(run_dir)
    best = max(probes, key=lambda p: p["probe"]["probe_score"]) if probes else None
    bp = best["probe"] if best else {}
    raw = manifest["raw_knn_score"]
    return {
        "run": run_dir.name, "preset": cfg["preset"],
        "encoder_params": manifest["encoder_params"],
        "epochs_run": epochs[-1]["epoch"] if epochs else 0,
        "hours": sum(e["seconds"] for e in epochs) / 3600,
        "best_epoch": best["epoch"] if best else None,
        "probe_score": bp.get("probe_score"), "knn_score": bp.get("knn_score"),
        "logreg_score": bp.get("logreg_score"), "per_task": bp.get("per_task"),
        "rankme": bp.get("rankme"), "source_id_bacc": bp.get("source_id_bacc"),
        "raw_knn_score": raw,
        "g1_pass": bool(best) and bp["knn_score"] > raw,
        "end": end["end"] if end else "running or interrupted",
    }


def recommend(summaries):
    """Highest best probe score; a smaller preset wins if within TIE of the best."""
    done = [s for s in summaries if s["probe_score"] is not None]
    if not done:
        return None
    top = max(s["probe_score"] for s in done)
    close = [s for s in done if s["probe_score"] >= top - TIE]
    return min(close, key=lambda s: SIZE_ORDER.index(s["preset"]))


def plot_curves(run_dir, out):
    epochs, probes, _ = read_metrics(run_dir)
    raw = json.loads((Path(run_dir) / "manifest.json").read_text())["raw_knn_score"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot([e["epoch"] for e in epochs], [e["loss"] for e in epochs])
    ax[0].set(title="reconstruction loss", xlabel="epoch")
    pe = [p["epoch"] for p in probes]
    for key in ("probe_score", "knn_score", "logreg_score"):
        ax[1].plot(pe, [p["probe"][key] for p in probes], marker="o", label=key)
    ax[1].axhline(raw, ls="--", c="k", label="raw-spectra k-NN (G1 bar)")
    ax[1].set(title="probe macro-F1", xlabel="epoch", ylim=(0, 1))
    ax[1].legend(fontsize=8)
    ax[2].plot(pe, [p["probe"]["rankme"] for p in probes], marker="o", c="tab:green")
    ax[2].set(title="RankMe (green) / source-ID bacc (red)", xlabel="epoch")
    tw = ax[2].twinx()
    tw.plot(pe, [p["probe"]["source_id_bacc"] for p in probes], marker="s", c="tab:red")
    tw.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def plot_recon(npz_path, out, n=4):
    r = np.load(npz_path)
    picks = np.linspace(0, len(r["clean"]) - 1, n).astype(int)   # spread over probe tasks
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.4 * n), sharex=True)
    for i, ax in zip(picks, np.atleast_1d(axes)):
        m = r["mask"][i]
        ax.plot(GRID[m], r["clean"][i][m], c="k", lw=1, label="clean target")
        cm = r["corrupted_mask"][i]
        ax.plot(GRID[cm], r["corrupted"][i][cm], c="tab:gray", lw=0.6, alpha=0.7, label="corrupted input")
        hid = np.repeat(r["hidden_tokens"][i], PATCH)[:GRID.size] & m
        ax.plot(GRID, np.where(hid, r["reconstruction"][i], np.nan), c="tab:red", lw=1,
                label="reconstruction (hidden)")
        for j in np.flatnonzero(r["hidden_tokens"][i]):
            lo = GRID[min(j * PATCH, GRID.size - 1)]
            ax.axvspan(lo, lo + PATCH * (GRID[1] - GRID[0]), color="tab:red", alpha=0.06, lw=0)
    np.atleast_1d(axes)[0].legend(fontsize=8, loc="upper right")
    np.atleast_1d(axes)[-1].set_xlabel("Raman shift (cm$^{-1}$)")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def plot_embedding(run_dir, corpus, out, device):
    ck = torch.load(Path(run_dir) / "ckpt_best.pt", map_location=device, weights_only=True)
    enc = SpectralEncoder(PRESETS[ck["preset"]]).to(device)
    enc.load_state_dict(ck["encoder"])
    ps = build_probe_set(corpus.meta)
    emb = embed(enc, corpus.X, corpus.M, ps.sample_idx, device)
    emb = emb - emb.mean(axis=0)
    xy = emb @ np.linalg.svd(emb, full_matrices=False)[2][:2].T
    domain = corpus.meta["domain"].to_numpy(dtype=object)[ps.sample_idx]
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    for k, (title, groups) in enumerate((("domain", domain), ("source", ps.sample_source))):
        for g in sorted(set(groups)):
            sel = groups == g
            ax[k].scatter(xy[sel, 0], xy[sel, 1], s=4, label=g, alpha=0.6)
        ax[k].set(title=f"PCA of embeddings by {title}")
        ax[k].legend(fontsize=6, markerscale=3, ncol=2 if k else 1)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def report_run(run_dir, corpus=None, device=None):
    run_dir = Path(run_dir)
    out = run_dir / "analysis"
    out.mkdir(exist_ok=True)
    s = summarize(run_dir)
    plot_curves(run_dir, out / "curves.png")
    if s["best_epoch"] is not None:
        rec = run_dir / f"recon_epoch{s['best_epoch']:04d}.npz"
        if rec.exists():
            plot_recon(rec, out / "recon.png")
        if corpus is not None:
            plot_embedding(run_dir, corpus, out / "embedding.png", device)
    (out / "summary.json").write_text(json.dumps(s, indent=2))
    return s


def table(summaries):
    head = ("| run | preset | params | epochs | hours | best epoch | probe | k-NN | logreg | "
            "raw k-NN | G1 | RankMe | source-ID | end |")
    rows = [head, "|" + "---|" * 14]
    f = lambda v, d=3: "-" if v is None else f"{v:.{d}f}"
    for s in summaries:
        rows.append(f"| {s['run']} | {s['preset']} | {s['encoder_params'] / 1e6:.2f}M | "
                    f"{s['epochs_run']} | {s['hours']:.1f} | {s['best_epoch']} | "
                    f"{f(s['probe_score'])} | {f(s['knn_score'])} | {f(s['logreg_score'])} | "
                    f"{f(s['raw_knn_score'])} | {'PASS' if s['g1_pass'] else 'FAIL'} | "
                    f"{f(s['rankme'], 1)} | {f(s['source_id_bacc'])} | {s['end']} |")
    return "\n".join(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out")
    ap.add_argument("--no-embedding", action="store_true", help="skip the PCA plot (no GPU work)")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load((Path(args.runs[0]) / "config.yaml").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corpus = None if args.no_embedding else Corpus(cfg["corpus_dir"])
    summaries = [report_run(r, corpus, device) for r in args.runs]
    text = table(summaries)
    rec = recommend(summaries)
    text += ("\n\nRecommended size (probe score only; smaller wins ties within "
             f"{TIE}): {rec['preset']} ({rec['run']})" if rec else "\n\nNo finished probe yet.")
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `venv/Scripts/python.exe -m pytest tests/foundation/test_report.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add foundation/analysis/__init__.py foundation/analysis/report.py tests/foundation/test_report.py
git commit -m "foundation: pretraining run report, size recommendation and G1 check"
```

---

### Task 7: README, full suite and a real-corpus smoke run

**Files:**
- Modify: `foundation/README.md` (append a section)

**Interfaces:**
- Consumes: everything above. Produces: documentation plus a verified end-to-end run on the real corpus and GPU.

- [ ] **Step 1: Append to `foundation/README.md`**

````markdown

## Phases 1-2: pretrain model A

Size trials (same 50 epochs each; about 0.8 / 1.2 / 2.0 h on an RTX 3060):
```
venv/Scripts/python.exe -m foundation.pretrain --config foundation/configs/pretrain_trial_tiny.yaml
venv/Scripts/python.exe -m foundation.pretrain --config foundation/configs/pretrain_trial_small.yaml
venv/Scripts/python.exe -m foundation.pretrain --config foundation/configs/pretrain_trial_base.yaml
```
Understand and compare them (plots land in each run's `analysis/`):
```
venv/Scripts/python.exe -m foundation.analysis.report foundation/runs/pretrain/<tiny-run> foundation/runs/pretrain/<small-run> foundation/runs/pretrain/<base-run> --out foundation/runs/pretrain/trials.md
```
Runs stop only at `max_epochs` or when the probe score is flat for 3 checkpoints, never on
time. Continue one that was still improving:
```
venv/Scripts/python.exe -m foundation.pretrain --resume foundation/runs/pretrain/<run> --max-epochs 300
```
Plan 3 loads `<run>/ckpt_best.pt` (`{"encoder", "preset", "epoch", "probe"}`).
````

- [ ] **Step 2: Run the whole test suite**

Run: `venv/Scripts/python.exe -m pytest tests/foundation -v`
Expected: every test passes, including all Plan 1 tests, with no warnings.

- [ ] **Step 3: Smoke-run on the real corpus and GPU**

Run, from the repo root:
```bash
venv/Scripts/python.exe -c "import foundation.pretrain as pt; cfg = pt.load_config('foundation/configs/pretrain_trial_small.yaml', steps_per_epoch=30, max_epochs=1, warmup_epochs=1, probe_every=1, tag='smoke'); print(pt.run(cfg)[1])"
```
Expected:
- an `epoch 1/1 (0 left)` line;
- a `[probe]` line with `raw-knn 0.6xxx`;
- `finished 1/1: probe score still improving, ...`;
- a `track` dict.

Then run `venv/Scripts/python.exe -m foundation.analysis.report <that run dir>`. Expected: a one-row table and `analysis/*.png` in the run directory.

Delete the smoke run directory afterwards. It lives under the git-ignored `foundation/runs/`.

- [ ] **Step 4: Commit**

```bash
git add foundation/README.md
git commit -m "foundation: document pretraining, size trials and the run report"
```

---

### Task 8: Phase 1, the size trials (controller-run; ends at a user review)

**Files:** none (run outputs go to the git-ignored `foundation/runs/pretrain/`)

**Interfaces:** consumes Tasks 5–6. Produces three finished trial runs plus `foundation/runs/pretrain/trials.md`.

- [ ] **Step 1: Run the three trials one after another, in the background**

```bash
venv/Scripts/python.exe -u -m foundation.pretrain --config foundation/configs/pretrain_trial_tiny.yaml
venv/Scripts/python.exe -u -m foundation.pretrain --config foundation/configs/pretrain_trial_small.yaml
venv/Scripts/python.exe -u -m foundation.pretrain --config foundation/configs/pretrain_trial_base.yaml
```

Expected: about 4 h in total. Each run ends with a `[pretrain] stopped at ...` or `[pretrain] finished ...` line.

If a run crashes, resume it with `--resume <run dir>`. Don't restart it from scratch.

- [ ] **Step 2: Report**

Run:
```bash
venv/Scripts/python.exe -m foundation.analysis.report <tiny-run> <small-run> <base-run> --out foundation/runs/pretrain/trials.md
```

- [ ] **Step 3: STOP. Show the user the trials and let them reconfigure.**

The user asked to understand what the model does after training before choosing its configuration. Show:
- the comparison table and the recommended size;
- each run's `curves.png`, `recon.png` and `embedding.png`.

Point out:
1. how the probe score moved over the 50 epochs, against the raw-k-NN bar (G1);
2. whether the reconstructions recover peaks, not just backgrounds;
3. whether the embedding clusters by domain and chemistry or by source, with the source-ID accuracy;
4. RankMe, where a collapse shows as a value near 1.
5. The spec §7 re-check of **no baseline correction**:
   - whether the reconstructions reproduce each source's background rather than its peaks;
   - whether a high source-ID accuracy goes with flat chemistry probes. Airborne arrives already baseline-corrected (Plan 1 deviation D2), while the other sources are raw.

   If either holds, offer baseline correction for all sources as the reconfiguration. That's a Plan 1 change: a new `HARMONIZE_VERSION` and a rebuild from the cached downloads.

Then ask the user to choose:
- the size;
- whether to change the mask ratio or the corruption strengths;
- whether to run one extra short round of trials (spec §7 allows one).

Selection uses the probe score only, never the Bacteria-ID data.

---

### Task 9: Phase 2, main pretraining and gate G1 (after the user's choice)

**Files:**
- Create: `foundation/configs/pretrain_main.yaml`

**Interfaces:** consumes the Task 8 decision. Produces the main run, whose `ckpt_best.pt` is model A for Plan 3.

- [ ] **Step 1: Write the main config from the user's choice**

Copy the chosen trial config to `foundation/configs/pretrain_main.yaml`. Then set:
- `tag: main`;
- `max_epochs: 200`;
- any mask or corruption changes the user approved in Task 8.

All other values stay as in the trial.

- [ ] **Step 2: Run the main pretraining in the background**

```bash
venv/Scripts/python.exe -u -m foundation.pretrain --config foundation/configs/pretrain_main.yaml
```

If the final line says the probe score was still improving, ask the user whether to resume with a larger `--max-epochs`.

- [ ] **Step 3: Report and gate G1**

Run:
```bash
venv/Scripts/python.exe -m foundation.analysis.report <main-run> --out foundation/runs/pretrain/main.md
```

**G1 PASS** (best k-NN probe score above the raw-spectra k-NN score): model A is `<main-run>/ckpt_best.pt`. Plan 3 can start.

**G1 FAIL:** stop and diagnose with the user before any adaptation, as spec §5.4 requires. Look at the per-task scores, because RRUFF and airborne can disagree, and at the reconstructions and embedding map.

- [ ] **Step 4: Commit the main config and the run summary**

```bash
mkdir -p foundation/results
cp foundation/runs/pretrain/main.md foundation/results/pretrain_main.md
cp foundation/runs/pretrain/trials.md foundation/results/pretrain_trials.md
git add foundation/configs/pretrain_main.yaml foundation/results/pretrain_main.md foundation/results/pretrain_trials.md
git commit -m "foundation: main pretraining run and G1 result"
```
