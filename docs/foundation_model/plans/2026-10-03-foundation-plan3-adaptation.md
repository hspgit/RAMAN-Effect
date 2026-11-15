# Foundation Model — Plan 3: Adaptation (S / LP / B1 / B2) and the Single Final Evaluation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adapt the frozen model A to our two Bacteria-ID tasks with four arms (S, LP, B1 = full fine-tuning, B2 = LoRA). Tune each arm's learning rate on validation only, train 5 paired seeds, then score the test sets **once** and answer the pre-registered questions Q1–Q3.

**Architecture:**
- **Four focused modules:**
  - `model/lora.py`: LoRA wrapper and injection by layer name.
  - `arms.py`: what each arm initialises from, trains, and at which learning rate.
  - `adapt.py`: the trainer and the sweep / seeds / curve queues. It never touches a test set.
  - `stats.py`: McNemar, the bootstrap and the Q1/Q3 rules.
- **`evaluate.py`:** the one test pass, behind a lock file, and the report.
- **Run folders:** each run writes to a deterministic folder (`runs/adapt/<task>/<arm>/lr<lr>_seed<s>/`), and a finished run is skipped. So every command is safe to re-run after an interruption.

**Tech Stack:** the Python 3.12 repo venv with torch 2.11 + CUDA (RTX 3060, 12 GB), numpy, scikit-learn 1.9, scipy 1.18, matplotlib, pyyaml.

**Spec:** `docs/superpowers/specs/2026-10-01-raman-foundation-model-design.md`. This plan implements:
- §6: the arms, hyperparameter fairness and the reuse deliverable;
- §8: metrics, statistics, Q1–Q3, the limitation and the optional curve;
- phases 3–4 of §7;
- the LoRA tests of §10.

**Inputs:**
- **Model A:** `foundation/runs/pretrain/20261003-123546_small_main/ckpt_best.pt` (git-ignored).
  - Its contents: `{"encoder": state_dict, "preset": "small", "epoch": 150, "probe": {...}}`.
  - The encoder has the masked-GroupNorm stem. G1 passed (k-NN 0.765 vs raw 0.625; commit `c364567`).
- **Tasks:** from `foundation.data.tasks` (Plan 1). All splits are balanced.

| Task | Train | Validation | Test | Classes |
|---|---|---|---|---|
| T1 stage i | `ref_train` 54,000 | `ref_val` 6,000 | — | 30 |
| T1 stage ii | `ft_train` 2,700 (90/class) | `ft_val` 300 (10/class) | `test` 3,000 | 30 |
| T2 | `train` 9,000 (2018clinical) | `val` 1,000 | `test` 2,500 (2019clinical) | 5 |

## Global Constraints

- **Separate code:** nothing in `foundation/` imports from `src/`, `legacy_baseline/`, `main.py`, `evaluate_kfold.py` or `generate_heatmaps.py`.
- **No comparison with the earlier project's models** (spec §12).
- **A is read-only:** it is loaded with `torch.load(..., weights_only=True)` and never written.
- **Arms** (spec §6.1):

  | Arm | Starts from | Trains |
  |---|---|---|
  | S | random | everything |
  | LP | A | a new linear head only (A frozen) |
  | B1 | A | the head for **5 epochs, once**, at the start of the first stage; then everything with **layer-wise LR decay 0.75** (stem smallest) |
  | B2 | A | LoRA on every `q`, `k`, `v`, `o`, `fc1`, `fc2`, plus norm parameters plus the head (A frozen) |

  - B2's LoRA: **α = 2r**, **LoRA dropout 0.05**, and r is the largest of **{8, 4, 2}** that keeps the trainable fraction **≤ 5%**.
  - B2's learning rate is **10× B1's**.
  - S uses the same schedule as B1's unfrozen phase, without the warm-up.
- **Early stopping:** on **validation loss** with **patience 5**.
- **Augmentation during adaptation:** the `mild` preset (shift ±2 cm⁻¹, scale 0.9–1.1, SNR 30–100), no masking.
- **Protocols:**
  - T1 is stage i (reference) and then stage ii (finetune). The arm's freeze rule applies in both stages.
  - T2 is a single stage on 2018clinical.
- **Fairness** (spec §6.2): every arm gets a **3-value LR sweep on seed 1**. Selection uses validation only (T1: the stage-ii `ft_val`). The selected LR is fixed for all seeds.
- **Seeds 1–5, paired across arms by seed.**
- **Statistics:**
  - McNemar per paired seed on the same test set.
  - Bootstrap 95% CI of the accuracy difference, resampling test samples **and** seeds, **10,000** resamples.
  - Mean ± sd per arm.
- **Q1–Q3** exactly as spec §8.3; Q3's margin is **1.0 accuracy point** and its limit is **≤ 5% trainable**.
- **The test sets are scored once,** after every run is frozen. Only `foundation/evaluate.py` predicts on `splits["test"]`.
- **No time limits:** runs end at a stage's `max_epochs` or on early stopping.
  - Progress lines: `[B1 T1 stage-ii seed 3] epoch 4/50 (46 left) | ... | ~… ETA`.
  - The ETA is display only.
  - The final line says why the stage ended.
- **Each run folder holds:**
  - the config snapshot with git SHA and library versions;
  - `metrics.jsonl`;
  - per-sample validation predictions;
  - the add-on;
  - `summary.json`.
- **Python:** `venv/Scripts/python.exe` from the repo root. Every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Decisions where the spec is silent (made while building and testing the prototype)

| # | Topic | Decision | Why |
|---|---|---|---|
| D1 | The three LR values | S and B1: {1e-4, 3e-4, 1e-3}. LP and B2: {1e-3, 3e-3, 1e-2} | B2 = 10× B1 (spec §6.1). LP is head-only like B2, so it gets the same scale |
| D2 | Selection metric | Validation accuracy, ties broken by lower validation loss | The spec says "select on validation". T2's validation is near ceiling (pilot: 98.3–99.5% for every arm), so the loss usually decides there |
| D3 | B1 warm-up | 5 epochs at 10× the arm's LR, no early stopping | Head-only training needs a head-sized LR (the same factor as B2) |
| D4 | Optimiser inside a stage | AdamW β (0.9, 0.999), weight decay 0.05 on matrices only, batch 256, grad clip 1.0, fp16 autocast. 1 epoch of linear LR warm-up, then cosine to the stage's `max_epochs`. Fresh optimiser per stage | Spec §5.3's β, batch and warm-up are pretraining settings; these are standard fine-tuning ones |
| D5 | `max_epochs` | stage i 30, stage ii 50, T2 50. The best-validation-loss epoch is restored at the end of each stage | Early stopping ends most stages long before (pilot: T2 stopped at 13–40) |
| D6 | LLRD on S | S also gets LLRD 0.75 | The spec says "same schedule as B1's unfrozen phase". LLRD slightly favours a pretrained start; S's own LR sweep partly offsets this, and it is stated in the report |
| D7 | Trainable fraction | trainable / all parameters of the adapted model (A + LoRA + head). For small, r = 8: **4.52% (T1)**, **4.40% (T2)** | It counts what is actually deployed |
| D8 | B2 "norm parameters" | Every LayerNorm plus the stem's masked GroupNorm affine | They are all norm parameters |
| D9 | Dropout with a frozen A | LP and the B1 warm-up run the frozen encoder in eval mode. B2 trains with the encoder's dropout on, as is usual for PEFT | A frozen feature extractor should be deterministic |
| D10 | "Scored once" | `evaluate.py` writes `runs/adapt/final_test/scored.json` and refuses a second scoring pass. `--report-only` re-reads the saved predictions | Makes the rule mechanical |
| D11 | Optional curve (§8.5) | Sizes 5/10/25/50 per class plus "all" = the main runs (**90**, not 100: `ft_train` holds 90 per class after the 10% validation split). Each arm uses its full-data LR. Stage i is reused from the full-data run (`stage1.pt`) because stage i never sees finetune data | Cuts the curve from about 7 h to about 40 min |
| D12 | Curve vs the single pass | `evaluate.py` refuses to score until the curve is finished, unless `--skip-curve` is given | Otherwise the test lock would make the curve impossible later |
| D13 | Tests | McNemar is the exact binomial on discordant pairs; the bootstrap CI is a percentile interval | Exact at any n; standard |

**Note on `mild` scale 0.9–1.1:**
- This was Plan 2's existing behaviour and is kept as specified.
- `corrupt()` re-normalises min–max after scaling. So the scale factor only changes the effective noise level by about ±10%.

## Measured on this machine (real data, model A, RTX 3060)

| | S / B1 | B2 | LP |
|---|---|---|---|
| T1 stage-i epoch (211 steps) | ~30 s | ~40 s | ~9 s |
| T1 stage-ii epoch | ~1.5 s | ~2 s | <1 s |
| T2 epoch | ~5 s | ~7 s | ~1.5 s |

**T2 pilot at seed 1** (full settings):

| Arm | LR | Stopped at | Validation accuracy | Time |
|---|---|---|---|---|
| B1 | 3e-4 | epoch 17/50 | 99.5% | 1m37s |
| B2 | 3e-3 | epoch 13/50 | 98.8% | 1m25s |
| S | 3e-4 | epoch 22/50 | 98.3% | 1m54s |
| LP | 3e-3 | epoch 40/50 | 98.3% | 1m07s |

**Estimated total:**
- Sweep plus seeds: 24 + 32 runs, about 5 h (T1 about 4 h, T2 about 45 min).
- Curve: about 40 min.
- Final pass: under 5 min.

These are estimates only; nothing stops on time.

## Review Focus

1. **Re-running after an interruption must not redo or overwrite finished runs.** A run counts as finished only once its `summary.json` exists (written last, atomically). Pinned by `test_finished_run_is_skipped` and `test_sweep_selects_then_seeds_reuse_the_sweep_run` (Task 3).
2. **A second test-scoring pass must be refused.** Pinned by `test_single_test_pass_and_report` (Task 4).
3. **The add-on reloaded for testing must be exactly the trained model.** If it is not, every test number is wrong while looking plausible. Pinned by `test_scored_predictions_match_the_trained_models` (Task 4) and `test_addon_round_trip_is_exact` (Task 1).
4. **Comparisons must pair runs by seed.** Unpaired seeds raise an error. Pinned by `test_unpaired_seeds_raise` (Task 4).
5. **A frozen A must really stay frozen and deterministic.** Base weights must not move under LoRA, and LP must see no dropout. Pinned by `test_frozen_base_weights_do_not_move_during_training` (Task 1) and `test_head_phase_runs_the_frozen_encoder_without_dropout` (Task 3).

---

### Task 1: LoRA and the four arms

**Files:**
- Create: `foundation/model/lora.py`, `foundation/arms.py`
- Test: `tests/foundation/test_lora.py`, `tests/foundation/test_arms.py`

**Interfaces:**
- Consumes (from Plan 2):
  - `foundation.model.encoder`: `SpectralEncoder(cfg)` with `.cfg.width`, `.cfg.layers`, `.embed(x, m)`, `.blocks[i].attn.{q,k,v,o}` and `.blocks[i].mlp.{fc1,fc2}`; `PRESETS`.
  - `foundation.model.stem.MaskedGroupNorm`.
- Produces:
  - `lora.py`: `TARGETS`, `RANKS = (8, 4, 2)`, `MAX_FRACTION = 0.05`, `LoRALinear(base, r, alpha, dropout)` with `.base`, `.lora_a` (r × in) and `.lora_b` (out × r, zeros); `inject_lora(module, r, dropout=0.05, targets=TARGETS) -> list[str]` (α = 2r; raises `ValueError` if nothing matches); `lora_parameter_count(module) -> int`.
  - `arms.py`:
    - `ARMS = ("S", "LP", "B1", "B2")`;
    - `Classifier(encoder, n_classes)` with `.encoder` and `.head`;
    - `load_a(path) -> (preset, state)`;
    - `choose_rank(encoder, n_classes) -> (r, fraction)`;
    - `build_model(arm, n_classes, preset, a_state=None, seed=0, lora_rank=None) -> (model, rank|None)`;
    - `set_trainable(model, phase)` with phase in `"head" | "full" | "lora"`;
    - `phases(arm) -> [(phase, is_warmup)]`;
    - `trainable_count`, `total_count`;
    - `lr_scale(name, n_layers, decay)`;
    - `param_groups(model, lr, weight_decay, llrd=None)`;
    - `addon_state(model) -> {name: tensor}`;
    - `load_addon(model, state)` (raises `KeyError` on unknown names and `ValueError` on a shape mismatch).

- [ ] **Step 1: Write the failing tests**

<!-- file: tests/foundation/test_lora.py -->
```python
import copy

import pytest
import torch
import torch.nn as nn

from foundation.arms import Classifier, choose_rank, set_trainable, total_count, trainable_count
from foundation.data.harmonize import GRID
from foundation.model.encoder import PRESETS, SpectralEncoder
from foundation.model.lora import TARGETS, LoRALinear, inject_lora, lora_parameter_count


def _batch(n=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, GRID.size, generator=g)
    m = torch.ones(n, GRID.size, dtype=torch.bool)
    m[0, :200] = False
    return x * m, m


def test_zero_initialised_adapters_leave_outputs_unchanged():
    torch.manual_seed(0)
    enc = SpectralEncoder(PRESETS["tiny"]).eval()
    x, m = _batch()
    before = enc.embed(x, m)
    inject_lora(enc, r=4)
    enc.eval()
    torch.testing.assert_close(enc.embed(x, m), before, rtol=0, atol=0)


def test_injects_every_target_in_every_block():
    enc = SpectralEncoder(PRESETS["tiny"])
    names = inject_lora(enc, r=4)
    assert len(names) == len(TARGETS) * PRESETS["tiny"].layers
    assert "blocks.0.attn.q" in names and "blocks.3.mlp.fc2" in names
    assert type(enc.proj) is nn.Linear                      # the patch projection is not a target
    assert lora_parameter_count(enc) == PRESETS["tiny"].layers * 4 * (
        4 * (128 + 128) + 2 * (128 + 512))


def test_inject_without_targets_raises():
    with pytest.raises(ValueError, match="no nn.Linear"):
        inject_lora(nn.Sequential(nn.ReLU()), r=2)


def test_injecting_twice_raises():
    enc = SpectralEncoder(PRESETS["tiny"])
    inject_lora(enc, r=4)
    with pytest.raises(ValueError, match="no nn.Linear"):
        inject_lora(enc, r=4)


def test_lora_update_is_scaled_by_alpha_over_r():
    torch.manual_seed(0)
    base = nn.Linear(6, 5)
    layer = LoRALinear(base, r=2, alpha=4, dropout=0.0)
    with torch.no_grad():
        layer.lora_b.fill_(1.0)
    x = torch.randn(3, 6)
    expected = base(x) + 2.0 * (x @ layer.lora_a.t() @ layer.lora_b.t())
    torch.testing.assert_close(layer(x), expected)
    assert not any(p.requires_grad for p in layer.base.parameters())


@pytest.mark.parametrize("preset", ["tiny", "small", "base"])
def test_trainable_fraction_at_most_5_percent(preset):
    enc = SpectralEncoder(PRESETS[preset])
    for n_classes in (30, 5):
        r, frac = choose_rank(enc, n_classes)
        assert r in (8, 4, 2) and frac <= 0.05
        m = Classifier(copy.deepcopy(enc), n_classes)
        inject_lora(m.encoder, r)
        set_trainable(m, "lora")
        assert trainable_count(m) / total_count(m) == pytest.approx(frac)


def test_choose_rank_takes_the_largest_rank_that_fits():
    assert choose_rank(SpectralEncoder(PRESETS["small"]), 30)[0] == 8
    assert choose_rank(SpectralEncoder(PRESETS["tiny"]), 30)[0] == 4


def test_frozen_base_weights_do_not_move_during_training():
    torch.manual_seed(0)
    m = Classifier(SpectralEncoder(PRESETS["tiny"]), 3)
    inject_lora(m.encoder, 4)
    set_trainable(m, "lora")
    q = m.encoder.blocks[0].attn.q
    w0, b0 = q.lora_b.detach().clone(), q.base.weight.detach().clone()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    x, mask = _batch()
    nn.functional.cross_entropy(m(x, mask), torch.tensor([0, 1, 2])).backward()
    opt.step()
    assert torch.equal(q.base.weight, b0)
    assert not torch.equal(q.lora_b, w0)
```

<!-- file: tests/foundation/test_arms.py -->
```python
import pytest
import torch

from foundation.arms import (ARMS, addon_state, build_model, load_addon, lr_scale, param_groups,
                             phases, set_trainable, trainable_count)
from foundation.data.harmonize import GRID
from foundation.model.encoder import PRESETS, SpectralEncoder


@pytest.fixture
def a_state():
    torch.manual_seed(123)
    return SpectralEncoder(PRESETS["tiny"]).state_dict()


def _batch(n=4):
    g = torch.Generator().manual_seed(1)
    return torch.rand(n, GRID.size, generator=g), torch.ones(n, GRID.size, dtype=torch.bool)


def test_arms_other_than_s_start_from_a(a_state):
    x, m = _batch()
    ref = SpectralEncoder(PRESETS["tiny"])
    ref.load_state_dict(a_state)
    ref.eval()
    for arm in ("LP", "B1", "B2"):
        model, _ = build_model(arm, 3, "tiny", a_state, seed=1)
        model.eval()
        torch.testing.assert_close(model.encoder.embed(x, m), ref.embed(x, m))
    s, _ = build_model("S", 3, "tiny", a_state, seed=1)
    s.eval()
    assert not torch.allclose(s.encoder.embed(x, m), ref.embed(x, m))


def test_build_is_seeded_and_validates(a_state):
    a1, _ = build_model("S", 3, "tiny", None, seed=5)
    a2, _ = build_model("S", 3, "tiny", None, seed=5)
    b, _ = build_model("S", 3, "tiny", None, seed=6)
    assert torch.equal(a1.head.weight, a2.head.weight)
    assert not torch.equal(a1.head.weight, b.head.weight)
    with pytest.raises(ValueError, match="needs model A"):
        build_model("B1", 3, "tiny", None)
    with pytest.raises(ValueError, match="unknown arm"):
        build_model("B3", 3, "tiny", a_state)
    _, rank = build_model("B2", 3, "tiny", a_state)
    assert rank == 4
    assert build_model("B2", 3, "tiny", a_state, lora_rank=2)[1] == 2


def test_set_trainable_phases(a_state):
    model, _ = build_model("B2", 3, "tiny", a_state)
    set_trainable(model, "lora")
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert names and all(n.startswith("head.") or n.endswith(("lora_a", "lora_b"))
                         or ".ln" in n or n.startswith("encoder.norm.")
                         or n.startswith("encoder.stem.norm.") for n in names)
    assert any("stem" in n for n in names)            # the masked GroupNorm's affine is trained
    assert not any(n.endswith("base.weight") for n in names)
    set_trainable(model, "head")
    assert [n for n, p in model.named_parameters() if p.requires_grad] == ["head.weight", "head.bias"]
    set_trainable(model, "full")
    assert all(p.requires_grad for p in model.parameters())
    with pytest.raises(ValueError):
        set_trainable(model, "half")


def test_phases_per_arm():
    assert phases("B1") == [("head", True), ("full", False)]
    assert phases("LP") == [("head", False)]
    assert phases("S") == [("full", False)]
    assert phases("B2") == [("lora", False)]
    assert set(ARMS) == {"S", "LP", "B1", "B2"}


def test_layerwise_decay_scales():
    d, L = 0.75, 6
    assert lr_scale("head.weight", L, d) == 1.0
    assert lr_scale("encoder.norm.weight", L, d) == 1.0
    assert lr_scale("encoder.blocks.5.attn.q.weight", L, d) == pytest.approx(0.75)
    assert lr_scale("encoder.blocks.0.mlp.fc1.bias", L, d) == pytest.approx(0.75 ** 6)
    assert lr_scale("encoder.proj.weight", L, d) == pytest.approx(0.75 ** 7)
    assert lr_scale("encoder.stem.convs.0.weight", L, d) == pytest.approx(0.75 ** 7)


def test_param_groups_cover_trainable_once_with_decay_rules(a_state):
    model, _ = build_model("B1", 3, "tiny", a_state)
    set_trainable(model, "full")
    groups = param_groups(model, 1e-3, 0.05, llrd=0.75)
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == sum(1 for p in model.parameters())
    for g in groups:
        assert all((p.ndim >= 2) == (g["weight_decay"] == 0.05) for p in g["params"])
    lrs = sorted({g["lr"] for g in groups})
    assert lrs[-1] == pytest.approx(1e-3) and lrs[0] == pytest.approx(1e-3 * 0.75 ** 5)
    set_trainable(model, "head")
    groups = param_groups(model, 1e-2, 0.05)
    assert sum(len(g["params"]) for g in groups) == 2
    assert {g["lr"] for g in groups} == {1e-2}


def test_addon_round_trip_is_exact(a_state):
    x, m = _batch()
    model, rank = build_model("B2", 3, "tiny", a_state, seed=3)
    set_trainable(model, "lora")
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.add_(0.01 * torch.randn_like(p))
    state = addon_state(model)
    assert len(state) and sum(v.numel() for v in state.values()) == trainable_count(model)
    fresh, _ = build_model("B2", 3, "tiny", a_state, seed=99, lora_rank=rank)
    load_addon(fresh, state)
    model.eval(), fresh.eval()
    torch.testing.assert_close(fresh(x, m), model(x, m), rtol=0, atol=0)


def test_load_addon_rejects_foreign_state(a_state):
    model, _ = build_model("LP", 3, "tiny", a_state)
    with pytest.raises(KeyError, match="does not"):
        load_addon(model, {"encoder.blocks.0.attn.q.lora_a": torch.zeros(4, 128)})
    with pytest.raises(ValueError, match="shape mismatch"):
        load_addon(model, {"head.weight": torch.zeros(5, 128)})
```

- [ ] **Step 2: Run them to verify they fail**

Run: `venv/Scripts/python.exe -m pytest tests/foundation/test_lora.py tests/foundation/test_arms.py -q`
Expected: collection errors, `ModuleNotFoundError: No module named 'foundation.model.lora'` / `'foundation.arms'`.

- [ ] **Step 3: Implement**

<!-- file: foundation/model/lora.py -->
```python
"""LoRA adapters for the encoder's Linear layers (spec section 6.1, arm B2).

inject_lora() swaps each target nn.Linear for a LoRALinear that keeps the
original layer frozen as `.base` and adds a low-rank update scaled by
alpha / r. B starts at zero, so a freshly injected model computes exactly
what A computed. Load A's weights BEFORE injecting: injection renames
`...q.weight` to `...q.base.weight`.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

TARGETS = ("q", "k", "v", "o", "fc1", "fc2")
RANKS = (8, 4, 2)                 # spec: the largest that keeps trainable <= MAX_FRACTION
MAX_FRACTION = 0.05


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.scale = r, alpha / r
        self.lora_a = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        update = (self.drop(x) @ self.lora_a.t().to(x.dtype)) @ self.lora_b.t().to(x.dtype)
        return self.base(x) + self.scale * update


def inject_lora(module: nn.Module, r: int, dropout: float = 0.05, targets=TARGETS):
    """Replace every nn.Linear whose attribute name is in `targets` with a
    LoRALinear (alpha = 2r). Returns the dotted names that were replaced."""
    replaced = []
    for parent_name, parent in list(module.named_modules()):
        for name, child in list(parent.named_children()):
            if name in targets and type(child) is nn.Linear:
                setattr(parent, name, LoRALinear(child, r, alpha=2 * r, dropout=dropout))
                replaced.append(f"{parent_name}.{name}" if parent_name else name)
    if not replaced:
        raise ValueError(f"no nn.Linear named any of {targets} found")
    return replaced


def lora_parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for n, p in module.named_parameters()
               if n.endswith(("lora_a", "lora_b")))
```

<!-- file: foundation/arms.py -->
```python
"""The four adaptation arms (spec section 6.1): what each one initialises from,
which parameters it trains, and the learning rate each parameter gets.

    S   random init, everything trained
    LP  A, a new linear head only (A frozen, in eval mode)
    B1  A, head-only warm-up, then everything with layer-wise LR decay
    B2  A, LoRA adapters + norm parameters + head (A frozen)

A run's add-on is the state of its trainable parameters: the head (LP), LoRA
plus norms plus head (B2), or the whole model (S, B1).
"""
from __future__ import annotations

import copy
import re

import torch
import torch.nn as nn

from .model.encoder import PRESETS, SpectralEncoder
from .model.lora import MAX_FRACTION, RANKS, inject_lora
from .model.stem import MaskedGroupNorm

ARMS = ("S", "LP", "B1", "B2")
NORM_TYPES = (nn.LayerNorm, MaskedGroupNorm)


class Classifier(nn.Module):
    """Encoder -> mean-pooled embedding -> linear head."""

    def __init__(self, encoder: SpectralEncoder, n_classes: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.cfg.width, n_classes)

    def forward(self, x, m):
        return self.head(self.encoder.embed(x, m))


def load_a(path):
    """Model A's checkpoint -> (preset, encoder state_dict)."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    return ck["preset"], ck["encoder"]


def choose_rank(encoder: SpectralEncoder, n_classes: int):
    """Largest rank in RANKS whose B2 model trains <= MAX_FRACTION of its
    parameters. Returns (rank, fraction)."""
    for r in RANKS:
        m = Classifier(copy.deepcopy(encoder), n_classes)
        inject_lora(m.encoder, r)
        set_trainable(m, "lora")
        frac = trainable_count(m) / total_count(m)
        if frac <= MAX_FRACTION:
            return r, frac
    raise ValueError(f"no LoRA rank in {RANKS} keeps the trainable fraction <= {MAX_FRACTION}")


def build_model(arm, n_classes, preset, a_state=None, seed=0, lora_rank=None):
    """Seeded construction. Arms other than S start from A's weights; B2 then
    gets LoRA (rank from choose_rank unless given). Returns (model, rank)."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    if arm != "S" and a_state is None:
        raise ValueError(f"arm {arm} needs model A's weights")
    torch.manual_seed(seed)
    enc = SpectralEncoder(PRESETS[preset])
    if arm != "S":
        enc.load_state_dict(a_state)
    rank = None
    if arm == "B2":
        rank = lora_rank or choose_rank(enc, n_classes)[0]
    model = Classifier(enc, n_classes)
    if arm == "B2":
        inject_lora(model.encoder, rank)
    return model, rank


def set_trainable(model: Classifier, phase: str):
    """phase 'head': head only. 'full': everything. 'lora': LoRA + norms + head."""
    for p in model.parameters():
        p.requires_grad_(phase == "full")
    if phase == "full":
        return
    if phase not in ("head", "lora"):
        raise ValueError(f"unknown phase {phase!r}")
    for p in model.head.parameters():
        p.requires_grad_(True)
    if phase == "lora":
        for name, p in model.named_parameters():
            if name.endswith(("lora_a", "lora_b")):
                p.requires_grad_(True)
        for mod in model.modules():
            if isinstance(mod, NORM_TYPES):
                for p in mod.parameters():
                    p.requires_grad_(True)


def phases(arm):
    """[(phase, is_warmup)] in training order for the FIRST stage of a task;
    later stages (T1 stage ii) use only the last entry."""
    return {"S": [("full", False)], "LP": [("head", False)],
            "B1": [("head", True), ("full", False)], "B2": [("lora", False)]}[arm]


def trainable_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def total_count(model):
    return sum(p.numel() for p in model.parameters())


_BLOCK = re.compile(r"^encoder\.blocks\.(\d+)\.")


def lr_scale(name, n_layers, decay):
    """Layer-wise LR decay: head and final norm x1, block i x decay^(L-i),
    stem and patch projection x decay^(L+1) (the smallest)."""
    if name.startswith("head.") or name.startswith("encoder.norm."):
        return 1.0
    m = _BLOCK.match(name)
    if m:
        return decay ** (n_layers - int(m.group(1)))
    return decay ** (n_layers + 1)


def param_groups(model, lr, weight_decay, llrd=None):
    """AdamW groups over trainable parameters. Matrices get weight decay,
    vectors (biases, norms) do not. With llrd, each parameter's LR is
    lr * lr_scale(name)."""
    groups = {}
    n_layers = model.encoder.cfg.layers
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        scale = lr_scale(name, n_layers, llrd) if llrd else 1.0
        wd = weight_decay if p.ndim >= 2 else 0.0
        g = groups.setdefault((scale, wd), {"params": [], "lr": lr * scale, "weight_decay": wd})
        g["params"].append(p)
    return list(groups.values())


def addon_state(model):
    """The trainable parameters (what this run adds on top of A), on the CPU."""
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def load_addon(model, state):
    """Load an add-on into a model built the same way (build_model with A's
    weights). Every key must be one of the model's parameters."""
    names = dict(model.named_parameters())
    unknown = sorted(set(state) - set(names))
    if unknown:
        raise KeyError(f"add-on has parameters the model does not: {unknown[:5]}")
    with torch.no_grad():
        for k, v in state.items():
            if names[k].shape != v.shape:
                raise ValueError(f"shape mismatch for {k}: {tuple(v.shape)} vs "
                                 f"{tuple(names[k].shape)}")
            names[k].copy_(v)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `venv/Scripts/python.exe -m pytest tests/foundation/test_lora.py tests/foundation/test_arms.py -q`
Expected: `18 passed`.

- [ ] **Step 5: Commit**

```bash
git add foundation/model/lora.py foundation/arms.py tests/foundation/test_lora.py tests/foundation/test_arms.py
git commit -m "foundation: LoRA adapters and the four adaptation arms

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Paired statistics and the pre-registered questions

**Files:**
- Create: `foundation/stats.py`
- Test: `tests/foundation/test_stats.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (numpy, `scipy.stats.binomtest`).
- Produces:
  - `N_BOOT = 10_000`, `Q3_MARGIN = 0.01`, `Q3_MAX_FRACTION = 0.05`;
  - `mcnemar(correct_a, correct_b) -> {"b", "c", "p"}`;
  - `bootstrap_diff(correct_a, correct_b, n_boot=N_BOOT, seed=0) -> {"diff", "lo", "hi"}`, taking two (seeds, samples) bool arrays paired by row;
  - `mean_sd(values) -> (mean, sd)`;
  - `verdict(ci) -> str`;
  - `q1(diffs)`, where `diffs` is `{task: {(arm, other): ci}}` and the result is `{task: {"B1 vs S": {..., "verdict", "success"}}}`;
  - `q3(mean_acc, fraction, candidates=("B2", "LP")) -> {"candidates": {...}, "pass": bool}`.

- [ ] **Step 1: Write the failing tests**

<!-- file: tests/foundation/test_stats.py -->
```python
import numpy as np
import pytest

from foundation.stats import bootstrap_diff, mcnemar, mean_sd, q1, q3, verdict


def test_mcnemar_counts_and_exact_p():
    a = np.array([1] * 10 + [0] * 5 + [1] * 5, bool)
    b = np.array([0] * 10 + [0] * 5 + [1] * 5, bool)
    r = mcnemar(a, b)
    assert (r["b"], r["c"]) == (10, 0)
    assert r["p"] == pytest.approx(2 * 0.5 ** 10)
    assert mcnemar(a, a) == {"b": 0, "c": 0, "p": 1.0}
    with pytest.raises(ValueError):
        mcnemar(a, b[:-1])


def test_bootstrap_degenerate_cases():
    ones, zeros = np.ones((5, 200), bool), np.zeros((5, 200), bool)
    assert bootstrap_diff(ones, ones, n_boot=200) == {"diff": 0.0, "lo": 0.0, "hi": 0.0}
    assert bootstrap_diff(ones, zeros, n_boot=200) == {"diff": 1.0, "lo": 1.0, "hi": 1.0}
    with pytest.raises(ValueError):
        bootstrap_diff(ones, zeros[:, :10])
    with pytest.raises(ValueError):
        bootstrap_diff(ones[0], zeros[0])


def test_bootstrap_ci_covers_a_known_difference():
    rng = np.random.default_rng(0)
    a = rng.random((5, 3000)) < 0.90
    b = rng.random((5, 3000)) < 0.85
    r = bootstrap_diff(a, b, n_boot=2000)
    assert r["diff"] == pytest.approx(a.mean() - b.mean())
    assert r["lo"] < r["diff"] < r["hi"]
    assert r["lo"] > 0 and r["hi"] - r["lo"] < 0.05
    assert bootstrap_diff(a, b, n_boot=2000) == r                # seeded


def test_seed_resampling_widens_the_ci():
    """Seeds that disagree must widen the interval beyond sample noise alone."""
    rng = np.random.default_rng(1)
    b = rng.random((5, 2000)) < 0.8
    a_stable = b.copy()
    a_stable[:, :100] = True
    a_noisy = b.copy()
    for s, k in enumerate([0, 0, 100, 200, 200]):
        a_noisy[s, :k] = True
    w_stable = np.ptp(list(bootstrap_diff(a_stable, b, n_boot=2000).values())[1:])
    w_noisy = np.ptp(list(bootstrap_diff(a_noisy, b, n_boot=2000).values())[1:])
    assert w_noisy > w_stable


def test_mean_sd_and_verdict():
    assert mean_sd([1.0, 3.0]) == (2.0, pytest.approx(np.sqrt(2)))
    assert mean_sd([0.5]) == (0.5, 0.0)
    assert verdict({"lo": 0.01, "hi": 0.02}) == "A helps"
    assert verdict({"lo": -0.02, "hi": -0.01}) == "A hurts"
    assert verdict({"lo": -0.01, "hi": 0.02}) == "no clear difference"


def test_q1_only_compares_against_s():
    diffs = {"T1": {("B1", "S"): {"diff": 0.02, "lo": 0.01, "hi": 0.03},
                    ("B2", "S"): {"diff": 0.01, "lo": -0.01, "hi": 0.03},
                    ("B1", "B2"): {"diff": 0.01, "lo": 0.0, "hi": 0.02}}}
    r = q1(diffs)["T1"]
    assert set(r) == {"B1 vs S", "B2 vs S"}
    assert r["B1 vs S"]["success"] and not r["B2 vs S"]["success"]


def test_q3_margin_fraction_and_both_tasks():
    acc = {"T1": {"S": 0.90, "LP": 0.80, "B1": 0.95, "B2": 0.94},
           "T2": {"S": 0.90, "LP": 0.85, "B1": 0.95, "B2": 0.95}}
    frac = {t: {"S": 1.0, "LP": 0.001, "B1": 1.0, "B2": 0.045} for t in acc}
    r = q3(acc, frac)
    assert r["candidates"]["B2"]["pass"] and not r["candidates"]["LP"]["pass"] and r["pass"]
    assert r["candidates"]["B2"]["tasks"]["T1"]["gap"] == pytest.approx(0.01)
    acc["T2"]["B2"] = 0.93                                     # 2 points below on one task
    assert not q3(acc, frac)["pass"]
    acc["T2"]["B2"] = 0.95
    frac["T1"]["B2"] = 0.06                                    # too many trainable parameters
    assert not q3(acc, frac)["candidates"]["B2"]["pass"]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `venv/Scripts/python.exe -m pytest tests/foundation/test_stats.py -q`
Expected: `ModuleNotFoundError: No module named 'foundation.stats'`.

- [ ] **Step 3: Implement**

<!-- file: foundation/stats.py -->
```python
"""Paired statistics and the pre-registered questions (spec section 8).

Correctness arrays are (seeds, test samples) bool, rows paired by seed.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest

N_BOOT = 10_000
Q3_MARGIN = 0.01          # 1.0 accuracy point
Q3_MAX_FRACTION = 0.05


def mcnemar(correct_a, correct_b):
    """Exact two-sided McNemar test on one paired seed. b: only A right,
    c: only B right."""
    a, b_ = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    if a.shape != b_.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b_.shape}")
    b = int((a & ~b_).sum())
    c = int((~a & b_).sum())
    p = 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue)
    return {"b": b, "c": c, "p": p}


def bootstrap_diff(correct_a, correct_b, n_boot=N_BOOT, seed=0, chunk=500):
    """Accuracy difference A - B: point estimate (mean over seeds) and a 95%
    percentile CI from resampling seeds and test samples together."""
    a = np.asarray(correct_a, bool)
    b = np.asarray(correct_b, bool)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError(f"need equal (seeds, samples) arrays, got {a.shape} and {b.shape}")
    d = a.astype(np.float64) - b.astype(np.float64)              # (S, N)
    S, N = d.shape
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for s in range(0, n_boot, chunk):
        k = min(chunk, n_boot - s)
        w = rng.multinomial(N, np.full(N, 1.0 / N), size=k)       # (k, N) sample counts
        per_seed = w @ d.T / N                                    # (k, S)
        pick = rng.integers(0, S, size=(k, S))
        out[s:s + k] = np.take_along_axis(per_seed, pick, axis=1).mean(axis=1)
    lo, hi = np.percentile(out, [2.5, 97.5])
    return {"diff": float(d.mean()), "lo": float(lo), "hi": float(hi)}


def mean_sd(values):
    v = np.asarray(values, float)
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0


def verdict(ci):
    if ci["lo"] > 0:
        return "A helps"
    if ci["hi"] < 0:
        return "A hurts"
    return "no clear difference"


def q1(diffs):
    """diffs[task][(arm, 'S')] -> bootstrap dict. Success = mean > 0 and CI
    excluding 0 (spec 8.3)."""
    return {task: {f"{a} vs {b}": {**ci, "verdict": verdict(ci),
                                   "success": ci["diff"] > 0 and ci["lo"] > 0}
                   for (a, b), ci in pairs.items() if b == "S"}
            for task, pairs in diffs.items()}


def q3(mean_acc, fraction, candidates=("B2", "LP")):
    """mean_acc[task][arm], fraction[task][arm]. A candidate passes if, on every
    task, its mean accuracy is within Q3_MARGIN of that task's best arm and it
    trains <= Q3_MAX_FRACTION of its parameters."""
    res = {}
    for c in candidates:
        per_task = {}
        for task, accs in mean_acc.items():
            best = max(accs.values())
            per_task[task] = {"gap": best - accs[c], "fraction": fraction[task][c],
                              "ok": accs[c] >= best - Q3_MARGIN - 1e-12
                              and fraction[task][c] <= Q3_MAX_FRACTION}
        res[c] = {"tasks": per_task, "pass": all(t["ok"] for t in per_task.values())}
    return {"candidates": res, "pass": any(r["pass"] for r in res.values())}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `venv/Scripts/python.exe -m pytest tests/foundation/test_stats.py -q`
Expected: `7 passed`.

- [ ] **Step 5: Commit**

```bash
git add foundation/stats.py tests/foundation/test_stats.py
git commit -m "foundation: McNemar, seed+sample bootstrap and the Q1/Q3 rules

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: The adaptation trainer and its queues

**Files:**
- Create: `foundation/adapt.py`, `foundation/configs/adapt.yaml`, `tests/foundation/test_adapt.py`
- Modify: `tests/foundation/conftest.py` (append only)

**Interfaces:**
- Consumes:
  - **Task 1:** `ARMS`, `build_model`, `load_a`, `set_trainable`, `phases`, `param_groups`, `addon_state`, `trainable_count`, `total_count`.
  - **Plan 1:** `foundation.data.tasks.{Split, load_t1, load_t2}` and `foundation.data.corrupt.{PRESETS, corrupt}`.
  - **Plan 2:** `foundation.pretrain.{_fmt_secs, _git_sha, _save_atomic, lr_factor}`.
- Produces:
  - **Constants:** `TASKS = ("T1", "T2")`, `STAGES`, `DEFAULTS`.
  - **Config and data:**
    - `load_config(path=None, **overrides)`;
    - `load_task(name, cfg) -> Task`, which applies `cfg["ft_per_class"]` to T1's `ft_train`;
    - `subsample_per_class(split, n, seed=0)`;
    - `run_dir_for(cfg, task, arm, lr, seed) -> Path`;
    - `GPUSplit(split, device)` with `.X`, `.M`, `.y` and `len()`.
  - **Training and prediction:**
    - `predict(model, data, device) -> logits`;
    - `val_metrics(model, data, device) -> (loss, acc, pred)`;
    - `train_phase(model, phase, train, val, lr, max_epochs, early_stop, tag, cfg, gen, device, log, metrics) -> info dict`;
    - `stage_label(stage, warmup)`;
    - `run_one(cfg, task_name, arm, lr, seed, task=None, a=None, log=print) -> summary dict`. Its run folder holds `config.json`, `metrics.jsonl`, `val_pred.npy`, `addon.pt`, `summary.json`, and `stage1.pt` for curve arms and seeds.
  - **Queues:**
    - `select_lr(summaries)`;
    - `selection_path(cfg, task)`;
    - `sweep(cfg, tasks, arms, log)`;
    - `seeds(cfg, tasks, arms, log)`;
    - `curve(cfg, log)`;
    - `status(cfg, log)`.
  - **CLI:** `python -m foundation.adapt {run,sweep,seeds,curve,status}`.
  - **`summary.json` keys:** `task`, `arm`, `lr`, `seed`, `lora_rank`, `ft_per_class`, `val_acc`, `val_loss`, `val_macro_f1`, `trainable_params`, `total_params`, `trainable_fraction`, `addon_bytes`, `seconds`, `stages`.
  - **Test fixtures** (`conftest.py`): `make_fake_ours(d)`, `make_fake_a(path, preset="tiny")`, and the `adapt_cfg` fixture (CPU, 1 epoch per stage, one LR per arm, seeds [1, 2], curve [2] per class on seed 1).

- [ ] **Step 1: Append the shared fixtures to `tests/foundation/conftest.py`**

Leave the existing contents unchanged and append:

<!-- file: tests/foundation/conftest.py (append) -->
```python


OURS_AXIS = np.linspace(1792.4, 381.98, 1000)       # descending, like data/wavenumbers.npy


def make_fake_ours(d, seed=0):
    """data/ look-alike: 3 isolate classes (T1) and 5 clinical classes (T2),
    one Gaussian band per class."""
    rng = np.random.default_rng(seed)
    d.mkdir()
    np.save(d / "wavenumbers.npy", OURS_AXIS)
    sets = {"reference": np.repeat([0.0, 1.0, 2.0], 20), "finetune": np.repeat([0.0, 1.0, 2.0], 10),
            "test": np.repeat([0.0, 1.0, 2.0], 10),
            "2018clinical": np.repeat([0.0, 2.0, 3.0, 5.0, 6.0], 10),
            "2019clinical": np.repeat([0.0, 2.0, 3.0, 5.0, 6.0], 4)}
    for name, y in sets.items():
        centres = {lab: 500 + 150 * i for i, lab in enumerate(sorted(set(y)))}
        X = np.stack([0.2 + np.exp(-0.5 * ((OURS_AXIS - centres[v]) / 6) ** 2)
                      + rng.normal(0, 0.01, OURS_AXIS.size) for v in y])
        np.save(d / f"X_{name}.npy", X)
        np.save(d / f"y_{name}.npy", y)
    return d


def make_fake_a(path, preset="tiny", seed=0):
    """A model-A checkpoint shaped like pretrain.py's ckpt_best.pt."""
    import torch

    from foundation.model.encoder import PRESETS, SpectralEncoder
    torch.manual_seed(seed)
    enc = SpectralEncoder(PRESETS[preset])
    torch.save({"encoder": enc.state_dict(), "preset": preset, "epoch": 1, "probe": {}}, path)
    return path


@pytest.fixture
def adapt_cfg(tmp_path):
    """A CPU-sized adaptation config over fake data and a fake tiny A."""
    from foundation.adapt import load_config
    data = make_fake_ours(tmp_path / "data")
    a = make_fake_a(tmp_path / "a.pt")
    cfg = load_config(None, a_ckpt=str(a), data_dir=str(data), cache_dir=str(tmp_path / "cache"),
                      run_root=str(tmp_path / "runs"), device="cpu", amp=False, batch_size=16)
    cfg.update(max_epochs={"stage-i": 1, "stage-ii": 1, "single": 1}, head_warmup_epochs=1,
               lr_grid={"S": [1e-3], "B1": [1e-3], "LP": [1e-2], "B2": [1e-2]},
               seeds=[1, 2], curve_per_class=[2], curve_seeds=[1])
    return cfg
```

- [ ] **Step 2: Write the failing tests**

<!-- file: tests/foundation/test_adapt.py -->
```python
import json

import numpy as np
import pytest
import torch

from foundation import adapt
from foundation.adapt import (GPUSplit, curve, load_task, run_dir_for, run_one, seeds,
                              select_lr, stage_label, status, subsample_per_class, sweep,
                              train_phase)
from foundation.arms import build_model, load_a
from foundation.data.tasks import Split


def test_b1_t1_smoke_run_writes_everything(adapt_cfg):
    logs = []
    s = run_one(adapt_cfg, "T1", "B1", 1e-3, 1, log=logs.append)
    out = run_dir_for(adapt_cfg, "T1", "B1", 1e-3, 1)
    for f in ("config.json", "metrics.jsonl", "val_pred.npy", "addon.pt", "summary.json"):
        assert (out / f).exists(), f
    assert [x["stage"] for x in s["stages"]] == ["stage-i-warmup", "stage-i", "stage-ii"]
    assert s["trainable_fraction"] == 1.0 and s["addon_bytes"] > 0 and s["lora_rank"] is None
    assert 0.0 <= s["val_acc"] <= 1.0
    assert any(l.startswith("[B1 T1 stage-i-warmup seed 1] epoch 1/1 (0 left)") for l in logs)
    assert any(l.startswith("[B1 T1 stage-ii seed 1] epoch 1/1 (0 left)") for l in logs)
    assert np.load(out / "val_pred.npy").shape == (3,)            # ft_val: 1 per class
    cfg_saved = json.loads((out / "config.json").read_text())
    assert cfg_saved["arm"] == "B1" and "git_sha" in cfg_saved and "torch" in cfg_saved
    tags = {json.loads(l)["tag"] for l in (out / "metrics.jsonl").read_text().splitlines()}
    assert tags == {"B1 T1 stage-i-warmup seed 1", "B1 T1 stage-i seed 1", "B1 T1 stage-ii seed 1"}


def test_finished_run_is_skipped(adapt_cfg):
    run_one(adapt_cfg, "T2", "LP", 1e-2, 1, log=lambda *_: None)
    logs = []
    run_one(adapt_cfg, "T2", "LP", 1e-2, 1, log=logs.append)
    assert logs and logs[0].startswith("[skip]")


def test_b2_trains_only_its_addon_and_t2_tags(adapt_cfg):
    logs = []
    s = run_one(adapt_cfg, "T2", "B2", 1e-2, 2, log=logs.append)
    assert s["lora_rank"] == 4 and s["trainable_fraction"] <= 0.05
    assert any(l.startswith("[B2 T2 seed 2] epoch 1/1") for l in logs)
    addon = torch.load(run_dir_for(adapt_cfg, "T2", "B2", 1e-2, 2) / "addon.pt", weights_only=True)
    assert any(k.endswith("lora_a") for k in addon) and "head.weight" in addon
    assert not any(k.endswith("base.weight") for k in addon)
    assert sum(v.numel() for v in addon.values()) == s["trainable_params"]


def test_b1_t2_warmup_tag():
    assert stage_label("single", True) == "warmup"
    assert stage_label("single", False) == ""
    assert stage_label("stage-i", True) == "stage-i-warmup"
    assert stage_label("stage-ii", False) == "stage-ii"


def test_lp_keeps_a_frozen(adapt_cfg):
    run_one(adapt_cfg, "T2", "LP", 1e-2, 1, log=lambda *_: None)
    addon = torch.load(run_dir_for(adapt_cfg, "T2", "LP", 1e-2, 1) / "addon.pt", weights_only=True)
    assert set(addon) == {"head.weight", "head.bias"}


def test_early_stopping_restores_best_epoch(adapt_cfg, monkeypatch):
    preset, a_state = load_a(adapt_cfg["a_ckpt"])
    model, _ = build_model("LP", 3, preset, a_state, seed=0)
    task = load_task("T1", adapt_cfg)
    tr = GPUSplit(task.splits["ft_train"], torch.device("cpu"))
    va = GPUSplit(task.splits["ft_val"], torch.device("cpu"))
    losses = iter([0.5, 0.6, 0.7, 0.8, 0.9])
    snapshots = []

    def fake_val(m, data, device):
        snapshots.append(m.head.weight.detach().clone())
        return next(losses), 0.5, np.zeros(3, int)

    monkeypatch.setattr(adapt, "val_metrics", fake_val)
    cfg = dict(adapt_cfg, patience=2)
    logs = []
    with open(adapt_cfg["run_root"] + "_m.jsonl", "w") as metrics:
        info = train_phase(model, "head", tr, va, 1e-2, 10, True, "LP T1 stage-ii seed 0", cfg,
                           torch.Generator().manual_seed(0), torch.device("cpu"), logs.append,
                           metrics)
    assert info["epochs_run"] == 3 and info["best_epoch"] == 1
    assert "val loss flat for 2 epochs" in info["reason"]
    assert torch.equal(model.head.weight, snapshots[0])


def test_select_lr_accuracy_then_loss():
    rs = [{"lr": 1e-4, "val_acc": 0.9, "val_loss": 0.3},
          {"lr": 3e-4, "val_acc": 0.95, "val_loss": 0.4},
          {"lr": 1e-3, "val_acc": 0.95, "val_loss": 0.2}]
    assert select_lr(rs) == 1e-3


def test_subsample_per_class():
    y = np.repeat([0, 1, 2], 9)
    s = Split(np.arange(27)[:, None].astype(np.float32), np.ones((27, 1), bool), y)
    sub = subsample_per_class(s, 2)
    assert np.bincount(sub.y).tolist() == [2, 2, 2]
    assert np.array_equal(subsample_per_class(s, 2).X, sub.X)       # fixed rows
    with pytest.raises(ValueError, match="fewer than"):
        subsample_per_class(s, 10)


def test_sweep_selects_then_seeds_reuse_the_sweep_run(adapt_cfg):
    adapt_cfg["lr_grid"]["LP"] = [1e-2, 1e-3]
    sweep(adapt_cfg, tasks=["T2"], arms=["LP"], log=lambda *_: None)
    sel = json.loads((run_dir_for(adapt_cfg, "T2", "LP", 1e-2, 1).parents[1]
                      / "selection.json").read_text())
    assert sel["LP"]["lr"] in (1e-2, 1e-3) and len(sel["LP"]["grid"]) == 2
    logs = []
    seeds(adapt_cfg, tasks=["T2"], arms=["LP"], log=logs.append)
    assert sum(l.startswith("[skip]") for l in logs) == 1          # seed 1 came from the sweep
    assert (run_dir_for(adapt_cfg, "T2", "LP", sel["LP"]["lr"], 2) / "summary.json").exists()
    status_lines = []
    status(adapt_cfg, log=status_lines.append)
    assert any("T2: sweep 2/" in l and "final seeds 2/8" in l for l in status_lines)


def test_seeds_without_sweep_explains(adapt_cfg):
    with pytest.raises(FileNotFoundError, match="sweep"):
        seeds(adapt_cfg, tasks=["T1"], arms=["S"])


def test_curve_reuses_stage_one(adapt_cfg):
    adapt_cfg.update(curve_arms=["B2"])
    sweep(adapt_cfg, tasks=["T1"], arms=["B2"], log=lambda *_: None)
    assert (run_dir_for(adapt_cfg, "T1", "B2", 1e-2, 1) / "stage1.pt").exists()
    logs = []
    curve(adapt_cfg, log=logs.append)
    assert any("reused from" in l for l in logs)
    out = run_dir_for(dict(adapt_cfg, ft_per_class=2), "T1", "B2", 1e-2, 1)
    s = json.loads((out / "summary.json").read_text())
    assert s["ft_per_class"] == 2 and "reused_from" in s["stages"][0]
    assert [x["stage"] for x in s["stages"]] == ["stage-i", "stage-ii"]


def test_head_phase_runs_the_frozen_encoder_without_dropout(adapt_cfg, tmp_path):
    preset, a_state = load_a(adapt_cfg["a_ckpt"])
    model, _ = build_model("LP", 3, preset, a_state, seed=0)
    task = load_task("T1", adapt_cfg)
    cpu = torch.device("cpu")
    tr, va = GPUSplit(task.splits["ft_train"], cpu), GPUSplit(task.splits["ft_val"], cpu)
    modes = []
    model.encoder.blocks[0].register_forward_hook(lambda m, i, o: modes.append(m.training))
    with open(tmp_path / "m.jsonl", "w") as metrics:
        args = (tr, va, 1e-3, 1, False, "t", adapt_cfg, torch.Generator().manual_seed(0), cpu,
                lambda *_: None, metrics)
        train_phase(model, "head", *args)
        assert modes and not any(modes)
        modes.clear()
        train_phase(model, "full", *args)
        assert any(modes)
```

- [ ] **Step 3: Run them to verify they fail**

Run: `venv/Scripts/python.exe -m pytest tests/foundation/test_adapt.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'foundation.adapt'`.

- [ ] **Step 4: Implement the trainer and its config**

<!-- file: foundation/adapt.py -->
```python
"""Adapt model A to our tasks with arms S / LP / B1 / B2 (spec section 6).

    python -m foundation.adapt run   --task T1 --arm B1 --lr 3e-4 --seed 1
    python -m foundation.adapt sweep [--task T1 T2] [--arm S LP B1 B2]
    python -m foundation.adapt seeds [--task T1 T2] [--arm ...]
    python -m foundation.adapt curve                (optional, spec 8.5)
    python -m foundation.adapt status

`sweep` trains every arm at each learning rate of its grid on the sweep seed
and writes <run_root>/<task>/selection.json (best validation accuracy, ties
-> lower validation loss). `seeds` trains the remaining seeds at the selected
learning rate. A run whose summary.json exists is skipped, so both commands
can be re-run after an interruption. Test sets are never touched here
(foundation.evaluate scores them once).

Runs end at a stage's max_epochs or when validation loss has not improved for
`patience` epochs; never on elapsed time.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score

from .arms import (ARMS, addon_state, build_model, load_a, param_groups, phases,
                   set_trainable, total_count, trainable_count)
from .data.corrupt import PRESETS as CORRUPTIONS
from .data.corrupt import corrupt
from .data.tasks import Split, load_t1, load_t2
from .pretrain import _fmt_secs, _git_sha, _save_atomic, lr_factor

TASKS = ("T1", "T2")
# (stage name, train split, validation split). The last stage's validation
# split is the one selection uses (spec 6.2: T1 -> stage-ii finetune val).
STAGES = {"T1": [("stage-i", "ref_train", "ref_val"), ("stage-ii", "ft_train", "ft_val")],
          "T2": [("single", "train", "val")]}

DEFAULTS = {
    "a_ckpt": "foundation/runs/pretrain/20261003-123546_small_main/ckpt_best.pt",
    "data_dir": "data", "cache_dir": "foundation/data/cache",
    "run_root": "foundation/runs/adapt", "device": "cuda", "amp": True,
    "batch_size": 256, "weight_decay": 0.05, "betas": [0.9, 0.999],
    "lr_warmup_epochs": 1, "llrd": 0.75, "patience": 5,
    "head_warmup_epochs": 5, "head_warmup_lr_mult": 10.0,
    "max_epochs": {"stage-i": 30, "stage-ii": 50, "single": 50},
    "augment": "mild", "lora_dropout": 0.05,
    "lr_grid": {"S": [1.0e-4, 3.0e-4, 1.0e-3], "B1": [1.0e-4, 3.0e-4, 1.0e-3],
                "LP": [1.0e-3, 3.0e-3, 1.0e-2], "B2": [1.0e-3, 3.0e-3, 1.0e-2]},
    "sweep_seed": 1, "seeds": [1, 2, 3, 4, 5],
    "ft_per_class": None,
    "curve_per_class": [5, 10, 25, 50], "curve_arms": ["S", "B1", "B2"], "curve_seeds": [1, 2, 3],
}


def load_config(path=None, **overrides):
    cfg = json.loads(json.dumps(DEFAULTS))
    if path:
        cfg.update(yaml.safe_load(Path(path).read_text()) or {})
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def load_task(name, cfg):
    loader = {"T1": load_t1, "T2": load_t2}[name]
    task = loader(cfg["data_dir"], cfg["cache_dir"])
    if cfg.get("ft_per_class") and name == "T1":
        task.splits["ft_train"] = subsample_per_class(task.splits["ft_train"],
                                                      cfg["ft_per_class"])
    return task


def subsample_per_class(split: Split, n, seed=0):
    """First n spectra of each class after a fixed shuffle (optional data-efficiency
    curve, spec 8.5). Same rows for every arm and seed."""
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(split.y):
        idx = np.flatnonzero(split.y == c)
        rng.shuffle(idx)
        if idx.size < n:
            raise ValueError(f"class {c} has {idx.size} spectra, fewer than {n}")
        keep.append(idx[:n])
    keep = np.sort(np.concatenate(keep))
    return Split(split.X[keep], split.M[keep], split.y[keep])


def run_dir_for(cfg, task, arm, lr, seed):
    t = task if not cfg.get("ft_per_class") else f"{task}_pc{cfg['ft_per_class']}"
    return Path(cfg["run_root"]) / t / arm / f"lr{lr:.0e}_seed{seed}"


class GPUSplit:
    def __init__(self, split: Split, device):
        self.X = torch.as_tensor(split.X, dtype=torch.float32, device=device)
        self.M = torch.as_tensor(split.M, device=device)
        self.y = torch.as_tensor(split.y, dtype=torch.long, device=device)

    def __len__(self):
        return self.y.numel()


@torch.no_grad()
def predict(model, data: GPUSplit, device, batch=1024):
    """Logits for every row, in order (eval mode, no augmentation)."""
    model.eval()
    out = []
    for s in range(0, len(data), batch):
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out.append(model(data.X[s:s + batch], data.M[s:s + batch]).float())
    return torch.cat(out)


def val_metrics(model, data: GPUSplit, device):
    logits = predict(model, data, device)
    loss = F.cross_entropy(logits, data.y).item()
    pred = logits.argmax(dim=1)
    return loss, (pred == data.y).float().mean().item(), pred.cpu().numpy()


def train_phase(model, phase, train, val, lr, max_epochs, early_stop, tag, cfg, gen,
                device, log, metrics):
    """Train one phase. With early_stop, ends after `patience` epochs without a
    lower validation loss and restores the best epoch's trainable weights."""
    set_trainable(model, phase)
    groups = param_groups(model, lr, cfg["weight_decay"],
                          cfg["llrd"] if phase == "full" else None)
    opt = torch.optim.AdamW(groups, betas=tuple(cfg["betas"]))
    bs = cfg["batch_size"]
    steps = math.ceil(len(train) / bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(
        s, cfg["lr_warmup_epochs"] * steps, max_epochs * steps))
    use_amp = cfg["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    aug = CORRUPTIONS[cfg["augment"]]
    params = [p for p in model.parameters() if p.requires_grad]
    best = {"loss": math.inf, "epoch": 0, "state": None}
    bad, secs, reason = 0, [], None
    for epoch in range(1, max_epochs + 1):
        te = time.time()
        model.train()
        if phase == "head":
            model.encoder.eval()          # frozen features: no dropout
        perm = torch.randperm(len(train), generator=gen, device=device)
        total, seen = 0.0, 0
        for s in range(0, len(train), bs):
            idx = perm[s:s + bs]
            x, m = corrupt(train.X[idx], train.M[idx], aug, generator=gen)
            with torch.autocast(device.type, enabled=use_amp):
                loss = F.cross_entropy(model(x, m).float(), train.y[idx])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * idx.numel()
            seen += idx.numel()
        vloss, vacc, _ = val_metrics(model, val, device)
        secs.append(time.time() - te)
        improved = vloss < best["loss"] - 1e-4
        if improved:
            best = {"loss": vloss, "epoch": epoch,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
            bad = 0
        else:
            bad += 1
        rec = {"tag": tag, "phase": phase, "epoch": epoch, "max_epochs": max_epochs,
               "train_loss": total / max(1, seen), "val_loss": vloss, "val_acc": vacc,
               "lr": opt.param_groups[0]["lr"], "seconds": secs[-1]}
        metrics.write(json.dumps(rec) + "\n")
        metrics.flush()
        left = max_epochs - epoch
        log(f"[{tag}] epoch {epoch}/{max_epochs} ({left} left) | train loss "
            f"{rec['train_loss']:.4f} | val loss {vloss:.4f} | val acc {vacc:.4f} | best "
            f"{best['loss']:.4f} @ {best['epoch']} | ~{_fmt_secs(np.mean(secs) * left)} ETA")
        if early_stop and bad >= cfg["patience"]:
            reason = (f"stopped at epoch {epoch}/{max_epochs}: val loss flat for "
                      f"{cfg['patience']} epochs")
            break
    else:
        reason = f"finished {max_epochs}/{max_epochs}"
    if early_stop and best["state"] is not None:
        model.load_state_dict(best["state"])
        reason += f"; restored epoch {best['epoch']}"
    log(f"[{tag}] {reason}")
    return {"phase": phase, "epochs_run": epoch, "best_epoch": best["epoch"],
            "best_val_loss": best["loss"], "reason": reason, "seconds": float(sum(secs))}


def stage_label(stage, warmup):
    """'stage-i', 'stage-i-warmup', 'stage-ii'; T2's single stage has no label
    except 'warmup'. Used in progress tags: [B1 T1 stage-ii seed 3]."""
    parts = ([] if stage == "single" else [stage]) + (["warmup"] if warmup else [])
    return "-".join(parts)


def _keeps_stage1(cfg, arm, seed):
    """Full-data T1 runs that the optional curve will reuse save their state
    after stage i (stage i never sees finetune data, so it is identical)."""
    return not cfg.get("ft_per_class") and arm in cfg["curve_arms"] and seed in cfg["curve_seeds"]


def _stage1_source(cfg, task_name, arm, lr, seed):
    if task_name != "T1" or not cfg.get("ft_per_class"):
        return None
    path = run_dir_for(dict(cfg, ft_per_class=None), task_name, arm, lr, seed) / "stage1.pt"
    return path if path.exists() else None


def run_one(cfg, task_name, arm, lr, seed, task=None, a=None, log=print):
    """One arm x task x LR x seed. Returns the summary dict (also saved)."""
    out = run_dir_for(cfg, task_name, arm, lr, seed)
    if (out / "summary.json").exists():
        log(f"[skip] {out} already finished")
        return json.loads((out / "summary.json").read_text())
    device = torch.device(cfg["device"] if cfg["device"] != "cuda" or torch.cuda.is_available()
                          else "cpu")
    task = task or load_task(task_name, cfg)
    preset, a_state = a or load_a(cfg["a_ckpt"])
    model, rank = build_model(arm, len(task.classes), preset, a_state, seed)
    if arm == "B2" and cfg["lora_dropout"] != 0.05:
        raise ValueError("spec 6.1 fixes LoRA dropout at 0.05")
    model.to(device)
    gen = torch.Generator(device=device).manual_seed(seed)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({
        **cfg, "task": task_name, "arm": arm, "lr": lr, "seed": seed, "lora_rank": rank,
        "preset": preset, "git_sha": _git_sha(), "torch": torch.__version__,
        "python": platform.python_version(), "device": str(device)}, indent=2))
    t0 = time.time()
    stages = []
    reuse = _stage1_source(cfg, task_name, arm, lr, seed)
    with open(out / "metrics.jsonl", "w", encoding="utf-8") as metrics:
        for i, (stage, tr_name, va_name) in enumerate(STAGES[task_name]):
            if stage == "stage-i" and reuse is not None:
                ck = torch.load(reuse, map_location="cpu", weights_only=True)
                model.load_state_dict(ck["model"])
                gen.set_state(ck["gen"].to(gen.get_state().device))
                stages += [dict(s, reused_from=str(reuse)) for s in ck["stages"]]
                log(f"[{arm} {task_name} stage-i seed {seed}] reused from {reuse}")
                continue
            train = GPUSplit(task.splits[tr_name], device)
            val = GPUSplit(task.splits[va_name], device)
            todo = phases(arm) if i == 0 else phases(arm)[-1:]
            for phase, warmup in todo:
                where = stage_label(stage, warmup)
                tag = " ".join(filter(None, (arm, task_name, where, f"seed {seed}")))
                if warmup:
                    info = train_phase(model, phase, train, val, lr * cfg["head_warmup_lr_mult"],
                                       cfg["head_warmup_epochs"], False, tag, cfg, gen, device,
                                       log, metrics)
                else:
                    info = train_phase(model, phase, train, val, lr, cfg["max_epochs"][stage],
                                       True, tag, cfg, gen, device, log, metrics)
                stages.append({"stage": where or stage, **info})
            del train, val
            if stage == "stage-i" and _keeps_stage1(cfg, arm, seed):
                _save_atomic({"model": model.state_dict(), "gen": gen.get_state(),
                              "stages": stages}, out / "stage1.pt")
    vloss, vacc, vpred = val_metrics(model, GPUSplit(task.splits[STAGES[task_name][-1][2]],
                                                       device), device)
    vy = task.splits[STAGES[task_name][-1][2]].y
    np.save(out / "val_pred.npy", vpred)
    _save_atomic(addon_state(model), out / "addon.pt")
    summary = {
        "task": task_name, "arm": arm, "lr": lr, "seed": seed, "lora_rank": rank,
        "ft_per_class": cfg.get("ft_per_class"),
        "val_acc": vacc, "val_loss": vloss,
        "val_macro_f1": float(f1_score(vy, vpred, average="macro")),
        "trainable_params": trainable_count(model), "total_params": total_count(model),
        "trainable_fraction": trainable_count(model) / total_count(model),
        "addon_bytes": os.path.getsize(out / "addon.pt"),
        "seconds": time.time() - t0, "stages": stages,
    }
    tmp = out / "summary.json.tmp"
    tmp.write_text(json.dumps(summary, indent=2))
    os.replace(tmp, out / "summary.json")
    log(f"[{arm} {task_name} seed {seed}] done | val acc {vacc:.4f} | val loss {vloss:.4f} | "
        f"trainable {summary['trainable_params']:,} ({100 * summary['trainable_fraction']:.2f}%)"
        f" | {_fmt_secs(summary['seconds'])}")
    return summary


def select_lr(summaries):
    """Best validation accuracy; ties broken by lower validation loss."""
    return min(summaries, key=lambda s: (-round(s["val_acc"], 6), s["val_loss"]))["lr"]


def selection_path(cfg, task):
    """Selections are made on full data only; curve runs reuse T1's."""
    return Path(cfg["run_root"]) / task / "selection.json"


def _queue(cfg, jobs, log):
    """Run (task, arm, lr, seed) jobs in order, loading each task and A once."""
    a = load_a(cfg["a_ckpt"])
    tasks, results = {}, []
    for n, (t, arm, lr, seed) in enumerate(jobs, 1):
        log(f"[queue] run {n}/{len(jobs)}: {arm} {t} lr {lr:.0e} seed {seed}")
        if t not in tasks:
            tasks[t] = load_task(t, cfg)
        results.append(run_one(cfg, t, arm, lr, seed, task=tasks[t], a=a, log=log))
    return results


def sweep(cfg, tasks=TASKS, arms=ARMS, log=print):
    s = cfg["sweep_seed"]
    jobs = [(t, arm, lr, s) for t in tasks for arm in arms for lr in cfg["lr_grid"][arm]]
    results = _queue(cfg, jobs, log)
    for t in tasks:
        path = selection_path(cfg, t)
        sel = json.loads(path.read_text()) if path.exists() else {}
        for arm in arms:
            rs = [r for r in results if r["task"] == t and r["arm"] == arm]
            sel[arm] = {"lr": select_lr(rs), "grid": [
                {"lr": r["lr"], "val_acc": r["val_acc"], "val_loss": r["val_loss"]} for r in rs]}
            log(f"[sweep] {t} {arm}: selected lr {sel[arm]['lr']:.0e}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sel, indent=2))
    return results


def seeds(cfg, tasks=TASKS, arms=ARMS, log=print):
    jobs = [(t, arm, _selected(cfg, t, arm), s) for t in tasks for arm in arms
            for s in cfg["seeds"]]
    return _queue(cfg, jobs, log)


def _selected(cfg, task, arm):
    path = selection_path(cfg, task)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `python -m foundation.adapt sweep` first")
    sel = json.loads(path.read_text())
    if arm not in sel:
        raise KeyError(f"{path} has no selected LR for arm {arm}")
    return sel[arm]["lr"]


def curve(cfg, log=print):
    """Optional data-efficiency curve (spec 8.5): T1 with stage ii cut to n
    spectra per class, at each arm's full-data LR. The full-data point (90 per
    class) is the seeds run itself."""
    results = []
    for n in cfg["curve_per_class"]:
        c = dict(cfg, ft_per_class=n)
        jobs = [("T1", arm, _selected(cfg, "T1", arm), s)
                for arm in cfg["curve_arms"] for s in cfg["curve_seeds"]]
        log(f"[curve] {n} spectra per class")
        results += _queue(c, jobs, log)
    return results


def status(cfg, log=print):
    """How many sweep and seed runs are finished, per task."""
    for t in TASKS:
        sweep_runs = [(arm, lr, cfg["sweep_seed"]) for arm in ARMS for lr in cfg["lr_grid"][arm]]
        done = sum((run_dir_for(cfg, t, *r) / "summary.json").exists() for r in sweep_runs)
        line = f"[status] {t}: sweep {done}/{len(sweep_runs)}"
        path = selection_path(cfg, t)
        if path.exists():
            sel = json.loads(path.read_text())
            final = [(arm, sel[arm]["lr"], s) for arm in ARMS if arm in sel for s in cfg["seeds"]]
            fdone = sum((run_dir_for(cfg, t, *r) / "summary.json").exists() for r in final)
            line += f" | final seeds {fdone}/{len(ARMS) * len(cfg['seeds'])}"
        log(line)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("command", choices=["run", "sweep", "seeds", "curve", "status"])
    ap.add_argument("--config", default="foundation/configs/adapt.yaml")
    ap.add_argument("--task", nargs="+", choices=TASKS, default=list(TASKS))
    ap.add_argument("--arm", nargs="+", choices=ARMS, default=list(ARMS))
    ap.add_argument("--lr", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--ft-per-class", type=int, help="T1 stage-ii spectra per class (spec 8.5)")
    ap.add_argument("--device")
    args = ap.parse_args(argv)
    cfg = load_config(args.config if Path(args.config).exists() else None,
                      device=args.device, ft_per_class=args.ft_per_class)
    if args.command == "run":
        if args.lr is None or args.seed is None or len(args.task) != 1 or len(args.arm) != 1:
            ap.error("run needs exactly one --task, one --arm, --lr and --seed")
        run_one(cfg, args.task[0], args.arm[0], args.lr, args.seed)
    elif args.command == "sweep":
        sweep(cfg, args.task, args.arm)
    elif args.command == "seeds":
        seeds(cfg, args.task, args.arm)
    elif args.command == "curve":
        curve(cfg)
    else:
        status(cfg)


if __name__ == "__main__":
    main()
```

<!-- file: foundation/configs/adapt.yaml -->
```yaml
# Adaptation of model A to our tasks (spec section 6). Read by
# python -m foundation.adapt and python -m foundation.evaluate.
a_ckpt: foundation/runs/pretrain/20261003-123546_small_main/ckpt_best.pt
data_dir: data
cache_dir: foundation/data/cache
run_root: foundation/runs/adapt
device: cuda
amp: true
batch_size: 256
weight_decay: 0.05
betas: [0.9, 0.999]
lr_warmup_epochs: 1          # linear LR warm-up inside each stage, then cosine to max_epochs
llrd: 0.75                   # layer-wise LR decay for full fine-tuning (S, B1)
patience: 5                  # early stopping on validation loss
head_warmup_epochs: 5        # B1: head only, once, at the start of the first stage
head_warmup_lr_mult: 10.0    # B1 warm-up LR = 10 x the arm's LR
max_epochs: {stage-i: 30, stage-ii: 50, single: 50}
augment: mild                # shift +-2 cm^-1, scale 0.9-1.1, SNR 30-100, no masking
lora_dropout: 0.05
lr_grid:                     # 3 values per arm (spec 6.2); B2 is 10x B1 (spec 6.1)
  S: [1.0e-4, 3.0e-4, 1.0e-3]
  B1: [1.0e-4, 3.0e-4, 1.0e-3]
  LP: [1.0e-3, 3.0e-3, 1.0e-2]
  B2: [1.0e-3, 3.0e-3, 1.0e-2]
sweep_seed: 1
seeds: [1, 2, 3, 4, 5]
curve_per_class: [5, 10, 25, 50]     # optional curve (spec 8.5); "all" = the 90-per-class runs
curve_arms: [S, B1, B2]
curve_seeds: [1, 2, 3]
```

- [ ] **Step 5: Run the tests to verify they pass, and check the config mirrors the defaults**

Run: `venv/Scripts/python.exe -m pytest tests/foundation/test_adapt.py -q`
Expected: `12 passed`.

Run: `venv/Scripts/python.exe -c "from foundation.adapt import load_config, DEFAULTS; c = load_config('foundation/configs/adapt.yaml'); assert c == DEFAULTS, [k for k in c if c[k] != DEFAULTS.get(k)]; print('ok')"`
Expected: `ok`.

- [ ] **Step 6: Commit**

```bash
git add foundation/adapt.py foundation/configs/adapt.yaml tests/foundation/test_adapt.py tests/foundation/conftest.py
git commit -m "foundation: adaptation trainer (S/LP/B1/B2 on T1/T2), LR sweep, seeds and curve queues

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: The single final test pass, the report, and the README

**Files:**
- Create: `foundation/evaluate.py`, `tests/foundation/test_evaluate.py`
- Modify: `foundation/README.md` (append only)

**Interfaces:**
- Consumes:
  - **Task 1:** `ARMS`, `build_model`, `load_a`, `load_addon`.
  - **Task 2:** `bootstrap_diff`, `mcnemar`, `mean_sd`, `q1`, `q3`.
  - **Task 3:** `TASKS`, `GPUSplit`, `load_config`, `load_task`, `predict`, `run_dir_for`, `selection_path`, and the `adapt_cfg` fixture.
- Produces:
  - **Functions:**
    - `planned_runs(cfg)`;
    - `curve_runs(cfg)`, both giving `[(key, task, arm, lr, seed, run_dir)]`;
    - `unfinished(runs)`;
    - `score(cfg, runs, out)`;
    - `collect(cfg, runs, out)`;
    - `analyse(res)`, which raises `ValueError("... not paired ...")`;
    - `render(...)`;
    - `plot_confusions`;
    - `curve_results`;
    - `curve_table`.
  - **CLI:** `python -m foundation.evaluate [--skip-curve | --report-only] [--results DIR]`.
  - **Outputs:**
    - `runs/adapt/final_test/{scored.json, report.json, <task>/y_test.npy, <task>/<arm>_seed<s>.npy, T1_pc<n>/...}`;
    - `foundation/results/adaptation.md`;
    - `foundation/results/adaptation/confusion_<task>.png`.

- [ ] **Step 1: Write the failing tests**

<!-- file: tests/foundation/test_evaluate.py -->
```python
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from foundation import evaluate
from foundation.adapt import seeds, sweep
from foundation.arms import ARMS


def _write_cfg(cfg, tmp_path):
    p = tmp_path / "adapt.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return str(p)


@pytest.fixture
def finished(adapt_cfg, tmp_path):
    """Every planned run (4 arms x 2 tasks x 2 seeds) finished on fake data."""
    sweep(adapt_cfg, log=lambda *_: None)
    seeds(adapt_cfg, log=lambda *_: None)
    return adapt_cfg, _write_cfg(adapt_cfg, tmp_path)


def test_refuses_before_runs_finish(adapt_cfg, tmp_path):
    with pytest.raises(FileNotFoundError, match="sweep"):
        evaluate.main(["--config", _write_cfg(adapt_cfg, tmp_path), "--skip-curve"])
    sweep(adapt_cfg, tasks=["T1", "T2"], arms=["LP"], log=lambda *_: None)
    sel = Path(adapt_cfg["run_root"]) / "T1" / "selection.json"
    s = json.loads(sel.read_text())
    s.update({a: {"lr": 1e-3, "grid": []} for a in ARMS if a != "LP"})
    sel.write_text(json.dumps(s))
    t2 = Path(adapt_cfg["run_root"]) / "T2" / "selection.json"
    t2.write_text(sel.read_text())
    with pytest.raises(SystemExit, match="not finished"):
        evaluate.main(["--config", _write_cfg(adapt_cfg, tmp_path), "--skip-curve"])


def test_curve_must_be_finished_or_explicitly_skipped(finished):
    cfg, path = finished
    with pytest.raises(SystemExit, match="--skip-curve"):
        evaluate.main(["--config", path])


def test_single_test_pass_and_report(finished, tmp_path):
    cfg, path = finished
    results = tmp_path / "results"
    evaluate.main(["--config", path, "--skip-curve", "--results", str(results)])
    out = Path(cfg["run_root"]) / "final_test"
    scored = json.loads((out / "scored.json").read_text())
    assert len(scored["runs"]) == 16 and scored["curve"] is False
    assert np.load(out / "T1" / "B2_seed2.npy").shape == (30,)
    assert np.load(out / "T2" / "y_test.npy").shape == (20,)
    text = (results / "adaptation.md").read_text(encoding="utf-8")
    for s in ("## T1", "## T2", "B1 vs S", "B2 vs S", "B1 vs B2", "**Q1**", "**Q2**",
              "Q3 overall", "Reuse deliverable", "Known limitation", "Per-class recall"):
        assert s in text, s
    assert (results / "adaptation" / "confusion_T1.png").exists()
    rep = json.loads((out / "report.json").read_text())
    assert set(rep["tasks"]["T2"]) == set(ARMS)
    assert len(rep["diffs"]["T1"]["B1 vs S"]["mcnemar"]) == 2       # one test per paired seed
    with pytest.raises(SystemExit, match="already scored"):
        evaluate.main(["--config", path, "--skip-curve", "--results", str(results)])
    (results / "adaptation.md").unlink()
    evaluate.main(["--config", path, "--report-only", "--results", str(results)])
    assert (results / "adaptation.md").read_text(encoding="utf-8") == text


def test_report_only_needs_a_scored_pass(finished):
    _, path = finished
    with pytest.raises(SystemExit, match="not been scored"):
        evaluate.main(["--config", path, "--report-only"])


def test_scored_predictions_match_the_trained_models(finished):
    """The add-on reloaded for testing must reproduce the run's own validation predictions."""
    import torch

    from foundation.adapt import GPUSplit, load_task, predict, run_dir_for
    from foundation.arms import build_model, load_a, load_addon
    cfg, _ = finished
    preset, a_state = load_a(cfg["a_ckpt"])
    task = load_task("T2", cfg)
    sel = json.loads((Path(cfg["run_root"]) / "T2" / "selection.json").read_text())
    for arm in ARMS:
        rdir = run_dir_for(cfg, "T2", arm, sel[arm]["lr"], 2)
        s = json.loads((rdir / "summary.json").read_text())
        model, _ = build_model(arm, 5, preset, a_state, 2, lora_rank=s["lora_rank"])
        load_addon(model, torch.load(rdir / "addon.pt", weights_only=True))
        pred = predict(model, GPUSplit(task.splits["val"], torch.device("cpu")),
                       torch.device("cpu")).argmax(1).numpy()
        assert np.array_equal(pred, np.load(rdir / "val_pred.npy")), arm


def test_unpaired_seeds_raise():
    c = np.ones(4, bool)
    res = {"T1": {a: {"lr": 1e-3, "seeds": [1, 2] if a != "S" else [1, 3], "correct": [c, c],
                      "acc": [1.0, 1.0], "f1": [1.0, 1.0], "cm": np.eye(2, dtype=int),
                      "summ": [{"val_acc": 1, "trainable_params": 10, "trainable_fraction": 0.01,
                                "addon_bytes": 100, "seconds": 1.0}] * 2} for a in ARMS}}
    with pytest.raises(ValueError, match="not paired"):
        evaluate.analyse(res)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `venv/Scripts/python.exe -m pytest tests/foundation/test_evaluate.py -q`
Expected: `ImportError: cannot import name 'evaluate' from 'foundation'`.

- [ ] **Step 3: Implement**

<!-- file: foundation/evaluate.py -->
```python
"""Single final test pass and report (spec section 7 phase 4, section 8).

    python -m foundation.evaluate                 # needs every planned run finished
    python -m foundation.evaluate --skip-curve    # proceed without the optional curve
    python -m foundation.evaluate --report-only   # rebuild the report from saved predictions

Scores T1's test set and T2's 2019clinical set ONCE with every finished,
frozen run (4 arms x 5 seeds per task at the selected LR, plus the optional
curve runs), saves per-sample predictions under <run_root>/final_test/ and
writes scored.json. A second scoring pass is refused; --report-only only
re-reads the saved predictions.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score  # noqa: E402

from .adapt import (TASKS, GPUSplit, load_config, load_task, predict,  # noqa: E402
                    run_dir_for, selection_path)
from .arms import ARMS, build_model, load_a, load_addon  # noqa: E402
from .pretrain import _git_sha  # noqa: E402
from .stats import bootstrap_diff, mcnemar, mean_sd, q1, q3  # noqa: E402

PAIRS = (("B1", "S"), ("B2", "S"), ("B1", "B2"))
FINAL = "final_test"


def planned_runs(cfg):
    """[(key, task, arm, lr, seed, run_dir)] for the main comparison."""
    runs = []
    for t in TASKS:
        path = selection_path(cfg, t)
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run the LR sweep first")
        sel = json.loads(path.read_text())
        for arm in ARMS:
            for s in cfg["seeds"]:
                lr = sel[arm]["lr"]
                runs.append((f"{t}/{arm}_seed{s}", t, arm, lr, s, run_dir_for(cfg, t, arm, lr, s)))
    return runs


def curve_runs(cfg):
    sel = json.loads(selection_path(cfg, "T1").read_text())
    runs = []
    for n in cfg["curve_per_class"]:
        c = dict(cfg, ft_per_class=n)
        for arm in cfg["curve_arms"]:
            for s in cfg["curve_seeds"]:
                lr = sel[arm]["lr"]
                runs.append((f"T1_pc{n}/{arm}_seed{s}", "T1", arm, lr, s,
                             run_dir_for(c, "T1", arm, lr, s)))
    return runs


def unfinished(runs):
    return [r[0] for r in runs if not (r[5] / "summary.json").exists()]


def score(cfg, runs, out, log=print):
    """Predict each task's test set once per run; save predictions and labels."""
    device = torch.device(cfg["device"] if cfg["device"] != "cuda" or torch.cuda.is_available()
                          else "cpu")
    preset, a_state = load_a(cfg["a_ckpt"])
    tasks = {t: load_task(t, cfg) for t in sorted({r[1] for r in runs})}
    for t, task in tasks.items():
        (out / t).mkdir(parents=True, exist_ok=True)
        np.save(out / t / "y_test.npy", task.splits["test"].y)
    test = {t: GPUSplit(task.splits["test"], device) for t, task in tasks.items()}
    for n, (key, t, arm, lr, seed, rdir) in enumerate(runs, 1):
        summary = json.loads((rdir / "summary.json").read_text())
        model, _ = build_model(arm, len(tasks[t].classes), preset, a_state, seed,
                               lora_rank=summary["lora_rank"])
        load_addon(model, torch.load(rdir / "addon.pt", map_location="cpu", weights_only=True))
        model.to(device)
        pred = predict(model, test[t], device).argmax(dim=1).cpu().numpy()
        (out / key).parent.mkdir(parents=True, exist_ok=True)
        np.save(out / f"{key}.npy", pred)
        log(f"[test] {n}/{len(runs)} {key}: acc {(pred == tasks[t].splits['test'].y).mean():.4f}")


def _load(out, key):
    return np.load(out / f"{key}.npy")


def collect(cfg, runs, out):
    """Per task and arm: seed-ordered predictions, summaries and metrics."""
    res = {}
    for key, t, arm, lr, seed, rdir in runs:
        y = np.load(out / t / "y_test.npy")
        pred = _load(out, key)
        s = json.loads((rdir / "summary.json").read_text())
        r = res.setdefault(t, {}).setdefault(arm, {"lr": lr, "seeds": [], "correct": [],
                                                  "acc": [], "f1": [], "cm": 0, "summ": []})
        r["seeds"].append(seed)
        r["correct"].append(pred == y)
        r["acc"].append(float((pred == y).mean()))
        r["f1"].append(float(f1_score(y, pred, average="macro")))
        r["cm"] = r["cm"] + confusion_matrix(y, pred, labels=np.arange(int(y.max()) + 1))
        r["summ"].append(s)
    return res


def analyse(res):
    """Numbers for every table in the report (JSON-serialisable)."""
    out = {"tasks": {}, "diffs": {}}
    for t, arms in res.items():
        rows = {}
        for arm, r in arms.items():
            rows[arm] = {
                "lr": r["lr"], "seeds": r["seeds"],
                "acc": mean_sd(r["acc"]), "f1": mean_sd(r["f1"]),
                "val_acc": mean_sd([s["val_acc"] for s in r["summ"]]),
                "trainable": r["summ"][0]["trainable_params"],
                "fraction": r["summ"][0]["trainable_fraction"],
                "addon_bytes": r["summ"][0]["addon_bytes"],
                "seconds": mean_sd([s["seconds"] for s in r["summ"]]),
                "recall": (np.diag(r["cm"]) / r["cm"].sum(axis=1).clip(min=1)).tolist(),
                "confusion": r["cm"].tolist(),
            }
        out["tasks"][t] = rows
        diffs = {}
        for a, b in PAIRS:
            if a in arms and b in arms:
                if arms[a]["seeds"] != arms[b]["seeds"]:
                    raise ValueError(f"{t}: {a} and {b} are not paired by seed")
                ca, cb = np.stack(arms[a]["correct"]), np.stack(arms[b]["correct"])
                diffs[f"{a} vs {b}"] = {**bootstrap_diff(ca, cb),
                                        "mcnemar": [mcnemar(x, y) for x, y in zip(ca, cb)]}
        out["diffs"][t] = diffs
    out["q1"] = q1({t: {tuple(k.split(" vs ")): v for k, v in d.items()}
                    for t, d in out["diffs"].items()})
    out["q2"] = {t: {**out["diffs"][t]["B1 vs B2"],
                     "trainable_ratio": rows["B1"]["trainable"] / rows["B2"]["trainable"]}
                 for t, rows in out["tasks"].items()}
    out["q3"] = q3({t: {a: r["acc"][0] for a, r in rows.items()} for t, rows in out["tasks"].items()},
                   {t: {a: r["fraction"] for a, r in rows.items()} for t, rows in out["tasks"].items()})
    return out


def _pct(v, d=2):
    return f"{100 * v:.{d}f}"


def _size(b):
    return f"{b / 1e6:.2f} MB" if b >= 1e6 else f"{b / 1e3:.1f} kB"


def render(a, classes, a_ckpt, curve=None, figs=None):
    """Markdown report from analyse() output."""
    L = [f"# Foundation experiment: adaptation results", "",
         f"Model A: `{a_ckpt}` ({_size(os.path.getsize(a_ckpt))}). Test sets scored once "
         "(spec section 7, phase 4). Accuracies in %; mean +- sd over seeds "
         f"{a['tasks'][TASKS[0]]['S']['seeds']}.", "",
         "**Known limitation (spec 8.4):** one pretraining run of A; variation between "
         "pretraining seeds is not measured.", ""]
    for t, rows in a["tasks"].items():
        L += [f"## {t}", "", "| arm | LR | val acc | test acc | macro-F1 | trainable | add-on | "
              "time per run |", "|---|---|---|---|---|---|---|---|"]
        for arm in ARMS:
            r = rows[arm]
            L.append(f"| {arm} | {r['lr']:.0e} | {_pct(r['val_acc'][0])} | "
                     f"{_pct(r['acc'][0])} +- {_pct(r['acc'][1])} | "
                     f"{_pct(r['f1'][0])} +- {_pct(r['f1'][1])} | "
                     f"{r['trainable']:,} ({_pct(r['fraction'])}%) | {_size(r['addon_bytes'])} | "
                     f"{r['seconds'][0] / 60:.1f} min |")
        L += ["", "| comparison | diff (points) | 95% bootstrap CI | McNemar p per seed |",
              "|---|---|---|---|"]
        for k, d in a["diffs"][t].items():
            ps = ", ".join(f"{m['p']:.3g}" for m in d["mcnemar"])
            L.append(f"| {k} | {_pct(d['diff'])} | [{_pct(d['lo'])}, {_pct(d['hi'])}] | {ps} |")
        names = [str(int(c)) if float(c).is_integer() else str(c) for c in classes[t]]
        L += ["", "Per-class recall (%, pooled over seeds):", "",
              "| class | " + " | ".join(ARMS) + " |", "|---|" + "---|" * len(ARMS)]
        for i, n in enumerate(names):
            L.append(f"| {n} | " + " | ".join(_pct(rows[arm]["recall"][i], 1) for arm in ARMS) + " |")
        if figs and t in figs:
            L += ["", f"![confusion matrices {t}]({figs[t]})"]
        L.append("")
    L += ["## Pre-registered questions (spec 8.3)", ""]
    for t, qs in a["q1"].items():
        for k, v in qs.items():
            L.append(f"- **Q1** {t} {k}: {_pct(v['diff'])} points, CI [{_pct(v['lo'])}, "
                     f"{_pct(v['hi'])}] -> {v['verdict']} ({'success' if v['success'] else 'not met'})")
    for t, v in a["q2"].items():
        L.append(f"- **Q2** {t} B1 vs B2: {_pct(v['diff'])} points, CI [{_pct(v['lo'])}, "
                 f"{_pct(v['hi'])}]; B1 trains {v['trainable_ratio']:.1f}x B2's parameters")
    for c, v in a["q3"]["candidates"].items():
        det = "; ".join(f"{t}: {_pct(x['gap'])} points below best, {_pct(x['fraction'])}% trainable"
                        for t, x in v["tasks"].items())
        L.append(f"- **Q3** {c}: {'PASS' if v['pass'] else 'FAIL'} ({det})")
    L += [f"- **Q3 overall: {'PASS' if a['q3']['pass'] else 'FAIL'}**", "",
          "## Reuse deliverable (spec 6.3)", "",
          f"One A file ({_size(os.path.getsize(a_ckpt))}) serves both tasks; each task stores only "
          "its add-on.", "", "| task | arm | add-on | trainable | test acc |", "|---|---|---|---|---|"]
    for t, rows in a["tasks"].items():
        for arm in ("LP", "B2"):
            r = rows[arm]
            L.append(f"| {t} | {arm} | {_size(r['addon_bytes'])} | {r['trainable']:,} | "
                     f"{_pct(r['acc'][0])} |")
    if curve:
        L += ["", "## Data-efficiency curve (spec 8.5, T1 stage ii)", "",
              "| spectra per class | " + " | ".join(curve["arms"]) + " |",
              "|---|" + "---|" * len(curve["arms"])]
        for n, row in curve["rows"].items():
            L.append(f"| {n} | " + " | ".join(f"{_pct(m)} +- {_pct(s)}" for m, s in row) + " |")
    return "\n".join(L) + "\n"


def plot_confusions(a, t, path):
    rows = a["tasks"][t]
    fig, axes = plt.subplots(1, len(ARMS), figsize=(5 * len(ARMS), 4.6))
    for ax, arm in zip(axes, ARMS):
        cm = np.asarray(rows[arm]["confusion"], float)
        ax.imshow(cm / cm.sum(axis=1, keepdims=True).clip(min=1), vmin=0, vmax=1, cmap="Blues")
        ax.set(title=f"{t} {arm} ({_pct(rows[arm]['acc'][0])}%)", xlabel="predicted",
               ylabel="true")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def curve_results(cfg, crun, out):
    """{per-class size: collect() result for T1} over the curve runs."""
    return {n: collect(cfg, [r for r in crun if r[0].startswith(f"T1_pc{n}/")], out)["T1"]
            for n in cfg["curve_per_class"]}


def curve_table(cfg, by_size, res_main):
    """Mean +- sd accuracy per size and curve arm; the last row is the full
    stage-ii set (the main runs, restricted to the curve seeds)."""
    rows = {n: [mean_sd(by_size[n][arm]["acc"]) for arm in cfg["curve_arms"]]
            for n in cfg["curve_per_class"]}
    main_t1 = res_main["T1"]
    rows["all"] = [mean_sd([acc for s, acc in zip(main_t1[arm]["seeds"], main_t1[arm]["acc"])
                                 if s in cfg["curve_seeds"]]) for arm in cfg["curve_arms"]]
    return {"arms": cfg["curve_arms"], "rows": rows}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="foundation/configs/adapt.yaml")
    ap.add_argument("--skip-curve", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--results", default="foundation/results")
    ap.add_argument("--device")
    args = ap.parse_args(argv)
    cfg = load_config(args.config if Path(args.config).exists() else None, device=args.device)
    out = Path(cfg["run_root"]) / FINAL
    runs = planned_runs(cfg)
    missing = unfinished(runs)
    if missing:
        raise SystemExit(f"{len(missing)} planned runs are not finished, e.g. {missing[:3]}; "
                         "run `python -m foundation.adapt sweep` and `... seeds` first")
    crun = curve_runs(cfg)
    have_curve = not unfinished(crun)
    if not have_curve and not args.skip_curve and not args.report_only:
        done = len(crun) - len(unfinished(crun))
        raise SystemExit(f"the optional curve has {done}/{len(crun)} runs finished. The test sets "
                         "are scored only once, so finish it (`python -m foundation.adapt curve`) "
                         "or pass --skip-curve to go without it")
    lock = out / "scored.json"
    if args.report_only:
        if not lock.exists():
            raise SystemExit(f"{lock} missing: the test sets have not been scored yet")
        scored = json.loads(lock.read_text())
        have_curve = scored["curve"]
    else:
        if lock.exists():
            raise SystemExit(f"the test sets were already scored ({lock}); use --report-only")
        score(cfg, runs + (crun if have_curve else []), out)
        lock.write_text(json.dumps({"date": datetime.now().isoformat(timespec="seconds"),
                                    "git_sha": _git_sha(), "curve": have_curve,
                                    "runs": [r[0] for r in runs + (crun if have_curve else [])]},
                                   indent=2))
    res = collect(cfg, runs, out)
    a = analyse(res)
    curve = curve_table(cfg, curve_results(cfg, crun, out), res) if have_curve else None
    results = Path(args.results)
    figdir = results / "adaptation"
    figdir.mkdir(parents=True, exist_ok=True)
    figs = {}
    for t in a["tasks"]:
        plot_confusions(a, t, figdir / f"confusion_{t}.png")
        figs[t] = f"adaptation/confusion_{t}.png"
    classes = {t: load_task(t, cfg).classes for t in a["tasks"]}
    text = render(a, classes, cfg["a_ckpt"], curve, figs)
    (results / "adaptation.md").write_text(text, encoding="utf-8")
    (out / "report.json").write_text(json.dumps({**a, "curve": curve}, indent=2, default=str))
    print(text)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Append the adaptation section to `foundation/README.md`**

<!-- file: foundation/README.md (append) -->
````markdown

## Phases 3-4: adapt A (arms S / LP / B1 / B2) and score the test sets once

All settings live in `foundation/configs/adapt.yaml`. Every run ends at a stage's
`max_epochs` or when validation loss is flat for 5 epochs, never on time; finished runs
are skipped, so any command can simply be re-run after an interruption.
```
venv/Scripts/python.exe -m foundation.adapt sweep     # 3 LRs per arm on seed 1 -> <task>/selection.json
venv/Scripts/python.exe -m foundation.adapt seeds     # seeds 2-5 at the selected LR
venv/Scripts/python.exe -m foundation.adapt curve     # optional data-efficiency curve (T1 stage ii)
venv/Scripts/python.exe -m foundation.adapt status    # what is finished
```
One run by hand: `venv/Scripts/python.exe -m foundation.adapt run --task T1 --arm B2 --lr 3e-3 --seed 1`.
Each run folder (`foundation/runs/adapt/<task>/<arm>/lr<lr>_seed<s>/`) holds `config.json`
(with git SHA and versions), `metrics.jsonl`, `val_pred.npy`, `addon.pt` (the trainable
parameters: head for LP; LoRA + norms + head for B2; everything for S and B1) and
`summary.json`.

Final evaluation scores T1 test and T2 2019clinical **once** and writes
`foundation/results/adaptation.md`; a second scoring pass is refused:
```
venv/Scripts/python.exe -m foundation.evaluate                # or --skip-curve without the curve
venv/Scripts/python.exe -m foundation.evaluate --report-only  # rebuild the report from saved predictions
```
````

- [ ] **Step 5: Run the new tests, then the whole suite**

Run: `venv/Scripts/python.exe -m pytest tests/foundation/test_evaluate.py -q`
Expected: `6 passed` (about 1 min; each test trains 16 tiny CPU runs).

Run: `venv/Scripts/python.exe -m pytest tests/foundation -q`
Expected: `245 passed, 1 skipped`.

- [ ] **Step 6: Commit**

```bash
git add foundation/evaluate.py tests/foundation/test_evaluate.py foundation/README.md
git commit -m "foundation: single locked test pass, paired statistics and the adaptation report

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5 (controller, GPU): LR sweep — phase 3, part 1

This task runs on the GPU; it is not a subagent task. Run it in the background and rely on the completion notification rather than a polling monitor.

- [ ] **Step 1: Start the sweep (24 runs: 4 arms × 3 LRs × 2 tasks, seed 1; about 3 h)**

Run (in the background): `venv/Scripts/python.exe -m foundation.adapt sweep > foundation/runs/adapt_sweep.log 2>&1`

Expected: progress lines such as `[queue] run 1/24: S T1 lr 1e-04 seed 1` and `[S T1 stage-i seed 1] epoch 1/30 (29 left) | ...`. At the end it prints `[sweep] T1 S: selected lr ...` for each of the 8 task × arm pairs.

- [ ] **Step 2: Check the selection**

Run: `venv/Scripts/python.exe -m foundation.adapt status`
Expected: `[status] T1: sweep 12/12 | final seeds 4/20` and the same for T2. Seed 1 of each arm comes from the sweep.

Read `foundation/runs/adapt/T1/selection.json` and `T2/selection.json`. Note every arm whose selected LR is at the **edge** of its grid. The spec fixes the budget at 3 values per arm, so the grid is **not** extended (that would give one arm more tuning than the others). Each edge selection is stated in the final report's notes (Task 8).

---

### Task 6 (controller, GPU): Seeds 2–5 — phase 3, part 2

- [ ] **Step 1: Train the remaining seeds (32 runs, about 2 h)**

Run (in the background): `venv/Scripts/python.exe -m foundation.adapt seeds > foundation/runs/adapt_seeds.log 2>&1`

- [ ] **Step 2: Confirm completeness**

Run: `venv/Scripts/python.exe -m foundation.adapt status`
Expected: `[status] T1: sweep 12/12 | final seeds 20/20` and `[status] T2: sweep 12/12 | final seeds 20/20`.

---

### Task 7 (controller, GPU): Optional data-efficiency curve (spec §8.5)

The spec runs the curve "if budget remains". With stage i reused (D11), it costs about 40 min, so run it. If the user declines it, skip this task and pass `--skip-curve` in Task 8.

- [ ] **Step 1: Run the curve (36 runs: 4 sizes × S/B1/B2 × seeds 1–3)**

Run (in the background): `venv/Scripts/python.exe -m foundation.adapt curve > foundation/runs/adapt_curve.log 2>&1`

Expected: each run logs `[<arm> T1 stage-i seed <s>] reused from foundation\runs\adapt\T1\<arm>\...\stage1.pt` and then trains stage ii only.

---

### Task 8 (controller): The single final evaluation — phase 4

- [ ] **Step 1: Score the test sets once and write the report**

Run: `venv/Scripts/python.exe -m foundation.evaluate` (or add `--skip-curve` if Task 7 was skipped).

Expected:
- 76 `[test] n/76 ...` lines (40 main runs + 36 curve runs);
- then the report printed;
- `foundation/results/adaptation.md` and `foundation/results/adaptation/confusion_T1.png` / `confusion_T2.png` written;
- `foundation/runs/adapt/final_test/scored.json` created.

**Do not run the scoring pass again.** Use `--report-only` to re-render.

- [ ] **Step 2: Add the run notes to the report**

Append a `## Notes` section to `foundation/results/adaptation.md`. List:
- the selected LR per arm and task, marking any edge selections from Task 5;
- D6 (S also uses LLRD);
- the T2 validation ceiling (D2).

Do not change any number.

- [ ] **Step 3: Commit the results**

```bash
git add foundation/results/adaptation.md foundation/results/adaptation/
git commit -m "foundation: adaptation results (S/LP/B1/B2 on T1/T2, single test pass)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 4: Report to the user**

Give, per task:
- the mean ± sd test accuracy of the four arms;
- the Q1 verdicts with their CIs;
- Q2;
- Q3 overall;
- the reuse table (A size vs add-on sizes);
- the curve if run;
- the stated limitation (one pretraining run).
