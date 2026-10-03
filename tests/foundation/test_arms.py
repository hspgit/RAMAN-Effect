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
