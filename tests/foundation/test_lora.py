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
