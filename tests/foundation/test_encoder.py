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
