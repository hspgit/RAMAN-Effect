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
