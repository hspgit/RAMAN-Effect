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
