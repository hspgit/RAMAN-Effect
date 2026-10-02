import numpy as np
import pytest

from foundation.data.harmonize import (GRID, N_GRID, AxisError, despike, harmonize,
                                       harmonize_many, is_usable)


def gaussian(x, center, sigma, height=1.0):
    return height * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def peak_center(spec):
    """Sub-grid peak position by a parabola through the max and its neighbours."""
    i = int(np.argmax(spec))
    y0, y1, y2 = spec[i - 1], spec[i], spec[i + 1]
    offset = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    return GRID[i] + offset * (GRID[1] - GRID[0])


def test_grid_is_1001_points_350_to_1850():
    assert N_GRID == 1001
    assert GRID[0] == 350.0 and GRID[-1] == 1850.0
    assert np.allclose(np.diff(GRID), 1.5)


@pytest.mark.parametrize("step", [0.7, 1.41, 3.0])
def test_peak_position_preserved_within_half_wavenumber(step):
    x = np.arange(300.0, 2000.0, step)
    y = 0.1 + gaussian(x, 1001.3, 4.0)
    spec, _ = harmonize(x, y)
    assert abs(peak_center(spec) - 1001.3) < 0.5


def test_descending_axis_gives_identical_output():
    x = np.linspace(381.98, 1792.4, 1000)
    y = 0.2 + gaussian(x, 1450.0, 6.0)
    a_spec, a_mask = harmonize(x, y)
    b_spec, b_mask = harmonize(x[::-1], y[::-1])
    np.testing.assert_array_equal(a_spec, b_spec)
    np.testing.assert_array_equal(a_mask, b_mask)


def test_never_extrapolates_outside_source_range():
    x = np.arange(600.0, 1800.0, 1.0)
    spec, mask = harmonize(x, 1 + gaussian(x, 1000, 5))
    assert not mask[GRID < 600].any()
    assert not mask[GRID > 1800].any()
    assert mask[(GRID >= 601) & (GRID <= 1799)].all()
    assert (spec[~mask] == 0).all()


def test_gap_in_source_sampling_is_masked():
    x = np.arange(400.0, 1800.0, 1.0)
    x = x[(x < 1000) | (x > 1050)]
    _, mask = harmonize(x, 1 + gaussian(x, 1200, 5))
    assert not mask[np.argmin(np.abs(GRID - 1025.0))]
    assert mask[np.argmin(np.abs(GRID - 900.0))]


def test_bacteria_id_axis_covers_940_grid_points():
    x = np.linspace(1792.4, 381.98, 1000)          # descending, like data/wavenumbers.npy
    _, mask = harmonize(x, 1 + gaussian(x, 1000, 5))
    assert mask.sum() == 940                        # grid points 383.0 .. 1791.5


def test_minmax_uses_valid_points_only():
    x = np.arange(600.0, 1800.0, 1.0)
    spec, mask = harmonize(x, 5 + 3 * gaussian(x, 1000, 5))
    assert spec[mask].min() == pytest.approx(0.0, abs=1e-6)
    assert spec[mask].max() == pytest.approx(1.0, abs=1e-6)


def test_despike_removes_cosmic_spike_and_keeps_real_band():
    rng = np.random.default_rng(0)
    x = np.arange(1000)
    clean = gaussian(x, 300, 10) + rng.normal(0, 0.01, x.size)
    spiky = clean.copy()
    spiky[700] += 1.0
    out = despike(spiky)
    assert abs(out[700] - clean[700]) < 0.05
    np.testing.assert_allclose(out[250:350], clean[250:350])


def test_nan_and_inf_points_are_dropped_not_propagated():
    x = np.arange(400.0, 1800.0, 1.0)
    y = 1 + gaussian(x, 1000, 5)
    y[100] = np.nan
    y[200] = np.inf
    spec, mask = harmonize(x, y)
    assert np.isfinite(spec).all()
    assert mask.sum() > 900


def test_absolute_wavenumber_axis_is_rejected():
    x = np.linspace(11000.0, 12700.0, 500)          # absolute cm^-1, not Raman shift
    with pytest.raises(AxisError):
        harmonize(x, np.ones_like(x))


def test_flat_or_short_spectra_are_not_usable():
    x = np.arange(400.0, 1800.0, 1.0)
    spec, mask = harmonize(x, np.ones_like(x))
    assert not is_usable(spec, mask)
    xs = np.arange(1000.0, 1100.0, 1.0)
    spec, mask = harmonize(xs, 1 + gaussian(xs, 1050, 3))
    assert not is_usable(spec, mask)
    spec, mask = harmonize(x, 1 + gaussian(x, 1000, 5))
    assert is_usable(spec, mask)


def test_harmonize_many_shared_axis_matches_per_spectrum_axes():
    x = np.arange(400.0, 1800.0, 1.2)
    Y = np.stack([1 + gaussian(x, c, 5) for c in (700, 1000, 1500)])
    S1, M1 = harmonize_many(x, Y)
    S2, M2 = harmonize_many([x, x, x], list(Y))
    assert S1.shape == (3, 1001) and S1.dtype == np.float32 and M1.dtype == bool
    np.testing.assert_array_equal(S1, S2)
    np.testing.assert_array_equal(M1, M2)
