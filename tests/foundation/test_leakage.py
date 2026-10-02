import numpy as np

from foundation.data.harmonize import harmonize, harmonize_many
from foundation.data.leakage import LeakageGuard, spectrum_hash

AXIS = np.linspace(1792.4, 381.98, 1000)


def _random_raw(n, seed):
    rng = np.random.default_rng(seed)
    g = lambda c, s: np.exp(-0.5 * ((AXIS - c) / s) ** 2)
    return np.stack([0.2 + g(rng.uniform(500, 1700), rng.uniform(3, 15))
                     + g(rng.uniform(500, 1700), 8) + rng.normal(0, 0.01, AXIS.size)
                     for _ in range(n)])


def _random_spectra(n, seed):
    return harmonize_many(AXIS, _random_raw(n, seed))


def test_hash_is_stable_and_sensitive():
    S, M = _random_spectra(2, 0)
    assert spectrum_hash(S[0], M[0]) == spectrum_hash(S[0].copy(), M[0].copy())
    assert spectrum_hash(S[0], M[0]) != spectrum_hash(S[1], M[1])


def test_exact_copy_is_caught():
    S, M = _random_spectra(50, 0)
    guard = LeakageGuard(S, M, device="cpu")
    assert guard.find(S[:3], M[:3]).all()


def test_noisy_copy_is_caught_by_cosine():
    S, M = _random_spectra(50, 0)
    rng = np.random.default_rng(1)
    noisy = (S[:5] + rng.normal(0, 0.002, S[:5].shape) * M[:5]).astype(np.float32)
    guard = LeakageGuard(S, M, device="cpu")
    assert guard.find(noisy, M[:5]).all()


def test_unrelated_spectra_are_not_flagged():
    S, M = _random_spectra(50, 0)
    other_S, other_M = _random_spectra(20, 99)
    guard = LeakageGuard(S, M, device="cpu")
    assert not guard.find(other_S, other_M).any()


def test_small_overlap_is_not_flagged():
    S, M = _random_spectra(50, 0)
    from foundation.data.harmonize import GRID
    window = (GRID >= 1000) & (GRID <= 1100)
    crop_M = M[:1] & window
    crop_S = np.where(crop_M, S[:1], 0).astype(np.float32)
    guard = LeakageGuard(S, M, device="cpu")
    assert not guard.find(crop_S, crop_M).any()


def test_empty_input_returns_empty():
    S, M = _random_spectra(5, 0)
    guard = LeakageGuard(S, M, device="cpu")
    out = guard.find(S[:0], M[:0])
    assert out.shape == (0,) and out.dtype == bool


def test_regridded_copy_is_caught():
    raw = _random_raw(50, 0)
    S, M = harmonize_many(AXIS, raw)
    new_axis = np.arange(381.98, 1792.4, 1.0)
    # np.interp needs an ascending axis; AXIS is descending. Every reference
    # is re-gridded: cosine alone lets a fraction of them escape.
    copies = [harmonize(new_axis, np.interp(new_axis, AXIS[::-1], r[::-1])) for r in raw]
    cs, cm = np.stack([c[0] for c in copies]), np.stack([c[1] for c in copies])
    guard = LeakageGuard(S, M, device="cpu")
    assert guard.find(cs, cm).all()


def test_cropped_renormalised_copy_is_caught():
    raw = _random_raw(50, 0)
    S, M = harmonize_many(AXIS, raw)
    keep = (AXIS >= 500) & (AXIS <= 1500)
    copies = [harmonize(AXIS[keep], r[keep]) for r in raw]
    cs, cm = np.stack([c[0] for c in copies]), np.stack([c[1] for c in copies])
    guard = LeakageGuard(S, M, device="cpu")
    assert guard.find(cs, cm).all()


def test_scores_shapes_and_range():
    S, M = _random_spectra(30, 0)
    other_S, other_M = _random_spectra(7, 5)
    guard = LeakageGuard(S, M, device="cpu")
    cos, corr = guard.scores(other_S, other_M)
    for a in (cos, corr):
        assert a.shape == (7,) and a.dtype == np.float32
        assert a.min() >= -1.0 - 1e-6 and a.max() <= 1.0 + 1e-6
