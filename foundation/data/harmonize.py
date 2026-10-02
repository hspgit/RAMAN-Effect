"""Put any Raman spectrum onto the shared wavenumber grid (spec section 4.3).

Order: clean axis -> despike -> anti-alias + resample -> min-max over valid
points. No baseline correction, by design (spec section 4.3).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter

HARMONIZE_VERSION = 2  # bump whenever output changes; part of every cache key

GRID_START = 350.0
GRID_STOP = 1850.0
GRID_STEP = 1.5
GRID = np.linspace(GRID_START, GRID_STOP, 1001)
N_GRID = GRID.size

MIN_VALID_POINTS = 200            # under 300 cm^-1 of coverage is not worth keeping
AXIS_MIN, AXIS_MAX = -100.0, 5000.0  # plausible Raman-shift range in cm^-1


class AxisError(ValueError):
    """The axis is not a plausible Raman-shift axis in cm^-1."""


def clean_axis(x, y):
    """Drop non-finite points, sort ascending, drop duplicate wavenumbers."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError(f"axis has {x.size} points but spectrum has {y.size}")
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size and (x.min() < AXIS_MIN or x.max() > AXIS_MAX):
        raise AxisError(f"axis spans {x.min():.1f}..{x.max():.1f}; "
                        "expected a Raman shift in cm^-1")
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    x, first = np.unique(x, return_index=True)
    return x, y[first]


def despike(y, threshold=8.0, window=7, max_width=2):
    """Cosmic-ray removal.

    A robust z-score of the residual from a running median flags positive
    outliers; only runs of at most `max_width` consecutive flagged points are
    spikes (cosmic rays are 1-2 pixels wide), and they are replaced by the
    running median. Wider excursions are real bands and are left untouched.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.size < window:
        return y.copy()
    base = median_filter(y, size=window, mode="nearest")
    r = y - base
    med = np.median(r)
    mad = np.median(np.abs(r - med))
    if mad == 0:
        return y.copy()
    cand = (0.6745 * (r - med) / mad) > threshold
    edges = np.diff(np.concatenate([[0], cand.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    out = y.copy()
    for s, e in zip(starts, ends):
        if e - s <= max_width:
            out[s:e] = base[s:e]
    return out


def _resample_clean(x, y, grid):
    """x must already be sorted, unique and finite (see clean_axis)."""
    out = np.zeros(grid.size, dtype=np.float32)
    mask = np.zeros(grid.size, dtype=bool)
    if x.size < 2:
        return out, mask
    step = float(np.median(np.diff(x)))
    grid_step = float(grid[1] - grid[0])
    if step < grid_step:
        # anti-alias: smooth to the grid's resolution (FWHM = one grid step)
        y = gaussian_filter1d(y, sigma=(grid_step / 2.3548) / step, mode="nearest")
    inside = np.flatnonzero((grid >= x[0]) & (grid <= x[-1]))
    g = grid[inside]
    pos = np.searchsorted(x, g)
    d_left = np.abs(g - x[np.clip(pos - 1, 0, x.size - 1)])
    d_right = np.abs(x[np.clip(pos, 0, x.size - 1)] - g)
    near = np.minimum(d_left, d_right) <= max(1.5 * step, grid_step)  # gaps stay masked
    keep = inside[near]
    out[keep] = np.interp(g[near], x, y)
    mask[keep] = True
    return out, mask


def resample(x, y, grid=GRID):
    x, y = clean_axis(x, y)
    return _resample_clean(x, y, grid)


def minmax_valid(y, mask):
    out = np.zeros(np.shape(y), dtype=np.float32)
    if not mask.any():
        return out
    v = np.asarray(y, dtype=np.float64)[mask]
    lo, hi = v.min(), v.max()
    if hi - lo <= 0:
        return out
    out[mask] = (v - lo) / (hi - lo)
    return out


def harmonize(x, y):
    x, y = clean_axis(x, y)
    if x.size < 2:
        return np.zeros(N_GRID, dtype=np.float32), np.zeros(N_GRID, dtype=bool)
    y = despike(y)
    spec, mask = _resample_clean(x, y, GRID)
    return minmax_valid(spec, mask), mask


def harmonize_many(axes, spectra):
    """axes: one shared 1D axis, or a sequence (or 2D array) of per-spectrum axes."""
    n = len(spectra)
    S = np.zeros((n, N_GRID), dtype=np.float32)
    M = np.zeros((n, N_GRID), dtype=bool)
    shared = isinstance(axes, np.ndarray) and axes.ndim == 1
    for i in range(n):
        S[i], M[i] = harmonize(axes if shared else axes[i], spectra[i])
    return S, M


def is_usable(spectrum, mask):
    return int(mask.sum()) >= MIN_VALID_POINTS and float(spectrum[mask].max()) > 0.0
