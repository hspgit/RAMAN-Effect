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
