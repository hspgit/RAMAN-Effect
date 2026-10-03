import numpy as np
import pandas as pd
import pytest
import torch

from foundation.model.encoder import PRESETS, SpectralEncoder
from foundation.probe import (build_probe_set, embed, evaluate, knn_f1, logreg_f1,
                              rankme, raw_knn_score, source_id_bacc)


def _clusters(n_per=10, classes=3, dim=8, seed=0, spread=0.05):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(classes, dim))
    X = np.concatenate([c + spread * rng.normal(size=(n_per, dim)) for c in centres])
    y = np.repeat([f"c{i}" for i in range(classes)], n_per)
    return X, y


def test_knn_and_logreg_separate_clean_clusters():
    X, y = _clusters()
    Xt, yt = _clusters(seed=0)
    assert knn_f1(X, y, Xt, yt) == pytest.approx(1.0)
    assert logreg_f1(X, y, Xt, yt) == pytest.approx(1.0)


def test_knn_works_with_three_examples_per_class():
    # RRUFF-like: many classes, ~3 training spectra each, k=20 > class size
    X, y = _clusters(n_per=3, classes=40)
    Xt, yt = _clusters(n_per=3, classes=40, seed=0)
    assert knn_f1(X, y, Xt, yt) > 0.95


def test_rankme_counts_effective_dimensions():
    rng = np.random.default_rng(0)
    full = rng.normal(size=(500, 16))
    collapsed = np.outer(rng.normal(size=500), rng.normal(size=16))
    assert rankme(full) > 12 and rankme(collapsed) < 1.5


def test_source_id_is_nan_with_a_single_source():
    X, _ = _clusters()
    assert np.isnan(source_id_bacc(X, np.array(["a"] * len(X))))
    Xs, ys = _clusters(n_per=20)
    assert source_id_bacc(Xs, ys) > 0.9


def test_build_probe_set_uses_probe_rows_and_a_balanced_pretrain_sample():
    meta = pd.DataFrame({
        "source": ["big"] * 1000 + ["small"] * 5 + ["p"] * 8,
        "split": ["pretrain"] * 1005 + ["probe_train"] * 4 + ["probe_test"] * 4,
        "label": [None] * 1005 + ["x", "y"] * 4})
    ps = build_probe_set(meta, per_source=300)
    assert [t.name for t in ps.tasks] == ["p"]
    assert len(ps.tasks[0].train_idx) == 4 and len(ps.tasks[0].test_idx) == 4
    counts = pd.Series(ps.sample_source).value_counts().to_dict()
    assert counts == {"big": 300, "small": 5}
    assert (meta.loc[ps.sample_idx, "split"] == "pretrain").all()


def test_evaluate_and_raw_baseline_on_fake_corpus(fake_corpus):
    X = np.load(fake_corpus / "spectra.npy")
    M = np.load(fake_corpus / "mask.npy")
    meta = pd.read_parquet(fake_corpus / "meta.parquet")
    ps = build_probe_set(meta)
    enc = SpectralEncoder(PRESETS["tiny"])
    out = evaluate(enc, X, M, ps, torch.device("cpu"))
    for key in ("probe_score", "knn_score", "logreg_score", "rankme", "source_id_bacc"):
        assert key in out
    assert out["probe_score"] == pytest.approx((out["knn_score"] + out["logreg_score"]) / 2)
    assert raw_knn_score(X, M, ps) > 0.9          # fake classes are easy on raw spectra
    assert embed(enc, X, M, ps.sample_idx, torch.device("cpu")).shape == (len(ps.sample_idx), 128)


def test_evaluate_aggregates_two_probe_tasks_and_two_sources():
    from foundation.data.harmonize import GRID

    rng = np.random.default_rng(0)
    n_wn = GRID.size  # 1001 points
    X, M, rows = [], [], []

    # Helper to add spectra
    def add_rows(source, split, label, n_rows, peak_centres):
        for i in range(n_rows):
            centre = rng.choice(peak_centres)
            y = np.exp(-0.5 * ((GRID - centre) / 15) ** 2) + rng.normal(0, 0.01, n_wn)
            y = (y - y.min()) / (y.max() - y.min() + 1e-8)
            X.append(y.astype(np.float16))
            M.append(np.ones(n_wn, bool))
            rows.append({"source": source, "split": split, "label": label})

    # Two pretrain sources with clearly different peak regions
    add_rows("s1", "pretrain", None, 40, [600, 700, 800])
    add_rows("s2", "pretrain", None, 40, [1200, 1300, 1400])

    # Probe source "pa": 3 classes, 4 train + 4 test per class
    for cls, centre in enumerate([900, 1000, 1100]):
        add_rows("pa", "probe_train", f"c{cls}", 4, [centre])
        add_rows("pa", "probe_test", f"c{cls}", 4, [centre])

    # Probe source "pb": 4 classes, 4 train + 4 test per class
    for cls, centre in enumerate([500, 550, 600, 650]):
        add_rows("pb", "probe_train", f"d{cls}", 4, [centre])
        add_rows("pb", "probe_test", f"d{cls}", 4, [centre])

    X = np.stack(X)
    M = np.stack(M)
    meta = pd.DataFrame(rows)

    torch.manual_seed(0)
    ps = build_probe_set(meta)
    enc = SpectralEncoder(PRESETS["tiny"])
    out = evaluate(enc, X, M, ps, torch.device("cpu"))

    # Assert per_task structure
    assert set(out["per_task"]) == {"pa", "pb"}

    # Assert per-task scores are in [0, 1]
    for task_name, task_scores in out["per_task"].items():
        assert 0 <= task_scores["knn_f1"] <= 1
        assert 0 <= task_scores["logreg_f1"] <= 1

    # Assert aggregation
    knn_mean = np.mean([out["per_task"]["pa"]["knn_f1"], out["per_task"]["pb"]["knn_f1"]])
    lr_mean = np.mean([out["per_task"]["pa"]["logreg_f1"], out["per_task"]["pb"]["logreg_f1"]])
    assert out["knn_score"] == pytest.approx(knn_mean)
    assert out["logreg_score"] == pytest.approx(lr_mean)
    assert out["probe_score"] == pytest.approx((out["knn_score"] + out["logreg_score"]) / 2)

    # Assert rankme is finite and >= 1
    assert np.isfinite(out["rankme"])
    assert out["rankme"] >= 1

    # Assert source_id_bacc is in [0, 1] and not NaN (two sources present)
    assert 0 <= out["source_id_bacc"] <= 1
    assert not np.isnan(out["source_id_bacc"])
