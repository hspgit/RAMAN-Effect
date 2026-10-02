from pathlib import Path

import numpy as np
import pytest

from foundation.data.tasks import load_ours, load_t1, load_t2

AXIS = np.linspace(1792.4, 381.98, 1000)       # descending, like the real file


def _spectra(labels, rng):
    centers = {lab: 500 + 37 * (i + 1) for i, lab in enumerate(sorted(set(labels)))}
    return np.stack([0.2 + np.exp(-0.5 * ((AXIS - centers[l]) / 6) ** 2)
                     + rng.normal(0, 0.01, AXIS.size) for l in labels])


@pytest.fixture
def fake_data(tmp_path):
    rng = np.random.default_rng(0)
    d = tmp_path / "data"
    d.mkdir()
    np.save(d / "wavenumbers.npy", AXIS)
    sets = {
        "reference": np.repeat([0.0, 1.0, 2.0], 20),
        "finetune": np.repeat([0.0, 1.0, 2.0], 10),
        "test": np.repeat([0.0, 1.0, 2.0], 10),
        "2018clinical": np.repeat([0.0, 2.0, 3.0, 5.0, 6.0], 10),
        "2019clinical": np.repeat([0.0, 2.0, 3.0, 5.0, 6.0], 4),
    }
    for name, y in sets.items():
        np.save(d / f"X_{name}.npy", _spectra(y, rng))
        np.save(d / f"y_{name}.npy", y)
    return d


def test_t1_shapes_dtypes_and_mask(fake_data, tmp_path):
    t1 = load_t1(fake_data, cache_dir=tmp_path / "cache")
    assert t1.classes.tolist() == [0.0, 1.0, 2.0]
    assert set(t1.splits) == {"ref_train", "ref_val", "ft_train", "ft_val", "test"}
    s = t1.splits["ref_train"]
    assert s.X.shape[1] == 1001 and s.X.dtype == np.float32
    assert s.M.dtype == bool and s.y.dtype == np.int64
    assert (s.M.sum(axis=1) == 940).all()


def test_t2_maps_treatment_labels_to_0_through_4(fake_data, tmp_path):
    t2 = load_t2(fake_data, cache_dir=tmp_path / "cache")
    assert t2.classes.tolist() == [0.0, 2.0, 3.0, 5.0, 6.0]
    assert set(np.unique(t2.splits["test"].y)) == {0, 1, 2, 3, 4}
    assert len(t2.splits["test"].y) == 20


def test_validation_split_is_stratified_disjoint_and_fixed(fake_data, tmp_path):
    a = load_t1(fake_data, cache_dir=tmp_path / "cache")
    b = load_t1(fake_data, cache_dir=tmp_path / "cache")
    assert np.bincount(a.splits["ref_val"].y).tolist() == [2, 2, 2]     # 10% of 20 per class
    assert len(a.splits["ref_train"].y) + len(a.splits["ref_val"].y) == 60
    np.testing.assert_array_equal(a.splits["ref_val"].X, b.splits["ref_val"].X)


def test_unknown_test_label_raises(fake_data, tmp_path):
    y = np.load(fake_data / "y_test.npy")
    y[0] = 7.0
    np.save(fake_data / "y_test.npy", y)
    with pytest.raises(ValueError, match="not in task classes"):
        load_t1(fake_data, cache_dir=tmp_path / "cache")


def test_cache_is_reused_without_rereading_spectra(fake_data, tmp_path):
    load_ours(fake_data, tmp_path / "cache")
    for name in ("reference", "finetune", "test", "2018clinical", "2019clinical"):
        (fake_data / f"X_{name}.npy").unlink()
    ours = load_ours(fake_data, tmp_path / "cache")
    assert ours["reference"][0].shape == (60, 1001)


@pytest.mark.skipif(not Path("data/X_2018clinical.npy").exists(), reason="real data absent")
def test_real_clinical_task_loads(tmp_path):
    t2 = load_t2("data", cache_dir=tmp_path / "cache")
    assert len(t2.splits["train"].y) == 9000 and len(t2.splits["val"].y) == 1000
    assert len(t2.splits["test"].y) == 2500
    assert (t2.splits["test"].M.sum(axis=1) == 940).all()
