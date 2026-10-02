import numpy as np
import pytest

from foundation.data.sources import REGISTRY, base
from foundation.data.sources.base import ManualDownloadRequired
from foundation.data.sources.raman_data_export import RamanDataExport


def _npz(raw, ds_id, n, labels=True):
    d = raw / "raman_data"
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"{ds_id}.npz", spectra=np.ones((n, 300), np.float32),
             axis=np.linspace(400, 1800, 300),
             labels=np.array(["pos", "neg"] * (n // 2)) if labels else np.array([], dtype=str),
             group_ids=np.array([], dtype=str))


def test_whitelisted_sources_are_registered():
    assert "rd_covid19_salvia" in REGISTRY
    assert REGISTRY["rd_covid19_salvia"].domain == "bio"


def test_missing_export_asks_for_export_step(tmp_path):
    src = RamanDataExport("x", "bio", 785)
    with pytest.raises(ManualDownloadRequired, match="export_raman_data"):
        src.download(tmp_path / "rd_x")


def test_reads_labels_and_defaults_group(tmp_path):
    _npz(tmp_path, "x", 4)
    src = RamanDataExport("x", "bio", 785)
    src.download(tmp_path / "rd_x")
    (batch,) = list(src.iter_batches(tmp_path / "rd_x"))
    assert batch.label == ["pos", "neg", "pos", "neg"]
    assert set(batch.sample_id) == {"rd_x"}
    assert batch.laser_nm.tolist() == [785.0] * 4


def test_unlabelled_and_chunked(tmp_path, monkeypatch):
    _npz(tmp_path, "y", 10, labels=False)
    monkeypatch.setattr(base, "MAX_BATCH", 4)
    import foundation.data.sources.raman_data_export as rde
    monkeypatch.setattr(rde, "MAX_BATCH", 4)
    batches = list(RamanDataExport("y", "chem", None).iter_batches(tmp_path / "rd_y"))
    assert [len(b.spectra) for b in batches] == [4, 4, 2]
    assert all(l is None for b in batches for l in b.label)
    assert np.isnan(batches[0].laser_nm).all()


def test_fingerprint_paths_points_at_the_npz(tmp_path):
    _npz(tmp_path, "x", 4)
    src = RamanDataExport("x", "bio", 785)
    assert src.fingerprint_paths(tmp_path / "rd_x") == [tmp_path / "raman_data" / "x.npz"]
