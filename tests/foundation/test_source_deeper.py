import numpy as np
import pytest
from scipy.io import savemat

from foundation.data.sources.base import ManualDownloadRequired
from foundation.data.sources.deeper import REQUIRED, DeepeR


def _fixture(raw):
    raw.mkdir(parents=True)
    axis = np.linspace(498.265, 1799.9, 500)
    np.savetxt(raw / "axis.txt", axis, fmt="%.18e")
    rng = np.random.default_rng(0)
    savemat(raw / "Train_Outputs.mat", {"Train_Outputs": rng.random((40, 500))})
    savemat(raw / "Test_Outputs.mat", {"Test_Outputs": rng.random((10, 500))})


def test_missing_files_ask_for_manual_download(tmp_path):
    with pytest.raises(ManualDownloadRequired, match="Train_Outputs.mat"):
        DeepeR().download(tmp_path / "deeper")


def test_subsample_is_capped_unique_and_deterministic(tmp_path):
    _fixture(tmp_path / "deeper")
    DeepeR().download(tmp_path / "deeper")              # files present: no error
    (a,) = list(DeepeR(max_spectra=20, seed=0).iter_batches(tmp_path / "deeper"))
    (b,) = list(DeepeR(max_spectra=20, seed=0).iter_batches(tmp_path / "deeper"))
    assert len(a.spectra) == 20
    np.testing.assert_array_equal(a.spectra, b.spectra)
    assert len({row.tobytes() for row in a.spectra}) == 20
    assert a.axes.size == 500 and a.label == [None] * 20


def test_cap_larger_than_data_keeps_everything(tmp_path):
    _fixture(tmp_path / "deeper")
    (batch,) = list(DeepeR(max_spectra=1000).iter_batches(tmp_path / "deeper"))
    assert len(batch.spectra) == 50
    assert set(REQUIRED) == {"Train_Outputs.mat", "Test_Outputs.mat", "axis.txt"}
