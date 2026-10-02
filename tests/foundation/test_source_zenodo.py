import zipfile

import numpy as np
import pandas as pd
import pytest

from foundation.data.sources import REGISTRY
from foundation.data.sources.zenodo_guo import read_bacteria, read_oil


def _bacteria_zip(path, meta_rows=None):
    axis = np.arange(400.0, 1800.0, 2.0)
    spectra = np.vstack([axis, np.ones((3, axis.size)), 2 * np.ones((1, axis.size))])
    df = pd.DataFrame(spectra, index=[str(i) for i in range(1, 6)],
                      columns=[f"V{i}" for i in range(1, axis.size + 1)])
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("Bacterial_data/data.csv", df.to_csv())
        if meta_rows is not None:
            meta = pd.DataFrame({"labels": ["E.coli"] * meta_rows,
                                 "batches": [f"b{i % 2}" for i in range(meta_rows)]})
            zf.writestr("Bacterial_data/meta_all.csv", meta.to_csv(index=False))


def test_bacteria_with_metadata(tmp_path):
    _bacteria_zip(tmp_path / "Bacterial_data.zip", meta_rows=4)
    axis, X, labels, groups = read_bacteria(tmp_path / "Bacterial_data.zip")
    assert axis[0] == 400.0 and X.shape == (4, axis.size)
    assert labels == ["E.coli"] * 4 and groups == ["b0", "b1", "b0", "b1"]


def test_bacteria_without_metadata_is_unlabelled(tmp_path):
    _bacteria_zip(tmp_path / "Bacterial_data.zip")
    _, X, labels, groups = read_bacteria(tmp_path / "Bacterial_data.zip")
    assert labels == [None] * 4 and set(groups) == {"guo_bacteria"}


def test_bacteria_metadata_row_mismatch_falls_back_to_unlabelled(tmp_path, capsys):
    _bacteria_zip(tmp_path / "Bacterial_data.zip", meta_rows=3)
    _, X, labels, groups = read_bacteria(tmp_path / "Bacterial_data.zip")
    assert X.shape[0] == 4
    assert labels == [None] * 4 and groups == ["guo_bacteria"] * 4
    out = capsys.readouterr().out
    assert "meta_all.csv" in out and "3" in out and "4" in out


def test_oil_and_registered_sources(tmp_path):
    axis = np.arange(353.0, 1900.0, 3.0)
    spec = np.vstack([axis, np.ones((2, axis.size))])
    with zipfile.ZipFile(tmp_path / "Oil_data.zip", "w") as zf:
        zf.writestr("Oil_data/spec.csv", pd.DataFrame(spec).to_csv(header=False, index=False))
        zf.writestr("Oil_data/metadata.csv", "batches,labels,category\nbatch_0,olive,train\n"
                                             "batch_1,olive,test\n")
    axis2, X, labels, groups = read_oil(tmp_path / "Oil_data.zip")
    assert axis2[0] == 353.0 and X.shape == (2, axis.size)
    assert labels == ["olive", "olive"] and groups == ["batch_0", "batch_1"]
    with zipfile.ZipFile(tmp_path / "nometa.zip", "w") as zf:
        zf.writestr("Oil_data/spec.csv", "1,2\n3,4\n")
    with pytest.raises(ValueError, match="not found"):
        read_oil(tmp_path / "nometa.zip")
    (batch,) = list(REGISTRY["zenodo_oil"].iter_batches(tmp_path))
    assert REGISTRY["zenodo_oil"].domain == "chem" and len(batch.label) == 2
