import numpy as np

from foundation.data.sources import REGISTRY
from foundation.data.sources.api_flanagan import FILENAME, parse_api_csv


def _write_csv(path):
    axis = np.arange(150.0, 1650.0, 1.0)
    header = ",".join(f"{v:.1f}" for v in axis) + ",label"
    rows = []
    for name in ('"1,3-Dimethyl-2-imidazolidinone"', "Toluene"):
        rows.append(",".join(f"{v:.3f}" for v in 1 + np.sin(axis / 50)) + "," + name)
    path.write_text("\n".join([header] + rows) + "\n")


def test_parse_wide_csv_with_quoted_labels(tmp_path):
    _write_csv(tmp_path / FILENAME)
    axis, X, labels = parse_api_csv(tmp_path / FILENAME)
    assert axis[0] == 150.0 and axis.size == 1500
    assert X.shape == (2, 1500)
    assert labels == ["1,3-Dimethyl-2-imidazolidinone", "Toluene"]


def test_iter_batches_groups_by_compound(tmp_path):
    _write_csv(tmp_path / FILENAME)
    (batch,) = list(REGISTRY["api_flanagan"].iter_batches(tmp_path))
    assert batch.sample_id == ["compound:1,3-Dimethyl-2-imidazolidinone", "compound:Toluene"]
    assert batch.laser_nm.tolist() == [785.0, 785.0]
