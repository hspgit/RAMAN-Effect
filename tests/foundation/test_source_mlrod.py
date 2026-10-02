import numpy as np

from foundation.data.sources import REGISTRY
from foundation.data.sources.mlrod import FILES, LABELS, parse_header_axis, parse_mlrod_csv


def test_file_list_and_labels_are_complete():
    assert len(FILES) == 41
    assert LABELS[0] == "Quartz" and LABELS[15] == "Granite slab - 50% dust"


def test_header_tolerates_malformed_token():
    axis = parse_header_axis(["95.49", "1630.89", "1630.89.1", " 1631.5"])
    assert axis.tolist() == [95.49, 1630.89, 1630.89, 1631.5]


def test_parse_csv_and_iter_batches(tmp_path, monkeypatch):
    axis = np.linspace(95.49, 1100.97, 600)
    header = "og-idx," + ",".join(f"{v:.2f}" for v in axis) + ",label"
    rows = [f"hb_train_060s_104-{i}," + ",".join("1.0" for _ in axis) + ",2" for i in range(3)]
    name = "hb_train_060s_104.csv"
    (tmp_path / name).write_text("\n".join([header] + rows) + "\n")
    ax, X, y = parse_mlrod_csv(tmp_path / name)
    assert ax.size == 600 and X.shape == (3, 600) and y == [2, 2, 2]
    import foundation.data.sources.mlrod as mlrod
    monkeypatch.setattr(mlrod, "FILES", {name: "unused"})
    (batch,) = list(REGISTRY["mlrod"].iter_batches(tmp_path))
    assert batch.label == ["Amphibole - Hornblende"] * 3
    assert batch.sample_id == ["hb_train_060s_104"] * 3
