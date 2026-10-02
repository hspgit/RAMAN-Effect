import numpy as np
import pytest

from foundation.data.sources import REGISTRY
import foundation.data.sources.mlrod as mlrod
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


def _write_csv(path, n_axis=600, n_cols=None, bom=False):
    axis = np.linspace(95.49, 1100.97, n_axis)
    header = "og-idx," + ",".join(f"{v:.2f}" for v in axis) + ",label"
    width = n_axis if n_cols is None else n_cols
    rows = [f"r{i}," + ",".join("1.0" for _ in range(width)) + ",2" for i in range(2)]
    path.write_text("\n".join([header] + rows) + "\n", encoding="utf-8-sig" if bom else "utf-8")


def test_bom_header_is_accepted(tmp_path):
    _write_csv(tmp_path / "a.csv", bom=True)
    ax, X, y = parse_mlrod_csv(tmp_path / "a.csv")
    assert ax.size == 600 and X.shape == (2, 600)


def test_download_rejects_error_body(tmp_path, monkeypatch):
    monkeypatch.setattr(mlrod, "FILES", {"bad.csv": "unused"})

    def fake_fetch(url, dest, **kw):
        dest.write_text('{"error": "x"}')
        return dest

    monkeypatch.setattr(mlrod, "fetch", fake_fetch)
    with pytest.raises(IOError, match="bad.csv"):
        mlrod.MLROD().download(tmp_path)
    assert not (tmp_path / "bad.csv").exists()


def test_axis_width_mismatch_raises(tmp_path):
    _write_csv(tmp_path / "w.csv", n_axis=600, n_cols=590)
    with pytest.raises(ValueError, match="w.csv"):
        parse_mlrod_csv(tmp_path / "w.csv")
