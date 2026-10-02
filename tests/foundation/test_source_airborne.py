import pytest
import numpy as np

from foundation.data.sources import REGISTRY
from foundation.data.sources.airborne import classify, local_relpath


def test_local_relpath_handles_ampersand_and_apostrophe():
    url = ("https://download.scidb.cn/download?fileId=abc&path=%2FTraining_set_2%26Test_set_2"
           "%2Fnew_data%2Funknown%27&fileName=7.txt")
    assert local_relpath(url) == "Training_set_2&Test_set_2/new_data/unknown'/7.txt"


def test_local_relpath_handles_unencoded_ampersand_in_path():
    url = ("https://download.scidb.cn/download?fileId=abc&path=/Training_set_1&Test_set_1"
           "/new_data/train_test/E.coli&fileName=12.txt")
    assert local_relpath(url) == "Training_set_1&Test_set_1/new_data/train_test/E.coli/12.txt"


def test_local_relpath_does_not_repeat_filename_already_in_path():
    url = "https://download.scidb.cn/download?fileId=x&path=%2Fa%2Fb%2F3.txt&fileName=3.txt"
    assert local_relpath(url) == "a/b/3.txt"


def test_classify_labels_species_and_drops_augmented():
    assert classify("Training_set_1&Test_set_1/new_data/train_test/E.coli/1.txt") == (True, "E.coli")
    assert classify("Test_set_3/Pathogen/S.aureus/5.txt") == (True, "S.aureus")
    assert classify("Training_set_1&Test_set_1/new_data/train_test/Others/2.txt") == (True, None)
    assert classify("Test_set_3/Unknown/U_aug_ 12.txt") == (False, None)
    assert classify("Test_set_4/readme.md") == (False, None)


def test_iter_batches_reads_listed_files(tmp_path):
    rows = "\r\n".join(f"{x:.18e} {y:.18e}" for x, y in
                       zip(np.linspace(608, 1802.67, 225), np.linspace(0, 1, 225)))
    files = {"Test_set_3/Pathogen/S.aureus/5.txt": rows,
             "Training_set_1&Test_set_1/new_data/train_test/Others/2.txt": rows,
             "Test_set_3/Unknown/U_aug_ 12.txt": rows}
    urls = []
    for rel, content in files.items():
        p = tmp_path / "files" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        folder, name = rel.rsplit("/", 1)
        urls.append(f"https://download.scidb.cn/download?fileId=1&path=/{folder}&fileName={name}")
    (tmp_path / "urls.txt").write_text("\n".join(urls))
    (batch,) = list(REGISTRY["airborne"].iter_batches(tmp_path))
    assert batch.label == ["S.aureus", None]
    assert batch.sample_id[0] == "Test_set_3/Pathogen/S.aureus/5.txt"
    assert batch.axes[0].size == 225


def test_download_finishes_other_files_before_reporting_failures(tmp_path, monkeypatch):
    import foundation.data.sources.airborne as airborne

    names = ["1.txt", "2.txt", "3.txt"]
    urls = [f"https://download.scidb.cn/download?fileId={n}&path=/Test_set_3/Pathogen/E.coli"
            f"&fileName={n}" for n in names]
    (tmp_path / "urls.txt").write_text("\n".join(urls))

    def fake_fetch(url, dest, **kw):
        if url.endswith("fileName=2.txt"):
            raise ConnectionError("gave up")
        dest = __import__("pathlib").Path(dest)
        if not dest.exists():               # urls.txt is already in place
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text("x")
        return dest

    monkeypatch.setattr(airborne, "fetch", fake_fetch)
    with pytest.raises(RuntimeError, match="1 of 3"):
        airborne.Airborne().download(tmp_path)
    files = tmp_path / "files" / "Test_set_3" / "Pathogen" / "E.coli"
    assert (files / "1.txt").exists() and (files / "3.txt").exists()
