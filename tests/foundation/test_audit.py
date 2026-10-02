import json

from foundation.data import audit


def _write_manifest(path, n_pretrain):
    src = {"domain": "bio", "n_raw": n_pretrain + 10, "n_unusable": 5, "n_leaked": 1,
           "n_capped": 0, "n_duplicate": 2, "n_probe_isolation": 0,
           "n_pretrain": n_pretrain, "n_probe_train": 1, "n_probe_test": 1,
           "native_min": 500.0, "native_max": 1800.0, "lasers": [532.0], "licence": "CC-BY"}
    path.mkdir()
    (path / "manifest.json").write_text(json.dumps({
        "sources": {"cells": src},
        "totals": {"pretrain": n_pretrain, "probe_train": 1, "probe_test": 1}}))


def test_g0_passes_at_threshold(tmp_path, capsys):
    _write_manifest(tmp_path / "c", 150_000)
    assert audit.main(["--corpus", str(tmp_path / "c")]) == 0
    assert "G0 PASS" in capsys.readouterr().out


def test_g0_fails_below_threshold(tmp_path, capsys):
    _write_manifest(tmp_path / "c", 149_999)
    assert audit.main(["--corpus", str(tmp_path / "c")]) == 1
    out = capsys.readouterr().out
    assert "G0 FAIL" in out and "cells" in out


def test_summarize_has_one_row_per_source(tmp_path):
    _write_manifest(tmp_path / "c", 10)
    table, totals = audit.summarize(tmp_path / "c")
    assert list(table.index) == ["cells"] and totals["pretrain"] == 10
