import json

import numpy as np
import pandas as pd
import pytest

from foundation.data import build_corpus as bc
from foundation.data.harmonize import harmonize_many
from foundation.data.sources.base import RawBatch

OUR_AXIS = np.linspace(1792.4, 381.98, 1000)


def _g(x, c, s):
    return np.exp(-0.5 * ((x - c) / s) ** 2)


def _our_raw(n=10, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack([0.3 + _g(OUR_AXIS, rng.uniform(600, 1700), 6)
                     + _g(OUR_AXIS, rng.uniform(600, 1700), 10)
                     + rng.normal(0, 0.01, OUR_AXIS.size) for _ in range(n)])


class LabelledMinerals:
    """3 classes x 5 samples x 4 spectra, plus a flat spectrum and an exact
    duplicate of row 0. Labels are ints to test label normalisation."""
    name, domain, licence, url, probe_label_kind = "fake_min", "mineral", "CC0", "x", "species"

    def download(self, raw_dir):
        pass

    def iter_batches(self, raw_dir):
        rng = np.random.default_rng(1)
        x = np.arange(300.0, 1900.0, 2.0)
        spectra, ids, labels = [], [], []
        for cls, center in enumerate((500, 900, 1400)):
            for s in range(5):
                for _ in range(4):
                    spectra.append(0.1 + _g(x, center + rng.normal(0, 2), 5)
                                   + rng.normal(0, 0.01, x.size))
                    ids.append(f"c{cls}s{s}")
                    labels.append(cls)
        spectra.append(np.ones_like(x)); ids.append("flat"); labels.append(None)
        spectra.append(spectra[0].copy()); ids.append("c0s0"); labels.append(0)
        yield RawBatch(axes=x, spectra=np.stack(spectra), sample_id=ids,
                       label=labels, laser_nm=532)


class UnlabelledCells:
    """One big map (30 spectra, capped to 10) and one small map (5), plus one
    planted copy of an 'our' spectrum on our own axis."""
    name, domain, licence, url, probe_label_kind = "fake_cells", "bio", "?", "y", None

    def download(self, raw_dir):
        pass

    def iter_batches(self, raw_dir):
        rng = np.random.default_rng(2)
        x = np.arange(500.0, 1800.0, 1.0)
        big = [0.2 + _g(x, 1000 + 5 * i, 8) + rng.normal(0, 0.01, x.size) for i in range(30)]
        small = [0.2 + _g(x, 1300 + 7 * i, 8) + rng.normal(0, 0.01, x.size) for i in range(5)]
        yield RawBatch(axes=[x] * 35, spectra=big + small,
                       sample_id=["map_big"] * 30 + ["map_small"] * 5,
                       label=[None] * 35, laser_nm=785)
        yield RawBatch(axes=OUR_AXIS, spectra=_our_raw()[:1], sample_id=["planted"],
                       label=[None], laser_nm=633)


@pytest.fixture
def built(tmp_path, monkeypatch):
    our_S, our_M = harmonize_many(OUR_AXIS, _our_raw())
    y = np.zeros(len(our_S))
    ours = {n: (our_S, our_M, y) for n in ("reference", "finetune", "test",
                                            "2018clinical", "2019clinical")}
    monkeypatch.setattr(bc, "load_ours", lambda *a, **k: ours)
    monkeypatch.setattr(bc, "REGISTRY", {"fake_min": LabelledMinerals(),
                                         "fake_cells": UnlabelledCells()})
    cfg = {
        "version": "corpus_test", "out_dir": str(tmp_path / "corpus"),
        "raw_dir": str(tmp_path / "raw"), "cache_dir": str(tmp_path / "cache"),
        "our_data_dir": "unused", "seed": 0, "device": "cpu",
        "leakage_cos_threshold": 0.999, "probe_fraction": 0.4,
        "probe_test_fraction": 0.5, "min_probe_class_spectra": 5,
        "sources": [{"name": "fake_min", "probe": True, "max_per_sample": None},
                    {"name": "fake_cells", "probe": False, "max_per_sample": 10}],
    }
    manifest = bc.build(cfg)
    out = tmp_path / "corpus"
    meta = pd.read_parquet(out / "meta.parquet")
    return manifest, out, meta


def test_artefacts_are_consistent(built):
    manifest, out, meta = built
    S = np.load(out / "spectra.npy")
    M = np.load(out / "mask.npy")
    assert S.dtype == np.float16 and S.shape == (len(meta), 1001)
    assert M.dtype == bool and M.shape == S.shape
    splits = json.loads((out / "splits.json").read_text())
    assert sum(len(v) for v in splits.values()) == len(meta)
    assert set(meta.columns) == {"source", "domain", "laser_nm", "sample_id",
                                 "label", "split", "sample_weight"}


def test_planted_copy_of_our_data_is_removed(built):
    manifest, _, meta = built
    assert manifest["sources"]["fake_cells"]["n_leaked"] == 1
    assert "planted" not in set(meta["sample_id"])


def test_similarity_hist_counts_the_planted_copy(built):
    manifest, _, _ = built
    hist = manifest["sources"]["fake_cells"]["max_similarity_hist"]
    assert set(hist) == {"0.99", "0.995", "0.999"}
    assert hist["0.999"] >= 1


def test_similarity_hist_bins():
    h = bc.similarity_hist([0.5, 0.99, 0.994, 0.995, 0.9989, 0.999, 1.0])
    assert h == {"0.99": 2, "0.995": 2, "0.999": 2}


def test_flat_spectrum_is_dropped_and_duplicate_removed(built):
    manifest, _, meta = built
    src = manifest["sources"]["fake_min"]
    assert src["n_unusable"] == 1
    assert src["n_duplicate"] == 1
    assert (meta["source"] == "fake_min").sum() == 60


def test_per_sample_cap_is_respected(built):
    manifest, _, meta = built
    cells = meta[meta["source"] == "fake_cells"]
    assert (cells["sample_id"] == "map_big").sum() == 10
    assert (cells["sample_id"] == "map_small").sum() == 5
    assert manifest["sources"]["fake_cells"]["n_capped"] == 20


def test_probe_groups_never_shared_with_pretrain_or_each_other(built):
    _, _, meta = built
    probe = meta[meta["split"] != "pretrain"]
    assert len(probe) > 0
    by_split = {s: set(meta.loc[meta["split"] == s, "sample_id"]) for s in
                ("pretrain", "probe_train", "probe_test")}
    assert not by_split["probe_train"] & by_split["probe_test"]
    assert not (by_split["probe_train"] | by_split["probe_test"]) & by_split["pretrain"]


def test_labels_are_strings_or_null(built):
    _, _, meta = built
    mins = meta[meta["source"] == "fake_min"]["label"]
    assert set(mins) == {"0", "1", "2"}
    assert meta[meta["source"] == "fake_cells"]["label"].isna().all()


def test_sampling_weights_follow_sqrt_n(built):
    _, _, meta = built
    pre = meta[meta["split"] == "pretrain"]
    assert meta.loc[meta["split"] != "pretrain", "sample_weight"].eq(0).all()
    assert pre["sample_weight"].sum() == pytest.approx(1.0)
    totals = pre.groupby("source")["sample_weight"].sum()
    counts = pre.groupby("source").size()
    ratio = totals / np.sqrt(counts)
    assert ratio.max() == pytest.approx(ratio.min())


class NeedsManualDownload:
    name, domain, licence, url, probe_label_kind = "manual_src", "bio", "?", "z", None

    def download(self, raw_dir):
        from foundation.data.sources.base import ManualDownloadRequired
        raise ManualDownloadRequired("put files in raw/manual_src")

    def iter_batches(self, raw_dir):
        raise AssertionError("must not be called")


class CrashesMidway:
    """One good batch, then a ValueError on the second."""
    name, domain, licence, url, probe_label_kind = "crashy", "bio", "?", "w", None

    def download(self, raw_dir):
        pass

    def iter_batches(self, raw_dir):
        x = np.arange(500.0, 1800.0, 1.0)
        rng = np.random.default_rng(4)
        yield RawBatch(axes=x, spectra=[0.2 + _g(x, 900 + 9 * i, 8) + rng.normal(0, 0.01, x.size)
                                        for i in range(6)],
                       sample_id=["crash_a"] * 6, label=[None] * 6, laser_nm=785)
        raise ValueError("boom")


def _cfg(tmp_path, sources):
    return {"version": "t", "out_dir": str(tmp_path / "c2"), "raw_dir": str(tmp_path / "raw"),
            "cache_dir": str(tmp_path / "cache"), "our_data_dir": "unused", "seed": 0,
            "device": "cpu", "leakage_cos_threshold": 0.999, "probe_fraction": 0.4,
            "probe_test_fraction": 0.5, "min_probe_class_spectra": 5, "sources": sources}


def test_optional_manual_source_is_skipped_and_recorded(tmp_path, monkeypatch):
    our_S, our_M = harmonize_many(OUR_AXIS, _our_raw())
    ours = {n: (our_S, our_M, np.zeros(len(our_S))) for n in
            ("reference", "finetune", "test", "2018clinical", "2019clinical")}
    monkeypatch.setattr(bc, "load_ours", lambda *a, **k: ours)
    monkeypatch.setattr(bc, "REGISTRY", {"fake_min": LabelledMinerals(),
                                         "manual_src": NeedsManualDownload()})
    manifest = bc.build(_cfg(tmp_path, [
        {"name": "fake_min", "probe": True},
        {"name": "manual_src", "probe": False, "optional": True}]))
    assert manifest["sources"]["manual_src"]["status"].startswith("skipped")
    assert manifest["sources"]["fake_min"]["status"] == "ok"


def test_optional_source_that_crashes_is_dropped_cleanly(tmp_path, monkeypatch):
    our_S, our_M = harmonize_many(OUR_AXIS, _our_raw())
    ours = {n: (our_S, our_M, np.zeros(len(our_S))) for n in
            ("reference", "finetune", "test", "2018clinical", "2019clinical")}
    monkeypatch.setattr(bc, "load_ours", lambda *a, **k: ours)
    monkeypatch.setattr(bc, "REGISTRY", {"fake_min": LabelledMinerals(),
                                         "crashy": CrashesMidway()})
    manifest = bc.build(_cfg(tmp_path, [
        {"name": "crashy", "probe": False, "optional": True},
        {"name": "fake_min", "probe": True}]))
    assert manifest["sources"]["crashy"]["status"].startswith("failed: ValueError")
    meta = pd.read_parquet(tmp_path / "c2" / "meta.parquet")
    assert "crashy" not in set(meta["source"])
    assert manifest["sources"]["fake_min"]["status"] == "ok"


def test_required_source_that_crashes_fails_the_build(tmp_path, monkeypatch):
    our_S, our_M = harmonize_many(OUR_AXIS, _our_raw())
    ours = {n: (our_S, our_M, np.zeros(len(our_S))) for n in
            ("reference", "finetune", "test", "2018clinical", "2019clinical")}
    monkeypatch.setattr(bc, "load_ours", lambda *a, **k: ours)
    monkeypatch.setattr(bc, "REGISTRY", {"crashy": CrashesMidway()})
    with pytest.raises(ValueError, match="boom"):
        bc.build(_cfg(tmp_path, [{"name": "crashy", "probe": False}]))


def test_required_manual_source_fails_the_build(tmp_path, monkeypatch):
    our_S, our_M = harmonize_many(OUR_AXIS, _our_raw())
    ours = {n: (our_S, our_M, np.zeros(len(our_S))) for n in
            ("reference", "finetune", "test", "2018clinical", "2019clinical")}
    monkeypatch.setattr(bc, "load_ours", lambda *a, **k: ours)
    monkeypatch.setattr(bc, "REGISTRY", {"manual_src": NeedsManualDownload()})
    from foundation.data.sources.base import ManualDownloadRequired
    with pytest.raises(ManualDownloadRequired):
        bc.build(_cfg(tmp_path, [{"name": "manual_src", "probe": False}]))


def test_raw_fingerprint_changes_when_files_change(tmp_path):
    d = tmp_path / "raw" / "src"
    assert bc.raw_fingerprint(d) is None
    d.mkdir(parents=True)
    (d / "a.txt").write_text("1")
    first = bc.raw_fingerprint(d)
    assert first == bc.raw_fingerprint(d)
    (d / "a.txt").write_text("12")
    assert bc.raw_fingerprint(d) != first


def test_assign_splits_is_deterministic_and_skips_small_classes():
    source = np.array(["a"] * 12 + ["a"] * 4, dtype=object)
    label = np.array(["x"] * 12 + ["y"] * 4, dtype=object)
    sample = np.array([f"s{i // 2}" for i in range(12)] + ["t0", "t0", "t1", "t1"], dtype=object)
    args = dict(probe_sources={"a"}, probe_fraction=0.5, probe_test_fraction=0.5,
                min_class_spectra=2, seed=3)
    s1 = bc.assign_splits(source, label, sample, **args)
    s2 = bc.assign_splits(source, label, sample, **args)
    assert (s1 == s2).all()
    assert (s1[12:] == "pretrain").all()            # class y has only 2 groups
    assert set(s1[:12]) == {"pretrain", "probe_train", "probe_test"}
