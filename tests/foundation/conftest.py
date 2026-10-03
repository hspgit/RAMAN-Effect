"""Shared fixtures for the foundation tests."""
import json

import numpy as np
import pandas as pd
import pytest

from foundation.data.harmonize import GRID


def make_fake_corpus(d, seed=0):
    """A tiny corpus_v1 look-alike: 120 pretrain rows of one source and a
    3-class probe source with 18 probe_train and 18 probe_test rows."""
    rng = np.random.default_rng(seed)
    rows, X, M = [], [], []
    def add(source, split, label, centre):
        y = np.exp(-0.5 * ((GRID - centre) / 8) ** 2) + rng.normal(0, 0.02, GRID.size)
        y = (y - y.min()) / (y.max() - y.min())
        X.append(y.astype(np.float16)); M.append(np.ones(GRID.size, bool))
        rows.append({"source": source, "domain": "bio", "laser_nm": 532.0,
                     "sample_id": f"{source}{len(rows)}", "label": label, "split": split,
                     "weight_group": source, "sample_weight": 0.0})
    for i in range(120):
        add("bulk", "pretrain", None, rng.uniform(500, 1700))
    for split in ("probe_train", "probe_test"):
        for cls, c in enumerate((700, 1100, 1500)):
            for _ in range(6):
                add("probe", split, f"c{cls}", c + rng.normal(0, 3))
    meta = pd.DataFrame(rows)
    meta.loc[meta.split == "pretrain", "sample_weight"] = 1 / 120
    d.mkdir()
    np.save(d / "spectra.npy", np.stack(X)); np.save(d / "mask.npy", np.stack(M))
    meta.to_parquet(d / "meta.parquet", index=False)
    (d / "manifest.json").write_text(json.dumps({"version": "fake"}))
    return d



@pytest.fixture
def fake_corpus(tmp_path):
    return make_fake_corpus(tmp_path / "corpus")
