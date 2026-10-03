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


OURS_AXIS = np.linspace(1792.4, 381.98, 1000)       # descending, like data/wavenumbers.npy


def make_fake_ours(d, seed=0):
    """data/ look-alike: 3 isolate classes (T1) and 5 clinical classes (T2),
    one Gaussian band per class."""
    rng = np.random.default_rng(seed)
    d.mkdir()
    np.save(d / "wavenumbers.npy", OURS_AXIS)
    sets = {"reference": np.repeat([0.0, 1.0, 2.0], 20), "finetune": np.repeat([0.0, 1.0, 2.0], 10),
            "test": np.repeat([0.0, 1.0, 2.0], 10),
            "2018clinical": np.repeat([0.0, 2.0, 3.0, 5.0, 6.0], 10),
            "2019clinical": np.repeat([0.0, 2.0, 3.0, 5.0, 6.0], 4)}
    for name, y in sets.items():
        centres = {lab: 500 + 150 * i for i, lab in enumerate(sorted(set(y)))}
        X = np.stack([0.2 + np.exp(-0.5 * ((OURS_AXIS - centres[v]) / 6) ** 2)
                      + rng.normal(0, 0.01, OURS_AXIS.size) for v in y])
        np.save(d / f"X_{name}.npy", X)
        np.save(d / f"y_{name}.npy", y)
    return d


def make_fake_a(path, preset="tiny", seed=0):
    """A model-A checkpoint shaped like pretrain.py's ckpt_best.pt."""
    import torch

    from foundation.model.encoder import PRESETS, SpectralEncoder
    torch.manual_seed(seed)
    enc = SpectralEncoder(PRESETS[preset])
    torch.save({"encoder": enc.state_dict(), "preset": preset, "epoch": 1, "probe": {}}, path)
    return path


@pytest.fixture
def adapt_cfg(tmp_path):
    """A CPU-sized adaptation config over fake data and a fake tiny A."""
    from foundation.adapt import load_config
    data = make_fake_ours(tmp_path / "data")
    a = make_fake_a(tmp_path / "a.pt")
    cfg = load_config(None, a_ckpt=str(a), data_dir=str(data), cache_dir=str(tmp_path / "cache"),
                      run_root=str(tmp_path / "runs"), device="cpu", amp=False, batch_size=16)
    cfg.update(max_epochs={"stage-i": 1, "stage-ii": 1, "single": 1}, head_warmup_epochs=1,
               lr_grid={"S": [1e-3], "B1": [1e-3], "LP": [1e-2], "B2": [1e-2]},
               seeds=[1, 2], curve_per_class=[2], curve_seeds=[1])
    return cfg
