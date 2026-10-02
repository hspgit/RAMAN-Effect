"""DeepeR breast-cancer-cell spectra, Horgan et al. 2021 (denoising set), 532 nm.

MANUAL DOWNLOAD (SharePoint is not reliably scriptable): put Train_Outputs.mat,
Test_Outputs.mat and axis.txt from the 'Raman Spectral Denoising/Dataset'
folder of SHARE_URL into foundation/data/raw/deeper/. Only the high-SNR
*Outputs* are used; the *Inputs* are noisy copies of the same spectra. The set
is shuffled with no cell/image ids (11 cells), so the spec's 2,000-per-image
cap is approximated by a seeded subsample of 11 x 2,000 = 22,000 (deviation D1).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from .base import ManualDownloadRequired, RawBatch, register

SHARE_URL = ("https://emckclac-my.sharepoint.com/:f:/g/personal/k1919691_kcl_ac_uk/"
             "EqZaY-_FrGdImybIGuMCvb8Bo_YD1Bc9ATBxbLxdDIv0RA?e=5%3aHhLp91&fromShare=true&at=9")
REQUIRED = ("Train_Outputs.mat", "Test_Outputs.mat", "axis.txt")


class DeepeR:
    name = "deeper"
    domain = "bio"
    licence = "code MIT; data licence not stated (research use)"
    url = SHARE_URL
    probe_label_kind = None

    def __init__(self, max_spectra=22_000, seed=0):
        self.max_spectra, self.seed = max_spectra, seed

    def download(self, raw_dir):
        missing = [f for f in REQUIRED if not (Path(raw_dir) / f).exists()]
        if missing:
            raise ManualDownloadRequired(
                f"DeepeR needs a manual download of {', '.join(missing)} from {SHARE_URL} "
                f"(folder 'Raman Spectral Denoising/Dataset') into {Path(raw_dir).resolve()}")

    def iter_batches(self, raw_dir):
        raw_dir = Path(raw_dir)
        axis = np.loadtxt(raw_dir / "axis.txt")
        X = np.vstack([loadmat(raw_dir / "Train_Outputs.mat")["Train_Outputs"],
                       loadmat(raw_dir / "Test_Outputs.mat")["Test_Outputs"]])
        if X.shape[1] != axis.size:
            raise ValueError(f"DeepeR: {X.shape[1]} points per spectrum, axis has {axis.size}")
        n = min(self.max_spectra, len(X))
        idx = np.sort(np.random.default_rng(self.seed).choice(len(X), size=n, replace=False))
        yield RawBatch(axes=axis, spectra=X[idx], sample_id=["deeper_cells"] * n,
                       label=[None] * n, laser_nm=532)


register(DeepeR())
