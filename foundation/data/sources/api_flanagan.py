"""Pharmaceutical API Raman set, Flanagan et al. 2025 (figshare 27931131, CC BY 4.0).

One wide CSV: header '150.0,...,3425.0,label', one row per spectrum, compound
name in the last column. 785 nm, Kaiser Rxn2. There is no replicate/sample
column, so this source is pretrain-only (plan deviation D3).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .base import RawBatch, fetch, register

URL = "https://ndownloader.figshare.com/files/50881917"
FILENAME = "raman_spectra_api_compounds.csv"


def parse_api_csv(path):
    df = pd.read_csv(path)
    if df.columns[-1] != "label":
        raise ValueError(f"{path}: last column is {df.columns[-1]!r}, expected 'label'")
    axis = np.array([float(c) for c in df.columns[:-1]])
    X = df.iloc[:, :-1].to_numpy(dtype=np.float64)
    return axis, X, df["label"].astype(str).tolist()


class APIFlanagan:
    name = "api_flanagan"
    domain = "chem"
    licence = "CC BY 4.0"
    url = "https://doi.org/10.6084/m9.figshare.27931131"
    probe_label_kind = "compound"

    def download(self, raw_dir):
        fetch(URL, Path(raw_dir) / FILENAME, retries=5)

    def iter_batches(self, raw_dir):
        axis, X, labels = parse_api_csv(Path(raw_dir) / FILENAME)
        yield RawBatch(axes=axis, spectra=X, sample_id=[f"compound:{l}" for l in labels],
                       label=labels, laser_nm=785)


register(APIFlanagan())
