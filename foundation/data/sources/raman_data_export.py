"""Datasets from the raman_data package, read from .npz files exported by
foundation/tools/export_raman_data.py in its own venv (plan deviation D5).
One source per whitelisted dataset id, named rd_<id>.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from .base import MAX_BATCH, ManualDownloadRequired, RawBatch, register

WHITELIST = Path(__file__).resolve().parents[2] / "configs" / "raman_data_whitelist.yaml"
EXPORT_HINT = ("python -m venv foundation/.venv-raman-data; "
               "foundation/.venv-raman-data/Scripts/python.exe -m pip install "
               "\"git+https://github.com/ml-lab-htw/raman_data@c92289e\" pyyaml; "
               "foundation/.venv-raman-data/Scripts/python.exe foundation/tools/export_raman_data.py")


class RamanDataExport:
    url = "https://github.com/ml-lab-htw/raman_data"
    licence = "per dataset; see raw/raman_data/export_report.json"
    probe_label_kind = None

    def __init__(self, ds_id, domain, laser_nm):
        self.ds_id = ds_id
        self.name = f"rd_{ds_id}"
        self.domain = domain
        self.laser_nm = float("nan") if laser_nm is None else float(laser_nm)

    def _file(self, raw_dir):
        return Path(raw_dir).parent / "raman_data" / f"{self.ds_id}.npz"

    def download(self, raw_dir):
        if not self._file(raw_dir).exists():
            raise ManualDownloadRequired(
                f"{self._file(raw_dir)} missing; run the export_raman_data step: {EXPORT_HINT}")

    def iter_batches(self, raw_dir):
        z = np.load(self._file(raw_dir))
        X, axis = z["spectra"], z["axis"]
        n = len(X)
        labels = z["labels"].tolist() if z["labels"].size == n else [None] * n
        groups = z["group_ids"].tolist() if z["group_ids"].size == n else [self.name] * n
        for s in range(0, n, MAX_BATCH):
            e = min(n, s + MAX_BATCH)
            yield RawBatch(axes=axis, spectra=X[s:e], sample_id=groups[s:e],
                           label=labels[s:e], laser_nm=self.laser_nm)


for _id, _cfg in (yaml.safe_load(WHITELIST.read_text()) or {}).items():
    register(RamanDataExport(_id, _cfg["domain"], _cfg.get("laser_nm")))
