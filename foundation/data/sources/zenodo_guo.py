"""Single-cell bacteria and edible oils, Guo et al. (Zenodo 15279321, CC BY 4.0).

Bacterial_data.zip: Bacterial_data/data.csv in R write.csv style; the first
data row is the wavenumber axis, later rows are spectra. Labels/batches come
from a metadata CSV in the zip (meta_all.csv in the authors' code); without
one the spectra stay unlabelled (plan deviation D8).
Oil_data.zip: Oil_data/spec.csv (no header; row 0 = axis) and
Oil_data/metadata.csv (batches,labels,category), row-aligned.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from .base import RawBatch, fetch, register

BASE_URL = "https://zenodo.org/records/15279321/files/"
META_NAMES = ("meta_all.csv", "metadata.csv", "meta.csv")


def _member(zf, basename):
    for n in zf.namelist():
        if n.rsplit("/", 1)[-1] == basename:
            return n
    return None


def _labels_groups(zf, meta_name, n, default_group):
    labels, groups = [None] * n, [default_group] * n
    if meta_name is None:
        return labels, groups
    with zf.open(meta_name) as fh:
        meta = pd.read_csv(fh)
    if len(meta) != n:
        # plan D8: unlabelled beats a hard failure when the metadata is unusable
        print(f"[zenodo] WARNING {meta_name} has {len(meta)} rows for {n} spectra; "
              "treating the spectra as unlabelled")
        return labels, groups
    if "labels" in meta:
        labels = meta["labels"].tolist()
    if "batches" in meta:
        groups = meta["batches"].astype(str).tolist()
    return labels, groups


def read_bacteria(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        data = _member(zf, "data.csv")
        if data is None:
            raise ValueError(f"{zip_path}: data.csv not found")
        with zf.open(data) as fh:
            df = pd.read_csv(fh, index_col=0)
        axis = df.iloc[0].to_numpy(dtype=float)
        X = df.iloc[1:].to_numpy(dtype=float)
        meta = next((m for m in (_member(zf, b) for b in META_NAMES) if m), None)
        labels, groups = _labels_groups(zf, meta, len(X), "guo_bacteria")
    return axis, X, labels, groups


def read_oil(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        spec, meta = _member(zf, "spec.csv"), _member(zf, "metadata.csv")
        if spec is None or meta is None:
            raise ValueError(f"{zip_path}: spec.csv or metadata.csv not found")
        with zf.open(spec) as fh:
            df = pd.read_csv(fh, header=None)
        axis = df.iloc[0].to_numpy(dtype=float)
        X = df.iloc[1:].to_numpy(dtype=float)
        labels, groups = _labels_groups(zf, meta, len(X), "guo_oil")
    return axis, X, labels, groups


class _Guo:
    licence = "CC BY 4.0"
    url = "https://doi.org/10.5281/zenodo.15279321"
    probe_label_kind = None

    def download(self, raw_dir):
        fetch(BASE_URL + self.zip_name, Path(raw_dir) / self.zip_name, retries=5)

    def iter_batches(self, raw_dir):
        axis, X, labels, groups = self.reader(Path(raw_dir) / self.zip_name)
        yield RawBatch(axes=axis, spectra=X, sample_id=groups, label=labels,
                       laser_nm=float("nan"))


class ZenodoBacteria(_Guo):
    name, domain, zip_name = "zenodo_bacteria", "bio", "Bacterial_data.zip"
    reader = staticmethod(read_bacteria)


class ZenodoOil(_Guo):
    name, domain, zip_name = "zenodo_oil", "chem", "Oil_data.zip"
    reader = staticmethod(read_oil)


register(ZenodoBacteria())
register(ZenodoOil())
