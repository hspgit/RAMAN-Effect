"""RRUFF mineral spectra (spec section 4.1; plan deviation D4).

https://www.rruff.net/zipped_data_files/raman/ : flat .txt files named
Mineral__RRUFFID[-n]__{Raman|Broad_Scan}__LASER__ORIENT__POL__Raman_Data_{Processed|RAW|Raw}__hash.txt
with '##KEY=value' headers, 'x, y' rows and a final '##END='.

Only RAW files are kept: every specimen also ships a Processed copy, which is
baseline-corrected (spec: no baseline correction) and would duplicate it.
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

import numpy as np

from ..harmonize import AXIS_MAX, AXIS_MIN
from .base import RawBatch, fetch, register

BASE_URL = "https://www.rruff.net/zipped_data_files/raman/"
ZIPS = ("excellent_unoriented.zip", "fair_unoriented.zip", "LR-Raman.zip")
_LASER = re.compile(r"(\d{3})")
_SPLIT = re.compile(r"[,\s]+")


def parse_rruff_text(text):
    header, xs, ys = {}, [], []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("##"):
            key, _, val = line[2:].partition("=")
            if key.strip() == "END":
                break
            header[key.strip()] = val.strip()
            continue
        parts = [p for p in _SPLIT.split(line) if p]
        if len(parts) < 2:
            continue
        try:
            x, y = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        xs.append(x)
        ys.append(y)
    return header, np.asarray(xs), np.asarray(ys)


def parse_laser(*candidates):
    for c in candidates:
        m = _LASER.search(c or "")
        if m:
            return float(m.group(1))
    return float("nan")


def parse_filename(name):
    fields = Path(name).name.split("__")
    if len(fields) != 8:
        return None
    return {"mineral": fields[0], "rruff_id": fields[1], "laser": fields[3],
            "data_type": fields[6]}


class RRUFF:
    name = "rruff"
    domain = "mineral"
    licence = "CC BY-SA 4.0 per FAIRsharing (RRUFF's own terms unclear); cite Lafuente et al. 2015"
    url = BASE_URL
    probe_label_kind = "mineral species"

    def download(self, raw_dir):
        for z in ZIPS:
            fetch(BASE_URL + z, Path(raw_dir) / z, retries=5)

    def iter_batches(self, raw_dir):
        axes, spectra, ids, labels, lasers = [], [], [], [], []
        for z in ZIPS:
            with zipfile.ZipFile(Path(raw_dir) / z) as zf:
                for member in zf.namelist():
                    info = parse_filename(member)
                    if (not member.lower().endswith(".txt") or info is None
                            or not info["data_type"].lower().endswith("_raw")):
                        continue
                    header, x, y = parse_rruff_text(zf.read(member).decode("utf-8", "replace"))
                    if x.size < 2 or x.min() < AXIS_MIN or x.max() > AXIS_MAX:
                        continue
                    rid = header.get("RRUFFID") or info["rruff_id"]
                    axes.append(x)
                    spectra.append(y)
                    ids.append(re.sub(r"-\d+$", "", rid))
                    labels.append(header.get("NAMES") or info["mineral"])
                    lasers.append(parse_laser(header.get("RAMAN WAVELENGTH"), info["laser"]))
        yield RawBatch(axes=axes, spectra=spectra, sample_id=ids, label=labels, laser_nm=lasers)


register(RRUFF())
