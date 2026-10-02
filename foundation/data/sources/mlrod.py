"""MLROD, Mars-analogue minerals (Berlanga et al. 2022; CC BY-NC; DOI 10.48484/PWRB-R137).

41 CSVs from odr.io. File list and integer->mineral map copied from the
raman_data package (MIT, github.com/ml-lab-htw/raman_data @ c92289e,
MiscLoader._load_mlrod). Layout: header 'og-idx,<wavenumbers...>,label', one
row per spectrum; the axis differs per file; one header token is known to be
malformed ('1630.89.1'), so tokens are parsed by their leading number.
532 nm Horiba LabRAM HR Evolution, raw spectra. Group = file (one acquisition set).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .base import RawBatch, fetch, register

_ODR = "https://www.odr.io/view/downloadfile/{}?error_type=json"
FILES = {name: _ODR.format(i) for name, i in {
    "qtzalb_train_60s_3000.csv": 63653, "qtzalb_train_30s_2000.csv": 63652,
    "qtz_train_003s_4760.csv": 63758, "qtz_train_003s_1148.csv": 63757,
    "qtz_train_015s_625.csv": 63756, "qtz_train_005s_1000.csv": 63754,
    "qtz_train_005s_0900.csv": 63753, "qtz_train_003s-1599.csv": 63752,
    "ms_train_015s_5900.csv": 63755, "mc_train_015s_3243.csv": 63750,
    "mc_train_3s_2357.csv": 63651, "hb_train_120s_1225_01.csv": 63751,
    "hb_train_015s_2500.csv": 63749, "hb_train_120s_1225_02.csv": 63748,
    "hb_train_060s_104.csv": 63650, "gyp_train_12s_3320.csv": 63746,
    "gyp_train_15s_1680.csv": 63649, "foaug_train_60s_3520.csv": 63648,
    "foaug_train_30s_1640.csv": 63647, "foalb_train_15s_5220.csv": 63745,
    "fo_train_030s_5040.csv": 63747, "fo_train_015s_1100.csv": 63646,
    "en_train_015s_5600.csv": 63744, "cal_train_1s_2732.csv": 63743,
    "cal_train_1s_2300.csv": 63645, "bt_train_120s_1225.csv": 63741,
    "bt_train_015s_5600.csv": 63644, "bt_train_015s_1100.csv": 63643,
    "aug_train_20s_5103.csv": 63738, "aug_train_15s_1000.csv": 63642,
    "an_train_015s_5600.csv": 63740, "albite_train_015s_5250.csv": 63739,
    "albite_train_015s_625.csv": 63737, "albite_train_015s_1100.csv": 63641,
    "L_raw_gabbro50dust_test_015s_9740.csv": 63659,
    "L_raw_granite50dust_test_015s_10000.csv": 63245,
    "L_raw_granite0dust_test_015s_4084_final.csv": 63658,
    "L_raw_granite0dust_test_030s_1373_final.csv": 63657,
    "L_raw_granite0dust_test_015s_5571_final.csv": 63656,
    "L_raw_gabbro0dust_test_015s_8906_final.csv": 63655,
    "L_raw_gabbro0dust_test_015s_46_final.csv": 63654,
}.items()}

LABELS = {
    0: "Quartz", 1: "Feldspar - Albite", 2: "Amphibole - Hornblende", 3: "Mica - Biotite",
    4: "Mica - Muscovite", 5: "Olivine - Forsterite", 6: "Pyroxene - Augite",
    7: "Pyroxene - Enstatite", 8: "Feldspar - Anorthite", 9: "Feldspar - Microcline",
    10: "Carbonate - Calcite", 11: "Sulfate - Gypsum", 12: "50% Quartz / 50% Albite Mix",
    13: "50% Forsterite / 50% Augite Mix", 14: "50% Forsterite / 50% Albite Mix",
    15: "Granite slab - 50% dust",
}

_NUM = re.compile(r"^\s*(-?\d+(?:\.\d+)?)")


def parse_header_axis(tokens):
    vals = []
    for t in tokens:
        m = _NUM.match(t)
        if not m:
            raise ValueError(f"unparseable wavenumber {t!r}")
        vals.append(float(m.group(1)))
    return np.array(vals)


def parse_mlrod_csv(path):
    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        header = fh.readline().strip().split(",")
    if header[0].strip() != "og-idx" or header[-1].strip() != "label":
        raise ValueError(f"{path}: unexpected header {header[0]!r}...{header[-1]!r}")
    axis = parse_header_axis(header[1:-1])
    df = pd.read_csv(path, header=None, skiprows=1)
    X = df.iloc[:, 1:-1].to_numpy(dtype=np.float64)
    if X.shape[1] != axis.size:
        raise ValueError(f"{path}: {X.shape[1]} intensity columns for {axis.size} wavenumbers")
    y = df.iloc[:, -1].astype(int).tolist()
    return axis, X, y


class MLROD:
    name = "mlrod"
    domain = "mineral"
    licence = "CC BY-NC"
    url = "https://doi.org/10.48484/PWRB-R137"
    probe_label_kind = "mineral"

    def download(self, raw_dir):
        for fname, url in FILES.items():
            dest = fetch(url, Path(raw_dir) / fname, retries=5)
            # odr.io can answer HTTP 200 with an error body instead of the CSV
            with open(dest, encoding="utf-8-sig", errors="replace") as fh:
                ok = fh.readline().startswith("og-idx")
            if not ok:
                Path(dest).unlink()
                raise IOError(f"{fname}: download is not an MLROD CSV (no 'og-idx' header)")

    def iter_batches(self, raw_dir):
        for fname in FILES:
            axis, X, y = parse_mlrod_csv(Path(raw_dir) / fname)
            yield RawBatch(axes=axis, spectra=X, sample_id=[Path(fname).stem] * len(X),
                           label=[LABELS.get(v, str(v)) for v in y], laser_nm=532)


register(MLROD())
