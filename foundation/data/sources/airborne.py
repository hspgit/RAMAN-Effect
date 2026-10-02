"""Airborne pathogens, single-cell Raman (Sci. Adv. 2025; ScienceDB
10.57760/sciencedb.15628; CC BY-NC 4.0). 532 nm, LabRAM Aramis.

No bundle: one direct URL per file from the ScienceDB file-tree API. Each file
is one cell: 225 rows 'x y', axis 608-1802.67 cm^-1 at ~5.33 cm^-1. The
source already baseline-corrected and min-max scaled these (plan deviation
D2). Test_set_3/Unknown ('U_aug_ N.txt') holds augmented/blended spectra and
is excluded.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import unquote

import numpy as np

from .base import RawBatch, fetch, register

DATASET_ID = "5a1b7adb362d4064a3dedd01e5c40d90"
LIST_URL = ("https://www.scidb.cn/api/sdb-filetree-service/getAllUrl"
            f"?dataSetId={DATASET_ID}&type=&version=V1&global=")
SPECIES = ("A.baylyi", "E.coli", "P.aeruginosa", "S.aureus", "S.enterica")
# scidb.cn returned HTTP 429 after ~600 files at 8 parallel connections; one
# connection ran 300 files at ~0.46/s with no throttling. 3 stays well below
# the limit, and fetch() backs off if the server still throttles.
WORKERS = 3
# 'path' may contain a raw '&' (e.g. 'Training_set_1&Test_set_1'), so a normal
# query-string parser would cut it short. fileName is always the last parameter.
_PATH_NAME = re.compile(r"[?&]path=(?P<path>.*)&fileName=(?P<name>[^&]*)$")


def local_relpath(url):
    m = _PATH_NAME.search(url.strip())
    if not m:
        raise ValueError(f"unexpected download URL: {url}")
    path, name = unquote(m.group("path")), unquote(m.group("name"))
    parts = [p for p in path.replace("\\", "/").split("/") if p]
    if parts and parts[-1] == name:
        parts = parts[:-1]
    return "/".join(parts + [name])


def classify(relpath):
    parts = relpath.split("/")
    name = parts[-1]
    if not name.lower().endswith(".txt"):
        return False, None
    if "U_aug" in name or ("Test_set_3" in parts and "Unknown" in parts):
        return False, None
    label = next((p for p in parts[:-1] if p in SPECIES), None)
    return True, label


def _kept(raw_dir):
    lines = (Path(raw_dir) / "urls.txt").read_text(encoding="utf-8").splitlines()
    for url in (u.strip() for u in lines):
        if not url:
            continue
        rel = local_relpath(url)
        keep, label = classify(rel)
        if keep:
            yield url, rel, label


class Airborne:
    name = "airborne"
    domain = "bio"
    licence = "CC BY-NC 4.0"
    url = "https://doi.org/10.57760/sciencedb.15628"
    probe_label_kind = "pathogen species"

    def download(self, raw_dir):
        raw_dir = Path(raw_dir)
        fetch(LIST_URL, raw_dir / "urls.txt", retries=5)
        jobs = [(url, raw_dir / "files" / rel) for url, rel, _ in _kept(raw_dir)]
        todo = [(u, p) for u, p in jobs if not p.exists()]
        print(f"[airborne] {len(jobs)} files kept, {len(todo)} to download")
        failures = []

        def get(job):
            try:
                fetch(job[0], job[1], retries=5)
            except Exception as e:  # finish the rest first; report below
                failures.append((job[1], e))

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            for i, _ in enumerate(pool.map(get, todo), 1):
                if i % 500 == 0:
                    print(f"[airborne] {i}/{len(todo)}", flush=True)
        if failures:
            raise RuntimeError(
                f"[airborne] {len(failures)} of {len(todo)} downloads failed "
                f"(first: {failures[0][0].name}: {failures[0][1]}); "
                "rerun to resume, finished files are kept")

    def iter_batches(self, raw_dir):
        axes, spectra, ids, labels = [], [], [], []
        for _, rel, label in _kept(raw_dir):
            arr = np.loadtxt(Path(raw_dir) / "files" / rel)
            axes.append(arr[:, 0])
            spectra.append(arr[:, 1])
            ids.append(rel)
            labels.append(label)
        yield RawBatch(axes=axes, spectra=spectra, sample_id=ids, label=labels, laser_nm=532)


register(Airborne())
