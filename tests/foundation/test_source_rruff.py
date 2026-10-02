import math
import zipfile

import numpy as np

from foundation.data.sources import REGISTRY
from foundation.data.sources.rruff import parse_laser, parse_rruff_text


def _txt(header, xs, ys):
    lines = [f"##{k}={v}" for k, v in header.items()] + [""]
    lines += [f"{x:.4f}, {y:.7E}" for x, y in zip(xs, ys)] + ["##END="]
    return "\r\n".join(lines) + "\r\n"


def _make_zips(raw):
    raw.mkdir(parents=True)
    x = np.linspace(100.0, 1500.0, 800)
    corundum = _txt({"NAMES": "Corundum", "RRUFFID": "R060020", "RAMAN WAVELENGTH": "514"},
                    x, 1 + np.exp(-0.5 * ((x - 417) / 4) ** 2))
    with zipfile.ZipFile(raw / "excellent_unoriented.zip", "w") as zf:
        zf.writestr("Corundum__R060020-4__Raman__514__0-000____Raman_Data_RAW__5210.txt", corundum)
        zf.writestr("Corundum__R060020-4__Raman__514__0-000____Raman_Data_Processed__5211.txt",
                    corundum)
    with zipfile.ZipFile(raw / "fair_unoriented.zip", "w"):
        pass
    quartz = _txt({"NAMES": "Quartz"}, x, 1 + np.exp(-0.5 * ((x - 465) / 4) ** 2))
    bogus_x = np.linspace(11000.0, 12000.0, 300)
    bogus = _txt({"NAMES": "Bogus", "RRUFFID": "R1"}, bogus_x, np.ones_like(bogus_x))
    with zipfile.ZipFile(raw / "LR-Raman.zip", "w") as zf:
        zf.writestr("Quartz__R040031__Broad_Scan__532SP9__0____Raman_Data_Raw__9999.txt", quartz)
        zf.writestr("Bogus__R1__Raman__532__0____Raman_Data_RAW__1.txt", bogus)


def test_parse_text_reads_header_and_rows_until_end():
    text = _txt({"NAMES": "Corundum", "RRUFFID": "R060020"}, [100.0, 101.5], [3.0, 0.0])
    header, x, y = parse_rruff_text(text + "999, 999\r\n")
    assert header["NAMES"] == "Corundum"
    assert x.tolist() == [100.0, 101.5] and y.tolist() == [3.0, 0.0]


def test_parse_laser_handles_messy_values():
    assert parse_laser("632-8") == 632.0
    assert parse_laser("532nm") == 532.0
    assert parse_laser("", "785") == 785.0
    assert math.isnan(parse_laser("53", ""))


def test_iter_batches_keeps_raw_only_and_skips_implausible_axes(tmp_path):
    raw = tmp_path / "rruff"
    _make_zips(raw)
    (batch,) = list(REGISTRY["rruff"].iter_batches(raw))
    assert batch.label == ["Corundum", "Quartz"]
    assert batch.sample_id == ["R060020", "R040031"]
    assert batch.laser_nm.tolist() == [514.0, 532.0]
    assert np.all(np.diff(batch.axes[0]) > 0)
