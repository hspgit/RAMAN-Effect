"""Phase 0 audit and gate G0 (spec section 7).

    python -m foundation.data.audit --corpus foundation/data/corpus_v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

COLUMNS = ["status", "domain", "n_raw", "n_unusable", "n_leaked", "n_capped", "n_duplicate",
           "n_probe_isolation", "n_pretrain", "n_probe_train", "n_probe_test",
           "native_min", "native_max", "lasers", "licence"]


def summarize(corpus_dir):
    manifest = json.loads((Path(corpus_dir) / "manifest.json").read_text())
    table = pd.DataFrame.from_dict(manifest["sources"], orient="index")
    table = table.reindex(columns=COLUMNS)
    return table, manifest["totals"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--min-pretrain", type=int, default=150_000)
    args = ap.parse_args(argv)
    table, totals = summarize(args.corpus)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(table.to_string())
    print(f"\ntotals: {totals}")
    n = totals["pretrain"]
    if n >= args.min_pretrain:
        print(f"G0 PASS: {n:,} pretraining spectra >= {args.min_pretrain:,}")
        return 0
    print(f"G0 FAIL: {n:,} pretraining spectra < {args.min_pretrain:,}. "
          "Stop and revisit the source list (spec section 7).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
