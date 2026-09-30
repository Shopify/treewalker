#!/usr/bin/env python3
"""Build paper/experiments/data/expedia.parquet from the Kaggle training file.

The Expedia data ("Personalize Expedia Hotel Searches - ICDM 2013",
https://www.kaggle.com/competitions/expedia-personalized-sort) may not be
redistributed. Accept the competition rules on Kaggle, download data.zip, e.g.

    kaggle competitions download -c expedia-personalized-sort -f data.zip

and convert it (from the repository root):

    uv run python3 paper/experiments/scripts/fetch_expedia.py --train-csv data.zip

PATH may be train.csv or Kaggle's data.zip. The zip uses Deflate64, which
Python's zipfile cannot read, so the script extracts it with the system
`unzip` (Info-ZIP). The script keeps the 23 columns
that prepare.py uses, stores them with the dtypes the paper's runs used, and
checks a content fingerprint of the result against the file behind the
paper (9,917,530 rows, 399,344 search sessions).

    uv run python3 paper/experiments/scripts/fetch_expedia.py --check-only

verifies an existing parquet without rebuilding it.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "paper" / "experiments" / "data" / "expedia.parquet"

# Column order and dtypes of the parquet the paper was run on.
SCHEMA: dict[str, pl.DataType] = {
    "srch_id": pl.UInt32,
    "site_id": pl.UInt8,
    "visitor_location_country_id": pl.UInt8,
    "prop_country_id": pl.UInt8,
    "prop_id": pl.UInt32,
    "prop_starrating": pl.UInt8,
    "prop_review_score": pl.Float32,
    "prop_brand_bool": pl.Boolean,
    "prop_location_score1": pl.Float32,
    "prop_location_score2": pl.Float32,
    "prop_log_historical_price": pl.Float32,
    "position": pl.UInt8,
    "price_usd": pl.Float32,
    "promotion_flag": pl.Boolean,
    "srch_destination_id": pl.UInt16,
    "srch_length_of_stay": pl.UInt8,
    "srch_booking_window": pl.UInt16,
    "srch_adults_count": pl.UInt8,
    "srch_children_count": pl.UInt8,
    "srch_room_count": pl.UInt8,
    "srch_saturday_night_bool": pl.Boolean,
    "random_bool": pl.Boolean,
    "click_bool": pl.Boolean,
}

EXPECTED_ROWS = 9_917_530
EXPECTED_SESSIONS = 399_344
EXPECTED_FINGERPRINT = "471367a93277c91df3d99a815584c5f45a8acec2f17d84643f876989e29395a8"


def fingerprint(df: pl.DataFrame) -> str:
    """Library-version-independent content hash: names, dtypes, null masks, values."""
    h = hashlib.sha256()
    for c in df.columns:
        s = df[c]
        h.update(c.encode())
        h.update(str(s.dtype).encode())
        h.update(s.is_null().to_numpy().tobytes())
        v = s.fill_null(False) if s.dtype == pl.Boolean else s.fill_null(0)
        h.update(np.ascontiguousarray(v.to_numpy()).tobytes())
    return h.hexdigest()


def convert(csv_path: Path) -> pl.DataFrame:
    # Booleans are 0/1 in the CSV: parse as UInt8, then cast. Floats parse
    # directly to Float32. Missing values are the literal string "NULL".
    read_types = {c: (pl.UInt8 if t == pl.Boolean else t) for c, t in SCHEMA.items()}
    df = pl.read_csv(csv_path, columns=list(SCHEMA), schema_overrides=read_types,
                     null_values="NULL")
    return df.select([pl.col(c).cast(t) for c, t in SCHEMA.items()])


def check(df: pl.DataFrame) -> bool:
    rows, sessions = df.height, df["srch_id"].n_unique()
    fp = fingerprint(df)
    print(f"rows={rows:,} sessions={sessions:,} fingerprint={fp}")
    ok = (rows == EXPECTED_ROWS and sessions == EXPECTED_SESSIONS
          and fp == EXPECTED_FINGERPRINT)
    print("MATCH: identical to the data behind the paper" if ok else
          "MISMATCH: the Expedia results will not reproduce exactly")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--train-csv", type=Path, help="Kaggle train.csv or its .zip")
    g.add_argument("--check-only", action="store_true",
                   help=f"verify an existing {OUT.relative_to(ROOT)}")
    args = ap.parse_args()

    if args.check_only:
        return 0 if check(pl.read_parquet(OUT)) else 1

    src = args.train_csv
    with tempfile.TemporaryDirectory() as tmp:
        if src.suffix == ".zip":
            if shutil.which("unzip") is None:
                sys.exit("unzip not found; extract train.csv from the zip and pass its path")
            subprocess.run(["unzip", "-o", "-q", "-j", str(src), "train.csv", "-d", tmp],
                           check=True)
            src = Path(tmp) / "train.csv"
        df = convert(src)
    ok = check(df)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT)
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
