"""Competition data audit with no task-specific assumptions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def find_tables(data_dir: Path) -> list[Path]:
    return sorted(
        p for p in data_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".csv", ".tsv", ".parquet"}
    )


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, nrows=200_000)
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t", nrows=200_000)
    return pd.read_parquet(path).head(200_000)


def profile(path: Path) -> dict:
    frame = read_table(path)
    return {
        "file": str(path),
        "sampled_rows": int(len(frame)),
        "columns": list(frame.columns),
        "dtypes": {key: str(value) for key, value in frame.dtypes.items()},
        "missing_fraction": {
            key: round(float(value), 6) for key, value in frame.isna().mean().items()
        },
        "duplicate_rows": int(frame.duplicated().sum()),
        "unique_fraction": {
            key: round(float(value), 6) for key, value in frame.nunique(dropna=False).div(len(frame)).items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("outputs/data_audit.json"))
    args = parser.parse_args()

    tables = find_tables(args.data_dir)
    if not tables:
        raise SystemExit(f"No CSV, TSV, or Parquet files found under {args.data_dir}")

    report = {"tables": [profile(table) for table in tables]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
