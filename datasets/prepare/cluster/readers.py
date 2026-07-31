'''
Generic file readers, so a new benchmark is a registry entry rather than a
bespoke loader.

Paths are relative to `datasets/` and may be globs, which is what lets three
sources with the same underlying quirk — a categorical axis encoded as one file
per value — be handled by config alone: WMDP's bio/chem/cyber parquet, CySecBench's
ten per-category CSVs, InstrumentalEval's six task CSVs.
'''

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Sequence

import pandas as pd

DATASETS_DIR = Path(__file__).resolve().parent.parent.parent


def _frame(records, record_path: str | None) -> pd.DataFrame:
    '''Records may be dicts (one column each) or scalars (a single column).'''
    records = list(records)
    if records and not isinstance(records[0], dict):
        return pd.DataFrame({record_path or "value": records})
    return pd.DataFrame(records)


def _read_one(
    path: Path, reader: str, columns: Sequence[str] | None, record_path: str | None
) -> pd.DataFrame:
    if reader == "csv":
        if columns:
            return pd.read_csv(path, header=None, names=list(columns), dtype=str)
        return pd.read_csv(path, dtype=str)
    if reader == "parquet":
        return pd.read_parquet(path)
    if reader == "jsonl":
        with open(path, encoding="utf-8") as f:
            return pd.DataFrame([json.loads(line) for line in f if line.strip()])
    if reader == "json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return _frame(data[record_path] if record_path else data, record_path)
    if reader == "pylist":
        # Some upstreams ship their data inside a script rather than a data
        # file. Pull the named list literal out with ast so it stays in sync
        # with the source instead of being copied into this repo.
        source = path.read_text(encoding="utf-8")
        match = re.search(rf"{re.escape(record_path or '')}\s*=\s*(\[.*?\])", source, re.DOTALL)
        if not match:
            raise ValueError(f"no list literal named {record_path!r} in {path}")
        return _frame(ast.literal_eval(match.group(1)), record_path)
    raise ValueError(f"unknown reader {reader!r}")


def read(
    path: str,
    reader: str,
    *,
    columns: Sequence[str] | None = None,
    record_path: str | None = None,
    filename_field: str | None = None,
    dirname_field: str | None = None,
    first_row_field: str | None = None,
) -> pd.DataFrame:
    '''
    Load `path` (relative to datasets/, glob allowed) as a DataFrame.

    record_path     — key to index into before building the frame, for nested
                      JSON ({"leaders": [...]}) or a named list inside a .py.
    filename_field  — store each file's stem in this column. Turns "one file per
                      category" into an ordinary categorical column.
    dirname_field   — same, but from the parent directory name, for sources that
                      shard by folder instead (WMDP's wmdp-bio/, wmdp-chem/ ...).
    first_row_field — take the first row's first cell, store it in this column
                      for every row of that file, and drop it from the data.
                      InstrumentalEval's CSVs put the eval prompt there.
    '''
    matches = sorted(DATASETS_DIR.glob(path)) if any(c in path for c in "*?[") else [DATASETS_DIR / path]
    missing = [p for p in matches if not p.exists()]
    if not matches or missing:
        raise FileNotFoundError(f"no files matched {path!r} under {DATASETS_DIR}")

    frames = []
    for file_path in matches:
        frame = _read_one(file_path, reader, columns, record_path)
        if first_row_field:
            frame = frame.reset_index(drop=True)
            frame[first_row_field] = str(frame.iloc[0, 0])
            frame = frame.iloc[1:]
        if filename_field:
            frame[filename_field] = file_path.stem
        if dirname_field:
            frame[dirname_field] = file_path.parent.name
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)
