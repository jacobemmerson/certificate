#!/usr/bin/env python3
"""
Prepare flat CSV datasets for the LLM Human Rights (LHR) benchmark.

Sources (never modified):
  benchmarks/llm-human-rights/data/experiments/synthetic/udhr/scenarios/en.csv
  benchmarks/llm-human-rights/data/experiments/synthetic/echr/scenarios/en.csv

Outputs:
  benchmarks/datasets/public/lhr_udhr.csv   -- 144 UDHR scenarios (RQ: UDHR articles)
  benchmarks/datasets/public/lhr_echr.csv   -- 102 ECHR scenarios (RQ: ECHR articles)

The source CSVs are already clean and structured; this script copies them into the
shared public dataset directory so the Inspect script has a stable, canonical path.
No transformation is applied — the output is byte-identical to the source.

Usage:
  python scripts/prepare_lhr_data.py
"""
import csv
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LHR_DIR = REPO_ROOT / "benchmarks" / "llm-human-rights"
OUT_DIR = REPO_ROOT / "benchmarks" / "datasets" / "public"

SOURCES = {
    "lhr_udhr.csv": LHR_DIR / "data" / "experiments" / "synthetic" / "udhr" / "scenarios" / "en.csv",
    "lhr_echr.csv": LHR_DIR / "data" / "experiments" / "synthetic" / "echr" / "scenarios" / "en.csv",
}

EXPECTED_COLUMNS = {
    "limited_article",
    "limited_article_name",
    "limited_article_text",
    "actor",
    "affected_group",
    "severity",
    "state_of_emergency",
    "scenario_id",
    "scenario_text",
}


def _validate(path: Path, label: str) -> int:
    """Check that the source CSV has the expected columns and return its row count."""
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        missing = EXPECTED_COLUMNS - cols
        if missing:
            raise ValueError(f"{label}: missing expected columns: {missing}")
        rows = sum(1 for _ in reader)
    return rows


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating LHR datasets …")

    for dest_name, src_path in SOURCES.items():
        if not src_path.exists():
            raise FileNotFoundError(f"Source file not found: {src_path}")

        label = dest_name.replace(".csv", "").upper()
        rows = _validate(src_path, label)

        dest_path = OUT_DIR / dest_name
        shutil.copy2(src_path, dest_path)
        print(f"  wrote {rows:>5} rows → {dest_path.relative_to(REPO_ROOT)}")

    print("Done.")
