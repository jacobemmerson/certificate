'''
Split prompt CSVs from the PreservingHistoricalTruth benchmark into separate
flat files per push level for the AISI Inspect evaluation pipeline.

Input:
    benchmarks/PreservingHistoricalTruth/data/neutral_prompt.csv    (no_push)
    benchmarks/PreservingHistoricalTruth/data/revisionist_prompts.csv (explicit_push)

Output:
    benchmarks/datasets/public/pht_no_push.csv
    benchmarks/datasets/public/pht_explicit_push.csv

Run from repo root:
    python scripts/prepare_pht_data.py
'''

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "benchmarks" / "PreservingHistoricalTruth" / "data"
OUT_DIR = REPO_ROOT / "benchmarks" / "datasets" / "public"

SOURCES = [
    DATA_DIR / 'neutral_prompt.csv',
    DATA_DIR / 'revisionist_prompts.csv',
]

OUT_COLUMNS = [
    'sample_id', 'case_id', 'Historical Event', 'True Version', 'False Version',
    'Country/Region', 'Source', 'Historical Period', 'Push Level',
    'Scenario', 'Prompt', 'Dataset',
]

PUSH_LEVEL_FILES = {
    'no_push': OUT_DIR / 'pht_no_push.csv',
    'explicit_push': OUT_DIR / 'pht_explicit_push.csv',
}


def main():
    rows = []
    for src in SOURCES:
        with open(src, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                rows.append(row)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    buckets: dict[str, list[dict]] = {level: [] for level in PUSH_LEVEL_FILES}
    for row in rows:
        level = row.get('Push Level', '')
        if level in buckets:
            buckets[level].append(row)

    for level, out_path in PUSH_LEVEL_FILES.items():
        level_rows = buckets[level]
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=OUT_COLUMNS)
            writer.writeheader()
            for idx, row in enumerate(level_rows):
                writer.writerow({
                    'sample_id': idx,
                    'case_id': row.get('case_id', ''),
                    'Historical Event': row.get('Historical Event', ''),
                    'True Version': row.get('True Version', ''),
                    'False Version': row.get('False Version', ''),
                    'Country/Region': row.get('Country/Region', ''),
                    'Source': row.get('Source', ''),
                    'Historical Period': row.get('Historical Period', ''),
                    'Push Level': row.get('Push Level', ''),
                    'Scenario': row.get('Scenario', ''),
                    'Prompt': row.get('Prompt', ''),
                    'Dataset': row.get('Dataset', ''),
                })
        print(f'Written {len(level_rows)} rows to {out_path}')


if __name__ == '__main__':
    main()
