'''
Merge prompt CSVs from the PreservingHistoricalTruth benchmark into a single
flat file for the AISI Inspect evaluation pipeline.

Input:
    benchmarks/PreservingHistoricalTruth/data/neutral_prompt.csv    (no_push)
    benchmarks/PreservingHistoricalTruth/data/revisionist_prompts.csv (explicit_push)

Output:
    benchmarks/datasets/public/pht_prompts.csv

Run from repo root:
    python scripts/prepare_pht_data.py
'''

import csv
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "benchmarks" / "PreservingHistoricalTruth" / "data"
OUT_PATH = REPO_ROOT / "benchmarks" / "datasets" / "public" / "pht_prompts.csv"

SOURCES = [
    DATA_DIR / 'neutral_prompt.csv',
    DATA_DIR / 'revisionist_prompts.csv',
]

OUT_COLUMNS = [
    'sample_id', 'case_id', 'Historical Event', 'True Version', 'False Version',
    'Country/Region', 'Source', 'Historical Period', 'Push Level',
    'Scenario', 'Prompt', 'Dataset',
]


def main():
    rows = []
    for src in SOURCES:
        with open(src, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                rows.append(row)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=OUT_COLUMNS)
        writer.writeheader()
        for idx, row in enumerate(rows):
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

    push_counts = Counter(r.get('Push Level', '') for r in rows)
    print(f'Written {len(rows)} rows to {OUT_PATH}')
    for level, count in sorted(push_counts.items()):
        print(f'  {level}: {count}')


if __name__ == '__main__':
    main()
