'''
author: @tae

Run the rewrite validator over artifacts already on disk.

`rewrite_defect` gates generation from now on, but the artifact set that
produced the current models.json predates it. Regenerating that set (and
re-running every target model against it) is not affordable, so this reports
which stored variants the gate would have rejected. The output is the
quarantine list: the (task, family, sample) triples whose scores rest on a
prompt that is not a faithful rewrite of its source.

    uv run python3 scripts/audit_rewrites.py [--write quarantine.json]

Counts here are a lower bound on damage and an upper bound on cost: a rejected
variant costs one regeneration, an accepted bad one corrupts a published score.
'''

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.artifacts import REPEAT_FAMILIES
from pipeline.stage2_perturbation.rewrite import rewrite_defect

ROOT = Path(__file__).resolve().parent.parent
# Only the repeat families go through rewrite_defect: framing and persona are
# template-built, multilingual has its own translation_defect, and scenario has
# its own schema.


def load_items() -> tuple[dict[str, str], dict[str, list[str]]]:
    '''sample_id -> the span a perturbation was allowed to reword, and ->
    the constructs (if any) that span must still contain after rewriting.'''
    csv.field_size_limit(10 ** 7)
    items: dict[str, str] = {}
    must_survive: dict[str, list[str]] = {}
    for path in sorted((ROOT / "datasets" / "public").glob("*.csv")):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                sample_id = row.get("sample_id")
                if sample_id:
                    items[sample_id] = (row.get("item_text") or row.get("query") or "").strip()
                    metadata = json.loads(row.get("metadata") or "{}")
                    if metadata.get("must_survive"):
                        must_survive[sample_id] = metadata["must_survive"]
    return items, must_survive


def audit() -> tuple[list[dict], Counter, int]:
    items, must_survive = load_items()
    rejected: list[dict] = []
    seen = 0
    for path in sorted((ROOT / "datasets" / "generated").glob("*/*.jsonl")):
        family = path.stem
        if family not in REPEAT_FAMILIES:
            continue
        task = path.parent.name
        for line in path.open():
            row = json.loads(line)
            original = items.get(row["id"])
            if original is None:
                continue
            seen += 1
            # A fallback row already carries the original text: it was never a
            # rewrite, and re-flagging it would double-count a known no-op.
            if row.get("fallback"):
                continue
            defect = rewrite_defect(
                row.get("text") or "", original, must_survive.get(row["id"]) or ()
            )
            if defect:
                rejected.append({
                    "task": task, "family": family, "id": row["id"],
                    "variant": row.get("variant"), "defect": defect,
                })
    return rejected, Counter(r["defect"] for r in rejected), seen


def main() -> None:
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--write", type=Path, default=None,
                      help="Write the quarantine list to this JSON file.")
    options = args.parse_args()

    rejected, by_defect, seen = audit()
    print(f"{len(rejected)} of {seen} stored rewrites would be rejected "
          f"({100 * len(rejected) / seen:.1f}%)\n")

    print("by defect")
    for defect, count in by_defect.most_common():
        print(f"    {defect:10s} {count:5d}")

    per_task: dict[str, Counter] = defaultdict(Counter)
    for row in rejected:
        per_task[row["task"]][row["family"]] += 1
    print("\nby cluster and family")
    for task in sorted(per_task):
        for family, count in sorted(per_task[task].items()):
            print(f"    {task:16s} {family:14s} {count:5d}")

    affected = {(r["task"], r["id"]) for r in rejected}
    print(f"\n{len(affected)} distinct (cluster, sample) pairs affected")

    if options.write:
        options.write.write_text(json.dumps(rejected, indent=1) + "\n")
        print(f"\nwrote {options.write}")


if __name__ == "__main__":
    main()
