'''
Keeps datasets/BENCHMARKS.md in step with the data it describes.

A benchmark table is worth having only if its numbers are true. Sample counts
change whenever a quota moves or a dedup threshold is retuned, and a hand-edited
doc goes stale on the first rebuild — silently, because nothing reads it.

These tests check the mechanical claims: every registered source is documented,
every documented count matches the built CSV, and the scoring-shape table lists
the shapes the code actually dispatches on. The prose — what each original
benchmark does, and how we diverge — cannot be tested and is cited instead.

Run: uv run python3 -m unittest discover tests
'''

import csv
import json
import re
import unittest
from collections import Counter
from pathlib import Path

from datasets.prepare.cluster.schema import QUESTION_TYPES
from datasets.prepare.cluster.sources import RISKS, SOURCES

REPO_ROOT = Path(__file__).resolve().parent.parent
DOC = REPO_ROOT / "datasets" / "BENCHMARKS.md"
PUBLIC_DIR = REPO_ROOT / "datasets" / "public"

# `| `name` | 60 / 419 | graded | ...`
ROW = re.compile(r"^\|\s*`(?P<name>\w+)`\s*\|\s*(?P<kept>[\d,]+)\s*/\s*(?P<loaded>[\d,]+)\s*\|")


def documented_rows() -> dict[str, tuple[int, int]]:
    rows = {}
    for line in DOC.read_text().splitlines():
        match = ROW.match(line)
        if match:
            rows[match["name"]] = (
                int(match["kept"].replace(",", "")),
                int(match["loaded"].replace(",", "")),
            )
    return rows


class TestBenchmarksDoc(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not DOC.exists():
            raise unittest.SkipTest("BENCHMARKS.md not written")
        cls.documented = documented_rows()

    def test_every_registered_source_is_documented(self):
        self.assertEqual(
            {source.name for source in SOURCES} - set(self.documented), set()
        )

    def test_no_documented_source_has_been_removed(self):
        self.assertEqual(
            set(self.documented) - {source.name for source in SOURCES}, set()
        )

    def test_kept_counts_match_the_built_clusters(self):
        for risk in RISKS:
            path = PUBLIC_DIR / f"{risk}.csv"
            if not path.exists():
                continue
            with open(path, newline="") as f:
                built = Counter(row["source"] for row in csv.DictReader(f))
            for source, count in built.items():
                with self.subTest(source=source):
                    self.assertEqual(
                        self.documented[source][0], count,
                        f"BENCHMARKS.md says {self.documented[source][0]} "
                        f"for {source}, built CSV has {count}",
                    )

    def test_loaded_counts_match_the_build_metadata(self):
        for risk in RISKS:
            meta_path = PUBLIC_DIR / f"{risk}.meta.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text())["sources"]
            for source, report in meta.items():
                with self.subTest(source=source):
                    self.assertEqual(self.documented[source][1], report["loaded"])

    def test_cluster_totals_match(self):
        text = DOC.read_text()
        for risk in RISKS:
            path = PUBLIC_DIR / f"{risk}.csv"
            if not path.exists():
                continue
            with open(path, newline="") as f:
                total = sum(1 for _ in csv.DictReader(f))
            with self.subTest(risk=risk):
                self.assertRegex(
                    text, rf"## {risk} — {total} samples",
                    f"{risk} heading does not say {total}",
                )

    def test_every_question_type_is_explained(self):
        text = DOC.read_text()
        for question_type in QUESTION_TYPES:
            with self.subTest(question_type=question_type):
                self.assertIn(f"`{question_type}`", text)


if __name__ == "__main__":
    unittest.main()
