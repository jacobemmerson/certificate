'''
Keeps datasets/BENCHMARKS.md in step with the data it describes.

Sample counts live in each cluster's meta.json, not here: a hand-edited count
goes stale on the first rebuild, silently, because nothing reads it.

These tests check the mechanical claims: every registered source is documented
with its true question type, and the scoring-shape table lists
the shapes the code actually dispatches on. The prose — what each original
benchmark does, and how we diverge — cannot be tested and is cited instead.

Run: uv run python3 -m unittest discover tests
'''

import re
import unittest
from pathlib import Path

from datasets.prepare.cluster.schema import QUESTION_TYPES
from datasets.prepare.cluster.sources import SOURCES

REPO_ROOT = Path(__file__).resolve().parent.parent
DOC = REPO_ROOT / "datasets" / "BENCHMARKS.md"

# `| `name` | graded | ...`
ROW = re.compile(r"^\|\s*`(?P<name>\w+)`\s*\|\s*(?P<question_type>graded|mcq|likert|extraction|detection)\s*\|")


def documented_rows() -> dict[str, str]:
    rows = {}
    for line in DOC.read_text().splitlines():
        match = ROW.match(line)
        if match:
            rows[match["name"]] = match["question_type"]
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

    def test_documented_question_type_matches_the_source(self):
        for source in SOURCES:
            with self.subTest(source=source.name):
                self.assertEqual(self.documented[source.name], source.question_type)

    def test_every_question_type_is_explained(self):
        text = DOC.read_text()
        for question_type in QUESTION_TYPES:
            with self.subTest(question_type=question_type):
                self.assertIn(f"`{question_type}`", text)


if __name__ == "__main__":
    unittest.main()
