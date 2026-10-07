'''datasets/raw/manifest.toml stays consistent with annotations.csv and the datasets tree.'''
import csv
import json
import re
import subprocess
import sys
import unittest

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from datasets.prepare.cluster.manifest import HOSTS, MANIFEST_PATH, NAME_PATTERN, STATUSES, load_manifest

NO_REPO_STATUSES = {"unreleased", "human_study", "agentic", "excluded"}
ON_DISK_STATUSES = {"registered", "vendored"}


class ManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = load_manifest(MANIFEST_PATH)
        text = MANIFEST_PATH.read_text(encoding="utf-8")
        _, marker, added_section = text.partition("# Added 2026-10-07")
        assert marker, "manifest lost its '# Added 2026-10-07' marker"
        cls.added = set(re.findall(r'^name = "([^"]+)"', added_section, re.M))

    @unittest.skipUnless((REPO_ROOT / "annotations.csv").exists(),
                         "annotations.csv (annotation export) not in checkout")
    def test_ids_match_annotations(self):
        with open(REPO_ROOT / "annotations.csv", newline="", encoding="utf-8") as handle:
            annotation_ids = {row["id"] for row in csv.DictReader(handle)}
        ids = [row["id"] for row in self.rows if row["id"]]
        self.assertEqual(len(ids), len(set(ids)), "duplicate id")
        self.assertEqual(set(ids), annotation_ids)

    def test_names(self):
        names = [row["name"] for row in self.rows]
        self.assertEqual(len(names), len(set(names)), "duplicate name")
        for name in names:
            self.assertRegex(name, f"^{NAME_PATTERN}$")

    def test_rows(self):
        for row in self.rows:
            name = row["name"]
            with self.subTest(name=name):
                self.assertIn(row["status"], STATUSES)
                if not row["id"]:
                    # ids are blank for on-disk sources and for rows added after the export.
                    self.assertTrue(row["status"] in ON_DISK_STATUSES or name in self.added)
                self.assertIn(row["host"], HOSTS)
                if row["host"] != "none":
                    # files may be empty with a host: partial rows whose prompts are rendered (note says how).
                    self.assertTrue(row.get("repo"), "host set but repo empty")
                else:
                    # agentic rows may have no host: some never published a repo.
                    self.assertIn(row["status"], NO_REPO_STATUSES)
                has_path = "path" in row
                self.assertEqual(row["status"] in ON_DISK_STATUSES, has_path)
                if has_path:
                    self.assertTrue((REPO_ROOT / "datasets" / row["path"]).is_dir())
                self.assertIsInstance(row["files"], list)
                for entry in row["files"]:
                    self.assertIsInstance(entry, str)
                    self.assertTrue(entry)
                self.assertIsInstance(row["revision"], str)

    def test_excluded_rows_keep_their_join(self):
        for row in self.rows:
            if row["status"] == "excluded":
                with self.subTest(name=row["name"]):
                    self.assertTrue(row["id"])
                    self.assertTrue(row["note"].startswith("Excluded 2026-10-07:"))

    def test_fetched_revisions_match(self):
        # Committed records only, so an in-progress fetch cannot change the result.
        tracked = subprocess.run(
            ["git", "ls-files", "datasets/raw/*/fetch.json"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.split()
        revisions = {row["name"]: row["revision"] for row in self.rows}
        for path in tracked:
            name = Path(path).parent.name
            with self.subTest(name=name):
                if revisions[name]:  # empty means unpinned
                    recorded = json.loads((REPO_ROOT / path).read_text(encoding="utf-8"))
                    self.assertEqual(recorded["revision"], revisions[name])

    def test_sources_read_registered_rows(self):
        from datasets.prepare.cluster.sources import SOURCES
        on_disk = {row["path"] for row in self.rows if row["status"] in ON_DISK_STATUSES}
        for source in SOURCES:
            with self.subTest(source=source.name):
                self.assertIn("/".join(source.path.split("/")[:2]), on_disk)


if __name__ == "__main__":
    unittest.main()
