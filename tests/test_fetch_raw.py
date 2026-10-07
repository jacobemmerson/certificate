'''
Tests for scripts/fetch_raw.py. No network: only planning, selection,
dry-run and the up-to-date skip are exercised.
'''
import contextlib
import io
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import fetch_raw

MANIFEST = '''
[[benchmark]]
name = "gh"
id = "1"
host = "github"
repo = "owner/repo"
revision = "v1"
files = ["data/*.jsonl", "README.md"]
status = "prompt"

[[benchmark]]
name = "hub"
id = "2"
host = "hf"
repo = "org/set"
revision = ""
files = ["*.parquet"]
status = "partial"

[[benchmark]]
name = "gone"
id = "3"
host = "none"
files = []
status = "unreleased"
'''


class FetchRawTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.manifest = self.tmp / "manifest.toml"
        self.manifest.write_text(MANIFEST)
        self.entries = fetch_raw.load_manifest(self.manifest)

    def test_dry_run_names_entries_and_creates_nothing(self):
        raw = self.tmp / "raw"
        out = io.StringIO()
        with mock.patch.object(fetch_raw, "RAW_DIR", raw), contextlib.redirect_stdout(out):
            fetch_raw.main(["--manifest", str(self.manifest), "--dry-run",
                            "--status", "prompt", "partial", "unreleased"])
        for name in ("gh", "hub", "gone"):
            self.assertIn(name, out.getvalue())
        self.assertEqual(sorted(p.name for p in self.tmp.iterdir()), ["manifest.toml"])

    def test_select_by_status_and_only(self):
        names = lambda chosen: [entry["name"] for entry in chosen]
        self.assertEqual(names(fetch_raw.select(self.entries, [], ["prompt", "partial"])), ["gh", "hub"])
        self.assertEqual(names(fetch_raw.select(self.entries, [], ["partial"])), ["hub"])
        self.assertEqual(names(fetch_raw.select(self.entries, ["gone"], ["prompt"])), ["gone"])

    def test_plan_github(self):
        self.assertEqual(fetch_raw.plan(self.entries[0], self.tmp / "gh"), [
            ["git", "init", "-q"],
            ["git", "remote", "add", "origin", "https://github.com/owner/repo.git"],
            ["git", "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", "v1"],
            ["git", "sparse-checkout", "set", "--no-cone", "data/*.jsonl", "README.md"],
            ["git", "checkout", "-q", "FETCH_HEAD"],
        ])

    def test_plan_hf(self):
        dest = self.tmp / "hub"
        self.assertEqual(fetch_raw.plan(self.entries[1], dest), {
            "repo_id": "org/set", "repo_type": "dataset", "revision": None,
            "allow_patterns": ["*.parquet"], "local_dir": dest,
        })

    def test_up_to_date_skip(self):
        dest = self.tmp / "gh"
        dest.mkdir()
        (dest / "fetch.json").write_text(json.dumps({"revision": "v1", "files": ["data/*.jsonl", "README.md"]}))
        out = io.StringIO()
        with mock.patch.object(fetch_raw.subprocess, "run") as run, contextlib.redirect_stdout(out):
            fetch_raw.fetch(self.entries[0], dest)
        run.assert_not_called()
        self.assertIn("up to date", out.getvalue())

        # Unpinned request: any recorded revision counts as current.
        hub = self.tmp / "hub"
        hub.mkdir()
        (hub / "fetch.json").write_text(json.dumps({"revision": "abc", "files": ["*.parquet"]}))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            fetch_raw.fetch(self.entries[1], hub)
        self.assertIn("up to date", out.getvalue())

        # Tag pin: the stored SHA differs, but the recorded request matches.
        (dest / "fetch.json").write_text(json.dumps(
            {"revision": "f00d", "requested": "v1", "files": ["data/*.jsonl", "README.md"]}))
        out = io.StringIO()
        with mock.patch.object(fetch_raw.subprocess, "run") as run, contextlib.redirect_stdout(out):
            fetch_raw.fetch(self.entries[0], dest)
        run.assert_not_called()
        self.assertIn("up to date", out.getvalue())

    def test_files_change_refetches_and_failure_cleans_up(self):
        dest = self.tmp / "gh"
        dest.mkdir()
        (dest / "fetch.json").write_text(json.dumps({"revision": "v1", "files": ["data/*.jsonl"]}))
        failure = subprocess.CalledProcessError(128, ["git", "fetch"])
        with mock.patch.object(fetch_raw.subprocess, "run", side_effect=failure) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                fetch_raw.fetch(self.entries[0], dest)
        run.assert_called()
        self.assertFalse(dest.exists())

    def test_main_rejects_unknown_only_and_bad_slug(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            fetch_raw.main(["--manifest", str(self.manifest), "--dry-run", "--only", "missing"])
        self.manifest.write_text(MANIFEST.replace('name = "gh"', 'name = "../gh"'))
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            fetch_raw.main(["--manifest", str(self.manifest), "--dry-run"])

    def test_empty_files_skipped(self):
        self.manifest.write_text(MANIFEST.replace('files = ["*.parquet"]', 'files = []'))
        out = io.StringIO()
        with mock.patch.object(fetch_raw, "RAW_DIR", self.tmp / "raw"), contextlib.redirect_stdout(out):
            fetch_raw.main(["--manifest", str(self.manifest), "--dry-run"])
        self.assertIn("hub: no files, skipped", out.getvalue())
        self.assertNotIn("hub ->", out.getvalue())

    def test_unpinned_registered_skipped_only_when_committed(self):
        self.manifest.write_text(MANIFEST.replace('status = "partial"', 'status = "registered"'))
        raw = self.tmp / "raw"
        args = ["--manifest", str(self.manifest), "--status", "registered", "--dry-run"]
        out = io.StringIO()
        with mock.patch.object(fetch_raw, "RAW_DIR", raw), contextlib.redirect_stdout(out):
            fetch_raw.main(args)
        self.assertIn("hub ->", out.getvalue())

        (raw / "hub").mkdir(parents=True)
        (raw / "hub" / "data.csv").write_text("x\n")
        out = io.StringIO()
        with mock.patch.object(fetch_raw, "RAW_DIR", raw), contextlib.redirect_stdout(out):
            fetch_raw.main(args)
        self.assertIn("hub: committed data, skipped", out.getvalue())
        self.assertNotIn("hub ->", out.getvalue())


if __name__ == "__main__":
    unittest.main()
