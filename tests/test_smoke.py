'''
Offline end-to-end smoke tests: the two CLIs run as subprocesses against
inspect's mockllm provider, from a temp cwd so logs land there, with --limit
(certify never writes models/results) and CERTIFICATE_GENERATED_DIR (generate
never writes datasets/generated). Mock replies are unparseable to the judges,
so scores abstain; these tests check that the plumbing runs, not the numbers.
'''
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LEAF = "loc_oversight_evasion"  # the smallest leaf dataset


def run(script: str, *args: str, cwd: str, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO / script), *args],
        cwd=cwd, env={**os.environ, "INSPECT_DISPLAY": "log", "PYTHONDONTWRITEBYTECODE": "1", **env},
        capture_output=True, text=True, timeout=600,
    )


class TestCertifySmoke(unittest.TestCase):

    def test_every_stage_runs_against_a_mock_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = run(
                "certify.py", "-m", "mockllm/model", "-g", "mockllm/model",
                "--sim-classifier", "mockllm/model", "--simulate",
                "--only", LEAF, "--limit", "2", "--max-connections", "4",
                cwd=tmp, env={"INSPECT_CACHE_DIR": tmp},
            )
            self.assertEqual(proc.returncode, 0, proc.stdout[-3000:] + proc.stderr[-3000:])
            self.assertIn("samples   target    judge", proc.stdout)  # the preflight estimate table
            logs = list((Path(tmp) / "logs" / "model" / "current-limit2").glob("*.eval"))
            self.assertEqual(len(logs), 1, proc.stdout[-2000:])
            self.assertFalse((Path(tmp) / "models").exists())


class TestGenerateSmoke(unittest.TestCase):

    def test_deterministic_family_writes_partial_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = run(
                "generate.py", "-a", "mockllm/model", "--only", LEAF,
                "--perturb", "persona", "--limit", "2",
                cwd=tmp, env={"CERTIFICATE_GENERATED_DIR": tmp},
            )
            self.assertEqual(proc.returncode, 0, proc.stdout[-3000:] + proc.stderr[-3000:])
            rows = (Path(tmp) / LEAF / "persona.jsonl").read_text().splitlines()
            self.assertTrue(rows)
            meta = json.loads((Path(tmp) / LEAF / "persona.meta.json").read_text())
            self.assertTrue(meta["partial"])


if __name__ == "__main__":
    unittest.main()
