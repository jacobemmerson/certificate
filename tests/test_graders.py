'''
Tests for pipeline/utils/graders.py::validate_graders and write_json_atomic.
Run: uv run python3 -m unittest discover tests
'''

import tempfile
import unittest
from pathlib import Path

from pipeline.utils.graders import validate_graders, write_json_atomic


class TestValidateGraders(unittest.TestCase):
    '''
    A misconfigured judge is the suite's worst failure: the model under test
    answers fine, so the run pays for every sample and then dies on scoring —
    or worse, a judge returning garbage scores every sample as an abstention,
    which is the safe end, and reports a perfect certification.
    '''

    def test_passes_when_every_grader_answers(self):
        validate_graders(["mockllm/model", "mockllm/model"])

    def test_raises_naming_the_bad_grader(self):
        with self.assertRaises(SystemExit) as ctx:
            validate_graders(["mockllm/model", "openrouter/openai/claude-sonnet-4.5"])
        message = str(ctx.exception)
        self.assertIn("openrouter/openai/claude-sonnet-4.5", message)
        self.assertIn("no evals were started", message)
        # points at where graders come from, since this is nearly always config
        self.assertIn("GRADERS.md", message)

    def test_accepts_a_single_grader_string(self):
        validate_graders("mockllm/model")


class TestWriteJsonAtomic(unittest.TestCase):
    def test_file_is_world_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.json"
            write_json_atomic(path, {})
            self.assertEqual(path.stat().st_mode & 0o777, 0o644)


if __name__ == "__main__":
    unittest.main()
