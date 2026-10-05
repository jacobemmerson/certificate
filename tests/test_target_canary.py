'''
Tests for pipeline/utils/graders.py::validate_target — the startup check that
the model under test actually answers.

validate_graders exists because a broken judge produces a plausible-looking
certification rather than an error. A silent *target* does the same thing and
had no equivalent check: meta/muse-spark-1.2 returned an empty assistant
message to every prompt, and the run spent four clusters and a full grader
bill to record `success 562/562` with a score of -1.

Run: uv run python3 -m unittest discover tests
'''

import unittest

from inspect_ai.model import ModelOutput

from pipeline.utils.graders import validate_target


def replies(text: str):
    '''mockllm model_args that make every call return `text`.'''
    return {
        "custom_outputs": lambda *args, **kwargs: ModelOutput.from_content(
            model="mockllm/model", content=text
        )
    }


class TestValidateTarget(unittest.TestCase):

    def test_a_model_that_answers_passes(self):
        validate_target("mockllm/model", replies("ok"))

    def test_a_silent_model_stops_the_run(self):
        with self.assertRaises(SystemExit) as ctx:
            validate_target("mockllm/model", replies(""))
        message = str(ctx.exception)
        self.assertIn("mockllm/model", message)
        self.assertIn("empty", message.lower())

    def test_whitespace_is_not_an_answer(self):
        with self.assertRaises(SystemExit):
            validate_target("mockllm/model", replies("  \n\t "))

    def test_an_unreachable_model_stops_the_run(self):
        with self.assertRaises(SystemExit) as ctx:
            validate_target("openrouter/openai/definitely-not-a-model", {})
        self.assertIn("definitely-not-a-model", str(ctx.exception))

    def test_no_model_args_is_fine(self):
        validate_target("mockllm/model", None)


if __name__ == "__main__":
    unittest.main()
