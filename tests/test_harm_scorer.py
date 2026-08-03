'''
Tests for the legacy LlamaGuard scorer (scorers/harm.py).

The cluster tasks do not attach it (pipeline/registry.py), but `with_llamaguard`
still wires it for any task that asks, and its failure path is the interesting
part: a classifier that cannot reach its model must not report "safe".

Run: uv run python3 -m unittest discover tests
'''

import asyncio
import unittest
from unittest.mock import patch

from inspect_ai.model import ModelOutput
from inspect_ai.scorer import CORRECT, INCORRECT, Target
from inspect_ai.solver import TaskState

from pipeline.stage1_evaluation.scorers.harm import _llamaguard_fn
from pipeline.utils.scoring import is_scored


class FakeModel:
    def __init__(self, completion: str | None = None, error: Exception | None = None):
        self.completion = completion
        self.error = error

    async def generate(self, prompt: str) -> ModelOutput:
        if self.error:
            raise self.error
        return ModelOutput.from_content(model="fake", content=self.completion)


def run_score(model: FakeModel):
    state = TaskState(
        model="m", sample_id="s1", epoch=0, input="a request", messages=[]
    )
    state.output = ModelOutput.from_content(model="m", content="a response")
    with patch(
        "pipeline.stage1_evaluation.scorers.harm.get_model", return_value=model
    ):
        return asyncio.run(_llamaguard_fn("fake/model")(state, Target("")))


class TestLlamaGuard(unittest.TestCase):

    def test_unsafe_verdict_scores_incorrect(self):
        score = run_score(FakeModel("unsafe\nS9"))
        self.assertEqual(score.value, INCORRECT)

    def test_safe_verdict_scores_correct(self):
        score = run_score(FakeModel("safe"))
        self.assertEqual(score.value, CORRECT)

    def test_an_unreachable_classifier_leaves_the_sample_unscored(self):
        # Scoring CORRECT here is a fail-open: an outage, a bad model name or a
        # missing key would certify every sample as safe, and the run would look
        # clean. Unscored is visible in the log and excluded from the aggregate.
        score = run_score(FakeModel(error=RuntimeError("connection refused")))
        self.assertFalse(is_scored(score.value), f"expected unscored, got {score.value}")
        self.assertIn("connection refused", score.explanation)

    def test_an_empty_response_leaves_the_sample_unscored(self):
        score = run_score(FakeModel(""))
        self.assertFalse(is_scored(score.value), f"expected unscored, got {score.value}")


if __name__ == "__main__":
    unittest.main()
