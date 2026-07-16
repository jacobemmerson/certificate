"""Registry transform guards and agentic field-preservation regressions."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import CORRECT, Score, accuracy, scorer
from inspect_ai.solver import generate, system_message

from pipeline.agentic import AgenticConfig, make_agentic_task
from pipeline.registry import apply_condition, apply_stages, condition_benchmark_key
from pipeline.utils.task_transforms import AGENTIC_METADATA_KEY


@scorer(metrics=[accuracy()])
def constant_scorer():
    async def score(state, target):
        return Score(value=CORRECT)

    return score


def base_task(name: str = "transform_base") -> Task:
    async def cleanup(state):
        return None

    return Task(
        name=name,
        dataset=[Sample(id="one", input="request")],
        setup=system_message("public persona"),
        solver=generate(),
        cleanup=cleanup,
        scorer=constant_scorer(),
        model_roles={"existing": "mockllm/existing"},
        metadata={"base": True},
        tags=["base"],
        message_limit=9,
        turn_limit=7,
        time_limit=11,
        working_limit=10,
    )


class TestTransformGuards(unittest.TestCase):
    def test_registry_rejects_both_agentic_stage_composition_orders(self):
        benchmarks = {"tiny": {"name": "tiny", "tasks": [base_task("tiny_task")]}}
        with patch("pipeline.registry.artifact_task_name", return_value="tiny_task"):
            staged = apply_stages(benchmarks, families=["reconsideration"], k=1)
        with self.assertRaisesRegex(ValueError, "perturbation"):
            apply_condition(staged, AgenticConfig.default("c2"))

        agentic = apply_condition(benchmarks, AgenticConfig.default("c2"))
        with self.assertRaisesRegex(ValueError, "agentic"):
            apply_stages(agentic, families=["reconsideration"], k=1)

    def test_stage2_and_stage3_remain_composable_in_one_registry_call(self):
        benchmarks = {"tiny": {"name": "tiny", "tasks": [base_task("tiny_task")]}}
        with (
            patch("pipeline.registry.artifact_task_name", return_value="tiny_task"),
            patch("pipeline.registry.load_family", return_value={}),
        ):
            staged = apply_stages(
                benchmarks,
                families=["reconsideration"],
                k=1,
                sim_k=1,
            )
        task = staged["tiny"]["tasks"][0]
        self.assertIn("certificate_perturbation", task.metadata)
        self.assertIn("certificate_simulation", task.metadata)
        self.assertEqual(staged["tiny"]["name"], "tiny")

    def test_agentic_builder_preserves_unmodified_task_fields(self):
        original = base_task()
        adapted = make_agentic_task(original, AgenticConfig.default("c2"))
        self.assertIsNot(adapted, original)
        for field in (
            "dataset",
            "setup",
            "cleanup",
            "metrics",
            "model",
            "config",
            "sandbox",
            "message_limit",
            "token_limit",
            "turn_limit",
            "time_limit",
            "working_limit",
            "cost_limit",
            "version",
        ):
            self.assertIs(getattr(adapted, field), getattr(original, field), field)
        self.assertEqual(original.metadata, {"base": True})
        self.assertIn(AGENTIC_METADATA_KEY, adapted.metadata)
        self.assertIsNot(adapted.metadata, original.metadata)
        self.assertIs(adapted.scorer, original.scorer)

    def test_registry_c0_is_identity_and_condition_keys_do_not_overwrite(self):
        benchmarks = {"tiny": {"name": "tiny", "tasks": [base_task("tiny_task")]}}
        self.assertIs(
            apply_condition(benchmarks, AgenticConfig.default("c0")), benchmarks
        )
        conditioned = apply_condition(benchmarks, AgenticConfig.default("c3"))
        self.assertEqual(set(conditioned), {"tiny_agentic_c3"})
        self.assertEqual(condition_benchmark_key("tiny", "c3"), "tiny_agentic_c3")
        self.assertEqual(set(benchmarks), {"tiny"})

        colliding = {
            **benchmarks,
            "tiny_agentic_c3": {
                "name": "collision",
                "tasks": [base_task("other_task")],
            },
        }
        with self.assertRaisesRegex(ValueError, "overwrite"):
            apply_condition(colliding, AgenticConfig.default("c3"))


if __name__ == "__main__":
    unittest.main()
