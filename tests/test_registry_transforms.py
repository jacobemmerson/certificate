"""Registry/task transform guards and field-preservation regressions."""

from __future__ import annotations

import unittest

from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import CORRECT, Score, accuracy, scorer
from inspect_ai.solver import generate, system_message

from pipeline.agentic import AgenticConfig, make_agentic_task
from pipeline.registry import apply_condition, condition_benchmark_key
from pipeline.stage2_perturbation.build import build_perturbed_task
from pipeline.stage3_simulation.build import build_simulation_task
from pipeline.utils.task_transforms import (
    AGENTIC_METADATA_KEY,
    PERTURBATION_METADATA_KEY,
    SIMULATION_METADATA_KEY,
)


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


def transform(name: str, task: Task) -> Task:
    if name == "agentic":
        return make_agentic_task(task, AgenticConfig.default("c2"))
    if name == "perturbation":
        return build_perturbed_task(task, ["reconsideration"], "mockllm/rewrite", 1)
    if name == "simulation":
        return build_simulation_task(task, "mockllm/simulator", 1)
    raise AssertionError(name)


class TestTransformGuards(unittest.TestCase):
    def test_all_cross_layer_orders_and_repeats_are_rejected(self):
        names = ("agentic", "perturbation", "simulation")
        for first in names:
            transformed = transform(first, base_task())
            for second in names:
                with self.subTest(first=first, second=second):
                    with self.assertRaises(ValueError):
                        transform(second, transformed)

    def test_each_builder_preserves_unmodified_task_fields(self):
        marker_by_transform = {
            "agentic": AGENTIC_METADATA_KEY,
            "perturbation": PERTURBATION_METADATA_KEY,
            "simulation": SIMULATION_METADATA_KEY,
        }
        for name, marker in marker_by_transform.items():
            with self.subTest(transform=name):
                original = base_task()
                adapted = transform(name, original)
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
                    self.assertIs(
                        getattr(adapted, field), getattr(original, field), field
                    )
                self.assertEqual(original.metadata, {"base": True})
                self.assertIn(marker, adapted.metadata)
                self.assertIsNot(adapted.metadata, original.metadata)
                if name == "agentic":
                    self.assertIs(adapted.scorer, original.scorer)
                else:
                    self.assertEqual(len(adapted.scorer), 1)

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
