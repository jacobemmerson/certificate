"""
Tests for pipeline/utils/graders.py — consistency_rate and aggregate_score.
EvalLogs are faked with SimpleNamespace (both functions only touch
attributes), so no model calls or real log files are involved.

Run: uv run python3 -m unittest discover tests
"""

import unittest
from types import SimpleNamespace

from inspect_ai.scorer import Score

from pipeline.utils.graders import aggregate_score, consistency_rate


def fake_log(
    task_name: str, samples: list, results: SimpleNamespace | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        eval=SimpleNamespace(task=task_name), samples=samples, results=results
    )


def sample_with_conditions(per_scorer: dict[str, dict]) -> SimpleNamespace:
    return SimpleNamespace(
        scores={
            scorer_name: Score(value="C", metadata={"conditions": conditions})
            for scorer_name, conditions in per_scorer.items()
        }
    )


class TestConsistencyRate(unittest.TestCase):
    def test_rates_per_family(self):
        log = fake_log(
            "social_harm_bench",
            [
                sample_with_conditions(
                    {
                        "llm_judge_scorer": {
                            "control": {
                                "family": "control",
                                "value": "C",
                                "stable": True,
                            },
                            "paraphrase_variant_1": {
                                "family": "paraphrase",
                                "value": "I",
                                "stable": False,
                            },
                            "reconsidered": {
                                "family": "reconsideration",
                                "value": "C",
                                "stable": True,
                            },
                        }
                    }
                ),
                sample_with_conditions(
                    {
                        "llm_judge_scorer": {
                            "control": {
                                "family": "control",
                                "value": "I",
                                "stable": True,
                            },
                            "paraphrase_variant_1": {
                                "family": "paraphrase",
                                "value": "I",
                                "stable": True,
                            },
                            "reconsidered": {
                                "family": "reconsideration",
                                "value": "C",
                                "stable": False,
                            },
                        }
                    }
                ),
            ],
        )

        rates = consistency_rate([log])
        self.assertEqual(
            rates["consistency_rate"], {"paraphrase": 50.0, "reconsideration": 50.0}
        )
        # control excluded from consistency, included in lvr as the baseline
        self.assertNotIn("control", rates["consistency_rate"])
        self.assertEqual(
            rates["lvr"], {"control": 50.0, "paraphrase": 100.0, "reconsideration": 0.0}
        )
        # by_task carries the same per-family rates, per task and scorer
        self.assertEqual(
            rates["by_task"],
            {
                "social_harm_bench": {
                    "llm_judge_scorer": {
                        "consistency_rate": {
                            "paraphrase": 50.0,
                            "reconsideration": 50.0,
                        },
                        "lvr": {
                            "control": 50.0,
                            "paraphrase": 100.0,
                            "reconsideration": 0.0,
                        },
                    },
                },
            },
        )

    def test_lvr_respects_scorer_polarity(self):
        # role_model_bias_scorer is inverted (higher fraction-autocratic = worse,
        # failing above 0.5) — see pipeline/stage2_perturbation/scoring.py::SCORER_POLARITY.
        log = fake_log(
            "role_model_bias",
            [
                sample_with_conditions(
                    {
                        "role_model_bias_scorer": {
                            "control": {
                                "family": "control",
                                "value": 0.0,
                                "stable": True,
                            },
                            "paraphrase_variant_1": {
                                "family": "paraphrase",
                                "value": 0.75,
                                "stable": False,
                            },
                            "paraphrase_variant_2": {
                                "family": "paraphrase",
                                "value": 0.0,
                                "stable": True,
                            },
                        }
                    }
                ),
            ],
        )

        rates = consistency_rate([log])
        self.assertEqual(rates["lvr"], {"control": 0.0, "paraphrase": 50.0})

    def test_non_perturbed_logs_yield_empty_rates(self):
        log = fake_log(
            "social_harm_bench",
            [
                SimpleNamespace(scores={"llm_judge_scorer": Score(value="C")}),
                SimpleNamespace(scores=None),
            ],
        )
        self.assertEqual(
            consistency_rate([log]),
            {"consistency_rate": {}, "lvr": {}, "by_task": {}},
        )

    def test_by_task_separates_tasks_and_scorers(self):
        conditions = {
            "control": {"family": "control", "value": "C", "stable": True},
            "paraphrase_variant_1": {
                "family": "paraphrase",
                "value": "I",
                "stable": False,
            },
        }
        log_a = fake_log(
            "udhr", [sample_with_conditions({"human_rights_scorer": conditions})]
        )
        log_b = fake_log(
            "echr",
            [
                sample_with_conditions(
                    {
                        "human_rights_scorer": {
                            "control": {
                                "family": "control",
                                "value": "C",
                                "stable": True,
                            },
                            "paraphrase_variant_1": {
                                "family": "paraphrase",
                                "value": "C",
                                "stable": True,
                            },
                        },
                    }
                )
            ],
        )

        rates = consistency_rate([log_a, log_b])
        # overall pools both tasks; by_task keeps them separate
        self.assertEqual(rates["consistency_rate"], {"paraphrase": 50.0})
        self.assertEqual(
            rates["by_task"]["udhr"]["human_rights_scorer"]["lvr"],
            {"control": 0.0, "paraphrase": 100.0},
        )
        self.assertEqual(
            rates["by_task"]["echr"]["human_rights_scorer"]["lvr"],
            {"control": 0.0, "paraphrase": 0.0},
        )


class TestAggregateScore(unittest.TestCase):
    def metric_log(
        self,
        task_name: str,
        value: float,
        *,
        metadata: dict | None = None,
        samples: list | None = None,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            eval=SimpleNamespace(task=task_name, metadata=metadata or {}),
            samples=samples or [],
            results=SimpleNamespace(
                scores=[
                    SimpleNamespace(
                        name="primary",
                        scorer="primary",
                        metrics={
                            "accuracy": SimpleNamespace(
                                name="accuracy", value=value, params={}
                            )
                        },
                    )
                ]
            ),
        )

    def process_sample(
        self,
        value: float,
        *,
        valid: bool,
        violation: str | None = None,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            id=f"sample-{value}-{valid}-{violation}",
            metadata={
                "agentic_protocol_audit": {
                    "valid": valid,
                    "violations": [violation] if violation else [],
                }
            },
            scores={"primary": Score(value=value)},
        )

    def test_averages_first_metric_of_first_scorer(self):
        logs = [self.metric_log("udhr", 80.0), self.metric_log("echr", 60.0)]
        reported, by_task = aggregate_score(logs)
        self.assertEqual(reported, 70.0)
        self.assertEqual(by_task, {"udhr": 80.0, "echr": 60.0})

    def test_social_harm_bench_scaled_to_percent_for_c0_and_c1_c4(self):
        clean = self.metric_log("social_harm_bench", 0.75)
        reported, by_task = aggregate_score([clean])
        self.assertEqual(reported, 75.0)
        self.assertEqual(by_task["social_harm_bench"], 75.0)

        for condition in ("c1", "c2", "c3", "c4"):
            with self.subTest(condition=condition):
                transformed = self.metric_log(
                    f"social_harm_bench_agentic_{condition}",
                    0.0,
                    metadata={
                        "agentic_protocol": {
                            "condition": condition,
                            "base_task": "social_harm_bench",
                        }
                    },
                    samples=[self.process_sample(0.75, valid=True)],
                )
                reported, by_task = aggregate_score(
                    [transformed], valid_process_only=True
                )
                self.assertEqual(reported, 75.0)
                self.assertEqual(by_task, {"social_harm_bench": 75.0})

    def test_process_invalid_leakage_is_excluded_from_aggregate(self):
        log = self.metric_log(
            "social_harm_bench_agentic_c1",
            0.5,
            metadata={
                "agentic_protocol": {
                    "condition": "c1",
                    "base_task": "social_harm_bench",
                }
            },
            samples=[
                self.process_sample(1.0, valid=True),
                self.process_sample(
                    0.0,
                    valid=False,
                    violation="hidden_data_in_solving_payload",
                ),
            ],
        )
        reported, by_task = aggregate_score([log], valid_process_only=True)
        self.assertEqual(reported, 100.0)
        self.assertEqual(by_task, {"social_harm_bench": 100.0})

    def test_process_invalid_limit_is_excluded_and_all_invalid_is_unreported(self):
        invalid = self.process_sample(0.0, valid=False, violation="protocol_limit")
        mixed = self.metric_log(
            "fscale_agentic_c1",
            50.0,
            metadata={"agentic_protocol": {"condition": "c1", "base_task": "fscale"}},
            samples=[self.process_sample(100.0, valid=True), invalid],
        )
        reported, by_task = aggregate_score([mixed], valid_process_only=True)
        self.assertEqual(reported, 100.0)
        self.assertEqual(by_task, {"fscale": 100.0})

        all_invalid = self.metric_log(
            "fscale_agentic_c1",
            0.0,
            metadata=mixed.eval.metadata,
            samples=[invalid],
        )
        reported, by_task = aggregate_score([all_invalid], valid_process_only=True)
        self.assertIsNone(reported)
        self.assertEqual(by_task, {"fscale": None})


if __name__ == "__main__":
    unittest.main()
