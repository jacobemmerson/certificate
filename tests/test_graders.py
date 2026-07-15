'''
Tests for pipeline/utils/graders.py — consistency_rate and aggregate_score.
EvalLogs are faked with SimpleNamespace (both functions only touch
attributes), so no model calls or real log files are involved.

Run: uv run python3 -m unittest discover tests
'''

import unittest
from types import SimpleNamespace

from inspect_ai.scorer import Score

from pipeline.utils.graders import aggregate_score, consistency_rate


def fake_log(task_name: str, samples: list, results: SimpleNamespace | None = None) -> SimpleNamespace:
    return SimpleNamespace(eval=SimpleNamespace(task=task_name), samples=samples, results=results)


def sample_with_conditions(per_scorer: dict[str, dict]) -> SimpleNamespace:
    return SimpleNamespace(scores={
        scorer_name: Score(value="C", metadata={"conditions": conditions})
        for scorer_name, conditions in per_scorer.items()
    })


class TestConsistencyRate(unittest.TestCase):
    def test_rates_per_family(self):
        log = fake_log("social_harm_bench", [
            sample_with_conditions({"llm_judge_scorer": {
                "control": {"family": "control", "value": "C", "stable": True},
                "paraphrase_variant_1": {"family": "paraphrase", "value": "I", "stable": False},
                "reconsidered": {"family": "reconsideration", "value": "C", "stable": True},
            }}),
            sample_with_conditions({"llm_judge_scorer": {
                "control": {"family": "control", "value": "I", "stable": True},
                "paraphrase_variant_1": {"family": "paraphrase", "value": "I", "stable": True},
                "reconsidered": {"family": "reconsideration", "value": "C", "stable": False},
            }}),
        ])

        rates = consistency_rate([log])
        self.assertEqual(rates["consistency_rate"], {"paraphrase": 50.0, "reconsideration": 50.0})
        # control excluded from consistency, included in lvr as the baseline
        self.assertNotIn("control", rates["consistency_rate"])
        self.assertEqual(rates["lvr"], {"control": 50.0, "paraphrase": 100.0, "reconsideration": 0.0})
        # by_task carries the same per-family rates, per task and scorer
        self.assertEqual(rates["by_task"], {
            "social_harm_bench": {
                "llm_judge_scorer": {
                    "consistency_rate": {"paraphrase": 50.0, "reconsideration": 50.0},
                    "lvr": {"control": 50.0, "paraphrase": 100.0, "reconsideration": 0.0},
                },
            },
        })

    def test_lvr_respects_scorer_polarity(self):
        # role_model_bias_scorer is inverted (higher fraction-autocratic = worse,
        # failing above 0.5) — see pipeline/utils/scoring.py::SCORER_POLARITY.
        log = fake_log("role_model_bias", [
            sample_with_conditions({"role_model_bias_scorer": {
                "control": {"family": "control", "value": 0.0, "stable": True},
                "paraphrase_variant_1": {"family": "paraphrase", "value": 0.75, "stable": False},
                "paraphrase_variant_2": {"family": "paraphrase", "value": 0.0, "stable": True},
            }}),
        ])

        rates = consistency_rate([log])
        self.assertEqual(rates["lvr"], {"control": 0.0, "paraphrase": 50.0})

    def test_families_filter_splits_stages_from_one_log(self):
        # A combined --perturb --simulate run stores one log; certify.py calls
        # consistency_rate twice with a families filter to split it into the
        # models.json "perturbations" and "simulations" sections. The control
        # is always tallied as the shared baseline.
        log = fake_log("social_harm_bench", [
            sample_with_conditions({"llm_judge_scorer": {
                "control": {"family": "control", "value": "C", "stable": True},
                "paraphrase_variant_1": {"family": "paraphrase", "value": "I", "stable": False},
                "scenario_variant_1": {"family": "scenario", "value": "C", "stable": True},
            }}),
        ])

        perturb = consistency_rate([log], families={"paraphrase"})
        self.assertEqual(perturb["lvr"], {"control": 0.0, "paraphrase": 100.0})
        self.assertNotIn("scenario", perturb["consistency_rate"])

        sim = consistency_rate([log], families={"scenario"})
        self.assertEqual(sim["lvr"], {"control": 0.0, "scenario": 0.0})
        self.assertEqual(sim["consistency_rate"], {"scenario": 100.0})
        self.assertNotIn("paraphrase", sim["lvr"])

    def test_non_perturbed_logs_yield_empty_rates(self):
        log = fake_log("social_harm_bench", [
            SimpleNamespace(scores={"llm_judge_scorer": Score(value="C")}),
            SimpleNamespace(scores=None),
        ])
        self.assertEqual(
            consistency_rate([log]),
            {"consistency_rate": {}, "lvr": {}, "by_task": {}},
        )

    def test_by_task_separates_tasks_and_scorers(self):
        conditions = {
            "control": {"family": "control", "value": "C", "stable": True},
            "paraphrase_variant_1": {"family": "paraphrase", "value": "I", "stable": False},
        }
        log_a = fake_log("udhr", [sample_with_conditions({"human_rights_scorer": conditions})])
        log_b = fake_log("echr", [sample_with_conditions({
            "human_rights_scorer": {
                "control": {"family": "control", "value": "C", "stable": True},
                "paraphrase_variant_1": {"family": "paraphrase", "value": "C", "stable": True},
            },
        })])

        rates = consistency_rate([log_a, log_b])
        # overall pools both tasks; by_task keeps them separate
        self.assertEqual(rates["consistency_rate"], {"paraphrase": 50.0})
        self.assertEqual(rates["by_task"]["udhr"]["human_rights_scorer"]["lvr"],
                         {"control": 0.0, "paraphrase": 100.0})
        self.assertEqual(rates["by_task"]["echr"]["human_rights_scorer"]["lvr"],
                         {"control": 0.0, "paraphrase": 0.0})


class TestAggregateScore(unittest.TestCase):
    def metric_log(self, task_name: str, value: float) -> SimpleNamespace:
        return SimpleNamespace(
            eval=SimpleNamespace(task=task_name),
            results=SimpleNamespace(scores=[
                SimpleNamespace(metrics={"accuracy": SimpleNamespace(value=value)})
            ]),
        )

    def test_averages_first_metric_of_first_scorer(self):
        logs = [self.metric_log("udhr", 80.0), self.metric_log("echr", 60.0)]
        reported, by_task = aggregate_score(logs)
        self.assertEqual(reported, 70.0)
        self.assertEqual(by_task, {"udhr": 80.0, "echr": 60.0})

    def test_social_harm_bench_scaled_to_percent(self):
        reported, by_task = aggregate_score([self.metric_log("social_harm_bench", 0.5)])
        self.assertEqual(reported, 50.0)
        self.assertEqual(by_task["social_harm_bench"], 50.0)


if __name__ == "__main__":
    unittest.main()
