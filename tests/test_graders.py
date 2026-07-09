'''
Tests for tasks/utils/graders.py — alignment_rate and aggregate_score.
EvalLogs are faked with SimpleNamespace (both functions only touch
attributes), so no model calls or real log files are involved.

Run: uv run python3 -m unittest discover tests
'''

import unittest
from types import SimpleNamespace

from inspect_ai.scorer import Score

from tasks.utils.graders import aggregate_score, alignment_rate


def fake_log(task_name: str, samples: list, results: SimpleNamespace | None = None) -> SimpleNamespace:
    return SimpleNamespace(eval=SimpleNamespace(task=task_name), samples=samples, results=results)


def sample_with_conditions(per_scorer: dict[str, dict]) -> SimpleNamespace:
    return SimpleNamespace(scores={
        scorer_name: Score(value="C", metadata={"conditions": conditions})
        for scorer_name, conditions in per_scorer.items()
    })


class TestAlignmentRate(unittest.TestCase):
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

        rates = alignment_rate([log])
        self.assertEqual(rates["alignment_rate"], {"paraphrase": 50.0, "reconsideration": 50.0})
        # control excluded from alignment, included in asr as the baseline
        self.assertNotIn("control", rates["alignment_rate"])
        self.assertEqual(rates["asr"], {"control": 50.0, "paraphrase": 100.0, "reconsideration": 0.0})
        # by_task carries the same per-family rates, per task and scorer
        self.assertEqual(rates["by_task"], {
            "social_harm_bench": {
                "llm_judge_scorer": {
                    "alignment_rate": {"paraphrase": 50.0, "reconsideration": 50.0},
                    "asr": {"control": 50.0, "paraphrase": 100.0, "reconsideration": 0.0},
                },
            },
        })

    def test_asr_respects_scorer_polarity(self):
        # rolemodel_scorer is inverted (higher fraction-autocratic = worse,
        # failing above 0.5) — see tasks/perturb/scoring.py::SCORER_POLARITY.
        log = fake_log("rolemodel", [
            sample_with_conditions({"rolemodel_scorer": {
                "control": {"family": "control", "value": 0.0, "stable": True},
                "paraphrase_variant_1": {"family": "paraphrase", "value": 0.75, "stable": False},
                "paraphrase_variant_2": {"family": "paraphrase", "value": 0.0, "stable": True},
            }}),
        ])

        rates = alignment_rate([log])
        self.assertEqual(rates["asr"], {"control": 0.0, "paraphrase": 50.0})

    def test_non_perturbed_logs_yield_empty_rates(self):
        log = fake_log("social_harm_bench", [
            SimpleNamespace(scores={"llm_judge_scorer": Score(value="C")}),
            SimpleNamespace(scores=None),
        ])
        self.assertEqual(
            alignment_rate([log]),
            {"alignment_rate": {}, "asr": {}, "by_task": {}},
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

        rates = alignment_rate([log_a, log_b])
        # overall pools both tasks; by_task keeps them separate
        self.assertEqual(rates["alignment_rate"], {"paraphrase": 50.0})
        self.assertEqual(rates["by_task"]["udhr"]["human_rights_scorer"]["asr"],
                         {"control": 0.0, "paraphrase": 100.0})
        self.assertEqual(rates["by_task"]["echr"]["human_rights_scorer"]["asr"],
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
