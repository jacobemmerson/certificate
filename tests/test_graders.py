'''
Tests for pipeline/utils/graders.py — condition_metrics and aggregate_score.
EvalLogs are faked with SimpleNamespace (both functions only touch
attributes), so no model calls or real log files are involved.

Run: uv run python3 -m unittest discover tests
'''

import unittest
from types import SimpleNamespace

from inspect_ai.scorer import Score

from pipeline.utils.graders import aggregate_score, condition_metrics, validate_graders


def fake_log(task_name: str, samples: list, results: SimpleNamespace | None = None) -> SimpleNamespace:
    return SimpleNamespace(eval=SimpleNamespace(task=task_name), samples=samples, results=results)


def sample_with_conditions(
    per_scorer: dict[str, dict], source: str | None = None
) -> SimpleNamespace:
    """`metadata` mirrors EvalSample's, which always carries the cluster row's
    own fields — `source` is what the per-source breakdown groups on."""
    return SimpleNamespace(
        metadata={"source": source} if source else {},
        scores={
            scorer_name: Score(value="C", metadata={"conditions": conditions})
            for scorer_name, conditions in per_scorer.items()
        },
    )


class TestConditionMetrics(unittest.TestCase):
    def test_rates_per_family(self):
        log = fake_log("social_harm_bench", [
            sample_with_conditions({"llm_judge_scorer": {
                "control": {"family": "control", "value": "C", "stable": True, "drift": 0.0},
                "paraphrase_variant_1": {"family": "paraphrase", "value": "I", "stable": False, "drift": 1.0},
                "reconsidered": {"family": "reconsideration", "value": "C", "stable": True, "drift": 0.0},
            }}),
            sample_with_conditions({"llm_judge_scorer": {
                "control": {"family": "control", "value": "I", "stable": True, "drift": 0.0},
                "paraphrase_variant_1": {"family": "paraphrase", "value": "I", "stable": True, "drift": 0.0},
                "reconsidered": {"family": "reconsideration", "value": "C", "stable": False, "drift": 1.0},
            }}),
        ])

        rates = condition_metrics([log])
        self.assertEqual(rates["stability"], {"paraphrase": 50.0, "reconsideration": 50.0})
        # control excluded from stability, included in propensity as the baseline
        self.assertNotIn("control", rates["stability"])
        self.assertEqual(rates["harm_propensity"], {"control": 50.0, "paraphrase": 100.0, "reconsideration": 0.0})
        # by_task carries the same per-family rates, per task and scorer
        self.assertEqual(rates["by_task"], {
            "social_harm_bench": {
                "llm_judge_scorer": {
                    "stability": {"paraphrase": 50.0, "reconsideration": 50.0},
                    "harm_propensity": {"control": 50.0, "paraphrase": 100.0, "reconsideration": 0.0},
                },
            },
        })

    def test_by_source_splits_a_cluster_by_originating_benchmark(self):
        '''A cluster pools several benchmarks, so a pooled propensity can hide
        one source moving while the rest hold still.'''
        log = fake_log("manipulation", [
            sample_with_conditions({"cluster_scorer": {
                "control": {"family": "control", "value": 1.0, "drift": 0.0},
                "scenario_variant_1": {"family": "scenario", "value": 0.0, "drift": 1.0},
            }}, source="social_harm"),
            sample_with_conditions({"cluster_scorer": {
                "control": {"family": "control", "value": 1.0, "drift": 0.0},
                "scenario_variant_1": {"family": "scenario", "value": 1.0, "drift": 0.0},
            }}, source="leader_favorability"),
        ])

        rates = condition_metrics([log], families={"scenario"})

        # pooled hides the split: one source flipped, the other did not
        self.assertEqual(rates["harm_propensity"]["scenario"], 50.0)
        self.assertEqual(rates["by_source"]["social_harm"]["harm_propensity"]["scenario"], 100.0)
        self.assertEqual(rates["by_source"]["leader_favorability"]["harm_propensity"]["scenario"], 0.0)
        self.assertEqual(rates["by_source"]["social_harm"]["stability"]["scenario"], 0.0)
        self.assertEqual(rates["by_source"]["leader_favorability"]["stability"]["scenario"], 100.0)
        # control is tallied per source too, as the shared baseline
        self.assertEqual(rates["by_source"]["social_harm"]["harm_propensity"]["control"], 0.0)

    def test_unscored_conditions_do_not_crash_or_poison_the_rates(self):
        '''An abstaining judge yields Score.unscored() (NaN) and, in
        scoring.py::_wrap_scorer, an explicit "drift": None. Both must be
        excluded: float(None) raised, and NaN unsafety would turn every rate
        into NaN. Same rule as scoring.py's own metrics, which filter on
        is_scored.'''
        nan = float("nan")
        log = fake_log("cyber", [
            sample_with_conditions({"cluster_scorer": {
                "control": {"family": "control", "value": 1.0, "drift": 0.0},
                "paraphrase_variant_1": {"family": "paraphrase", "value": 0.0, "drift": 1.0},
                # judge abstained on this one
                "paraphrase_variant_2": {"family": "paraphrase", "value": nan, "drift": None},
            }}, source="cysecbench"),
        ])

        rates = condition_metrics([log], families={"paraphrase"})

        # the scored variant alone decides both figures
        self.assertEqual(rates["harm_propensity"]["paraphrase"], 100.0)
        self.assertEqual(rates["stability"]["paraphrase"], 0.0)
        self.assertEqual(rates["by_source"]["cysecbench"]["harm_propensity"]["paraphrase"], 100.0)

    def test_family_with_only_unscored_conditions_is_omitted(self):
        log = fake_log("cyber", [
            sample_with_conditions({"cluster_scorer": {
                "control": {"family": "control", "value": 1.0, "drift": 0.0},
                "paraphrase_variant_1": {"family": "paraphrase", "value": float("nan"), "drift": None},
            }}, source="cysecbench"),
        ])

        rates = condition_metrics([log], families={"paraphrase"})
        # nothing scored for paraphrase — reporting 0.0 would invent a clean pass
        self.assertNotIn("paraphrase", rates["harm_propensity"])
        self.assertNotIn("paraphrase", rates["stability"])
        self.assertEqual(rates["harm_propensity"]["control"], 0.0)

    def test_by_source_omits_samples_without_a_source(self):
        log = fake_log("legacy", [
            sample_with_conditions({"judge": {
                "control": {"family": "control", "value": 1.0, "drift": 0.0},
            }}),
        ])
        rates = condition_metrics([log])
        self.assertEqual(rates["by_source"], {})
        self.assertEqual(rates["harm_propensity"], {"control": 0.0})

    def test_no_per_scorer_polarity_special_casing(self):
        # role_model_bias_scorer used to need a polarity exception because its
        # value was the fraction of *autocratic* figures. It now reports the
        # democratic fraction, like everything else reports safety, so identical
        # values must yield identical numbers whatever the scorer is called.
        conditions = {
            "control": {"family": "control", "value": 1.0, "drift": 0.0},
            "paraphrase_variant_1": {"family": "paraphrase", "value": 0.25, "drift": 0.75},
            "paraphrase_variant_2": {"family": "paraphrase", "value": 1.0, "drift": 0.0},
        }
        inverted = condition_metrics([fake_log(
            "role_model_bias",
            [sample_with_conditions({"role_model_bias_scorer": conditions})],
        )])
        ordinary = condition_metrics([fake_log(
            "social_harm_bench",
            [sample_with_conditions({"llm_judge_scorer": conditions})],
        )])
        self.assertEqual(inverted["harm_propensity"], ordinary["harm_propensity"])

        # mean safety over paraphrase = (0.25 + 1.0) / 2 = 0.625
        self.assertEqual(
            inverted["harm_propensity"], {"control": 0.0, "paraphrase": 37.5}
        )
        # mean drift over paraphrase = (0.75 + 0.0) / 2 = 0.375
        self.assertEqual(inverted["stability"], {"paraphrase": 62.5})

    def test_families_filter_splits_stages_from_one_log(self):
        # A combined --perturb --simulate run stores one log; certify.py calls
        # condition_metrics twice with a families filter to split it into the
        # models.json "perturbations" and "simulations" sections. The control
        # is always tallied as the shared baseline.
        log = fake_log("social_harm_bench", [
            sample_with_conditions({"llm_judge_scorer": {
                "control": {"family": "control", "value": "C", "stable": True, "drift": 0.0},
                "paraphrase_variant_1": {"family": "paraphrase", "value": "I", "stable": False, "drift": 1.0},
                "scenario_variant_1": {"family": "scenario", "value": "C", "stable": True, "drift": 0.0},
            }}),
        ])

        perturb = condition_metrics([log], families={"paraphrase"})
        self.assertEqual(perturb["harm_propensity"], {"control": 0.0, "paraphrase": 100.0})
        self.assertNotIn("scenario", perturb["stability"])

        sim = condition_metrics([log], families={"scenario"})
        self.assertEqual(sim["harm_propensity"], {"control": 0.0, "scenario": 0.0})
        self.assertEqual(sim["stability"], {"scenario": 100.0})
        self.assertNotIn("paraphrase", sim["harm_propensity"])

    def test_non_perturbed_logs_yield_empty_rates(self):
        log = fake_log("social_harm_bench", [
            SimpleNamespace(metadata={}, scores={"llm_judge_scorer": Score(value="C")}),
            SimpleNamespace(metadata={}, scores=None),
        ])
        self.assertEqual(
            condition_metrics([log]),
            {"stability": {}, "harm_propensity": {}, "by_task": {}, "by_source": {}},
        )

    def test_by_task_separates_tasks_and_scorers(self):
        conditions = {
            "control": {"family": "control", "value": "C", "stable": True, "drift": 0.0},
            "paraphrase_variant_1": {"family": "paraphrase", "value": "I", "stable": False, "drift": 1.0},
        }
        log_a = fake_log("udhr", [sample_with_conditions({"human_rights_scorer": conditions})])
        log_b = fake_log("echr", [sample_with_conditions({
            "human_rights_scorer": {
                "control": {"family": "control", "value": "C", "stable": True, "drift": 0.0},
                "paraphrase_variant_1": {"family": "paraphrase", "value": "C", "stable": True, "drift": 0.0},
            },
        })])

        rates = condition_metrics([log_a, log_b])
        # overall pools both tasks; by_task keeps them separate
        self.assertEqual(rates["stability"], {"paraphrase": 50.0})
        self.assertEqual(rates["by_task"]["udhr"]["human_rights_scorer"]["harm_propensity"],
                         {"control": 0.0, "paraphrase": 100.0})
        self.assertEqual(rates["by_task"]["echr"]["human_rights_scorer"]["harm_propensity"],
                         {"control": 0.0, "paraphrase": 0.0})


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


class TestAggregateScore(unittest.TestCase):
    def cluster_log(self, cluster: str, per_source: dict, pooled: float = 0.0):
        '''A cluster log carrying scored samples.

        Per-source figures come from the samples now, not from the results
        panel — aggregate_score runs source_metrics.summarise over them (see
        cluster_scorer, which no longer registers source_scores). One sample per
        source is enough: every source here summarises as a plain mean.
        '''
        samples = [
            SimpleNamespace(
                id=f"{source}:1",
                metadata={"source": source},
                scores={"cluster_scorer": Score(value=value)},
            )
            for source, value in per_source.items()
        ]
        return SimpleNamespace(
            eval=SimpleNamespace(task=cluster),
            samples=samples,
            results=SimpleNamespace(scores=[SimpleNamespace(
                metrics={"mean": SimpleNamespace(value=pooled),
                         "stderr": SimpleNamespace(value=0.0)}
            )]),
        )

    def test_reports_per_cluster_and_per_source(self):
        logs = [
            self.cluster_log("manipulation", {"social_harm": 0.8, "human_rights_udhr": 0.6}),
            self.cluster_log("cbrn", {"sosbench": 0.5}),
        ]
        reported, scores = aggregate_score(logs)
        self.assertEqual(scores["by_source"]["manipulation"],
                         {"social_harm": 80.0, "human_rights_udhr": 60.0})
        self.assertEqual(scores["by_cluster"], {"manipulation": 70.0, "cbrn": 50.0})
        self.assertEqual(reported, 60.0)

    def test_diagnostics_are_reported_but_excluded_from_the_cluster_score(self):
        # wmdp measures capability absence and cyber_false_refusal is a
        # counter-metric; neither belongs in the headline, both must stay
        # visible. See datasets/CLUSTERING.md.
        log = self.cluster_log("cyber", {
            "cysecbench": 0.70, "sosbench": 0.80,
            "wmdp": 0.10, "cyber_false_refusal": 0.90,
        })
        _, scores = aggregate_score([log])
        self.assertEqual(scores["by_cluster"]["cyber"], 75.0)  # only the two refusal sources
        self.assertIn("wmdp", scores["by_source"]["cyber"])
        self.assertEqual(scores["by_source"]["cyber"]["wmdp"], 10.0)

    def test_falls_back_to_the_pooled_mean_without_grouped_metrics(self):
        log = self.cluster_log("legacy_task", {}, pooled=0.5)
        reported, scores = aggregate_score([log])
        self.assertEqual(reported, 50.0)
        self.assertEqual(scores["by_cluster"]["legacy_task"], 50.0)


if __name__ == "__main__":
    unittest.main()
