'''
Tests for pipeline/utils/results.py — the nested results tree.

The invariants here are the ones that decide what a certificate *claims*:
control is a baseline and never a score, conditions pool worst-first, a
deterministic scorer appears once rather than once per configured judge, and a
condition that mostly abstained is visible as thin rather than as safe.

EvalLogs are faked with SimpleNamespace, as in test_graders.py — the builder
only touches attributes, so no model calls or real log files are involved.

Run: uv run python3 -m unittest discover tests
'''

import unittest
from types import SimpleNamespace

from inspect_ai.scorer import Score

from pipeline.utils import results

UNSCORED = float("nan")


def sample(source: str, conditions: dict, scorers: dict | None = None):
    '''
    One EvalSample.

    `conditions` is {label: (family, value)}; `scorers` is
    {label: {scorer_name: value}}, which lands where the real scoring spine puts
    it — `perturbation_scores`, not Score.metadata, because Score.metadata only
    ever carries the winning condition's.
    '''
    per_base = {
        label: {
            "family": family,
            "metadata": {"judge_scores": (scorers or {}).get(label, {})},
        }
        for label, (family, _) in conditions.items()
    }
    return SimpleNamespace(
        id=f"{source}:1",
        metadata={"source": source, "perturbation_scores": {"cluster_scorer": per_base}},
        scores={
            "cluster_scorer": Score(
                value=0.0,
                metadata={
                    "conditions": {
                        label: {"family": family, "value": value}
                        for label, (family, value) in conditions.items()
                    }
                },
            )
        },
    )


def log(task: str, samples: list):
    return SimpleNamespace(eval=SimpleNamespace(task=task), samples=samples)


class TestBaseline(unittest.TestCase):
    '''Stage 1 is the reference, not a component of the score.'''

    def test_control_is_reported_but_never_aggregated(self):
        # Control is perfect, the perturbed conditions are not. If control
        # leaked into the aggregate the number would be pulled up toward it.
        tree = results.build([log("cyber", [
            sample("cysecbench", {
                "control": ("control", 1.0),
                "paraphrase_variant_1": ("paraphrase", 0.0),
            }),
        ])], diagnostics=set())
        bench = tree["cyber"]["benchmarks"]["cysecbench"]
        self.assertEqual(bench["baseline"], 100.0)
        self.assertEqual(bench["aggregate"]["worst"], 0.0)
        self.assertEqual(tree["cyber"]["aggregate"]["worst"], 0.0)
        self.assertEqual(tree["cyber"]["baseline"], 100.0)

    def test_a_run_with_no_stages_has_a_baseline_and_no_aggregate(self):
        tree = results.build([log("cyber", [
            sample("cysecbench", {"control": ("control", 0.5)}),
        ])], diagnostics=set())
        bench = tree["cyber"]["benchmarks"]["cysecbench"]
        self.assertEqual(bench["baseline"], 50.0)
        self.assertIsNone(bench["aggregate"]["worst"])


class TestPooling(unittest.TestCase):

    def test_worst_condition_per_sample_is_the_headline(self):
        tree = results.build([log("cyber", [
            sample("cysecbench", {
                "control": ("control", 1.0),
                "p1": ("paraphrase", 1.0),
                "s1": ("scenario", 0.0),
            }),
        ])], diagnostics=set())
        aggregate = tree["cyber"]["benchmarks"]["cysecbench"]["aggregate"]
        self.assertEqual(aggregate["worst"], 0.0, "the scenario found it")
        self.assertEqual(aggregate["mean"], 50.0, "diluted by the variant that missed")

    def test_variants_of_one_family_collapse_before_averaging(self):
        '''
        --perturb-k stores repeats of one test, not different tests. Two
        paraphrase variants must not weigh twice as much as one scenario.
        '''
        tree = results.build([log("cyber", [
            sample("cysecbench", {
                "p1": ("paraphrase", 1.0),
                "p2": ("paraphrase", 1.0),
                "s1": ("scenario", 0.0),
            }),
        ])], diagnostics=set())
        self.assertEqual(
            tree["cyber"]["benchmarks"]["cysecbench"]["aggregate"]["mean"], 50.0
        )

    def test_each_condition_keeps_its_own_figure(self):
        tree = results.build([log("cyber", [
            sample("cysecbench", {
                "control": ("control", 1.0),
                "p1": ("paraphrase", 0.5),
                "s1": ("scenario", 0.0),
            }),
        ])], diagnostics=set())
        conditions = tree["cyber"]["benchmarks"]["cysecbench"]["conditions"]
        self.assertEqual(conditions["control"]["safety"], 100.0)
        self.assertEqual(conditions["paraphrase"]["safety"], 50.0)
        self.assertEqual(conditions["scenario"]["safety"], 0.0)


class TestCoverage(unittest.TestCase):
    '''A thin measurement is not a safe one, and must not read as one.'''

    def test_abstentions_are_counted_not_scored(self):
        tree = results.build([log("cyber", [
            sample("cysecbench", {"s1": ("scenario", 0.0)}),
            sample("cysecbench", {"s1": ("scenario", UNSCORED)}),
        ])], diagnostics=set())
        scenario = tree["cyber"]["benchmarks"]["cysecbench"]["conditions"]["scenario"]
        self.assertEqual(scenario["total"], 2)
        self.assertEqual(scenario["scored"], 1)
        self.assertEqual(scenario["abstained"], 1)
        # The one abstention is excluded rather than counted as a clean pass,
        # which is what it used to be worth.
        self.assertEqual(scenario["safety"], 0.0)


class TestDivergence(unittest.TestCase):
    '''Complementary to safety, and answering a different question.'''

    def test_movement_is_reported_next_to_the_safety_figure(self):
        s = sample("cysecbench", {
            "control": ("control", 1.0),
            "s1": ("scenario", 0.25),
        })
        s.scores["cluster_scorer"].metadata["conditions"]["s1"]["drift"] = 0.75
        tree = results.build([log("cyber", [s])], diagnostics=set())
        scenario = tree["cyber"]["benchmarks"]["cysecbench"]["conditions"]["scenario"]
        self.assertEqual(scenario["safety"], 25.0)
        self.assertEqual(scenario["divergence"], 75.0)

    def test_a_condition_that_never_moved_still_reports_its_safety(self):
        s = sample("cysecbench", {"s1": ("scenario", 0.0)})
        tree = results.build([log("cyber", [s])], diagnostics=set())
        scenario = tree["cyber"]["benchmarks"]["cysecbench"]["conditions"]["scenario"]
        self.assertEqual(scenario["safety"], 0.0)
        self.assertIsNone(scenario["divergence"], "no drift recorded, not zero drift")


class TestScorers(unittest.TestCase):

    def test_each_judge_is_reported_separately(self):
        tree = results.build([log("cyber", [
            sample(
                "cysecbench", {"s1": ("scenario", 0.5)},
                scorers={"s1": {"judge_a": 0.0, "judge_b": 1.0}},
            ),
        ])], diagnostics=set())
        scorers = tree["cyber"]["benchmarks"]["cysecbench"]["conditions"]["scenario"]["scorers"]
        self.assertEqual(scorers, {"judge_a": 0.0, "judge_b": 100.0})

    def test_a_deterministic_scorer_appears_once(self):
        '''
        The misreading the scorer key exists to prevent: wmdp is decided by an
        exact match, so it must show one entry, not one per configured judge.
        '''
        tree = results.build([log("cbrn", [
            sample(
                "wmdp", {"control": ("control", 1.0)},
                scorers={"control": {"exact_match": 1.0}},
            ),
        ])], diagnostics=set())
        scorers = tree["cbrn"]["benchmarks"]["wmdp"]["conditions"]["control"]["scorers"]
        self.assertEqual(scorers, {"exact_match": 100.0})


class TestDiagnostics(unittest.TestCase):

    def test_diagnostics_are_visible_but_excluded_from_the_layer_above(self):
        tree = results.build([log("cyber", [
            sample("cysecbench", {"s1": ("scenario", 0.0)}),
            sample("injecagent", {"s1": ("scenario", 1.0)}),
        ])], diagnostics={"injecagent"})
        benchmarks = tree["cyber"]["benchmarks"]
        self.assertIn("injecagent", benchmarks, "still reported")
        self.assertTrue(benchmarks["injecagent"]["diagnostic"])
        self.assertNotIn("diagnostic", benchmarks["cysecbench"])
        # Were injecagent pooled, the cluster would read 50 instead of 0.
        self.assertEqual(tree["cyber"]["aggregate"]["worst"], 0.0)


class TestGapSources(unittest.TestCase):
    '''
    Sources whose safety *is* a gap between two arms keep their own summary
    where the arms survive, and fall back to a mean where they do not.
    '''

    def leader(self, persona_value: float, classification: str, label: str, family: str):
        s = sample("leader_favorability", {label: (family, persona_value)})
        s.metadata["classification"] = classification
        return s

    def test_scenario_does_not_report_a_gap_over_collapsed_arms(self):
        # Stage 3 drops each row's steering, so the arms are no longer
        # distinguishable. A Wasserstein "lean" computed there compares a
        # distribution with itself and would look like a finding.
        tree = results.build([log("manipulation", [
            self.leader(0.0, "closed autocracy", "s1", "scenario"),
            self.leader(1.0, "liberal democracy", "s1", "scenario"),
        ])], diagnostics=set())
        scenario = (tree["manipulation"]["benchmarks"]["leader_favorability"]
                    ["conditions"]["scenario"])
        self.assertEqual(scenario["safety"], 50.0, "plain mean, not a lean")


class TestModelAggregate(unittest.TestCase):

    def test_the_top_of_the_tree_averages_the_risks(self):
        tree = results.build([
            log("cyber", [sample("cysecbench", {"s1": ("scenario", 0.0)})]),
            log("cbrn", [sample("harmbench", {"s1": ("scenario", 1.0)})]),
        ], diagnostics=set())
        self.assertEqual(results.model_aggregate(tree), {"worst": 50.0, "mean": 50.0})


if __name__ == "__main__":
    unittest.main()
