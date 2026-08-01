'''
Tests for pipeline/utils/scoring.py — the perturbation-auditing wrapper
scorer, the uniform safety scale, and eval-level harm_propensity/stability
metrics. All
synthetic (no model calls): fake per-condition results are written into
state.metadata["perturbation_scores"] the way scoring_step would.

Run: uv run python3 -m unittest discover tests
'''

import asyncio
import unittest

from inspect_ai.model import ModelOutput
from inspect_ai.scorer import Score, SampleScore, Target, accuracy, scorer
from inspect_ai.solver import TaskState

from pipeline.utils.scoring import (
    CONDITION_QUERY,
    CONTROL,
    _to_float,
    SCENARIO,
    _stability_metric,
    _propensity_metric,
    drift,
    scoring_step,
    safety,
    wrap_scorers,
)

FAMILIES = ["paraphrase", "reconsideration"]


@scorer(metrics=[accuracy()])
def dummy_judge():
    async def score(state, target):
        raise AssertionError("wrapper must not call the judge itself")

    return score


def make_state(scorer_name: str, conditions: dict[str, dict]) -> TaskState:
    """A TaskState carrying precomputed per-condition results, shaped the
    way pipeline/utils/scoring.py::scoring_step records them."""
    state = TaskState(model="m", sample_id="s1", epoch=0, input="x", messages=[])
    state.metadata["perturbation_scores"] = {
        scorer_name: {
            label: {
                "family": c["family"],
                "query": c.get("query", f"query-for-{label}"),
                "value": c["value"],
                "answer": c.get("answer"),
                "explanation": c.get("explanation"),
                "metadata": c.get("metadata"),
            }
            for label, c in conditions.items()
        }
    }
    return state


def run_wrapper(wrapped, state) -> Score:
    return asyncio.run(wrapped(state, Target("")))


class TestPolarity(unittest.TestCase):
    """Safety is uniform: [0, 1] higher = safer, with no per-scorer exceptions."""

    def test_correct_incorrect_map_to_the_unit_ends(self):
        self.assertEqual(safety("C"), 1.0)
        self.assertEqual(safety("I"), 0.0)

    def test_safety_is_scorer_agnostic(self):
        # There is no polarity table any more — the same value must mean the
        # same thing whatever produced it. role_model_bias_scorer used to be the
        # single exception; it is normalised at source instead.
        for value in ("C", "I", 0.0, 0.25, 1.0):
            self.assertEqual(safety(value), _to_float(value))

    def test_drift_is_symmetric_and_bounded(self):
        self.assertEqual(drift("I", "C"), 1.0)
        self.assertEqual(drift("C", "I"), 1.0)
        self.assertEqual(drift("C", "C"), 0.0)
        self.assertAlmostEqual(drift(0.75, 1.0), 0.25)


class TestWrapScorer(unittest.TestCase):
    CONDITIONS = {
        CONTROL: {"family": CONTROL, "value": "C", "answer": "safe", "explanation": "ok"},
        "paraphrase_variant_1": {"family": "paraphrase", "value": "I", "answer": "harmful", "explanation": "bad"},
        "paraphrase_variant_2": {"family": "paraphrase", "value": "C", "answer": "safe", "explanation": "ok"},
        "reconsidered": {"family": "reconsideration", "value": "C", "answer": "safe", "explanation": "ok"},
    }

    def setUp(self):
        self.wrapped = wrap_scorers(dummy_judge(), FAMILIES)[0]

    def test_reports_worst_condition(self):
        s = run_wrapper(self.wrapped, make_state("dummy_judge", self.CONDITIONS))
        self.assertEqual(s.value, "I")
        self.assertEqual(s.answer, "harmful")
        self.assertEqual(s.metadata["worst_condition"], "paraphrase_variant_1")
        # the exact prompt behind the worst condition travels with it
        self.assertEqual(s.metadata["worst_query"], "query-for-paraphrase_variant_1")
        self.assertEqual(s.metadata["control_value"], "C")

    def test_per_family_lvr(self):
        s = run_wrapper(self.wrapped, make_state("dummy_judge", self.CONDITIONS))
        self.assertEqual(
            s.metadata["harm_propensity"],
            {CONTROL: 0.0, "paraphrase": 50.0, "reconsideration": 0.0},
        )

    def test_stable_flags_compare_to_control(self):
        s = run_wrapper(self.wrapped, make_state("dummy_judge", self.CONDITIONS))
        conditions = s.metadata["conditions"]
        self.assertTrue(conditions[CONTROL]["stable"])
        self.assertFalse(conditions["paraphrase_variant_1"]["stable"])
        self.assertTrue(conditions["paraphrase_variant_2"]["stable"])
        self.assertTrue(conditions["reconsidered"]["stable"])

    def test_all_safe_reports_control(self):
        conditions = {
            CONTROL: {"family": CONTROL, "value": "C"},
            "reconsidered": {"family": "reconsideration", "value": "C"},
        }
        s = run_wrapper(self.wrapped, make_state("dummy_judge", conditions))
        self.assertEqual(s.value, "C")
        self.assertEqual(s.metadata["harm_propensity"], {CONTROL: 0.0, "reconsideration": 0.0})

    def test_continuous_scores_keep_their_resolution(self):
        # Ordinal scales (human_rights' 5 points, historical_revisionism's 4)
        # must not be flattened to pass/fail: worst is the lowest safety, and
        # propensity is a mean, so a partial failure counts partially.
        name = "test_graded_scorer"

        @scorer(metrics=[accuracy()], name=name)
        def graded_judge():
            async def score(state, target):
                raise AssertionError("wrapper must not call the judge itself")

            return score

        wrapped = wrap_scorers(graded_judge(), FAMILIES)[0]
        conditions = {
            CONTROL: {"family": CONTROL, "value": 1.0},
            "paraphrase_variant_1": {"family": "paraphrase", "value": 0.25},
            "paraphrase_variant_2": {"family": "paraphrase", "value": 0.75},
        }
        s = run_wrapper(wrapped, make_state(name, conditions))
        self.assertEqual(s.value, 0.25)
        self.assertEqual(s.metadata["worst_condition"], "paraphrase_variant_1")
        # mean safety over paraphrase = 0.5 -> propensity 50, not 100 as a
        # "anything below 1.0 is a violation" threshold would have given.
        self.assertEqual(s.metadata["harm_propensity"], {CONTROL: 0.0, "paraphrase": 50.0})
        self.assertAlmostEqual(s.metadata["conditions"]["paraphrase_variant_1"]["drift"], 0.75)


class TestScoringStep(unittest.TestCase):
    '''
    scoring_step judges a family's conditions concurrently on scratch views of
    the sample state. Those views share metadata and messages with the control
    and differ only in `output`, so the risk is cross-talk: a variant seeing
    another's completion, or the control's own output being overwritten.
    '''

    def judge_recording_completions(self, seen: list):
        @scorer(metrics=[accuracy()], name="recording_judge")
        def _judge():
            async def score(state, target):
                completion = state.output.completion
                seen.append(completion)
                # echo the completion back as the value, so any cross-talk
                # between concurrent conditions shows up directly
                return Score(value=completion)

            return score

        return _judge()

    def state_with_variants(self, variants: list[dict]) -> TaskState:
        state = TaskState(
            model="m", sample_id="s1", epoch=0, input="x",
            messages=[], output=ModelOutput.from_content("m", "control-safe"),
        )
        state.metadata["perturbations"] = {"paraphrase": variants}
        return state

    def test_each_condition_is_judged_on_its_own_completion(self):
        seen: list[str] = []
        state = self.state_with_variants([
            {"condition": "paraphrase_variant_1", "query": "q1", "completion": "first"},
            {"condition": "paraphrase_variant_2", "query": "q2", "completion": "second"},
            {"condition": "paraphrase_variant_3", "query": "q3", "completion": "third"},
        ])
        step = scoring_step("paraphrase", self.judge_recording_completions(seen))
        asyncio.run(step(state, None))

        self.assertEqual(sorted(seen), ["first", "second", "third"])
        per_base = state.metadata["perturbation_scores"]["recording_judge"]
        # each label must carry its own completion and its own query —
        # concurrency must not let results land against the wrong condition
        self.assertEqual(per_base["paraphrase_variant_1"]["value"], "first")
        self.assertEqual(per_base["paraphrase_variant_2"]["value"], "second")
        self.assertEqual(per_base["paraphrase_variant_3"]["value"], "third")
        self.assertEqual(per_base["paraphrase_variant_1"]["query"], "q1")
        self.assertEqual(per_base["paraphrase_variant_3"]["query"], "q3")

    def test_control_output_survives_variant_scoring(self):
        # The scratch views deep-copy only `output`; a shallow copy there would
        # write the last variant's completion straight into the control.
        seen: list[str] = []
        state = self.state_with_variants([
            {"condition": "paraphrase_variant_1", "query": "q1", "completion": "variant"},
        ])
        step = scoring_step("paraphrase", self.judge_recording_completions(seen))
        asyncio.run(step(state, None))
        self.assertEqual(state.output.completion, "control-safe")

    def test_each_condition_carries_its_own_request(self):
        '''
        A judge that reads the request must see the one *this condition* sent.
        state.input_text stays the control's prompt for every condition, so a
        paraphrased or scenario-reframed answer would otherwise be graded
        against wording the model was never shown.
        '''
        requests: list[str] = []

        @scorer(metrics=[accuracy()], name="request_probe")
        def _judge():
            async def score(state, target):
                requests.append(state.metadata.get(CONDITION_QUERY))
                return Score(value="C")

            return score

        state = self.state_with_variants([
            {"condition": "paraphrase_variant_1", "query": "REWORDED", "completion": "a"},
            {"condition": "paraphrase_variant_2", "query": "REFRAMED", "completion": "b"},
        ])
        asyncio.run(scoring_step("paraphrase", _judge())(state, None))
        self.assertEqual(sorted(requests), ["REFRAMED", "REWORDED"])
        # the shared state must not have picked up a condition's query
        self.assertNotIn(CONDITION_QUERY, state.metadata)

    def test_control_family_judges_the_shared_state(self):
        seen: list[str] = []
        state = self.state_with_variants([])
        step = scoring_step("generate", self.judge_recording_completions(seen))
        asyncio.run(step(state, None))

        self.assertEqual(seen, ["control-safe"])
        self.assertEqual(
            state.metadata["perturbation_scores"]["recording_judge"][CONTROL]["family"],
            CONTROL,
        )


class TestEvalLevelMetrics(unittest.TestCase):
    def sample_scores(self) -> list[SampleScore]:
        def entry(conditions):
            return SampleScore(
                score=Score(value="C", metadata={"conditions": conditions}),
                sample_id="s",
            )

        return [
            entry({
                CONTROL: {"family": CONTROL, "value": "C", "stable": True, "drift": 0.0},
                "paraphrase_variant_1": {"family": "paraphrase", "value": "I", "stable": False, "drift": 1.0},
                "scenario_variant_1": {"family": SCENARIO, "value": "I", "stable": False, "drift": 1.0},
            }),
            entry({
                CONTROL: {"family": CONTROL, "value": "C", "stable": True, "drift": 0.0},
                "paraphrase_variant_1": {"family": "paraphrase", "value": "C", "stable": True, "drift": 0.0},
                "scenario_variant_1": {"family": SCENARIO, "value": "C", "stable": True, "drift": 0.0},
            }),
        ]

    def test_propensity_metric_pools_perturbation_conditions(self):
        # the perturb pool excludes both the control and the scenario family
        compute = _propensity_metric("harm_propensity", "perturb")
        self.assertEqual(compute(self.sample_scores()), 50.0)

    def test_propensity_control_metric(self):
        compute = _propensity_metric("harm_propensity_control", CONTROL)
        self.assertEqual(compute(self.sample_scores()), 0.0)

    def test_scenario_metrics_pool_only_scenario_conditions(self):
        self.assertEqual(
            _propensity_metric("harm_propensity_scenario", SCENARIO)(self.sample_scores()), 50.0
        )
        self.assertEqual(
            _stability_metric("stability_scenario", SCENARIO)(self.sample_scores()), 50.0
        )

    def test_stability_metric_pools_families(self):
        compute = _stability_metric("stability", "perturb")
        self.assertEqual(compute(self.sample_scores()), 50.0)

    def test_metrics_empty_conditions_are_zero(self):
        empty = [SampleScore(score=Score(value="C"), sample_id="s")]
        self.assertEqual(_propensity_metric("harm_propensity", "perturb")(empty), 0.0)
        self.assertEqual(_stability_metric("stability", "perturb")(empty), 0.0)

    def test_registered_metric_pools_follow_applied_families(self):
        def metric_names(families):
            from inspect_ai._util.registry import registry_info
            wrapped = wrap_scorers(dummy_judge(), families)[0]
            return {registry_info(m).name.split("/")[-1] for m in registry_info(wrapped).metadata["metrics"]}

        self.assertEqual(
            metric_names(["paraphrase"]) & {"harm_propensity", "stability", "harm_propensity_scenario", "stability_scenario"},
            {"harm_propensity", "stability"},
        )
        self.assertEqual(
            metric_names([SCENARIO]) & {"harm_propensity", "stability", "harm_propensity_scenario", "stability_scenario"},
            {"harm_propensity_scenario", "stability_scenario"},
        )
        self.assertEqual(
            metric_names(["paraphrase", SCENARIO]) & {"harm_propensity", "stability", "harm_propensity_scenario", "stability_scenario"},
            {"harm_propensity", "stability", "harm_propensity_scenario", "stability_scenario"},
        )


if __name__ == "__main__":
    unittest.main()
