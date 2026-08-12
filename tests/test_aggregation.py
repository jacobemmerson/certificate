'''
Tests for pipeline/stage4_aggregation/workflow.py — the Bradley-Terry
cross-benchmark aggregation. All cohorts are synthetic dicts shaped like
models.json entries; file I/O goes through temp directories, so no model
calls or real repo files are involved.

Run: uv run python3 -m unittest discover tests
'''

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pipeline.stage4_aggregation import workflow
from pipeline.stage4_aggregation.workflow import (
    conditional_robustness_features,
    descending_ranks,
    fit_bradley_terry,
    gpa,
    load_models,
    merge_bt_results,
    midranks,
    partition_models,
    percentiles,
    run_analysis,
    run_and_writeback,
    win_probability_scores,
)


def make_model(model_id: str, level: float, **overrides) -> dict:
    '''A complete models.json entry whose every task score is `level`.'''
    model = {
        "id": model_id,
        "name": model_id,
        "company": "test",
        "scores_meta": {
            "auth": {
                "fscale": level,
                "leader_favorability": level,
                "role_model_bias": level,
            },
            "harm": {"social_harm_bench": level},
            "hr": {
                "udhr": level,
                "udhr_individual": level,
                "udhr_government": level,
                "echr": level,
                "echr_individual": level,
                "echr_government": level,
            },
            "hist": {"history_no_push": level, "history_explicit_push": level},
        },
    }
    model.update(overrides)
    return model


def make_cohort(levels: dict[str, float]) -> list[dict]:
    return [make_model(model_id, level) for model_id, level in levels.items()]


def stage23_blocks(consistency: float, scenario_lvr: float) -> dict:
    '''Synthetic perturbations/simulations entries for all four benchmarks.'''
    return {
        "perturbations": {
            bench: {"stability": {"paraphrase": consistency, "framing": consistency + 2}}
            for bench in workflow.BENCHMARK_KEYS
        },
        "simulations": {
            bench: {"harm_propensity": {"control": 5.0, "scenario": scenario_lvr}}
            for bench in workflow.BENCHMARK_KEYS
        },
    }


class TestBradleyTerryFit(unittest.TestCase):
    def test_recovers_strict_ordering(self):
        ids = ["a", "b", "c", "d"]
        features = {
            "x": [90.0, 70.0, 50.0, 30.0],
            "y": [95.0, 75.0, 55.0, 35.0],
        }
        _, ranks, theta = fit_bradley_terry(ids, features, tie_threshold=2.0)
        self.assertEqual(ranks, [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(sorted(theta, reverse=True), theta)

    def test_identical_models_share_midrank(self):
        ids = ["a", "b", "c"]
        features = {"x": [90.0, 90.0, 30.0]}
        _, ranks, _ = fit_bradley_terry(ids, features)
        self.assertEqual(ranks[0], 1.5)
        self.assertEqual(ranks[1], 1.5)
        self.assertEqual(ranks[2], 3.0)

    def test_all_ties_give_flat_ratings(self):
        ids = ["a", "b"]
        features = {"x": [50.0, 51.0]}
        ratings, ranks, _ = fit_bradley_terry(ids, features, tie_threshold=2.0)
        self.assertEqual(ratings, [0.0, 0.0])
        self.assertEqual(ranks, [1.5, 1.5])

    def test_none_values_are_excluded(self):
        ids = ["a", "b", "c"]
        features = {"x": [90.0, None, 30.0], "y": [80.0, 50.0, 20.0]}
        _, ranks, _ = fit_bradley_terry(ids, features)
        self.assertEqual(ranks, [1.0, 2.0, 3.0])

    def test_no_comparisons_raises(self):
        with self.assertRaises(ValueError):
            fit_bradley_terry(["a", "b"], {"x": [None, None]})


class TestRankHelpers(unittest.TestCase):
    def test_midranks_with_ties(self):
        self.assertEqual(midranks([10, 20, 20, 30]), [1.0, 2.5, 2.5, 4.0])

    def test_percentiles_single_value(self):
        self.assertEqual(percentiles([42.0]), [50.0])

    def test_percentiles_two_values(self):
        self.assertEqual(percentiles([10.0, 20.0]), [0.0, 100.0])

    def test_percentiles_all_equal(self):
        self.assertEqual(percentiles([5.0, 5.0, 5.0]), [50.0, 50.0, 50.0])

    def test_descending_ranks_tie_averaging(self):
        self.assertEqual(descending_ranks([30.0, 30.0, 10.0]), [1.5, 1.5, 3.0])


class TestScores(unittest.TestCase):
    def test_win_probability_bounds_and_ordering(self):
        theta = [2.0, 0.5, -0.5, -2.0]
        scores = win_probability_scores(theta)
        for score in scores:
            self.assertGreater(score, 0.0)
            self.assertLess(score, 100.0)
        self.assertEqual(sorted(scores, reverse=True), scores)
        self.assertAlmostEqual(sum(scores) / len(scores), 50.0, delta=5.0)

    def test_win_probability_flat_cohort(self):
        self.assertEqual(win_probability_scores([0.0, 0.0]), [50.0, 50.0])

    def test_gpa_extremes(self):
        self.assertEqual(gpa([1.0] * 24, 16), 4.0)
        self.assertEqual(gpa([16.0] * 24, 16), 0.0)

    def test_gpa_midfield_in_range(self):
        value = gpa([4.0, 8.0, 12.0], 16)
        self.assertGreater(value, 0.0)
        self.assertLess(value, 4.0)


class TestPartitionModels(unittest.TestCase):
    def test_partition_and_warning(self):
        incomplete = make_model("broken", 50.0)
        del incomplete["scores_meta"]["hist"]["history_explicit_push"]
        bare = {"id": "bare"}
        cohort = [make_model("ok", 60.0), incomplete, bare]
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            complete, skipped = partition_models(cohort)
        self.assertEqual([m["id"] for m in complete], ["ok"])
        self.assertEqual([m["id"] for m in skipped], ["broken", "bare"])
        self.assertIn("broken", buffer.getvalue())
        self.assertIn("history_explicit_push", buffer.getvalue())

    def test_load_models_needs_two_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "models.json"
            path.write_text(json.dumps([make_model("only", 50.0), {"id": "bare"}]))
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer), self.assertRaises(ValueError):
                load_models(path)


class TestMergeBtResults(unittest.TestCase):
    def test_merge_updates_and_removes_stale(self):
        models = [
            make_model("a", 50.0) | {"bt": {"stale": True}, "status": "ok"},
            make_model("b", 60.0),
            {"id": "skipped", "bt": {"stale": True}, "scores": {"harm": 1.0}},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "models.json"
            path.write_text(json.dumps(models, indent=4))
            merge_bt_results(path, {"a": {"fresh": 1}, "b": {"fresh": 2}})
            written = json.loads(path.read_text())
        self.assertEqual(written[0]["bt"], {"fresh": 1})
        self.assertEqual(written[0]["status"], "ok")
        self.assertEqual(written[1]["bt"], {"fresh": 2})
        self.assertNotIn("bt", written[2])
        self.assertEqual(written[2]["scores"], {"harm": 1.0})


class TestRunAnalysis(unittest.TestCase):
    def test_outputs_and_block_shape(self):
        cohort = make_cohort({"top": 90.0, "high": 70.0, "low": 50.0, "bottom": 30.0})
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                bt_by_id = run_analysis(cohort, output_dir=out)
            for name in (
                "derived_model_scores.csv",
                "convergence_correlations.csv",
                "pressure_rankings.csv",
                "susceptibility_rankings.csv",
                "dab_diagnostics.csv",
                "benchmark_sensitivity.csv",
                "specification_robustness.csv",
                "pairwise_win_probabilities.csv",
                "summary.json",
            ):
                self.assertTrue((out / name).exists(), name)
            summary = json.loads((out / "summary.json").read_text())
        self.assertEqual(set(bt_by_id), {"top", "high", "low", "bottom"})
        block = bt_by_id["top"]
        self.assertEqual(block["pressure"]["rank"], 1.0)
        self.assertEqual(block["pressure"]["gpa"], 4.0)
        self.assertEqual(block["cohort_size"], 4)
        self.assertNotIn("conditional_robustness", block)
        for key in ("score", "log_strength", "rating", "rank"):
            self.assertIn(key, block["pressure"])
        self.assertIn("score", block["steering_robustness"])
        for key in ("median_rank", "best_rank", "worst_rank", "top_quartile_frequency"):
            self.assertIn(key, block["robustness"])
        top_score = block["pressure"]["score"]
        for other in ("high", "low", "bottom"):
            self.assertGreater(top_score, bt_by_id[other]["pressure"]["score"])
        self.assertIn("skipped", summary["conditional_robustness"])


class TestRunAndWriteback(unittest.TestCase):
    def _cohort_file(self, tmp: str) -> Path:
        cohort = [
            make_model("strong", 90.0),
            make_model("mid", 60.0),
            make_model("weak", 30.0),
            {"id": "partial", "scores_meta": {"harm": {"social_harm_bench": 99.0}}},
        ]
        path = Path(tmp) / "models.json"
        path.write_text(json.dumps(cohort, indent=4))
        return path

    def test_writeback_targets_complete_models_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._cohort_file(tmp)
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                run_and_writeback(input_path=path, output_dir=Path(tmp) / "out")
            written = json.loads(path.read_text())
        by_id = {m["id"]: m for m in written}
        self.assertIn("bt", by_id["strong"])
        self.assertEqual(by_id["strong"]["bt"]["pressure"]["rank"], 1.0)
        self.assertNotIn("bt", by_id["partial"])

    def test_failed_analysis_leaves_models_json_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._cohort_file(tmp)
            before = path.read_bytes()
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer), \
                    mock.patch.object(workflow, "run_analysis", side_effect=RuntimeError("boom")), \
                    self.assertRaises(RuntimeError):
                workflow.run_and_writeback(input_path=path, output_dir=Path(tmp) / "out")
            self.assertEqual(path.read_bytes(), before)

    def test_no_writeback_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._cohort_file(tmp)
            before = path.read_bytes()
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                run_and_writeback(input_path=path, output_dir=Path(tmp) / "out", writeback=False)
            self.assertEqual(path.read_bytes(), before)


class TestConditionalRobustness(unittest.TestCase):
    def _covered_cohort(self) -> list[dict]:
        return [
            make_model("steady", 90.0, **stage23_blocks(consistency=95.0, scenario_lvr=5.0)),
            make_model("middling", 60.0, **stage23_blocks(consistency=70.0, scenario_lvr=30.0)),
            make_model("brittle", 30.0, **stage23_blocks(consistency=40.0, scenario_lvr=80.0)),
        ]

    def test_gate_requires_full_coverage(self):
        cohort = self._covered_cohort()
        self.assertIsNotNone(conditional_robustness_features(cohort))
        del cohort[1]["simulations"]["hist"]
        self.assertIsNone(conditional_robustness_features(cohort))

    def test_features_use_worst_family_and_scenario_stability(self):
        features = conditional_robustness_features(self._covered_cohort())
        assert features is not None
        self.assertEqual(len(features), 8)
        self.assertEqual(features["perturb_consistency_harm"], [95.0, 70.0, 40.0])
        self.assertEqual(features["scenario_stability_harm"], [95.0, 70.0, 20.0])

    def test_active_construct_ranks_and_outputs(self):
        cohort = self._covered_cohort()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                bt_by_id = run_analysis(cohort, output_dir=out)
            self.assertTrue((out / "conditional_robustness_rankings.csv").exists())
            summary = json.loads((out / "summary.json").read_text())
        self.assertEqual(bt_by_id["steady"]["conditional_robustness"]["rank"], 1.0)
        self.assertEqual(bt_by_id["brittle"]["conditional_robustness"]["rank"], 3.0)
        self.assertEqual(summary["conditional_robustness"]["name"], "conditional_robustness")

    def test_gated_run_matches_plain_cohort(self):
        cohort = self._covered_cohort()
        del cohort[0]["perturbations"]["auth"]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                bt_by_id = run_analysis(cohort, output_dir=out)
            self.assertFalse((out / "conditional_robustness_rankings.csv").exists())
        for block in bt_by_id.values():
            self.assertNotIn("conditional_robustness", block)


if __name__ == "__main__":
    unittest.main()
