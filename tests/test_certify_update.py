'''
Tests for certify.py::update — the models.json merge.

Untested until now, and it holds two rules that are easy to break and expensive
to notice: a `--only` rerun must leave every other risk's results alone, and a
rerun that came back partial must never overwrite a previously complete result
with worse-quality data.

Run: uv run python3 -m unittest discover tests
'''

import json
import os
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.model._generate_config import active_generate_config, set_active_generate_config
from inspect_ai.scorer import match
from inspect_ai.solver import generate

import certify
from pipeline.utils import graders


def entry(model_id: str, risks: dict, statuses: dict | None = None) -> dict:
    '''A stored model record, with one benchmark subtree per risk.'''
    return {
        "id": model_id,
        "name": model_id,
        "scores": {risk: value for risk, value in risks.items()},
        "aggregate": {"worst": 0.0, "mean": 0.0},
        "results": {
            risk: {
                "aggregate": {"worst": value, "mean": value},
                "baseline": 100.0,
                "benchmarks": {f"{risk}_bench": {"aggregate": {"worst": value}}},
            }
            for risk, value in risks.items()
        },
        "status": statuses or {
            risk: {"status": "success"} for risk in risks
        },
    }


class TestUpdate(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.models_dir = Path(self.tmp.name) / "models"
        patcher = mock.patch.object(graders, "MODELS_DIR", self.models_dir)
        patcher.start()
        self.addCleanup(patcher.stop)

    def written(self) -> list:
        return json.loads((self.models_dir / "models.json").read_text())

    def test_a_rerun_of_one_risk_preserves_the_others(self):
        stored = [entry("m", {"cbrn": 50.0, "cyber": 60.0, "manipulation": 70.0})]
        rerun = entry("m", {"cbrn": 42.0})

        certify.update(rerun, stored, idx=0)

        results = self.written()[0]["results"]
        self.assertEqual(sorted(results), ["cbrn", "cyber", "manipulation"])
        self.assertEqual(results["cbrn"]["aggregate"]["worst"], 42.0, "rerun wins")
        self.assertEqual(results["cyber"]["aggregate"]["worst"], 60.0, "untouched")
        self.assertEqual(results["manipulation"]["aggregate"]["worst"], 70.0)

    def test_the_model_aggregate_is_recomputed_across_every_risk(self):
        '''
        Not just the ones this run touched — otherwise a --only rerun would
        report a headline covering a single risk.
        '''
        stored = [entry("m", {"cbrn": 0.0, "cyber": 100.0})]
        rerun = entry("m", {"cbrn": 50.0})

        certify.update(rerun, stored, idx=0)

        self.assertEqual(self.written()[0]["aggregate"]["worst"], 75.0)

    def test_a_partial_rerun_never_replaces_a_complete_result(self):
        stored = [entry("m", {"cbrn": 90.0})]
        partial = entry(
            "m", {"cbrn": 10.0},
            statuses={"cbrn": {"status": "partial", "completed_samples": 3}},
        )

        certify.update(partial, stored, idx=0)

        written = self.written()[0]
        self.assertEqual(written["results"]["cbrn"]["aggregate"]["worst"], 90.0)
        self.assertEqual(written["scores"]["cbrn"], 90.0)
        self.assertEqual(written["status"]["cbrn"]["status"], "success")

    def test_a_new_model_is_appended(self):
        stored = [entry("a", {"cbrn": 50.0})]
        graders.write_json_atomic(graders.model_result_path("a"), stored[0])
        certify.update(entry("b", {"cbrn": 60.0}), stored, idx=-1)
        self.assertEqual([m["id"] for m in self.written()], ["a", "b"])

    def test_the_per_model_file_is_the_source_of_truth(self):
        certify.update(entry("m", {"cbrn": 42.0}), [], idx=-1)
        per_model = json.loads((self.models_dir / "results" / "m.json").read_text())
        self.assertEqual(per_model["scores"]["cbrn"], 42.0)
        self.assertEqual(self.written(), [per_model])
        self.assertEqual(
            [p.name for p in self.models_dir.iterdir() if p.suffix != ".json" and p.is_file()],
            [], "no tmp file left behind",
        )

    def test_two_models_updated_from_stale_lists_both_survive(self):
        # Two array-job processes each loaded `models` before the other wrote.
        certify.update(entry("a", {"cbrn": 1.0}), [], idx=-1)
        certify.update(entry("b", {"cbrn": 2.0}), [], idx=-1)
        self.assertEqual([m["id"] for m in self.written()], ["a", "b"])
        models, idx = graders.load_models_with_check("b")
        self.assertEqual((len(models), idx), (2, 1))

    def test_result_path_is_one_path_component(self):
        path = graders.model_result_path("author/bar:free")
        self.assertEqual(path.parent, self.models_dir / "results")
        self.assertEqual(path.name, "author_bar_free.json")

    def test_rebuild_sorts_by_id_and_is_idempotent(self):
        certify.update(entry("Zed", {"cbrn": 1.0}), [], idx=-1)
        certify.update(entry("alpha", {"cbrn": 2.0}), [], idx=-1)
        first = (self.models_dir / "models.json").read_bytes()
        graders.rebuild_models_json()
        self.assertEqual((self.models_dir / "models.json").read_bytes(), first)
        self.assertEqual([m["id"] for m in self.written()], ["alpha", "Zed"])

    def test_split_script_is_idempotent_and_keeps_aa_fields(self):
        stored = [entry("b", {"cbrn": 1.0}), entry("a", {"cbrn": 2.0})]
        stored[0]["aa_intelligence_index"] = 55.3
        self.models_dir.mkdir()
        (self.models_dir / "models.json").write_text(json.dumps(stored))
        script = Path(certify.__file__).parent / "scripts" / "split_models_json.py"

        runpy.run_path(str(script), run_name="__main__")
        first = (self.models_dir / "models.json").read_bytes()
        runpy.run_path(str(script), run_name="__main__")

        self.assertEqual((self.models_dir / "models.json").read_bytes(), first)
        self.assertEqual([m["id"] for m in self.written()], ["a", "b"])
        self.assertEqual(self.written()[1]["aa_intelligence_index"], 55.3)
        self.assertEqual(sorted(p.name for p in (self.models_dir / "results").glob("*.json")), ["a.json", "b.json"])


class TestParse(unittest.TestCase):

    def parse(self, *argv):
        with mock.patch.object(sys, "argv", ["certify.py", "-m", "mockllm/model", *argv]):
            return certify.parse()

    def test_epochs_is_an_int(self):
        self.assertEqual(self.parse("--epochs", "2").epochs, 2)
        self.assertEqual(self.parse().epochs, 1)


def usage_log(usage: dict, status="success"):
    model_usage = {
        name: SimpleNamespace(input_tokens=i, output_tokens=o, total_cost=c)
        for name, (i, o, c) in usage.items()
    }
    return SimpleNamespace(
        status=status, samples=[], results=None,
        stats=SimpleNamespace(model_usage=model_usage),
    )


class TestUsage(unittest.TestCase):

    def test_usage_is_summed_per_model_across_logs(self):
        record = certify.check_status(
            [usage_log({"m": (10, 2, None), "judge": (5, 1, 0.5)}),
             usage_log({"m": (1, 1, None), "judge": (5, 1, 0.25)})],
            run_id="current",
        )
        self.assertEqual(record["usage"]["m"], {"input_tokens": 11, "output_tokens": 3, "total_cost": None})
        self.assertEqual(record["usage"]["judge"], {"input_tokens": 10, "output_tokens": 2, "total_cost": 0.75})
        self.assertEqual(record["run_id"], "current")

    def test_a_log_without_stats_yields_empty_usage(self):
        record = certify.check_status([SimpleNamespace(status="success", samples=[], results=None)])
        self.assertEqual(record["usage"], {})
        self.assertIsNone(record["run_id"])


def tiny_task(name: str) -> Task:
    return Task(
        dataset=[Sample(input="say ok", target="ok", id=f"{name}-{i}") for i in range(2)],
        solver=generate(), scorer=match(), name=name,
    )


def eval_args(**overrides):
    base = dict(limit=None, epochs=1, max_connections=4, max_retries=1, attempt_timeout=60,
                timeout=60, working_limit=60)
    base.update(overrides)
    return SimpleNamespace(**base)


class TestRunDir(unittest.TestCase):

    def test_default_and_limit_runs_use_their_own_dir(self):
        self.assertEqual(certify.run_dir("m", "current", None), Path("logs/m/current"))
        self.assertEqual(certify.run_dir("m", "current", 3), Path("logs/m/current-limit3"))

    def test_move_aside_keeps_the_old_dir_under_a_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            current = Path(tmp) / "current"
            current.mkdir()
            (current / "x.eval").write_text("")
            moved = certify.move_aside(current)
            self.assertFalse(current.exists())
            self.assertTrue(moved.name.startswith("current-"))
            self.assertTrue((moved / "x.eval").exists())
            self.assertIsNone(certify.move_aside(current), "nothing to move is not an error")


class TestEvalSetResume(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name) / "current"
        os.environ["INSPECT_DISPLAY"] = "none"
        # eval_set leaves its GenerateConfig (cache=True) as the process-wide
        # default, which would serve later mockllm tests cached answers
        self.addCleanup(set_active_generate_config, active_generate_config())

    def test_second_run_reruns_nothing_and_logs_have_samples(self):
        first = certify.start_eval([tiny_task("t")], "mockllm/model", {}, self.run, eval_args())
        second = certify.start_eval([tiny_task("t")], "mockllm/model", {}, self.run, eval_args())
        self.assertEqual(len(list(self.run.glob("*.eval"))), 1, "no new log on resume")
        self.assertEqual(first[0].eval.run_id, second[0].eval.run_id)
        self.assertEqual(len(second[0].samples or []), 2, "reused header was re-read")

    def test_changed_task_in_same_run_dir_exits_with_hint(self):
        certify.start_eval([tiny_task("t")], "mockllm/model", {}, self.run, eval_args())
        with self.assertRaises(SystemExit) as ctx:
            certify.start_eval([tiny_task("other")], "mockllm/model", {}, self.run, eval_args())
        self.assertIn("--rerun", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()


class TestPartialScores(TestUpdate):
    '''
    A benchmark whose logs errored past fail_on_error is not scored into
    `scores`, but the samples that did complete still carry judgments. Those
    land in `partial_scores` so the final scoring step can decide what a
    provider-blocked sample is worth, instead of the figure being discarded.

    The merge rules that matter: a partial figure must never survive alongside
    the complete one that replaces it, and a --only rerun must leave other
    risks' partial figures alone.
    '''

    def test_a_partial_figure_is_stored_and_kept_out_of_scores(self):
        rerun = entry("m", {})
        rerun["partial_scores"] = {"cyber": {"worst": 55.32, "mean": 77.45}}
        rerun["status"] = {"cyber": {"status": "failed"}}

        certify.update(rerun, [], idx=-1)

        stored = self.written()[0]
        self.assertEqual(stored["partial_scores"]["cyber"]["worst"], 55.32)
        self.assertNotIn("cyber", stored["scores"])

    def test_a_rerun_of_one_risk_preserves_other_partial_figures(self):
        stored = [entry("m", {"manipulation": 70.0})]
        stored[0]["partial_scores"] = {
            "cbrn": {"worst": 85.98, "mean": 95.26},
            "cyber": {"worst": 55.32, "mean": 77.45},
        }
        rerun = entry("m", {})
        rerun["partial_scores"] = {"cbrn": {"worst": 81.0, "mean": 90.0}}
        rerun["status"] = {"cbrn": {"status": "partial"}}

        certify.update(rerun, stored, idx=0)

        partial = self.written()[0]["partial_scores"]
        self.assertEqual(partial["cbrn"]["worst"], 81.0, "rerun wins")
        self.assertEqual(partial["cyber"]["worst"], 55.32, "untouched")

    def test_a_complete_rerun_clears_the_partial_figure_it_replaces(self):
        '''
        Otherwise models.json carries two numbers for one benchmark and the
        stale one — computed over fewer samples — reads as equally current.
        '''
        stored = [entry("m", {})]
        stored[0]["partial_scores"] = {"cbrn": {"worst": 85.98, "mean": 95.26}}
        rerun = entry("m", {"cbrn": 42.0})

        certify.update(rerun, stored, idx=0)

        written = self.written()[0]
        self.assertEqual(written["scores"]["cbrn"], 42.0)
        # The key itself is dropped when nothing partial is left, so that a
        # rerun does not add an empty one to every model in the file.
        self.assertNotIn("cbrn", written.get("partial_scores", {}))

    def test_a_demoted_rerun_drops_its_partial_figure_too(self):
        '''
        update() already discards a partial rerun over a previously complete
        result. The partial figure must go with it, or the discarded run
        reappears under another key.
        '''
        stored = [entry("m", {"cbrn": 50.0})]
        rerun = entry("m", {})
        rerun["partial_scores"] = {"cbrn": {"worst": 12.0, "mean": 20.0}}
        rerun["status"] = {"cbrn": {"status": "failed"}}

        certify.update(rerun, stored, idx=0)

        written = self.written()[0]
        self.assertEqual(written["scores"]["cbrn"], 50.0, "complete result kept")
        self.assertNotIn("cbrn", written.get("partial_scores", {}))
