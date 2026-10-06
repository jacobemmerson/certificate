'''
Tests for certify.py::update — the models.json merge.

Untested until now, and it holds two rules that are easy to break and expensive
to notice: a `--only` rerun must leave every other risk's results alone, and a
rerun that came back partial must never overwrite a previously complete result
with worse-quality data.

Run: uv run python3 -m unittest discover tests
'''

import contextlib
import io
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
    '''A stored model record, with one benchmark subtree per risk. `risks` maps
    risk -> tail (None for a risk whose run was not a success).'''
    def aggregate(value):
        return {"average": value, "worst": value, "tail": value, "n_items": 1}
    return {
        "id": model_id,
        "name": model_id,
        "scores": dict(risks),
        "aggregate": {"average": 0.0, "worst": 0.0, "tail": 0.0},
        "results": {
            risk: {
                "aggregate": aggregate(value),
                "baseline": 100.0,
                "by_family": {},
                "benchmarks": {f"{risk}_bench": {"aggregate": aggregate(value)}},
                "status": "ok" if value is not None else "empty",
            }
            for risk, value in risks.items()
        },
        "status": statuses or {
            risk: {"status": "success" if value is not None else "partial"}
            for risk, value in risks.items()
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
        self.assertEqual(self.written()[0]["aggregate"]["tail"], 75.0)

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

    def test_unowned_fields_and_identity_survive_a_rerun(self):
        stored = [entry("m", {"cbrn": 50.0})]
        stored[0].update(name="Custom Name", aa_intelligence_index=55.3, aa_model_match="M")
        rerun = entry("m", {"cbrn": 42.0})
        rerun["name"] = "cli-name"

        certify.update(rerun, stored, idx=0)

        written = self.written()[0]
        self.assertEqual(written["scores"]["cbrn"], 42.0)
        self.assertEqual(written["name"], "Custom Name")
        self.assertEqual((written["aa_intelligence_index"], written["aa_model_match"]), (55.3, "M"))

    def test_a_partial_run_stores_a_null_score_with_its_tree(self):
        partial = entry("m", {"cbrn": None})
        certify.update(partial, [], idx=-1)
        written = self.written()[0]
        self.assertIsNone(written["scores"]["cbrn"])
        self.assertEqual(written["status"]["cbrn"]["status"], "partial")
        self.assertIn("cbrn_bench", written["results"]["cbrn"]["benchmarks"], "tree still present")
        self.assertNotIn("partial_scores", written)

    def test_completed_risks_treats_an_old_schema_record_as_incomplete(self):
        stored = entry("m", {"cbrn": 50.0, "cyber": 60.0})
        stored["results"]["cyber"]["aggregate"] = {"mean": 70.0, "worst": 60.0}
        self.assertEqual(certify.completed_risks(stored), {"cbrn"})

    def test_completed_risks_treats_a_null_score_as_incomplete(self):
        stored = entry("m", {"cbrn": 50.0, "cyber": None})
        stored["status"]["manipulation"] = {"status": "failed"}
        self.assertEqual(certify.completed_risks(stored), {"cbrn"})


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
        env = mock.patch.dict(os.environ, {"INSPECT_DISPLAY": "none", "INSPECT_CACHE_DIR": self.tmp.name})
        env.start()
        self.addCleanup(env.stop)
        # eval_set leaves its GenerateConfig (cache=True) as the process-wide
        # default, which would serve later mockllm tests cached answers
        self.addCleanup(set_active_generate_config, active_generate_config())

    def test_second_run_reruns_nothing_and_logs_have_samples(self):
        first = certify.start_eval([tiny_task("t")], "mockllm/model", {}, self.run, eval_args())
        with contextlib.redirect_stdout(io.StringIO()) as out:
            second = certify.start_eval([tiny_task("t")], "mockllm/model", {}, self.run, eval_args())
        self.assertIn("[resume] reused finished log", out.getvalue())
        self.assertEqual(len(list(self.run.glob("*.eval"))), 1, "no new log on resume")
        self.assertEqual(first[0].eval.run_id, second[0].eval.run_id)
        self.assertEqual(len(second[0].samples or []), 2, "reused header was re-read")

    def test_other_task_in_same_run_dir_runs_and_keeps_first_log(self):
        certify.start_eval([tiny_task("t")], "mockllm/model", {}, self.run, eval_args())
        logs = certify.start_eval([tiny_task("other")], "mockllm/model", {}, self.run, eval_args())
        self.assertEqual(len(list(self.run.glob("*.eval"))), 2)
        self.assertEqual([log.eval.task for log in logs], ["other"])


class TestSimulationFlags(unittest.TestCase):
    def _parse(self, *argv):
        import sys
        with mock.patch.object(sys, "argv", ["certify.py", "-m", "openrouter/x", *argv]):
            return certify.parse()

    def test_sim_k_defaults_to_two(self):
        self.assertEqual(self._parse().sim_k, 2)

    def test_sim_classifier_default(self):
        self.assertEqual(self._parse().sim_classifier, "openrouter/google/gemini-3-flash-preview")
        self.assertEqual(self._parse("--sim-classifier", "openrouter/y").sim_classifier, "openrouter/y")

    def test_generate_sim_k_defaults_to_two(self):
        import sys
        import generate
        with mock.patch.object(sys, "argv", ["generate.py"]):
            self.assertEqual(generate.parse().sim_k, 2)


if __name__ == "__main__":
    unittest.main()
