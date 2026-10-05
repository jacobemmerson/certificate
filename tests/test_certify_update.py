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
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import certify


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
        (Path(self.tmp.name) / "models").mkdir()
        cwd = os.getcwd()
        os.chdir(self.tmp.name)
        self.addCleanup(os.chdir, cwd)

    def written(self) -> list:
        return json.loads((Path(self.tmp.name) / "models" / "models.json").read_text())

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
        certify.update(entry("b", {"cbrn": 60.0}), stored, idx=-1)
        self.assertEqual([m["id"] for m in self.written()], ["a", "b"])

    def test_the_previous_file_is_kept_as_a_safety_net(self):
        stored = [entry("m", {"cbrn": 90.0})]
        certify.update(entry("m", {"cbrn": 10.0}), stored, idx=0)
        previous = json.loads(
            (Path(self.tmp.name) / "models" / "models_previous.json").read_text()
        )
        self.assertEqual(previous[0]["results"]["cbrn"]["aggregate"]["worst"], 90.0)


class TestParse(unittest.TestCase):

    def parse(self, *argv):
        with mock.patch.object(sys, "argv", ["certify.py", "-m", "mockllm/model", *argv]):
            return certify.parse()

    def test_epochs_is_an_int(self):
        self.assertEqual(self.parse("--epochs", "2").epochs, 2)
        self.assertEqual(self.parse().epochs, 1)


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
