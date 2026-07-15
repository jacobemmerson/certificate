'''
Tests for pipeline/artifacts.py (the datasets/generated/ store + pre-run
validation) and pipeline/generation.py's offline rendering (SampleView through
the per-benchmark adapters, deterministic framing rows). All synthetic — no
model calls; artifact files are written to a temp dir by pointing
pipeline.artifacts.GENERATED_DIR at it.

Run: uv run python3 -m unittest discover tests
'''

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pipeline.artifacts as artifacts
from inspect_ai.dataset import Sample
from pipeline.artifacts import (
    load_family,
    sample_ids,
    task_name,
    validate_artifacts,
    write_family,
)
from pipeline.generation import SampleView, generate_framing
from pipeline.stage1_evaluation.evals.democratic_authoritarian_bias import fscale
from pipeline.stage2_perturbation.adapters import DEFAULT_ADAPTER, get_adapter


def rewrite_rows(ids, family: str = "paraphrase", k: int = 1) -> list[dict]:
    return [
        {
            "id": i,
            "variant": v,
            "condition": f"{family}_variant_{v}",
            "text": f"text-{i}-{v}",
            "query": f"query-{i}-{v}",
            "fallback": False,
        }
        for i in ids
        for v in range(1, k + 1)
    ]


class ArtifactStoreTestCase(unittest.TestCase):
    """Base: a real registry task (fscale — no model calls at construction)
    plus a temp GENERATED_DIR."""

    @classmethod
    def setUpClass(cls):
        cls.task = fscale(llamaguard_model=None)
        cls.name = task_name(cls.task)
        cls.ids = sample_ids(cls.task)

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        patcher = mock.patch.object(artifacts, "GENERATED_DIR", Path(self._tmp.name))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)
        self.benchmarks = {"auth": {"tasks": [self.task], "name": "x"}}


class TestRoundTrip(ArtifactStoreTestCase):
    def test_write_then_load_groups_and_orders_variants(self):
        rows = rewrite_rows(self.ids[:2], "paraphrase", k=3)
        rows.reverse()  # write_family must sort for stable diffs
        write_family(self.name, "paraphrase", rows, meta={"prompt_version": "1"})

        by_id = load_family(self.name, "paraphrase")
        self.assertEqual(set(by_id), set(self.ids[:2]))
        self.assertEqual([r["variant"] for r in by_id[self.ids[0]]], [1, 2, 3])
        self.assertEqual(by_id[self.ids[0]][0]["query"], f"query-{self.ids[0]}-1")

    def test_load_missing_family_raises_with_hint(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            load_family(self.name, "paraphrase")
        self.assertIn("generate.py", str(ctx.exception))

    def test_meta_sidecar_round_trip(self):
        write_family(self.name, "framing", [], meta={"prompt_version": "1", "partial": True})
        self.assertEqual(artifacts.family_meta(self.name, "framing")["partial"], True)
        self.assertIsNone(artifacts.family_meta(self.name, "paraphrase"))


class TestValidateArtifacts(ArtifactStoreTestCase):
    def test_complete_rewrite_family_passes(self):
        write_family(self.name, "paraphrase", rewrite_rows(self.ids), meta={"prompt_version": "1"})
        validate_artifacts(self.benchmarks, families=["paraphrase"], simulate=False)

    def test_missing_file_fails_with_generate_command(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            validate_artifacts(self.benchmarks, families=["paraphrase"], simulate=False)
        self.assertIn("--only auth --perturb paraphrase", str(ctx.exception))

    def test_missing_sample_fails(self):
        write_family(self.name, "paraphrase", rewrite_rows(self.ids[1:]), meta={"prompt_version": "1"})
        with self.assertRaises(FileNotFoundError) as ctx:
            validate_artifacts(self.benchmarks, families=["paraphrase"], simulate=False)
        self.assertIn("--missing-only", str(ctx.exception))

    def test_k_exceeding_stored_variants_fails(self):
        write_family(self.name, "paraphrase", rewrite_rows(self.ids, k=1), meta={"prompt_version": "1"})
        with self.assertRaises(FileNotFoundError):
            validate_artifacts(self.benchmarks, families=["paraphrase"], simulate=False, perturb_k=2)

    def test_limit_relaxes_rewrite_coverage_to_warning(self):
        # partial artifacts (generate.py --limit) must pass a certify --limit
        # smoke run — coverage shortfalls warn instead of failing...
        write_family(self.name, "paraphrase", rewrite_rows(self.ids[:1]), meta={"prompt_version": "1"})
        validate_artifacts(self.benchmarks, families=["paraphrase"], simulate=False, limit=2)

    def test_limit_still_requires_the_file_to_exist(self):
        # ...but the file must still exist at all
        with self.assertRaises(FileNotFoundError):
            validate_artifacts(self.benchmarks, families=["paraphrase"], simulate=False, limit=2)

    def test_reconsideration_is_live_only_and_never_validated(self):
        validate_artifacts(self.benchmarks, families=["reconsideration"], simulate=False)

    def test_incomplete_scenario_warns_but_passes(self):
        # scenario coverage gaps mirror dropped reframings: warn, don't fail
        rows = rewrite_rows(self.ids[1:], "scenario")
        write_family(self.name, "scenario", rows, meta={"prompt_version": "1"})
        validate_artifacts(self.benchmarks, families=None, simulate=True)

    def test_framing_not_required_for_generic_elicitation(self):
        # fscale is "opinion", so framing IS required — absence must fail...
        with self.assertRaises(FileNotFoundError):
            validate_artifacts(self.benchmarks, families=["framing"], simulate=False)
        # ...but a generic-family task (role_model_bias) is skipped entirely
        self.assertEqual(get_adapter("role_model_bias").elicitation_family, "generic")
        self.assertFalse(artifacts.framing_applies("role_model_bias"))


class TestSampleViewRendering(unittest.TestCase):
    def test_fscale_adapter_reads_statement_and_rerenders_contract(self):
        adapter = get_adapter("fscale")
        view = SampleView.of(Sample(
            input="full rendered prompt", id="s1",
            metadata={"statement": "Obedience is a virtue."},
        ))
        self.assertEqual(adapter.item_text(view), "Obedience is a virtue.")
        rendered = adapter.render(view, "REWRITTEN")
        self.assertIn("REWRITTEN", rendered)
        self.assertNotEqual(rendered, "REWRITTEN")  # format contract re-injected

    def test_default_adapter_uses_whole_input(self):
        view = SampleView.of(Sample(input="the whole prompt", id="s1"))
        self.assertEqual(DEFAULT_ADAPTER.item_text(view), "the whole prompt")
        self.assertEqual(DEFAULT_ADAPTER.render(view, "new"), "new")


class TestGenerateFraming(unittest.TestCase):
    def test_compliance_templates_render_one_row_each(self):
        adapter = get_adapter("social_harm_bench")  # compliance: 3 templates
        rows = generate_framing([Sample(input="do the thing", id="a")], adapter)
        self.assertEqual(len(rows), 3)
        self.assertEqual({r["id"] for r in rows}, {"a"})
        self.assertTrue(all(r["condition"].startswith("framing_") for r in rows))
        self.assertTrue(any("do the thing" in r["query"] for r in rows))

    def test_generic_elicitation_yields_no_rows(self):
        rows = generate_framing([Sample(input="x", id="a")], DEFAULT_ADAPTER)
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
