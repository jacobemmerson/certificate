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
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from pipeline.artifacts import (
    load_family,
    sample_ids,
    task_name,
    validate_artifacts,
    write_family,
)
from pipeline.generation import SampleView, generate_framing
from pipeline.stage2_perturbation.adapters import ITEM, elicitation_family, item_text, render


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


@task
def fixture_task():
    """A small task whose samples carry the cluster schema's perturbation
    split. Registered via @task because artifacts key off the registry name."""
    return Task(
        dataset=[
            Sample(
                input=f"Statement: item {i}\nAnswer on the scale:",
                id=f"s{i}",
                metadata={
                    "item_text": f"item {i}",
                    "prompt_template": f"Statement: {ITEM}\nAnswer on the scale:",
                    "elicitation_family": "opinion",
                },
            )
            for i in range(3)
        ],
    )


class ArtifactStoreTestCase(unittest.TestCase):
    """Base: the fixture task plus a temp GENERATED_DIR."""

    @classmethod
    def setUpClass(cls):
        cls.task = fixture_task()
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

    def test_framing_not_required_when_no_sample_qualifies(self):
        # the fixture task's samples are "opinion", so framing IS required...
        with self.assertRaises(FileNotFoundError):
            validate_artifacts(self.benchmarks, families=["framing"], simulate=False)
        # ...but a task whose every sample is generic is skipped entirely
        generic = Task(
            dataset=[Sample(input="x", id="a", metadata={"elicitation_family": "generic"})],
            name="all_generic",
        )
        self.assertFalse(artifacts.framing_applies(generic))


class TestPerturbationSplit(unittest.TestCase):
    """The split is per-sample data now, not a per-task adapter registry."""

    def test_wrapper_is_reinjected_around_a_rewrite(self):
        # An item whose elicitation contract is parsed directly (a scale +
        # JSON block) must only expose its statement to a rewrite.
        view = SampleView.of(Sample(
            input=f"Statement: Obedience is a virtue.\nAnswer on the scale:", id="s1",
            metadata={
                "item_text": "Obedience is a virtue.",
                "prompt_template": f"Statement: {ITEM}\nAnswer on the scale:",
                "elicitation_family": "opinion",
            },
        ))
        self.assertEqual(item_text(view), "Obedience is a virtue.")
        rendered = render(view, "REWRITTEN")
        self.assertIn("REWRITTEN", rendered)
        self.assertIn("Answer on the scale:", rendered)  # contract survives
        self.assertNotIn("Obedience", rendered)

    def test_bare_sample_falls_back_to_the_whole_prompt(self):
        view = SampleView.of(Sample(input="the whole prompt", id="s1"))
        self.assertEqual(item_text(view), "the whole prompt")
        self.assertEqual(render(view, "new"), "new")
        self.assertEqual(elicitation_family(view), "generic")


class TestGenerateFraming(unittest.TestCase):
    def test_compliance_templates_render_one_row_each(self):
        rows = generate_framing([Sample(
            input="do the thing", id="a",
            metadata={"elicitation_family": "compliance"},  # 3 templates
        )])
        self.assertEqual(len(rows), 3)
        self.assertEqual({r["id"] for r in rows}, {"a"})
        self.assertTrue(all(r["condition"].startswith("framing_") for r in rows))
        self.assertTrue(any("do the thing" in r["query"] for r in rows))

    def test_generic_elicitation_yields_no_rows(self):
        self.assertEqual(generate_framing([Sample(input="x", id="a")]), [])

    def test_mixed_families_skip_only_the_generic_samples(self):
        # The reason the adapter registry had to go: one cluster dataset holds
        # several elicitation families, so the skip is per sample.
        rows = generate_framing([
            Sample(input="do the thing", id="a", metadata={"elicitation_family": "compliance"}),
            Sample(input="list some people", id="b", metadata={"elicitation_family": "generic"}),
        ])
        self.assertEqual({r["id"] for r in rows}, {"a"})


if __name__ == "__main__":
    unittest.main()
