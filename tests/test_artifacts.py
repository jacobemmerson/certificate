'''
Tests for pipeline/artifacts.py (the datasets/generated/ store + pre-run
validation) and pipeline/generation.py's offline rendering (SampleView through
the per-sample perturbation split, deterministic framing rows). All synthetic — no
model calls; artifact files are written to a temp dir by pointing
pipeline.artifacts.GENERATED_DIR at it.

Run: uv run python3 -m unittest discover tests
'''

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import certify
import pipeline.artifacts as artifacts
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from pipeline.artifacts import (
    family_ids,
    load_family,
    sample_ids,
    task_name,
    validate_artifacts,
    write_family,
)
from pipeline.generation import SampleView, generate_framing
from pipeline.stage2_perturbation.adapters import ITEM, elicitation_family, item_text, render
from pipeline.stage2_perturbation.framing import FRAMING_TEMPLATES
from pipeline.stage3_simulation.prompts import TREE_PATHS


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
                    "elicitation_family": "compliance",
                    "families": ["paraphrase", "register", "framing"],
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
        # keyed by systemic risk, the way registry.py::init_benchmarks builds it
        self.benchmarks = {"manipulation": {"tasks": [self.task], "name": "manipulation"}}


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


class TestMissingOnly(ArtifactStoreTestCase):
    def test_fallback_rows_are_not_existing(self):
        import generate
        rows = rewrite_rows(self.ids[:2])
        rows[1]["fallback"] = True
        write_family(self.name, "paraphrase", rows, meta={"prompt_version": "3"})
        self.assertEqual(generate.existing_keys(self.name, "paraphrase"), {(self.ids[0], 1)})
        self.assertEqual(generate.existing_rows(self.name, "paraphrase"), [rows[0]])


class TestRegistryTruncation(ArtifactStoreTestCase):
    def test_k_truncates_repeat_families_only(self):
        """multilingual's variants are languages, not repeats: k=1 must keep all three."""
        import pipeline.registry as registry
        write_family(self.name, "multilingual", rewrite_rows(self.ids[:1], "multilingual", k=3), meta={})
        write_family(self.name, "paraphrase", rewrite_rows(self.ids[:1], "paraphrase", k=3), meta={})
        replayed = {}

        def recording(family):
            real = registry.REPLAY_SOLVERS[family]

            def build(rows):
                replayed[family] = rows
                return real(rows)
            return build

        # the fixture has no registered scorer to wrap; only the solver chain matters here
        with mock.patch.object(registry, "wrap_scorers", return_value=None), \
                mock.patch.object(registry, "family_ids", return_value=set(self.ids[:1])), \
                mock.patch.dict(registry.REPLAY_SOLVERS, {f: recording(f) for f in ("multilingual", "paraphrase")}):
            registry._build_task(self.task, ["multilingual", "paraphrase"], k=1)
        self.assertEqual(len(replayed["multilingual"][self.ids[0]]), 3)
        self.assertEqual(len(replayed["paraphrase"][self.ids[0]]), 1)


PARAPHRASE_VERSION = artifacts.PROMPT_VERSIONS["paraphrase"]


class TestValidateArtifacts(ArtifactStoreTestCase):
    def test_complete_rewrite_family_passes(self):
        write_family(self.name, "paraphrase", rewrite_rows(self.ids), meta={"prompt_version": PARAPHRASE_VERSION})
        validate_artifacts(self.benchmarks, families=["paraphrase"], simulate=False)

    def test_missing_file_fails_with_generate_command(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            validate_artifacts(self.benchmarks, families=["paraphrase"], simulate=False)
        self.assertIn("--only manipulation --perturb paraphrase", str(ctx.exception))

    def test_missing_sample_fails(self):
        write_family(self.name, "paraphrase", rewrite_rows(self.ids[1:]), meta={"prompt_version": PARAPHRASE_VERSION})
        with self.assertRaises(FileNotFoundError) as ctx:
            validate_artifacts(self.benchmarks, families=["paraphrase"], simulate=False)
        self.assertIn("--missing-only", str(ctx.exception))

    def test_all_fallback_sample_is_covered_with_a_warning(self):
        """Coordinator ruling (WS-B item 17): replay scores it as `missing`."""
        rows = rewrite_rows(self.ids)
        rows[0]["fallback"] = True
        write_family(self.name, "paraphrase", rows, meta={"prompt_version": PARAPHRASE_VERSION})
        with mock.patch("builtins.print") as printed:
            validate_artifacts(self.benchmarks, families=["paraphrase"], simulate=False)
        self.assertIn("1 id(s) fallback-only", " ".join(str(c.args[0]) for c in printed.call_args_list))

    def test_stale_prompt_version_fails_unless_limited(self):
        write_family(self.name, "paraphrase", rewrite_rows(self.ids), meta={"prompt_version": "0"})
        with self.assertRaises(FileNotFoundError) as ctx:
            validate_artifacts(self.benchmarks, families=["paraphrase"], simulate=False)
        self.assertIn("prompt version 0", str(ctx.exception))
        validate_artifacts(self.benchmarks, families=["paraphrase"], simulate=False, limit=2)

    def test_k_exceeding_stored_variants_fails(self):
        write_family(self.name, "paraphrase", rewrite_rows(self.ids, k=1), meta={"prompt_version": PARAPHRASE_VERSION})
        with self.assertRaises(FileNotFoundError):
            validate_artifacts(self.benchmarks, families=["paraphrase"], simulate=False, perturb_k=2)

    def test_limit_relaxes_rewrite_coverage_to_warning(self):
        # partial artifacts (generate.py --limit) must pass a certify --limit
        # smoke run — coverage shortfalls warn instead of failing...
        write_family(self.name, "paraphrase", rewrite_rows(self.ids[:1]), meta={"prompt_version": PARAPHRASE_VERSION})
        validate_artifacts(self.benchmarks, families=["paraphrase"], simulate=False, limit=2)

    def test_limit_still_requires_the_file_to_exist(self):
        # ...but the file must still exist at all
        with self.assertRaises(FileNotFoundError):
            validate_artifacts(self.benchmarks, families=["paraphrase"], simulate=False, limit=2)

    def test_reconsideration_is_live_only_and_never_validated(self):
        validate_artifacts(self.benchmarks, families=["reconsideration"], simulate=False)

    def test_family_ids_follow_the_families_column(self):
        task = fixture_task()
        ids = [str(s.id) for s in task.dataset]
        task.dataset[0].metadata["families"] = ["framing"]
        self.assertEqual(family_ids(task, "paraphrase"), set(ids[1:]))
        self.assertEqual(family_ids(task, "framing"), set(ids))

    def test_validation_expects_only_applicable_ids(self):
        task = fixture_task()
        ids = [str(s.id) for s in task.dataset]
        task.dataset[0].metadata["families"] = ["framing"]
        write_family(task_name(task), "paraphrase", rewrite_rows(ids[1:]), {})
        validate_artifacts({"x": {"tasks": [task]}}, ["paraphrase"], simulate=False)  # must not raise

    def test_orphan_ids_fail_validation(self):
        '''The 51 loss_of_control orphans: rows for ids no longer in the CSV.'''
        rows = rewrite_rows(self.ids + ["gone"])
        write_family(self.name, "paraphrase", rows, meta={"prompt_version": PARAPHRASE_VERSION})
        with self.assertRaises(FileNotFoundError) as ctx:
            validate_artifacts(self.benchmarks, families=["paraphrase"], simulate=False)
        self.assertIn("1 orphan", str(ctx.exception))

    def test_framing_not_required_when_no_sample_qualifies(self):
        with self.assertRaises(FileNotFoundError):
            validate_artifacts(self.benchmarks, families=["framing"], simulate=False)
        none = Task(
            dataset=[Sample(input="x", id="a", metadata={"elicitation_family": "compliance", "families": []})],
            name="none_apply",
        )
        self.assertEqual(family_ids(none, "framing"), set())
        validate_artifacts({"x": {"tasks": [none]}}, families=["framing"], simulate=False)


def tree_rows(ids, k: int = 1, drop_path: str | None = None, blank_path: str | None = None) -> list[dict]:
    turns = {p: f"turn {p or 'opening'}" for p in TREE_PATHS}
    if drop_path is not None:
        turns.pop(drop_path)
    if blank_path is not None:
        turns[blank_path] = "  "
    return [
        {"id": i, "variant": v, "condition": f"scenario_variant_{v}", "system": "S",
         "turns": dict(turns), "query": turns.get("", "")}
        for i in ids for v in range(1, k + 1)
    ]


class TestValidateScenario(ArtifactStoreTestCase):
    def setUp(self):
        super().setUp()
        for sample in self.task.dataset:
            sample.metadata["families"] = ["paraphrase", "scenario"]

    def test_complete_tree_passes(self):
        write_family(self.name, "scenario", tree_rows(self.ids, k=2), meta={"prompt_version": "4"})
        validate_artifacts(self.benchmarks, families=None, simulate=True, sim_k=2)

    def test_missing_id_fails_strictly(self):
        write_family(self.name, "scenario", tree_rows(self.ids[1:]), meta={})
        with self.assertRaises(FileNotFoundError) as ctx:
            validate_artifacts(self.benchmarks, families=None, simulate=True, sim_k=1)
        self.assertIn("1 sample(s) missing", str(ctx.exception))
        self.assertIn("--simulate", str(ctx.exception))

    def test_fewer_than_sim_k_variants_fails(self):
        write_family(self.name, "scenario", tree_rows(self.ids, k=1), meta={})
        with self.assertRaises(FileNotFoundError) as ctx:
            validate_artifacts(self.benchmarks, families=None, simulate=True, sim_k=2)
        self.assertIn("fewer than 2", str(ctx.exception))

    def test_missing_turn_path_fails_naming_it(self):
        write_family(self.name, "scenario", tree_rows(self.ids, drop_path="hedge/refuse"), meta={})
        with self.assertRaises(FileNotFoundError) as ctx:
            validate_artifacts(self.benchmarks, families=None, simulate=True, sim_k=1)
        self.assertIn("hedge/refuse", str(ctx.exception))

    def test_blank_turn_fails(self):
        write_family(self.name, "scenario", tree_rows(self.ids, blank_path="refuse"), meta={})
        with self.assertRaises(FileNotFoundError):
            validate_artifacts(self.benchmarks, families=None, simulate=True, sim_k=1)

    def test_missing_system_fails(self):
        rows = tree_rows(self.ids)
        del rows[0]["system"]
        write_family(self.name, "scenario", rows, meta={})
        with self.assertRaises(FileNotFoundError) as ctx:
            validate_artifacts(self.benchmarks, families=None, simulate=True, sim_k=1)
        self.assertIn("system", str(ctx.exception))

    def test_only_scenario_applicable_ids_are_expected(self):
        self.task.dataset[0].metadata["families"] = ["paraphrase"]
        write_family(self.name, "scenario", tree_rows(self.ids[1:]), meta={})
        validate_artifacts(self.benchmarks, families=None, simulate=True, sim_k=1)  # must not raise

    def test_orphan_ids_fail(self):
        # the 51-orphan bug: rows for ids not in the dataset must fail preflight
        write_family(self.name, "scenario", tree_rows(self.ids + ["ghost"]), meta={})
        with self.assertRaises(FileNotFoundError) as ctx:
            validate_artifacts(self.benchmarks, families=None, simulate=True, sim_k=1)
        self.assertIn("ghost", str(ctx.exception))

    def test_limit_relaxes_coverage_but_not_shape(self):
        write_family(self.name, "scenario", tree_rows(self.ids[1:], drop_path="hedge"), meta={})
        with self.assertRaises(FileNotFoundError) as ctx:
            validate_artifacts(self.benchmarks, families=None, simulate=True, sim_k=1, limit=2)
        self.assertIn("hedge", str(ctx.exception))
        write_family(self.name, "scenario", tree_rows(self.ids[1:]), meta={})
        validate_artifacts(self.benchmarks, families=None, simulate=True, sim_k=1, limit=2)  # warns only


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

    def test_framing_is_compliance_only(self):
        # judgment/opinion wrappers changed the ask on acceptability and Likert items
        self.assertEqual(list(FRAMING_TEMPLATES), ["compliance"])

    def test_a_row_without_framing_in_families_yields_no_rows(self):
        opted_out = Sample(input="write a story", id="a",
                           metadata={"elicitation_family": "compliance", "families": ["paraphrase"]})
        self.assertEqual(generate_framing([opted_out]), [])

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


class TestIdentityStripIsGone(unittest.TestCase):
    def test_no_code_path_knows_the_family(self):
        from pipeline.registry import ALL_PERTURB_FAMILIES
        from pipeline.stage2_perturbation.rewrite import FAMILY_SYSTEM_PROMPTS
        from pipeline.stage2_perturbation.solvers import REPLAY_SOLVERS
        import pipeline.utils.scoring as scoring
        for roster in (ALL_PERTURB_FAMILIES, FAMILY_SYSTEM_PROMPTS, REPLAY_SOLVERS, artifacts.PROMPT_VERSIONS):
            self.assertNotIn("identity_strip", roster)
        self.assertFalse(hasattr(scoring, "RESULT_FAMILIES"))

    def test_no_identity_strip_artifacts_on_disk(self):
        self.assertEqual(sorted(artifacts.GENERATED_DIR.glob("*/identity_strip*")), [])


class TestPastTense(unittest.TestCase):
    def test_prompt_asks_for_a_historical_rewrite_and_carries_the_text(self):
        from pipeline.stage2_perturbation.rewrite import FAMILY_SYSTEM_PROMPTS
        prompt = FAMILY_SYSTEM_PROMPTS["past_tense"].format(text="How is bread leavened?")
        self.assertIn("How is bread leavened?", prompt)
        self.assertIn("past tense", prompt)
        self.assertIn("output only the rewritten text", prompt)

    def test_family_is_wired_end_to_end(self):
        import asyncio
        from pipeline.registry import ALL_PERTURB_FAMILIES, PREGENERATED_FAMILIES
        from pipeline.stage2_perturbation.solvers import REPLAY_SOLVERS
        from tests.test_replay import make_state, stub_generate
        self.assertIn("past_tense", ALL_PERTURB_FAMILIES)
        self.assertIn("past_tense", PREGENERATED_FAMILIES)
        state = make_state({"families": ["past_tense"]})
        captured: list = []
        rows = {"s1": [dict(condition="past_tense_variant_1", query="How was bread leavened?")]}
        asyncio.run(REPLAY_SOLVERS["past_tense"](rows)(state, stub_generate(captured)))
        self.assertEqual(captured[0][0].text, "How was bread leavened?")
        self.assertEqual(state.metadata["perturbations"]["past_tense"][0]["condition"], "past_tense_variant_1")


class TestMultilingualRows(unittest.TestCase):
    def test_one_row_per_language_with_lang_and_condition(self):
        import asyncio
        from inspect_ai.model import ModelOutput, get_model
        from pipeline.generation import generate_rewrites
        # Same German output for every call: de is accepted, zh/ar fall back —
        # so the assertions hold whatever order the three calls complete in.
        attacker = get_model("mockllm/model", custom_outputs=[
            ModelOutput.from_content("mockllm/model", "Wie wird Brot gesäuert?") for _ in range(3)
        ])
        sample = Sample(input="How is bread leavened?", id="a",
                        metadata={"item_text": "How is bread leavened?", "families": ["multilingual"]})
        rows = asyncio.run(generate_rewrites([sample], "multilingual", attacker, k=1))
        self.assertEqual({r["condition"] for r in rows}, {"multilingual_de", "multilingual_zh", "multilingual_ar"})
        self.assertEqual({(r["variant"], r["lang"]) for r in rows}, {(1, "de"), (2, "zh"), (3, "ar")})
        by_lang = {r["lang"]: r for r in rows}
        self.assertFalse(by_lang["de"]["fallback"])
        self.assertEqual(by_lang["de"]["query"], "Wie wird Brot gesäuert?")
        self.assertTrue(by_lang["zh"]["fallback"] and by_lang["ar"]["fallback"])

    def test_rewrite_roster_matches_the_contract(self):
        from pipeline.stage2_perturbation.rewrite import FAMILY_SYSTEM_PROMPTS
        self.assertEqual(artifacts.REWRITE_FAMILIES, ("paraphrase", "register", "past_tense", "multilingual"))
        self.assertEqual(artifacts.REPEAT_FAMILIES, ("paraphrase", "register", "past_tense"))
        self.assertEqual(set(FAMILY_SYSTEM_PROMPTS), set(artifacts.REPEAT_FAMILIES))

class TestGeneratePersona(unittest.TestCase):
    def sample(self, **metadata):
        return Sample(input="How is bread leavened?", id="a",
                      metadata={"families": ["persona"], **metadata})

    def test_two_rows_per_sample_query_unchanged(self):
        from pipeline.generation import generate_persona
        from pipeline.stage2_perturbation.framing import PERSONA_TEMPLATES
        rows = generate_persona([self.sample()])
        self.assertEqual([r["condition"] for r in rows], ["persona_professional", "persona_researcher"])
        self.assertEqual({r["query"] for r in rows}, {"How is bread leavened?"})
        self.assertEqual([r["system"] for r in rows], list(PERSONA_TEMPLATES.values()))
        self.assertEqual(set(rows[0]), {"id", "condition", "system", "query"})

    def test_templates_are_benign_context_without_conduct_rules(self):
        from pipeline.stage2_perturbation.framing import PERSONA_TEMPLATES, PERSONA_VERSION
        self.assertEqual(PERSONA_VERSION, "1")
        for text in PERSONA_TEMPLATES.values():
            for forbidden in ("always comply", "never refuse", "no restrictions", "ignore"):
                self.assertNotIn(forbidden, text.lower())

    def test_row_with_its_own_system_prompt_gets_no_persona(self):
        '''Review focus 3: the persona solver replaces the system turn.'''
        from pipeline.generation import generate_persona
        self.assertEqual(generate_persona([self.sample(system_prompt="You advise a minister.")]), [])

    def test_families_gate(self):
        from pipeline.generation import generate_persona
        self.assertEqual(generate_persona([self.sample(families=["paraphrase"])]), [])

class TestFidelityAuditRoster(unittest.TestCase):
    def test_families_follow_the_rewrite_roster(self):
        import importlib
        audit = importlib.import_module("scripts.audit_variant_fidelity")
        self.assertEqual(audit.FAMILIES, (*artifacts.REWRITE_FAMILIES, "scenario"))
        system, prompt = audit.blind_prompt({"family": "multilingual", "lang": "zh", "original": "o", "text": "t"})
        self.assertIn("translation into Chinese (Simplified)", prompt)
        _, prompt = audit.blind_prompt({"family": "scenario", "original": "o", "system": "s",
                                        "turns": {"": "opening"}})
        self.assertIn("opening", prompt)

class TestPromptVersionsAndCli(unittest.TestCase):
    def test_prompt_versions_contract(self):
        from pipeline.stage3_simulation.prompts import PROMPT_VERSION as SCENARIO_VERSION
        self.assertEqual(artifacts.PROMPT_VERSIONS, {
            "paraphrase": "3", "register": "3", "past_tense": "3", "multilingual": "3",
            "framing": "3", "persona": "1", "scenario": SCENARIO_VERSION,
        })

    def test_perturb_choices(self):
        from pipeline.registry import ALL_PERTURB_FAMILIES, PREGENERATED_FAMILIES
        self.assertEqual(ALL_PERTURB_FAMILIES, {
            "paraphrase", "register", "past_tense", "multilingual", "framing", "persona", "reconsideration",
        })
        self.assertEqual(PREGENERATED_FAMILIES, ALL_PERTURB_FAMILIES - {"reconsideration"})


class TestEstimateCalls(ArtifactStoreTestCase):
    def test_counts_stored_rows_per_family(self):
        for sample in self.task.dataset:
            original = sample.metadata["families"]
            self.addCleanup(sample.metadata.__setitem__, "families", original)
            sample.metadata["families"] = [*original, "reconsideration", "scenario"]
        write_family(self.name, "paraphrase", rewrite_rows(self.ids, k=2), meta={})
        write_family(self.name, "framing",
                     [{"id": i, "condition": f"framing_{v}", "query": "q"} for i in self.ids[:2] for v in range(2)],
                     meta={})
        write_family(self.name, "scenario", rewrite_rows(self.ids, "scenario", k=2), meta={})

        estimate = certify.estimate_calls(
            self.benchmarks, families=["paraphrase", "framing", "reconsideration"],
            k=1, sim_k=2, graders=["a", "b"],
        )["manipulation"]

        # 3 control + 3 paraphrase (k=1 of 2) + 4 framing + 3 reconsideration + 3 ids x 2 scenarios x 3 turns
        self.assertEqual(estimate, {"samples": 3, "target": 31, "judge": 62, "classifier": 18})
        doubled = certify.estimate_calls(
            self.benchmarks, families=["paraphrase", "framing", "reconsideration"],
            k=1, sim_k=2, graders=["a", "b"], epochs=2,
        )["manipulation"]
        self.assertEqual(doubled, {"samples": 3, "target": 62, "judge": 124, "classifier": 36})

    def test_limit_scales_stored_counts(self):
        write_family(self.name, "paraphrase", rewrite_rows(self.ids), meta={})
        estimate = certify.estimate_calls(self.benchmarks, ["paraphrase"], k=1, sim_k=None, graders="a", limit=1)
        self.assertEqual(estimate["manipulation"], {"samples": 1, "target": 2, "judge": 2, "classifier": 0})

    def test_inapplicable_framing_is_skipped_without_an_artifact(self):
        generic = Task(
            dataset=[Sample(input="x", id="a", metadata={"families": ["paraphrase"]})],
            name="all_generic",
        )
        estimate = certify.estimate_calls({"generic": {"tasks": [generic]}}, ["framing"], k=1, sim_k=None, graders="a")
        self.assertEqual(estimate["generic"], {"samples": 1, "target": 1, "judge": 1, "classifier": 0})

    def test_multi_task_entry_accumulates(self):
        benchmarks = {"manipulation": {"tasks": [self.task, self.task]}}
        estimate = certify.estimate_calls(benchmarks, [], k=1, sim_k=None, graders="a")
        self.assertEqual(estimate["manipulation"], {"samples": 6, "target": 6, "judge": 6, "classifier": 0})


if __name__ == "__main__":
    unittest.main()
