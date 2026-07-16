"""CLI/configuration tests which never call external models."""

from __future__ import annotations

from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from inspect_ai import Task, eval, task_with
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import CORRECT, Score, accuracy, scorer
from inspect_ai.solver import generate

from certify import (
    _safe_exception,
    build_protocol_config,
    execution_modes,
    parse,
    persisted_config,
    result_benchmark_key,
    skip_benchmark_keys,
    summarize_process,
)
from pipeline.agentic import AUDIT_METADATA_KEY, AgenticConfig, Condition
from pipeline.registry import (
    ALL_PERTURB_FAMILIES,
    apply_condition,
    apply_stages,
    select_paired_samples,
    selected_ids_cover_benchmarks,
)
from pipeline.stage1_evaluation.evals.llm_human_rights import (
    OPEN_SYSTEM_PROMPT,
    udhr_individual,
)


@scorer(metrics=[accuracy()])
def output_scorer():
    async def score(state, target):
        return Score(value=CORRECT)

    return score


def synthetic_benchmarks() -> dict:
    return {
        "tiny": {
            "name": "tiny",
            "tasks": [
                Task(
                    name="tiny_task",
                    dataset=[
                        Sample(id=f"id-{index}", input=f"request {index}")
                        for index in range(10)
                    ],
                    solver=generate(),
                    scorer=output_scorer(),
                )
            ],
        }
    }


class TestCliValidation(unittest.TestCase):
    def test_omitted_condition_preserves_default_stage_behavior(self):
        args = parse(["--model", "mockllm/model"])
        self.assertIsNone(args.condition)
        self.assertEqual(set(args.perturb), set(ALL_PERTURB_FAMILIES))
        self.assertEqual(execution_modes(args), (True, False))
        self.assertEqual(args.sample_seed, 0)
        self.assertIsNone(build_protocol_config(args).budget)

    def test_explicit_c0_is_clean(self):
        args = parse(["--model", "mockllm/model", "--condition", "c0"])
        self.assertEqual(args.condition, "c0")
        self.assertEqual(execution_modes(args), (False, False))

    def test_removed_compatibility_flags_are_rejected(self):
        for removed in ("--agentic-level", "--specialist", "--analyst", "--critic"):
            with self.subTest(flag=removed), self.assertRaises(SystemExit):
                parse(["--model", "mockllm/model", removed, "value"])

    def test_unknown_only_keys_fail_during_parse(self):
        with self.assertRaises(SystemExit):
            parse(
                [
                    "--model",
                    "mockllm/model",
                    "--only",
                    "harm",
                    "does_not_exist",
                ]
            )

    def test_provider_exception_text_is_sanitized(self):
        safe = _safe_exception(
            RuntimeError("Authorization: Bearer SECRET raw request body")
        )
        self.assertIn("RuntimeError", safe)
        self.assertIn("details omitted", safe)
        self.assertNotIn("SECRET", safe)
        self.assertNotIn("request body", safe)

    def test_unsupported_combinations_fail_during_parse(self):
        with self.assertRaises(SystemExit):
            parse(
                [
                    "--model",
                    "mockllm/model",
                    "--condition",
                    "c2",
                    "--perturb",
                    "framing",
                ]
            )
        with self.assertRaises(SystemExit):
            parse(
                [
                    "--model",
                    "mockllm/model",
                    "--condition",
                    "c4",
                    "--simulate",
                ]
            )
        combined = parse(
            [
                "--model",
                "mockllm/model",
                "--perturb",
                "framing",
                "--simulate",
            ]
        )
        self.assertEqual(execution_modes(combined), (True, True))
        with self.assertRaises(SystemExit):
            parse(
                [
                    "--model",
                    "mockllm/model",
                    "--aggregate-output-tokens",
                    "1000",
                ]
            )


class TestBenchmarkConstruction(unittest.TestCase):
    def test_clean_c0_selection_preserves_direct_solver_and_scorer(self):
        source = synthetic_benchmarks()
        original = source["tiny"]["tasks"][0]
        selected, ids = select_paired_samples(source, limit=3, seed=0)
        chosen = selected["tiny"]["tasks"][0]

        self.assertIs(chosen.solver, original.solver)
        self.assertIs(chosen.scorer, original.scorer)
        self.assertNotIn("agentic_protocol", chosen.metadata or {})
        self.assertEqual([sample.id for sample in chosen.dataset], ids["tiny_task"])
        self.assertEqual(len(original.dataset), 10)

    def test_pairing_is_stable_after_different_seed_in_same_process(self):
        first, first_ids = select_paired_samples(
            synthetic_benchmarks(), limit=4, seed=0
        )
        _different, different_ids = select_paired_samples(
            synthetic_benchmarks(), limit=4, seed=91
        )
        repeated, repeated_ids = select_paired_samples(
            synthetic_benchmarks(), limit=4, seed=0
        )

        self.assertNotEqual(first_ids, different_ids)
        self.assertEqual(first_ids, repeated_ids)
        self.assertEqual(
            [sample.id for sample in first["tiny"]["tasks"][0].dataset],
            [sample.id for sample in repeated["tiny"]["tasks"][0].dataset],
        )
        self.assertEqual(len(synthetic_benchmarks()["tiny"]["tasks"][0].dataset), 10)

    def test_ids_map_can_be_reused_exactly(self):
        _selected, ids = select_paired_samples(synthetic_benchmarks(), limit=3, seed=8)
        reused, reused_ids = select_paired_samples(
            synthetic_benchmarks(), limit=3, seed=999, selected_ids=ids
        )
        self.assertEqual(reused_ids, ids)
        self.assertEqual(
            [sample.id for sample in reused["tiny"]["tasks"][0].dataset],
            ids["tiny_task"],
        )

    def test_bad_paired_id_maps_fail_with_canonical_task_context(self):
        _selected, valid = select_paired_samples(
            synthetic_benchmarks(), limit=3, seed=8
        )
        cases = {
            "missing task": ({}, "missing canonical task 'tiny_task'"),
            "unknown IDs": (
                {"tiny_task": ["id-0", "id-1", "missing"]},
                "unknown IDs for tiny_task",
            ),
            "duplicates": (
                {"tiny_task": ["id-0", "id-0", "id-1"]},
                "duplicates for tiny_task",
            ),
            "wrong size": (
                {"tiny_task": valid["tiny_task"][:2]},
                "expected 3",
            ),
            "extra task": (
                {**valid, "not_a_canonical_task": ["id-0"]},
                "unknown canonical tasks.*not_a_canonical_task",
            ),
        }
        for label, (selected_ids, message) in cases.items():
            with self.subTest(case=label):
                with self.assertRaisesRegex(ValueError, message):
                    select_paired_samples(
                        synthetic_benchmarks(),
                        limit=3,
                        seed=999,
                        selected_ids=selected_ids,
                    )

    def test_generalized_construction_rejects_both_composition_orders(self):
        with patch("pipeline.registry.artifact_task_name", return_value="tiny_task"):
            perturbed = apply_stages(
                synthetic_benchmarks(), families=["reconsideration"], k=1
            )
        with self.assertRaisesRegex(ValueError, "perturbation"):
            apply_condition(perturbed, AgenticConfig.default("c2"))

        agentic = apply_condition(synthetic_benchmarks(), AgenticConfig.default("c2"))
        with self.assertRaisesRegex(ValueError, "agentic"):
            apply_stages(agentic, families=["reconsideration"], k=1)

    def test_selected_id_coverage_controls_persistence_eligibility(self):
        source = synthetic_benchmarks()
        _full_view, full_ids = select_paired_samples(source, limit=None, seed=0)
        self.assertTrue(selected_ids_cover_benchmarks(source, full_ids))

        complete_explicit, reused_full = select_paired_samples(
            source, limit=None, seed=999, selected_ids=full_ids
        )
        self.assertEqual(len(complete_explicit["tiny"]["tasks"][0].dataset), 10)
        self.assertTrue(selected_ids_cover_benchmarks(source, reused_full))

        _partial_view, partial_ids = select_paired_samples(source, limit=3, seed=0)
        self.assertFalse(selected_ids_cover_benchmarks(source, partial_ids))
        partial_explicit, reused_partial = select_paired_samples(
            source, limit=None, seed=999, selected_ids=partial_ids
        )
        self.assertEqual(len(partial_explicit["tiny"]["tasks"][0].dataset), 3)
        self.assertFalse(selected_ids_cover_benchmarks(source, reused_partial))

    def test_agentic_conditions_configure_roles(self):
        config = AgenticConfig.default("c4")
        conditioned = apply_condition(
            synthetic_benchmarks(),
            config,
            analyst_model="mockllm/analyst",
            critic_model="mockllm/critic",
        )
        wrapped = conditioned["tiny_agentic_c4"]["tasks"][0]
        self.assertEqual(
            str(wrapped.model_roles[config.roles.analyst]), "mockllm/analyst"
        )
        self.assertEqual(
            str(wrapped.model_roles[config.roles.critic]), "mockllm/critic"
        )
        self.assertEqual(wrapped.metadata["agentic_protocol"]["base_task"], "tiny_task")


class TestResultIdentity(unittest.TestCase):
    def test_skip_identity_distinguishes_all_execution_modes(self):
        entry = {"name": "tiny", "tasks": []}
        conditioned_entry = {
            **entry,
            "base_benchmark": "tiny",
            "condition": "c1",
        }
        modes = {
            "clean": (
                "tiny",
                entry,
                parse(["--model", "mockllm/model", "--condition", "c0"]),
            ),
            "perturbation": (
                "tiny",
                entry,
                parse(
                    [
                        "--model",
                        "mockllm/model",
                        "--perturb",
                        "reconsideration",
                    ]
                ),
            ),
            "simulation": (
                "tiny",
                entry,
                parse(["--model", "mockllm/model", "--no-perturb", "--simulate"]),
            ),
            "agentic": (
                "tiny_agentic_c1",
                conditioned_entry,
                parse(["--model", "mockllm/model", "--condition", "c1"]),
            ),
        }
        result_keys = {
            label: result_benchmark_key(benchmark, mode_entry, args)
            for label, (benchmark, mode_entry, args) in modes.items()
        }
        self.assertEqual(
            result_keys,
            {
                "clean": "tiny",
                "perturbation": "tiny_perturbation",
                "simulation": "tiny_simulation",
                "agentic": "tiny_agentic_c1",
            },
        )

        for label, (benchmark, mode_entry, args) in modes.items():
            benchmarks = {benchmark: mode_entry}
            with self.subTest(mode=label, existing="other modes"):
                other_scores = {
                    result_key: 1.0
                    for other_label, result_key in result_keys.items()
                    if other_label != label
                }
                self.assertEqual(
                    skip_benchmark_keys(benchmarks, args, other_scores), set()
                )
            with self.subTest(mode=label, existing="same mode"):
                self.assertEqual(
                    skip_benchmark_keys(benchmarks, args, {result_keys[label]: 1.0}),
                    {benchmark},
                )

    def test_legacy_transformed_record_does_not_suppress_clean_mode(self):
        benchmarks = {"tiny": {"name": "tiny", "tasks": []}}
        clean = parse(["--model", "mockllm/model", "--condition", "c0"])
        perturb = parse(["--model", "mockllm/model", "--perturb", "reconsideration"])
        legacy_perturbation = {
            "scores": {"tiny": 42.0},
            "perturbations": {"tiny": {"consistency_rate": {}}},
            "simulations": {},
        }

        self.assertEqual(
            skip_benchmark_keys(benchmarks, clean, legacy_perturbation), set()
        )
        self.assertEqual(
            skip_benchmark_keys(benchmarks, perturb, legacy_perturbation),
            {"tiny"},
        )

    def test_only_selection_reruns_requested_mode(self):
        args = parse(["--model", "mockllm/model", "--no-perturb", "--simulate"])
        args.only = ["tiny"]
        benchmarks = {"tiny": {"name": "tiny", "tasks": []}}
        self.assertEqual(
            skip_benchmark_keys(
                benchmarks, args, {"tiny_simulation": 1.0, "tiny": 1.0}
            ),
            set(),
        )


class TestPersonaSetupAndPersistence(unittest.TestCase):
    def test_persona_is_setup_and_reaches_clean_c0_model_input(self):
        calls = []

        def output(messages, tools, tool_choice, config):
            calls.append([message.text for message in messages])
            return ModelOutput.from_content("mockllm/persona", "answer")

        task = udhr_individual(
            grader="mockllm/grader", llamaguard_model="mockllm/guard"
        )
        self.assertIsNotNone(task.setup)
        self.assertFalse(isinstance(task.solver, list))
        direct = task_with(
            task,
            dataset=[task.dataset[0]],
            scorer=output_scorer(),
        )
        with tempfile.TemporaryDirectory() as log_dir:
            log = eval(
                direct,
                model=get_model("mockllm/persona", custom_outputs=output),
                log_dir=log_dir,
                display="none",
            )[0]
        self.assertEqual(log.status, "success")
        self.assertEqual(calls[0][0], OPEN_SYSTEM_PROMPT["individual-rights"])

    def test_persisted_config_contains_pairing_budget_and_audit_usage(self):
        args = parse(
            [
                "--model",
                "mockllm/model",
                "--condition",
                "c3",
                "--sample-seed",
                "0",
                "--final-output-tokens",
                "900",
            ]
        )
        config = build_protocol_config(args)
        task = apply_condition(
            synthetic_benchmarks(),
            config,
            analyst_model="mockllm/analyst",
            critic_model="mockllm/critic",
        )["tiny_agentic_c3"]
        usage = {
            "model_events": 1,
            "events_with_usage": 1,
            "input_tokens": 7,
            "output_tokens": 11,
            "total_tokens": 18,
            "reasoning_tokens": 2,
            "input_tokens_cache_write": 0,
            "input_tokens_cache_read": 0,
            "events_with_cost": 1,
            "total_cost": 0.03,
            "usage_complete": True,
            "cost_complete": True,
        }
        audit = {
            "valid": True,
            "violations": [],
            "termination_reason": "submitted",
            "role_model_counts": {"agentic_analyst": 1},
            "phase_model_counts": {"agentic:c3:draft": 1},
            "dynamic_helper_count": 0,
            "helper_count": 0,
            "fixed_reviewer_count": 2,
            "usage_status": "verified",
            "aggregate_usage": usage,
            "phase_usage": {"agentic:c3:draft": usage},
            "role_usage": {"agentic_analyst": usage},
        }
        process = summarize_process(
            [
                SimpleNamespace(
                    samples=[SimpleNamespace(metadata={AUDIT_METADATA_KEY: audit})]
                )
            ]
        )
        record = persisted_config(
            benchmark="tiny_agentic_c3",
            entry=task,
            config=config,
            args=args,
            analyst_model="mockllm/analyst",
            critic_model="mockllm/critic",
            selected_ids={"tiny_task": ["id-0", "id-2"]},
            process=process,
        )

        self.assertEqual(record["condition"], Condition.C3.value)
        self.assertEqual(record["canonical_base_benchmark"], "tiny")
        self.assertEqual(record["canonical_base_tasks"], ["tiny_task"])
        self.assertEqual(record["sample_seed"], 0)
        self.assertEqual(record["selected_sample_ids"], {"tiny_task": ["id-0", "id-2"]})
        self.assertEqual(record["requested_budget"]["final_output_tokens"], 900)
        self.assertEqual(record["budget_overrides"], {"final_output_tokens": 900})
        self.assertEqual(record["effective_budget"]["final_output_tokens"], 900)
        self.assertEqual(
            record["process_validity"]["aggregate_policy"],
            "protocol_valid_samples_only",
        )
        self.assertEqual(record["process_validity"]["valid_samples"], 1)
        self.assertEqual(record["observed_usage"]["roles"], {"agentic_analyst": 1})
        self.assertEqual(record["observed_usage"]["fixed_reviewer_calls"], 2)
        self.assertEqual(record["observed_usage"]["usage_statuses"], {"verified": 1})
        self.assertEqual(
            record["observed_usage"]["aggregate_tokens_and_cost"]["output_tokens"],
            11,
        )
        self.assertAlmostEqual(
            record["observed_usage"]["phase_tokens_and_cost"]["agentic:c3:draft"][
                "total_cost"
            ],
            0.03,
        )


if __name__ == "__main__":
    unittest.main()
