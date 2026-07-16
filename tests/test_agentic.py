"""Deterministic tests for the C0-C4 agentic protocol core."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from typing import Callable
from unittest.mock import patch

import anyio

from inspect_ai import Epochs, Task, eval, task_with
from inspect_ai.dataset import Sample
from inspect_ai.event import ModelEvent, ToolEvent
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    ModelOutput,
    ModelUsage,
    get_model,
)
from inspect_ai.scorer import CORRECT, Score, accuracy, mean_score, scorer
from inspect_ai.solver import generate, system_message
from inspect_ai.tool import ToolCall
from inspect_ai.util import Store, TokenLimit

from pipeline.agentic import (
    AUDIT_METADATA_KEY,
    AgenticConfig,
    Condition,
    ModelRoles,
    ProtocolBudget,
    make_agentic_task,
)
from pipeline.agentic.eval import agentic as registered_agentic_task
from pipeline.stage1_evaluation.evals.democratic_authoritarian_bias import fscale
from pipeline.stage1_evaluation.evals.preserving_historical_truth import history_no_push
from pipeline.stage2_perturbation.build import build_perturbed_task

TARGET_SENTINEL = "TARGET_SENTINEL_DO_NOT_LEAK_123456"
METADATA_SENTINEL = "METADATA_SENTINEL_DO_NOT_LEAK_654321"
EXACT_ANSWER = "exact submitted answer"


@scorer(metrics=[accuracy()])
def output_scorer():
    async def score(state, target):
        return Score(value=CORRECT, answer=state.output.completion)

    return score


@scorer(metrics=[accuracy()])
def second_output_scorer():
    async def score(state, target):
        return Score(value=CORRECT)

    return score


@scorer(metrics=[accuracy()])
def native_empty_completion_scorer():
    """A benchmark-native score whose value proves invalidity did not replace it."""

    async def score(state, target):
        return Score(value=0.625, answer=state.output.completion)

    return score


@scorer(metrics=[])
def cancellation_observing_scorer(observations: list[dict], cancelled: list[bool]):
    async def score(state, target):
        observations.append(
            {
                "cancelled_before_scoring": cancelled[0],
                "audit": dict(state.metadata.get(AUDIT_METADATA_KEY) or {}),
            }
        )
        return Score(value=CORRECT)

    return score


@scorer(metrics=[])
def model_then_error_scorer(observations: list[dict]):
    """Exercise a real scoring ModelEvent, then expose the original error."""

    async def score(state, target):
        observations.append(
            {
                "audit": dict(state.metadata.get(AUDIT_METADATA_KEY) or {}),
                "completion": state.output.completion,
            }
        )
        grader = get_model(role="scoring_grader", required=True)
        await grader.generate("SCORER-ONLY INPUT")
        raise RuntimeError("original scorer exploded")

    return score


def base_task(*, scorers=None, metadata: dict | None = None) -> Task:
    return Task(
        dataset=[
            Sample(
                id="sample-one",
                input=[
                    ChatMessageSystem(content="PUBLIC SYSTEM"),
                    ChatMessageUser(content="PUBLIC REQUEST"),
                ],
                target=TARGET_SENTINEL,
                metadata=metadata
                or {"secret": METADATA_SENTINEL, "public_id": "visible"},
            )
        ],
        solver=generate(),
        scorer=scorers or [output_scorer()],
        model_roles={"existing_role": "mockllm/existing"},
        message_limit=17,
        turn_limit=11,
        time_limit=30,
        working_limit=20,
        metadata={"base_metadata": "kept"},
        tags=["base-tag"],
        name="synthetic_protocol_base",
    )


def tool_output(*calls: tuple[str, dict], content: str = "tool actions") -> ModelOutput:
    return ModelOutput(
        model="mockllm/model",
        choices=[
            ChatCompletionChoice(
                message=ChatMessageAssistant(
                    content=content,
                    model="mockllm/model",
                    tool_calls=[
                        ToolCall(
                            id=f"call-{index}",
                            function=function,
                            arguments=arguments,
                        )
                        for index, (function, arguments) in enumerate(calls)
                    ],
                ),
                stop_reason="tool_calls",
            )
        ],
    )


def submit_output(answer: str = EXACT_ANSWER) -> ModelOutput:
    return ModelOutput.for_tool_call(
        "mockllm/model", "submit", {"answer": answer}, tool_call_id="submit-call"
    )


def with_usage(
    output: ModelOutput,
    *,
    input_tokens: int = 1,
    output_tokens: int = 1,
    total_cost: float | None = None,
) -> ModelOutput:
    output.usage = ModelUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        total_cost=total_cost,
    )
    return output


def fixed_role_model(name: str, text: str, calls: list | None = None):
    def output(messages, tools, tool_choice, config):
        if calls is not None:
            calls.append((name, messages, config))
        return ModelOutput.from_content(f"mockllm/{name}", text)

    return get_model(f"mockllm/{name}", custom_outputs=output)


def evaluate_task(
    task: Task,
    target_output: Callable,
    *,
    roles: dict | None = None,
    target_config: GenerateConfig | None = None,
    eval_config: GenerateConfig | None = None,
):
    target_model = get_model(
        "mockllm/model",
        custom_outputs=target_output,
        config=target_config,
    )
    with tempfile.TemporaryDirectory() as log_dir:
        eval_kwargs = (
            eval_config.model_dump(exclude_none=True) if eval_config is not None else {}
        )
        log = eval(
            task,
            model=target_model,
            model_roles=roles,
            log_dir=log_dir,
            display="none",
            **eval_kwargs,
        )[0]
        sample = log.samples[0]
        # Eval logs lazily load samples; materialize before the temp dir closes.
        sample.events = list(sample.events)
        sample.messages = list(sample.messages)
        return log, sample


def run_protocol(
    condition: Condition | str,
    target_output: Callable,
    *,
    roles: dict | None = None,
    config: AgenticConfig | None = None,
    task: Task | None = None,
):
    config = config or AgenticConfig.default(condition)
    wrapped = make_agentic_task(task or base_task(), config)
    log, sample = evaluate_task(wrapped, target_output, roles=roles)
    assert log.status == "success", log.error
    return sample


def event_payload(sample, *, event_ids: set[str] | None = None) -> str:
    payload = []
    for event in sample.events:
        if event_ids is not None and getattr(event, "uuid", None) not in event_ids:
            continue
        if isinstance(event, ModelEvent):
            payload.append(
                {
                    "input": [
                        message.model_dump(mode="json") for message in event.input
                    ],
                    "tools": [tool.model_dump(mode="json") for tool in event.tools],
                }
            )
        elif isinstance(event, ToolEvent):
            payload.append({"arguments": event.arguments, "result": str(event.result)})
    return json.dumps(payload, sort_keys=True)


class TestAgenticConfiguration(unittest.TestCase):
    def test_c0_is_exact_unwrapped_task(self):
        base = base_task()
        self.assertIs(make_agentic_task(base, AgenticConfig.default("c0")), base)

    def test_safe_copy_preserves_task_fields_and_scorer_order(self):
        scorers = [output_scorer(), second_output_scorer()]
        base = base_task(scorers=scorers)
        original_solver = base.solver
        original_metadata = dict(base.metadata)
        original_roles = dict(base.model_roles)
        original_tags = list(base.tags)

        wrapped = make_agentic_task(base, AgenticConfig.default("c3"))

        self.assertIsNot(wrapped, base)
        self.assertIs(wrapped.dataset, base.dataset)
        self.assertIs(wrapped.scorer, base.scorer)
        self.assertEqual(wrapped.scorer, scorers)
        self.assertTrue(all(a is b for a, b in zip(wrapped.scorer, scorers)))
        for field in (
            "setup",
            "cleanup",
            "on_checkpoint",
            "on_resume",
            "metrics",
            "model",
            "config",
            "sandbox",
            "checkpoint",
            "approval",
            "epochs",
            "epochs_reducer",
            "fail_on_error",
            "continue_on_fail",
            "score_on_error",
            "message_limit",
            "token_limit",
            "turn_limit",
            "time_limit",
            "working_limit",
            "cost_limit",
            "early_stopping",
            "version",
            "viewer",
        ):
            self.assertEqual(getattr(wrapped, field), getattr(base, field), field)

        self.assertIs(base.solver, original_solver)
        self.assertEqual(base.metadata, original_metadata)
        self.assertEqual(base.model_roles, original_roles)
        self.assertEqual(base.tags, original_tags)
        self.assertIsNot(wrapped.metadata, base.metadata)
        self.assertIsNot(wrapped.model_roles, base.model_roles)
        self.assertIsNot(wrapped.tags, base.tags)

        other = make_agentic_task(base, AgenticConfig.default("c2"))
        wrapped.metadata["only_c3"] = True
        self.assertNotIn("only_c3", other.metadata)
        self.assertNotIn("only_c3", base.metadata)

    def test_complete_task_surface_preserves_non_default_hooks_reducer_and_token_limit(
        self,
    ):
        async def cleanup(state):
            return None

        async def on_checkpoint(state):
            return None

        async def on_resume(state):
            return None

        reducer = mean_score()
        token_cap = TokenLimit(tokens=321, type="output")
        setup = system_message("PUBLIC SETUP")
        metrics = [accuracy()]
        task = Task(
            dataset=[Sample(id="rich", input="request")],
            setup=setup,
            solver=generate(),
            cleanup=cleanup,
            scorer=[output_scorer(), second_output_scorer()],
            metrics=metrics,
            model="mockllm/model",
            config=GenerateConfig(temperature=0.2, max_tokens=77),
            model_roles={"existing_role": "mockllm/existing"},
            checkpoint=True,
            on_checkpoint=on_checkpoint,
            on_resume=on_resume,
            epochs=Epochs(2, reducer=reducer),
            fail_on_error=0.25,
            continue_on_fail=True,
            score_on_error=True,
            message_limit=13,
            token_limit=token_cap,
            turn_limit=7,
            time_limit=19,
            working_limit=17,
            cost_limit=1.5,
            metadata={"rich": True},
            tags=["rich"],
            version="rich-v1",
            name="rich_task",
        )

        wrapped = make_agentic_task(task, AgenticConfig.default("c2"))

        for field in (
            "dataset",
            "setup",
            "cleanup",
            "scorer",
            "metrics",
            "model",
            "config",
            "checkpoint",
            "on_checkpoint",
            "on_resume",
            "epochs",
            "epochs_reducer",
            "fail_on_error",
            "continue_on_fail",
            "score_on_error",
            "message_limit",
            "token_limit",
            "turn_limit",
            "time_limit",
            "working_limit",
            "cost_limit",
            "version",
        ):
            self.assertIs(getattr(wrapped, field), getattr(task, field), field)
        # Inspect normalizes a TokenLimit instance onto Task.token_limit; the
        # wrapper must preserve that non-default normalized value unchanged.
        self.assertEqual(task.token_limit, 321)
        self.assertEqual(wrapped.token_limit, 321)
        self.assertEqual(wrapped.epochs, 2)
        self.assertEqual(wrapped.epochs_reducer, [reducer])

    def test_identical_reviewer_role_identifiers_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be distinct"):
            ModelRoles(analyst="same_role", critic="same_role")

    def test_all_public_composition_boundaries_reject_unsupported_tasks(self):
        perturbed = build_perturbed_task(
            base_task(), ["reconsideration"], "mockllm/rewrite", 1
        )
        with self.assertRaisesRegex(ValueError, "perturbation"):
            make_agentic_task(perturbed, AgenticConfig.default("c1"))

        agentic = make_agentic_task(base_task(), AgenticConfig.default("c1"))
        with self.assertRaisesRegex(ValueError, "agentic"):
            build_perturbed_task(agentic, ["reconsideration"], "mockllm/rewrite", 1)

        repeated = make_agentic_task(base_task(), AgenticConfig.default("c2"))
        with self.assertRaisesRegex(ValueError, "again"):
            make_agentic_task(repeated, AgenticConfig.default("c3"))

    def test_c1_c4_require_finite_valid_budgets(self):
        with self.assertRaisesRegex(ValueError, "requires finite"):
            AgenticConfig(condition=Condition.C1)
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            ProtocolBudget(
                aggregate_output_tokens=100,
                final_output_tokens=50,
                draft_output_tokens=10,
                review_output_tokens=10,
                consultation_output_tokens=10,
                helper_output_tokens=10,
                max_consultation_turns=1,
                max_final_turns=1,
                max_helper_calls=1,
                time_seconds=float("inf"),
                working_seconds=1,
            )
        with self.assertRaisesRegex(ValueError, "pre-final caps"):
            AgenticConfig(
                condition=Condition.C4,
                budget=ProtocolBudget(
                    aggregate_output_tokens=100,
                    final_output_tokens=50,
                    draft_output_tokens=10,
                    review_output_tokens=10,
                    consultation_output_tokens=20,
                    helper_output_tokens=20,
                    max_consultation_turns=2,
                    max_final_turns=1,
                    max_helper_calls=2,
                    time_seconds=1,
                    working_seconds=1,
                ),
            )


class TestSubmissionProtocol(unittest.TestCase):
    def test_c1_exact_submit_preserves_messages_and_hides_private_fields(self):
        calls = []

        def target(messages, tools, tool_choice, config):
            calls.append((messages, [tool.name for tool in tools], config))
            return submit_output()

        sample = run_protocol("c1", target)
        audit = sample.metadata[AUDIT_METADATA_KEY]
        self.assertEqual(sample.output.completion, EXACT_ANSWER)
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["submit_count"], 1)
        self.assertEqual(calls[0][1], ["submit"])
        self.assertEqual(
            [message.text for message in sample.messages],
            [
                "PUBLIC SYSTEM",
                "PUBLIC REQUEST",
            ],
        )
        payload = event_payload(sample)
        self.assertNotIn(TARGET_SENTINEL, payload)
        self.assertNotIn(METADATA_SENTINEL, payload)
        self.assertNotIn(TARGET_SENTINEL, json.dumps(audit))
        self.assertNotIn(METADATA_SENTINEL, json.dumps(audit))

    def test_second_final_turn_submit_is_valid(self):
        calls = 0

        def target(messages, tools, tool_choice, config):
            nonlocal calls
            calls += 1
            if calls == 1:
                return ModelOutput.from_content(
                    "mockllm/model", "I will use the required submit tool."
                )
            return submit_output()

        sample = run_protocol("c1", target)
        audit = sample.metadata[AUDIT_METADATA_KEY]

        self.assertEqual(calls, 2)
        self.assertEqual(sample.output.completion, EXACT_ANSWER)
        self.assertTrue(audit["valid"], audit["violations"])
        self.assertEqual(audit["phase_model_counts"]["agentic:c1:final_submit"], 2)
        self.assertEqual(audit["submit_count"], 1)
        self.assertTrue(audit["submit_payload_exact"])

    def test_projection_failure_yields_empty_completion_and_invalid_audit(self):
        with patch(
            "pipeline.agentic.solvers._public_messages",
            side_effect=ValueError("malformed private projection payload"),
        ):
            sample = run_protocol("c1", lambda *args: submit_output())

        audit = sample.metadata[AUDIT_METADATA_KEY]
        self.assertEqual(sample.output.completion, "")
        self.assertFalse(audit["valid"])
        self.assertEqual(audit["caught_error_type"], "ValueError")
        self.assertNotIn("malformed private projection payload", json.dumps(audit))

    def test_store_failure_keeps_metadata_audit_and_sanitizes_error(self):
        with patch.object(
            Store,
            "set",
            side_effect=RuntimeError("store secret request body"),
        ):
            sample = run_protocol("c1", lambda *args: submit_output())

        audit = sample.metadata[AUDIT_METADATA_KEY]
        self.assertFalse(audit["valid"])
        self.assertTrue(audit["metadata_persisted"])
        self.assertFalse(audit["store_persisted"])
        self.assertEqual(audit["store_error_type"], "RuntimeError")
        self.assertIn("store_persistence_error", audit["violations"])
        self.assertNotIn("store secret request body", json.dumps(audit))

    def test_non_submit_and_error_use_empty_completion_with_audit(self):
        def free_text(messages, tools, tool_choice, config):
            return ModelOutput.from_content("mockllm/model", "must be discarded")

        sample = run_protocol("c1", free_text)
        self.assertEqual(sample.output.completion, "")
        limit_audit = sample.metadata[AUDIT_METADATA_KEY]
        self.assertFalse(limit_audit["valid"])
        self.assertEqual(limit_audit["termination_reason"], "limit")
        self.assertGreaterEqual(limit_audit["limit_event_count"], 1)
        self.assertIn("submit_count", limit_audit["violations"])

        def raises(messages, tools, tool_choice, config):
            raise RuntimeError("model failure")

        errored = run_protocol("c1", raises)
        self.assertEqual(errored.output.completion, "")
        self.assertEqual(
            errored.metadata[AUDIT_METADATA_KEY]["caught_error_type"],
            "RuntimeError",
        )

    def test_process_invalid_non_submit_retains_native_benchmark_score(self):
        target_calls = 0

        def free_text(messages, tools, tool_choice, config):
            nonlocal target_calls
            target_calls += 1
            return ModelOutput.from_content("mockllm/model", "not a submission")

        task = base_task(scorers=[native_empty_completion_scorer()])
        sample = run_protocol("c1", free_text, task=task)

        # Inspect 0.3.246 emits the SampleLimitEvent on the first over-limit
        # turn, so the finite model-call bound is configured turns plus one.
        self.assertLessEqual(
            target_calls,
            AgenticConfig.default("c1").budget.max_final_turns + 1,
        )
        self.assertEqual(sample.output.completion, "")
        self.assertEqual(list(sample.scores), ["native_empty_completion_scorer"])
        native = sample.scores["native_empty_completion_scorer"]
        self.assertEqual(native.value, 0.625)
        self.assertEqual(native.answer, "")
        self.assertFalse(sample.metadata[AUDIT_METADATA_KEY]["valid"])
        self.assertNotIn(AUDIT_METADATA_KEY, sample.scores)

    def test_audit_precedes_scorer_model_event_and_survives_scorer_error(self):
        observations: list[dict] = []
        grader = get_model(
            "mockllm/scoring-grader",
            custom_outputs=lambda messages, tools, tool_choice, config: (
                ModelOutput.from_content("mockllm/scoring-grader", "graded")
            ),
        )
        task = base_task(scorers=[model_then_error_scorer(observations)])
        task.model_roles = {**(task.model_roles or {}), "scoring_grader": grader}
        wrapped = make_agentic_task(task, AgenticConfig.default("c1"))

        log, sample = evaluate_task(wrapped, lambda *args: submit_output())

        self.assertEqual(log.status, "error")
        self.assertIn("original scorer exploded", str(log.error))
        audit = sample.metadata[AUDIT_METADATA_KEY]
        self.assertTrue(audit["valid"])
        self.assertEqual(observations[0]["audit"], audit)
        self.assertEqual(observations[0]["completion"], EXACT_ANSWER)
        self.assertEqual(sample.scores, {})

        solving_ids = set(audit["solving_event_ids"])
        grader_events = [
            event
            for event in sample.events
            if isinstance(event, ModelEvent) and event.role == "scoring_grader"
        ]
        self.assertEqual(len(grader_events), 1)
        self.assertNotIn(grader_events[0].uuid, solving_ids)
        event_indexes = {
            event.uuid: index
            for index, event in enumerate(sample.events)
            if getattr(event, "uuid", None)
        }
        self.assertTrue(
            all(
                event_indexes[event_id] < event_indexes[grader_events[0].uuid]
                for event_id in solving_ids
            )
        )
        self.assertNotIn("scoring_grader", audit["role_model_counts"])

    def test_original_scorer_order_metrics_and_names_remain_benchmark_owned(self):
        originals = [output_scorer(), second_output_scorer()]
        task = base_task(scorers=originals)
        wrapped = make_agentic_task(task, AgenticConfig.default("c1"))
        log, sample = evaluate_task(wrapped, lambda *args: submit_output())

        self.assertEqual(log.status, "success")
        self.assertIs(wrapped.scorer, task.scorer)
        self.assertEqual(
            [scorer.__qualname__ for scorer in wrapped.scorer],
            [scorer.__qualname__ for scorer in originals],
        )
        self.assertEqual(list(sample.scores), ["output_scorer", "second_output_scorer"])
        self.assertEqual(
            [result.name for result in log.results.scores],
            ["output_scorer", "second_output_scorer"],
        )
        self.assertEqual(
            [list(result.metrics) for result in log.results.scores],
            [["accuracy"], ["accuracy"]],
        )
        self.assertNotIn(AUDIT_METADATA_KEY, sample.scores)

    def test_multiple_and_mixed_final_tools_are_rejected(self):
        def multiple(messages, tools, tool_choice, config):
            return tool_output(
                ("submit", {"answer": "first"}),
                ("submit", {"answer": "second"}),
            )

        sample = run_protocol("c1", multiple)
        self.assertEqual(sample.output.completion, "")
        self.assertEqual(sample.metadata[AUDIT_METADATA_KEY]["submit_count"], 2)

        def mixed(messages, tools, tool_choice, config):
            return tool_output(
                ("submit", {"answer": EXACT_ANSWER}),
                ("not_permitted", {}),
            )

        mixed_sample = run_protocol("c1", mixed)
        self.assertEqual(mixed_sample.output.completion, "")
        self.assertIn(
            "mixed_final_tool_turn",
            mixed_sample.metadata[AUDIT_METADATA_KEY]["violations"],
        )


class TestFixedReviewProtocols(unittest.TestCase):
    def test_c2_has_distinct_uncached_parallel_self_reviews_and_final_reserve(self):
        calls = []

        def target(messages, tools, tool_choice, config):
            system = messages[0].text
            calls.append((system, tuple(tool.name for tool in tools), config))
            if any(tool.name == "submit" for tool in tools):
                return submit_output()
            if "adversarially review" in system:
                return ModelOutput.from_content("mockllm/model", "self review B")
            if "Independently review" in system:
                return ModelOutput.from_content("mockllm/model", "self review A")
            return ModelOutput.from_content("mockllm/model", "draft")

        sample = run_protocol("c2", target)
        audit = sample.metadata[AUDIT_METADATA_KEY]
        self.assertEqual(sample.output.completion, EXACT_ANSWER)
        self.assertTrue(audit["valid"])
        self.assertEqual(
            audit["phase_model_counts"],
            {
                "agentic:c2:draft": 1,
                "agentic:c2:final_submit": 1,
                "agentic:c2:review_a": 1,
                "agentic:c2:review_b": 1,
            },
        )
        self.assertEqual(len(calls), 4)
        self.assertNotEqual(calls[1][0], calls[2][0])
        self.assertTrue(all(call[2].cache is False for call in calls))
        self.assertEqual(calls[-1][2].max_tokens, 1024)
        pre_final_caps = sum(call[2].max_tokens for call in calls[:-1])
        self.assertLessEqual(pre_final_caps, 4096 - 1024)

    def test_phase_generation_config_overrides_eval_task_model_and_role_configs(self):
        target_calls = []
        role_calls = []

        def target(messages, tools, tool_choice, config):
            target_calls.append(config)
            if any(tool.name == "submit" for tool in tools):
                return submit_output()
            return ModelOutput.from_content("mockllm/model", "draft")

        analyst = fixed_role_model("analyst", "analysis", role_calls)
        critic = fixed_role_model("critic", "critique", role_calls)
        analyst.config = GenerateConfig(max_tokens=3001, cache=True)
        critic.config = GenerateConfig(max_tokens=3002, cache=True)
        task = task_with(
            base_task(), config=GenerateConfig(max_tokens=2001, cache=True)
        )
        wrapped = make_agentic_task(task, AgenticConfig.default("c3"))
        log, sample = evaluate_task(
            wrapped,
            target,
            roles={"agentic_analyst": analyst, "agentic_critic": critic},
            target_config=GenerateConfig(max_tokens=4001, cache=True),
            eval_config=GenerateConfig(max_tokens=5001, cache=True),
        )

        self.assertEqual(log.status, "success")
        self.assertTrue(sample.metadata[AUDIT_METADATA_KEY]["valid"])
        self.assertEqual(
            [(call.max_tokens, call.cache) for call in target_calls],
            [(768, False), (1024, False)],
        )
        self.assertCountEqual(
            [(call[2].max_tokens, call[2].cache) for call in role_calls],
            [(512, False), (512, False)],
        )

    def test_structured_review_collection_cancels_sibling_before_audit_and_scoring(
        self,
    ):
        cancelled = [False]
        observations: list[dict] = []

        def target(messages, tools, tool_choice, config):
            return ModelOutput.from_content("mockllm/model", "draft")

        async def failing_reviewer(messages, tools, tool_choice, config):
            await anyio.sleep(0.02)
            raise RuntimeError("review failed with private provider body")

        async def blocked_reviewer(messages, tools, tool_choice, config):
            try:
                await anyio.sleep_forever()
            finally:
                cancelled[0] = True

        task = base_task(
            scorers=[cancellation_observing_scorer(observations, cancelled)]
        )
        sample = run_protocol(
            "c3",
            target,
            task=task,
            roles={
                "agentic_analyst": get_model(
                    "mockllm/failing-reviewer", custom_outputs=failing_reviewer
                ),
                "agentic_critic": get_model(
                    "mockllm/blocked-reviewer", custom_outputs=blocked_reviewer
                ),
            },
        )

        audit = sample.metadata[AUDIT_METADATA_KEY]
        self.assertTrue(cancelled[0])
        self.assertEqual(sample.output.completion, "")
        self.assertFalse(audit["valid"])
        self.assertEqual(audit["caught_error_type"], "RuntimeError")
        self.assertEqual(observations[0]["audit"], audit)
        self.assertTrue(observations[0]["cancelled_before_scoring"])

    def test_c3_uses_one_analyst_and_one_critic_in_parallel_phases(self):
        target_calls = []
        role_calls = []

        def target(messages, tools, tool_choice, config):
            target_calls.append(tuple(tool.name for tool in tools))
            if any(tool.name == "submit" for tool in tools):
                return submit_output()
            return ModelOutput.from_content("mockllm/model", "draft")

        roles = {
            "agentic_analyst": fixed_role_model("analyst", "analysis", role_calls),
            "agentic_critic": fixed_role_model("critic", "critique", role_calls),
        }
        sample = run_protocol("c3", target, roles=roles)
        audit = sample.metadata[AUDIT_METADATA_KEY]
        self.assertEqual(sample.output.completion, EXACT_ANSWER)
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["role_model_counts"]["agentic_analyst"], 1)
        self.assertEqual(audit["role_model_counts"]["agentic_critic"], 1)
        self.assertCountEqual([call[0] for call in role_calls], ["analyst", "critic"])
        self.assertEqual(len(target_calls), 2)


class TestAuditUsageAndIdentity(unittest.TestCase):
    def test_audit_records_phase_role_tokens_cost_and_ceiling_verification(self):
        call_number = 0

        def target(messages, tools, tool_choice, config):
            nonlocal call_number
            call_number += 1
            if any(tool.name == "submit" for tool in tools):
                output = submit_output()
            else:
                output = ModelOutput.from_content(
                    "mockllm/model", f"phase {call_number}"
                )
            return with_usage(
                output,
                input_tokens=call_number,
                output_tokens=call_number * 10,
                total_cost=call_number / 100,
            )

        sample = run_protocol("c2", target)
        audit = sample.metadata[AUDIT_METADATA_KEY]

        self.assertTrue(audit["valid"])
        self.assertEqual(audit["usage_status"], "verified")
        self.assertTrue(audit["usage_verifiable"])
        self.assertTrue(audit["usage_within_configured_ceiling"])
        self.assertEqual(audit["aggregate_usage"]["output_tokens"], 100)
        self.assertAlmostEqual(audit["aggregate_usage"]["total_cost"], 0.10)
        self.assertEqual(
            audit["phase_usage"]["agentic:c2:final_submit"]["output_tokens"],
            40,
        )
        self.assertEqual(audit["role_usage"]["target"]["output_tokens"], 100)
        self.assertEqual(audit["fixed_reviewer_count"], 2)
        self.assertEqual(audit["dynamic_helper_count"], 0)
        self.assertEqual(audit["helper_count"], 0)

    def test_reported_aggregate_usage_over_ceiling_is_invalid(self):
        sample = run_protocol(
            "c1",
            lambda *args: with_usage(
                submit_output(), input_tokens=1, output_tokens=2000
            ),
        )
        audit = sample.metadata[AUDIT_METADATA_KEY]
        self.assertFalse(audit["valid"])
        self.assertEqual(audit["usage_status"], "exceeded")
        self.assertFalse(audit["usage_within_configured_ceiling"])
        self.assertIn("aggregate_output_ceiling_exceeded", audit["violations"])

    def test_missing_usage_is_explicitly_unknown_not_verified(self):
        sample = run_protocol("c1", lambda *args: submit_output())
        audit = sample.metadata[AUDIT_METADATA_KEY]
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["usage_status"], "unknown")
        self.assertFalse(audit["usage_verifiable"])
        self.assertIsNone(audit["usage_within_configured_ceiling"])

    def test_audit_rejects_wrong_fixed_reviewer_identity_despite_phase_counts(self):
        from pipeline.agentic.helpers import analyst_agent as real_analyst_agent

        def target(messages, tools, tool_choice, config):
            if any(tool.name == "submit" for tool in tools):
                return submit_output()
            return ModelOutput.from_content("mockllm/model", "draft")

        roles = {
            "agentic_analyst": fixed_role_model("analyst", "analysis"),
            "agentic_critic": fixed_role_model("critic", "critique"),
        }
        with patch(
            "pipeline.agentic.solvers.analyst_agent",
            side_effect=lambda role, max_tokens: real_analyst_agent(
                "agentic_critic", max_tokens
            ),
        ):
            sample = run_protocol("c3", target, roles=roles)

        audit = sample.metadata[AUDIT_METADATA_KEY]
        self.assertEqual(audit["phase_model_counts"]["agentic:c3:review_a"], 1)
        self.assertEqual(audit["phase_model_counts"]["agentic:c3:review_b"], 1)
        self.assertFalse(audit["valid"])
        self.assertIn("fixed_reviewer_identity", audit["violations"])


class TestSolvingPayloadPrivacy(unittest.TestCase):
    def test_c2_c3_c4_actual_model_and_tool_payloads_hide_private_sentinels(self):
        for condition in (Condition.C2, Condition.C3, Condition.C4):
            with self.subTest(condition=condition.value):
                consultation_turn = 0

                def target(messages, tools, tool_choice, config):
                    nonlocal consultation_turn
                    names = [tool.name for tool in tools]
                    if "submit" in names:
                        return submit_output()
                    if condition is Condition.C4:
                        consultation_turn += 1
                        if consultation_turn == 1:
                            return ModelOutput.for_tool_call(
                                "mockllm/model",
                                "agentic_analyst",
                                {"input": "review only this public draft"},
                            )
                        return ModelOutput.for_tool_call(
                            "mockllm/model",
                            "finish_consultation",
                            {"draft": "public consultation draft"},
                        )
                    return ModelOutput.from_content(
                        "mockllm/model", "public draft or review"
                    )

                private = "CUSTOM_PRIVATE_SENTINEL_2468101214"
                task = base_task(
                    metadata={
                        "private_answer": private,
                        "secret": METADATA_SENTINEL,
                        "public_id": "visible",
                    }
                )
                config = AgenticConfig.default(
                    condition,
                    private_metadata_keys=("private_answer", "secret"),
                )
                roles = {
                    "agentic_analyst": fixed_role_model("analyst", "public analysis"),
                    "agentic_critic": fixed_role_model("critic", "public critique"),
                }
                sample = run_protocol(
                    condition,
                    target,
                    roles=roles,
                    config=config,
                    task=task,
                )
                audit = sample.metadata[AUDIT_METADATA_KEY]
                solving_payload = event_payload(
                    sample, event_ids=set(audit["solving_event_ids"])
                )
                for sentinel in (TARGET_SENTINEL, METADATA_SENTINEL, private):
                    self.assertNotIn(sentinel, solving_payload)
                    self.assertNotIn(sentinel, json.dumps(audit))
                self.assertFalse(audit["hidden_leakage_detected"])

    def test_benchmark_aware_private_fields_detect_deliberate_leaks(self):
        cases = (
            ("history_no_push", "True Version"),
            ("leader_favorability", "classification"),
            ("role_model_bias", "nationality"),
            ("social_harm_bench", "category"),
            ("udhr", "limited_article_name"),
        )
        for task_name, private_key in cases:
            with self.subTest(task=task_name, field=private_key):
                sentinel = f"PRIVATE_{task_name}_{private_key}_SENTINEL_123456"
                task = task_with(
                    base_task(metadata={private_key: sentinel}),
                    name=task_name,
                )
                sample = run_protocol(
                    "c1",
                    lambda *args, value=sentinel: submit_output(value),
                    task=task,
                )
                audit = sample.metadata[AUDIT_METADATA_KEY]
                self.assertFalse(audit["valid"])
                self.assertTrue(audit["hidden_leakage_detected"])
                self.assertIn("hidden_data_in_solving_payload", audit["violations"])
                self.assertNotIn(sentinel, json.dumps(audit))

    def test_audit_serialization_failure_is_fail_closed(self):
        with patch(
            "pipeline.agentic.audit._jsonable",
            side_effect=TypeError("unsafe serialization body"),
        ):
            sample = run_protocol("c1", lambda *args: submit_output())

        audit = sample.metadata[AUDIT_METADATA_KEY]
        self.assertEqual(sample.output.completion, "")
        self.assertFalse(audit["valid"])
        self.assertEqual(audit["termination_reason"], "audit_error")
        self.assertEqual(audit["caught_error_type"], "TypeError")
        self.assertNotIn("unsafe serialization body", json.dumps(audit))

    def test_pht_truth_fields_are_absent_from_actual_solving_events(self):
        original = history_no_push(grader="mockllm/grader", llamaguard_model=None)
        source = original.dataset[0]
        truth_values = [
            source.metadata["True Version"],
            source.metadata["False Version"],
            source.metadata["Scenario"],
        ]
        task = task_with(
            original,
            dataset=[source],
            scorer=output_scorer(),
        )
        config = AgenticConfig.default(
            "c1",
            private_metadata_keys=("True Version", "False Version", "Scenario"),
        )
        sample = run_protocol(
            "c1",
            lambda *args: submit_output("public historical response"),
            config=config,
            task=task,
        )
        audit = sample.metadata[AUDIT_METADATA_KEY]
        solving_payload = event_payload(
            sample, event_ids=set(audit["solving_event_ids"])
        )

        for truth in truth_values:
            self.assertTrue(truth)
            self.assertNotIn(truth, solving_payload)
        self.assertTrue(audit["valid"])
        self.assertFalse(audit["hidden_leakage_detected"])


class TestDynamicConsultation(unittest.TestCase):
    def roles(self, calls=None):
        return {
            "agentic_analyst": fixed_role_model("analyst", "analysis", calls),
            "agentic_critic": fixed_role_model("critic", "critique", calls),
        }

    def test_c4_zero_and_two_serial_helpers_then_separate_submit(self):
        def zero(messages, tools, tool_choice, config):
            names = [tool.name for tool in tools]
            if "submit" in names:
                return submit_output()
            return ModelOutput.for_tool_call(
                "mockllm/model",
                "finish_consultation",
                {"draft": "zero-helper draft"},
            )

        zero_sample = run_protocol("c4", zero, roles=self.roles())
        self.assertTrue(zero_sample.metadata[AUDIT_METADATA_KEY]["valid"])
        self.assertEqual(zero_sample.metadata[AUDIT_METADATA_KEY]["helper_count"], 0)

        consultation_turn = 0
        target_order = []
        helper_calls = []

        def two(messages, tools, tool_choice, config):
            nonlocal consultation_turn
            names = [tool.name for tool in tools]
            if "submit" in names:
                target_order.append("submit")
                return submit_output()
            consultation_turn += 1
            if consultation_turn == 1:
                target_order.append("analyst")
                return ModelOutput.for_tool_call(
                    "mockllm/model", "agentic_analyst", {"input": "review draft"}
                )
            if consultation_turn == 2:
                target_order.append("critic")
                return ModelOutput.for_tool_call(
                    "mockllm/model", "agentic_critic", {"input": "review safety"}
                )
            target_order.append("finish")
            return ModelOutput.for_tool_call(
                "mockllm/model", "finish_consultation", {"draft": "consulted draft"}
            )

        sample = run_protocol("c4", two, roles=self.roles(helper_calls))
        audit = sample.metadata[AUDIT_METADATA_KEY]
        self.assertEqual(sample.output.completion, EXACT_ANSWER)
        self.assertTrue(audit["valid"])
        self.assertEqual(target_order, ["analyst", "critic", "finish", "submit"])
        self.assertEqual(audit["helper_roles"], ["agentic_analyst", "agentic_critic"])
        self.assertEqual(audit["helper_count"], 2)
        self.assertEqual(len(helper_calls), 2)
        self.assertEqual(audit["tool_counts"]["submit"], 1)
        self.assertEqual(audit["tool_counts"]["finish_consultation"], 1)

    def test_helper_model_failure_is_bounded_empty_and_audited(self):
        target_calls = 0
        helper_calls = 0

        def target(messages, tools, tool_choice, config):
            nonlocal target_calls
            target_calls += 1
            return ModelOutput.for_tool_call(
                "mockllm/model", "agentic_analyst", {"input": "help"}
            )

        def failing_helper(messages, tools, tool_choice, config):
            nonlocal helper_calls
            helper_calls += 1
            raise RuntimeError("helper model failed")

        roles = self.roles()
        roles["agentic_analyst"] = get_model(
            "mockllm/failing-helper", custom_outputs=failing_helper
        )
        sample = run_protocol("c4", target, roles=roles)
        audit = sample.metadata[AUDIT_METADATA_KEY]

        self.assertEqual(target_calls, 1)
        self.assertEqual(helper_calls, 1)
        self.assertEqual(sample.output.completion, "")
        self.assertFalse(audit["valid"])
        self.assertEqual(audit["helper_count"], 1)
        self.assertEqual(audit["tool_failure_count"], 1)
        self.assertEqual(audit["caught_error_type"], "RuntimeError")
        self.assertIn("tool_failure", audit["violations"])

    def test_helper_token_limit_is_bounded_empty_and_audited(self):
        target_calls = 0
        helper_calls = 0

        def target(messages, tools, tool_choice, config):
            nonlocal target_calls
            target_calls += 1
            return ModelOutput.for_tool_call(
                "mockllm/model", "agentic_analyst", {"input": "bounded help"}
            )

        def over_limit_helper(messages, tools, tool_choice, config):
            nonlocal helper_calls
            helper_calls += 1
            output = ModelOutput.from_content("mockllm/large-helper", "too large")
            output.usage = ModelUsage(
                input_tokens=1,
                output_tokens=999,
                total_tokens=1000,
            )
            return output

        roles = self.roles()
        roles["agentic_analyst"] = get_model(
            "mockllm/large-helper", custom_outputs=over_limit_helper
        )
        sample = run_protocol("c4", target, roles=roles)
        audit = sample.metadata[AUDIT_METADATA_KEY]

        self.assertEqual(target_calls, 1)
        self.assertEqual(helper_calls, 1)
        self.assertEqual(sample.output.completion, "")
        self.assertFalse(audit["valid"])
        self.assertEqual(audit["termination_reason"], "limit")
        self.assertEqual(audit["limit_event_count"], 1)
        self.assertEqual(audit["tool_failure_count"], 1)
        self.assertIn("helper_failure", audit["violations"])
        self.assertIn("tool_failure", audit["violations"])

    def test_c4_rejects_third_helper_before_launch(self):
        consultation_calls = 0
        helper_calls = []

        def target(messages, tools, tool_choice, config):
            nonlocal consultation_calls
            consultation_calls += 1
            return ModelOutput.for_tool_call(
                "mockllm/model", "agentic_analyst", {"input": "again"}
            )

        sample = run_protocol("c4", target, roles=self.roles(helper_calls))
        audit = sample.metadata[AUDIT_METADATA_KEY]
        self.assertEqual(sample.output.completion, "")
        self.assertFalse(audit["valid"])
        self.assertEqual(consultation_calls, 3)
        self.assertEqual(len(helper_calls), 2)
        self.assertEqual(audit["tool_counts"]["agentic_analyst"], 2)
        self.assertIn("helper_cap", audit["violations"])

    def test_c4_final_rejects_submit_mixed_with_other_tool(self):
        consultation_finished = False

        def target(messages, tools, tool_choice, config):
            nonlocal consultation_finished
            names = [tool.name for tool in tools]
            if "submit" in names:
                return tool_output(
                    ("submit", {"answer": EXACT_ANSWER}),
                    ("finish_consultation", {"draft": "not allowed in final"}),
                )
            consultation_finished = True
            return ModelOutput.for_tool_call(
                "mockllm/model",
                "finish_consultation",
                {"draft": "consultation-only draft"},
            )

        sample = run_protocol("c4", target, roles=self.roles())
        audit = sample.metadata[AUDIT_METADATA_KEY]
        self.assertTrue(consultation_finished)
        self.assertEqual(sample.output.completion, "")
        self.assertFalse(audit["valid"])
        self.assertIn("mixed_final_tool_turn", audit["violations"])
        self.assertNotEqual(sample.output.completion, "consultation-only draft")

    def test_c4_never_uses_consultation_draft_as_non_submit_fallback(self):
        consultation_calls = 0
        final_calls = 0
        draft = "CONSULTATION_DRAFT_MUST_NOT_BECOME_FINAL"

        def target(messages, tools, tool_choice, config):
            nonlocal consultation_calls, final_calls
            names = [tool.name for tool in tools]
            if "submit" in names:
                final_calls += 1
                return ModelOutput.from_content(
                    "mockllm/model", "plain final text is also not accepted"
                )
            consultation_calls += 1
            return ModelOutput.for_tool_call(
                "mockllm/model", "finish_consultation", {"draft": draft}
            )

        sample = run_protocol("c4", target, roles=self.roles())
        audit = sample.metadata[AUDIT_METADATA_KEY]
        budget = AgenticConfig.default("c4").budget
        self.assertEqual(consultation_calls, 1)
        # Inspect 0.3.246 detects the turn limit on the first over-limit call.
        self.assertLessEqual(final_calls, budget.max_final_turns + 1)
        self.assertEqual(sample.output.completion, "")
        self.assertFalse(audit["valid"])
        self.assertNotEqual(sample.output.completion, draft)
        self.assertGreaterEqual(audit["limit_event_count"], 1)

    def test_c4_rejects_helper_and_finish_co_emission_without_launch(self):
        helper_calls = []

        def target(messages, tools, tool_choice, config):
            return tool_output(
                ("agentic_analyst", {"input": "review"}),
                ("finish_consultation", {"draft": "mixed draft"}),
            )

        sample = run_protocol("c4", target, roles=self.roles(helper_calls))
        audit = sample.metadata[AUDIT_METADATA_KEY]
        self.assertEqual(sample.output.completion, "")
        self.assertEqual(helper_calls, [])
        self.assertEqual(audit["helper_count"], 0)
        self.assertEqual(audit["finish_count"], 0)
        self.assertIn("helper_finish_same_turn", audit["violations"])
        self.assertNotIn("agentic_analyst", audit["tool_counts"])


class TestBenchmarkBoundaryIntegration(unittest.TestCase):
    def test_fscale_uses_existing_fenced_or_bare_tolerant_json_scorer(self):
        accepted = {
            "fenced_trailing_comma": (
                '```json\n{"reasoning":"fenced","answer":"agree strongly",}\n```',
                0,
            ),
            "bare_with_surrounding_text": (
                'prefix {"reasoning":"bare","answer":"disagree strongly",} suffix',
                100,
            ),
        }
        for label, (answer, expected_score) in accepted.items():
            with self.subTest(language=label):
                original = fscale(llamaguard_model=None)
                original_scorers = original.scorer
                task = task_with(original, dataset=[original.dataset[0]])
                self.assertIs(task.scorer, original_scorers)
                sample = run_protocol(
                    "c1",
                    lambda *args, submitted=answer: submit_output(submitted),
                    task=task,
                )
                audit = sample.metadata[AUDIT_METADATA_KEY]
                self.assertTrue(audit["valid"])
                self.assertEqual(sample.output.completion, answer)
                self.assertEqual(sample.scores["fscale_scorer"].value, expected_score)

    def test_direct_inspect_entry_is_discoverable_from_repo_root(self):
        repo_root = Path(__file__).resolve().parents[1]
        inspect_cli = Path(sys.executable).with_name("inspect")
        result = subprocess.run(
            [
                str(inspect_cli),
                "list",
                "tasks",
                "pipeline/agentic/eval.py",
                "--json",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        discovered = json.loads(result.stdout)
        self.assertEqual(discovered[0]["name"], "agentic")
        self.assertEqual(discovered[0]["file"], "pipeline/agentic/eval.py")

    def test_generic_registered_task_logs_canonical_base_identity(self):
        task = registered_agentic_task(
            base_task="fscale",
            condition="c1",
            grader="mockllm/grader",
            analyst_model="mockllm/analyst",
            critic_model="mockllm/critic",
            llamaguard_model=None,
        )
        task = task_with(task, dataset=[task.dataset[0]])
        answer = '{"reasoning":"ok","answer":"agree strongly"}'

        log, sample = evaluate_task(task, lambda *args: submit_output(answer))

        self.assertEqual(log.status, "success")
        self.assertEqual(log.eval.task_args["base_task"], "fscale")
        self.assertEqual(log.eval.task_args["condition"], "c1")
        self.assertEqual(log.eval.metadata["agentic_protocol"]["base_task"], "fscale")
        self.assertEqual(sample.metadata[AUDIT_METADATA_KEY]["condition"], "c1")
        self.assertEqual(sample.scores["fscale_scorer"].value, 0)


if __name__ == "__main__":
    unittest.main()
