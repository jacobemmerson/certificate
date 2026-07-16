"""Independent regressions for agentic privacy and C4 event attestation."""

from __future__ import annotations

import tempfile
import unittest

from inspect_ai import Task, eval, task_with
from inspect_ai.dataset import Sample
from inspect_ai.event import ModelEvent, SpanBeginEvent, ToolEvent
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import CORRECT, Score, scorer
from inspect_ai.solver import generate
from pipeline.agentic import AUDIT_METADATA_KEY, AgenticConfig, make_agentic_task
from pipeline.agentic.audit import ProtocolTrace, _validate_c4_causality
from pipeline.stage1_evaluation.evals.democratic_authoritarian_bias import (
    leader_favorability,
)
from pipeline.stage1_evaluation.evals.llm_human_rights import udhr
from pipeline.stage1_evaluation.evals.preserving_historical_truth import history_no_push
from pipeline.stage1_evaluation.evals.socialharmbench import social_harm_bench


@scorer(metrics=[])
def _constant_scorer():
    async def score(state, target):
        return Score(value=CORRECT)

    return score


def _submit(answer: str) -> ModelOutput:
    return ModelOutput.for_tool_call(
        "mockllm/model",
        "submit",
        {"answer": answer},
        tool_call_id="submit-call",
    )


def _evaluate(task: Task, target_output, *, roles: dict | None = None):
    wrapped = make_agentic_task(task, AgenticConfig.default("c1"))
    model = get_model("mockllm/model", custom_outputs=target_output)
    with tempfile.TemporaryDirectory() as log_dir:
        log = eval(
            wrapped,
            model=model,
            model_roles=roles,
            log_dir=log_dir,
            display="none",
        )[0]
        assert log.status == "success", log.error
        sample = log.samples[0]
        sample.events = list(sample.events)
        return sample


def _production_sample_task(task: Task, private_key: str, secret: str) -> Task:
    source = task.dataset[0]
    metadata = dict(source.metadata or {})
    metadata[private_key] = secret
    sample = Sample(
        id=source.id,
        input=source.input,
        target=source.target,
        metadata=metadata,
    )
    return task_with(task, dataset=[sample], scorer=_constant_scorer())


def _c4_sample(helper_count: int):
    task = Task(
        dataset=[Sample(id="c4", input="Give a public answer.")],
        solver=generate(),
        scorer=_constant_scorer(),
        name="c4_causal_regression",
    )
    turn = 0

    def target(messages, tools, tool_choice, config):
        nonlocal turn
        if any(tool.name == "submit" for tool in tools):
            return _submit("final")
        turn += 1
        if turn <= helper_count:
            function = "agentic_analyst" if turn == 1 else "agentic_critic"
            return ModelOutput.for_tool_call(
                "mockllm/model", function, {"input": f"review {turn}"}
            )
        return ModelOutput.for_tool_call(
            "mockllm/model",
            "finish_consultation",
            {"draft": "consulted draft"},
        )

    roles = {
        "agentic_analyst": get_model(
            "mockllm/analyst",
            custom_outputs=lambda *args: ModelOutput.from_content(
                "mockllm/analyst", "analysis"
            ),
        ),
        "agentic_critic": get_model(
            "mockllm/critic",
            custom_outputs=lambda *args: ModelOutput.from_content(
                "mockllm/critic", "critique"
            ),
        ),
    }
    wrapped = make_agentic_task(task, AgenticConfig.default("c4"))
    with tempfile.TemporaryDirectory() as log_dir:
        log = eval(
            wrapped,
            model=get_model("mockllm/model", custom_outputs=target),
            model_roles=roles,
            log_dir=log_dir,
            display="none",
        )[0]
        assert log.status == "success", log.error
        sample = log.samples[0]
        sample.events = list(sample.events)
        return sample


class TestPrivateScalarRegressions(unittest.TestCase):
    def test_production_shaped_short_private_leaks_invalidate_all_policies(self):
        cases = (
            (
                "PHT",
                history_no_push(grader="mockllm/grader", llamaguard_model=None),
                "True Version",
                "truth",
            ),
            (
                "DAB",
                leader_favorability(llamaguard_model=None),
                "classification",
                "autocracy",
            ),
            (
                "SocialHarmBench",
                social_harm_bench(grader="mockllm/grader", llamaguard_model=None),
                "category",
                "violence",
            ),
            (
                "HR",
                udhr(grader="mockllm/grader", llamaguard_model=None),
                "limited_article_name",
                "privacy",
            ),
        )
        for label, base, private_key, secret in cases:
            with self.subTest(benchmark=label, private_key=private_key):
                task = _production_sample_task(base, private_key, secret)
                sample = _evaluate(task, lambda *args, value=secret: _submit(value))
                audit = sample.metadata[AUDIT_METADATA_KEY]
                self.assertEqual(sample.output.completion, secret)
                self.assertTrue(audit["submit_payload_exact"])
                self.assertFalse(audit["valid"])
                self.assertTrue(audit["hidden_leakage_detected"])
                self.assertIn("hidden_data_in_solving_payload", audit["violations"])

    def test_configured_non_string_private_scalar_is_audited(self):
        task = Task(
            dataset=[
                Sample(
                    id="numeric-private",
                    input="Give a generic response.",
                    metadata={"secret": 7},
                )
            ],
            solver=generate(),
            scorer=_constant_scorer(),
            name="numeric_private_regression",
        )
        sample = _evaluate(task, lambda *args: _submit("private value: 7"))
        audit = sample.metadata[AUDIT_METADATA_KEY]
        self.assertFalse(audit["valid"])
        self.assertTrue(audit["hidden_leakage_detected"])

    def test_short_target_is_audited_but_token_substrings_do_not_match(self):
        def task_for(answer: str) -> Task:
            return Task(
                dataset=[
                    Sample(
                        id="short-target",
                        input="Discuss whether a claim is truthful.",
                        target="truth",
                    )
                ],
                solver=generate(),
                scorer=_constant_scorer(),
                name="short_target_regression",
            )

        clean = _evaluate(task_for("truthful"), lambda *args: _submit("truthful"))
        self.assertTrue(clean.metadata[AUDIT_METADATA_KEY]["valid"])
        self.assertFalse(clean.metadata[AUDIT_METADATA_KEY]["hidden_leakage_detected"])

        leaked = _evaluate(task_for("truth"), lambda *args: _submit("truth"))
        audit = leaked.metadata[AUDIT_METADATA_KEY]
        self.assertFalse(audit["valid"])
        self.assertTrue(audit["hidden_leakage_detected"])

    def test_exact_short_value_in_initial_public_projection_is_not_private(self):
        base = social_harm_bench(grader="mockllm/grader", llamaguard_model=None)
        source = base.dataset[0]
        sample = Sample(
            id=source.id,
            input="Discuss violence without endorsing harm.",
            metadata={**(source.metadata or {}), "category": "violence"},
        )
        task = task_with(base, dataset=[sample], scorer=_constant_scorer())
        result = _evaluate(task, lambda *args: _submit("safe response"))
        audit = result.metadata[AUDIT_METADATA_KEY]
        self.assertTrue(audit["valid"])
        self.assertFalse(audit["hidden_leakage_detected"])


class TestC4CausalRegressions(unittest.TestCase):
    def test_runtime_zero_one_and_two_helper_chains_remain_valid(self):
        for helper_count in (0, 1, 2):
            with self.subTest(helper_count=helper_count):
                sample = _c4_sample(helper_count)
                audit = sample.metadata[AUDIT_METADATA_KEY]
                self.assertTrue(audit["valid"], audit["violations"])
                self.assertEqual(audit["helper_count"], helper_count)

    def test_submit_producer_before_finish_completion_violates(self):
        sample = _c4_sample(1)
        events = list(sample.events)
        finish_index = next(
            index
            for index, event in enumerate(events)
            if isinstance(event, ToolEvent) and event.function == "finish_consultation"
        )
        submit_producer_index = next(
            index
            for index, event in enumerate(events)
            if isinstance(event, ModelEvent)
            and any(
                call.function == "submit"
                for call in (event.output.message.tool_calls or [])
            )
        )
        submit_producer = events.pop(submit_producer_index)
        if submit_producer_index < finish_index:
            finish_index -= 1
        events.insert(finish_index, submit_producer)

        spans = {
            event.id: event for event in events if isinstance(event, SpanBeginEvent)
        }
        model_events = [event for event in events if isinstance(event, ModelEvent)]
        tool_events = [event for event in events if isinstance(event, ToolEvent)]
        trace = ProtocolTrace(
            helper_launches=["agentic_analyst"],
            finish_count=1,
            termination_reason="submitted",
        )
        _validate_c4_causality(
            events=events,
            model_events=model_events,
            tool_events=tool_events,
            spans=spans,
            config=AgenticConfig.default("c4"),
            trace=trace,
        )
        self.assertIn("submit_causal_order", trace.violations)


if __name__ == "__main__":
    unittest.main()
