'''
Tests for pipeline/utils/replay.py — the message list a replayed variant
actually sends to the target.

Cluster rows may carry their own system prompt (the human_rights persona arms
and persusafety's setup, datasets/public/*.csv), which stage 1 sends as a
two-message input. A stage-2 variant has to send it too, or the perturbed
condition is measured against a control it does not match. Stage 3 is the
deliberate exception: the reframed deployment brings its own system prompt.

All synthetic (no model calls): the target `generate` is a stub that records
the messages it was handed.

Run: uv run python3 -m unittest discover tests
'''

import asyncio
import unittest

from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageUser,
    ModelOutput,
)
from inspect_ai.solver import TaskState

from pipeline.stage3_simulation.solvers import _scenario_messages
from pipeline.utils.replay import replay

SYSTEM_PROMPT = "You are advising a government minister."


def make_state(metadata: dict) -> TaskState:
    state = TaskState(
        model="m", sample_id="s1", epoch=0, input="the original question", messages=[]
    )
    state.metadata.update(metadata)
    return state


def stub_generate(captured: list):
    """A target stub that records each variant's message list."""

    async def generate(state: TaskState, cache: bool = True) -> TaskState:
        captured.append(list(state.messages))
        state.output = ModelOutput.from_content("m", "the reply")
        return state

    return generate


def run_replay(state: TaskState, family: str, rows: list[dict], **kwargs):
    captured: list = []
    asyncio.run(
        replay(state, stub_generate(captured), family, {"s1": rows}, **kwargs)
    )
    return captured


class TestSystemPromptPreserved(unittest.TestCase):
    """A sample's own system prompt is part of what stage 1 measured, so a
    stage-2 variant must replay it alongside the perturbed query."""

    def test_system_prompt_is_sent_with_the_variant(self):
        state = make_state({"system_prompt": SYSTEM_PROMPT})
        captured = run_replay(
            state, "paraphrase", [{"condition": "paraphrase_variant_1", "query": "reworded"}]
        )

        self.assertEqual(len(captured), 1)
        messages = captured[0]
        self.assertEqual(len(messages), 2)
        self.assertIsInstance(messages[0], ChatMessageSystem)
        self.assertEqual(messages[0].text, SYSTEM_PROMPT)
        self.assertIsInstance(messages[1], ChatMessageUser)
        self.assertEqual(messages[1].text, "reworded")

    def test_no_system_prompt_stays_a_single_user_turn(self):
        state = make_state({})
        captured = run_replay(
            state, "paraphrase", [{"condition": "paraphrase_variant_1", "query": "reworded"}]
        )

        self.assertEqual([type(m) for m in captured[0]], [ChatMessageUser])
        self.assertEqual(captured[0][0].text, "reworded")

    def test_empty_system_prompt_is_not_sent(self):
        """The cluster schema stores an absent system prompt as "", not null."""
        state = make_state({"system_prompt": ""})
        captured = run_replay(
            state, "framing", [{"condition": "framing_imperative", "query": "framed"}]
        )

        self.assertEqual([type(m) for m in captured[0]], [ChatMessageUser])

    def test_every_variant_carries_it(self):
        state = make_state({"system_prompt": SYSTEM_PROMPT})
        rows = [
            {"condition": "paraphrase_variant_1", "query": "reworded one"},
            {"condition": "paraphrase_variant_2", "query": "reworded two"},
        ]
        captured = run_replay(state, "paraphrase", rows)

        self.assertEqual(len(captured), 2)
        for messages in captured:
            self.assertIsInstance(messages[0], ChatMessageSystem)
            self.assertEqual(messages[0].text, SYSTEM_PROMPT)
        self.assertEqual(
            sorted(m[1].text for m in captured), ["reworded one", "reworded two"]
        )


class TestScenarioOverride(unittest.TestCase):
    """Stage 3 replaces the deployment wholesale, so the sample's own system
    prompt is intentionally *not* replayed."""

    def test_reframed_system_prompt_replaces_the_samples_own(self):
        state = make_state({"system_prompt": SYSTEM_PROMPT})
        row = {
            "condition": "scenario_variant_1",
            "query": "composite",
            "system": "You are a compliance assistant at a logistics firm.",
            "context": "It is Tuesday morning.",
            "scenario": "Draft the memo.",
        }
        captured = run_replay(state, "scenario", [row], messages=_scenario_messages)

        messages = captured[0]
        self.assertIsInstance(messages[0], ChatMessageSystem)
        self.assertEqual(messages[0].text, "You are a compliance assistant at a logistics firm.")
        self.assertNotIn(SYSTEM_PROMPT, messages[0].text)
        self.assertEqual(messages[1].text, "It is Tuesday morning.\n\nDraft the memo.")


class TestRecordedVariants(unittest.TestCase):
    """The recorded shape is the scoring spine's contract and must not shift."""

    def test_records_condition_query_and_completion(self):
        state = make_state({"system_prompt": SYSTEM_PROMPT})
        run_replay(
            state, "paraphrase", [{"condition": "paraphrase_variant_1", "query": "reworded"}]
        )

        variants = state.metadata["perturbations"]["paraphrase"]
        self.assertEqual(
            variants,
            [{"condition": "paraphrase_variant_1", "query": "reworded", "completion": "the reply"}],
        )

    def test_shared_state_messages_are_untouched(self):
        state = make_state({"system_prompt": SYSTEM_PROMPT})
        run_replay(
            state, "paraphrase", [{"condition": "paraphrase_variant_1", "query": "reworded"}]
        )

        self.assertEqual(state.messages, [])


if __name__ == "__main__":
    unittest.main()
