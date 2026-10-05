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
from pipeline.utils.replay import replay, truncated

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


class TestFallbackRowsAreNotScored(unittest.TestCase):
    '''
    A fallback row is the original text: the attacker refused, failed, or
    returned something rewrite.py::rewrite_defect rejected, so generation
    persisted the control prompt under a perturbation's label. Replaying it
    would credit the family with the control's own result and drag the family's
    score toward the baseline it is meant to be compared against — and would
    pay for a target call to re-ask a question already asked.
    '''

    def test_fallback_row_is_neither_sent_nor_recorded(self):
        state = make_state({})
        captured = run_replay(state, "paraphrase", [
            {"condition": "paraphrase_variant_1", "query": "the original question",
             "fallback": True},
        ])

        self.assertEqual(captured, [])
        self.assertEqual(state.metadata.get("perturbations", {}).get("paraphrase"), [])

    def test_genuine_variants_survive_alongside_a_fallback(self):
        state = make_state({})
        captured = run_replay(state, "paraphrase", [
            {"condition": "paraphrase_variant_1", "query": "the original question",
             "fallback": True},
            {"condition": "paraphrase_variant_2", "query": "a real rewording",
             "fallback": False},
        ])

        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0][-1].text, "a real rewording")
        recorded = state.metadata["perturbations"]["paraphrase"]
        self.assertEqual([v["condition"] for v in recorded], ["paraphrase_variant_2"])

    def test_rows_without_the_key_are_kept(self):
        '''Stage-3 scenario and stage-2 framing rows carry no `fallback` field.'''
        state = make_state({})
        captured = run_replay(state, "framing", [
            {"condition": "framing_interrogative", "query": "How would one do it?"},
        ])

        self.assertEqual(len(captured), 1)
        self.assertEqual(
            [v["condition"] for v in state.metadata["perturbations"]["framing"]],
            ["framing_interrogative"],
        )


class TestTruncated(unittest.TestCase):
    def test_truncation_counts_only_real_rows(self):
        rows = {"s": [dict(condition="p_1", fallback=True, query="a"),
                      dict(condition="p_2", fallback=False, query="b")]}
        self.assertEqual([r["condition"] for r in truncated(rows, 1)["s"]], ["p_2"])


class TestAbsentVersusEmptyFamily(unittest.TestCase):
    '''
    `truncated` keeps a sample id whose rows were all fallback as an empty
    list, and drops nothing else — so "id absent" means the family does not
    apply to this sample (framing on a sample with no templates, a tolerated
    scenario gap) while "id present with []" means rows existed but none were
    replayable. Only the second is a gap worth recording: an absent family
    recorded as [] becomes a `missing: True` condition in
    pipeline/utils/scoring.py and inflates the family's denominator to every
    sample.
    '''

    def test_absent_id_records_nothing(self):
        state = make_state({})
        captured: list = []
        asyncio.run(replay(state, stub_generate(captured), "framing", {}))

        self.assertEqual(captured, [])
        self.assertNotIn("framing", state.metadata.get("perturbations", {}))

    def test_all_fallback_id_records_an_empty_family(self):
        state = make_state({})
        rows = {"s1": [dict(condition="paraphrase_variant_1", query="q", fallback=True)]}
        captured: list = []
        asyncio.run(replay(state, stub_generate(captured), "paraphrase", truncated(rows, 1)))

        self.assertEqual(captured, [])
        self.assertEqual(state.metadata["perturbations"]["paraphrase"], [])


class TestFamiliesGate(unittest.TestCase):
    '''
    `metadata["families"]` is the one applicability gate (spec C2). Artifacts
    generated before a row's families changed still hold rows for it, so the
    gate has to hold at replay time, for every family, not just rewrites.
    '''

    def test_stored_rows_are_skipped_when_family_not_applicable(self):
        from pipeline.stage2_perturbation.solvers import register

        state = make_state({"families": ["paraphrase", "reconsideration"]})
        captured: list = []
        rows = {"s1": [dict(condition="register_variant_1", query="reworded")]}
        asyncio.run(register(rows)(state, stub_generate(captured)))

        self.assertEqual(captured, [])
        self.assertNotIn("register", state.metadata.get("perturbations", {}))

    def test_framing_rows_are_gated_too(self):
        from pipeline.stage2_perturbation.solvers import framing

        state = make_state({"families": ["paraphrase"]})
        captured: list = []
        rows = {"s1": [dict(condition="framing_imperative", query="framed")]}
        asyncio.run(framing(rows)(state, stub_generate(captured)))
        self.assertEqual(captured, [])

    def test_applicable_family_still_replays(self):
        from pipeline.stage2_perturbation.solvers import paraphrase

        state = make_state({"families": ["paraphrase"]})
        captured: list = []
        rows = {"s1": [dict(condition="paraphrase_variant_1", query="reworded")]}
        asyncio.run(paraphrase(rows)(state, stub_generate(captured)))
        self.assertEqual(len(captured), 1)

    def test_absent_families_means_everything_applies(self):
        '''Logs and fixtures predating the column.'''
        from pipeline.utils.replay import family_applies
        self.assertTrue(family_applies({}, "framing"))
        self.assertTrue(family_applies(None, "persona"))
        self.assertFalse(family_applies({"families": []}, "framing"))


class TestPersonaMessages(unittest.TestCase):
    def test_persona_system_prompt_then_original_query(self):
        from pipeline.stage2_perturbation.solvers import persona

        state = make_state({"families": ["persona"]})
        captured: list = []
        rows = {"s1": [{"condition": "persona_professional", "system": "You work at a firm.",
                        "query": "the original question"}]}
        asyncio.run(persona(rows)(state, stub_generate(captured)))

        messages = captured[0]
        self.assertEqual([type(m) for m in messages], [ChatMessageSystem, ChatMessageUser])
        self.assertEqual(messages[0].text, "You work at a firm.")
        self.assertEqual(messages[1].text, "the original question")
        recorded = state.metadata["perturbations"]["persona"][0]
        self.assertEqual(recorded["condition"], "persona_professional")
        self.assertEqual(recorded["query"], "the original question")
