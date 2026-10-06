'''
Tests for pipeline/utils/replay.py — the message list a replayed variant
actually sends to the target.

Cluster rows may carry their own system prompt (the human_rights persona arms
and persusafety's setup, datasets/public/*.csv), which stage 1 sends as a
two-message input. A stage-2 variant has to send it too, or the perturbed
condition is measured against a control it does not match. Stage 3 (the
multi-turn scenario walk) is tested here too: it brings its own system prompt
and records under `simulations`.

All synthetic (no model calls): the target `generate` is a stub that records
the messages it was handed.

Run: uv run python3 -m unittest discover tests
'''

import asyncio
import unittest
import unittest.mock

from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    ModelOutput,
)
from inspect_ai.solver import TaskState

from pipeline.stage3_simulation.solvers import render_transcript, scenario
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
        self.assertEqual(
            [v["condition"] for v in state.metadata["perturbations"]["paraphrase"]],
            ["paraphrase_variant_1"],
        )

    def test_reconsideration_is_gated_too(self):
        from pipeline.stage2_perturbation.solvers import reconsideration

        state = make_state({"families": ["paraphrase"]})
        captured: list = []
        asyncio.run(reconsideration()(state, stub_generate(captured)))
        self.assertEqual(captured, [])
        self.assertNotIn("reconsideration", state.metadata.get("perturbations", {}))

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


TURNS = {
    "": "Opening ask.",
    "refuse": "After refuse.",
    "hedge": "After hedge.",
    "refuse/refuse": "After refuse/refuse.",
    "refuse/hedge": "After refuse/hedge.",
    "hedge/refuse": "After hedge/refuse.",
    "hedge/hedge": "After hedge/hedge.",
}


def tree_row(variant: int = 1, turns: dict | None = None, **extra) -> dict:
    return {
        "id": "s1", "variant": variant, "condition": f"scenario_variant_{variant}",
        "system": f"Deployment {variant}.", "turns": dict(TURNS if turns is None else turns),
        "query": TURNS[""], **extra,
    }


def scripted_generate(replies: list[str], captured: list):
    """Target stub returning the next scripted reply; records each message list."""
    queue = list(replies)

    async def generate(state: TaskState, cache: bool = True) -> TaskState:
        captured.append(list(state.messages))
        state.output = ModelOutput.from_content("m", queue.pop(0))
        return state

    return generate


def scripted_classifier(labels: list[str]):
    queue = list(labels)
    seen = []

    async def classify(model, system, transcript, completion):
        seen.append(completion)
        return queue.pop(0), "judge"

    classify.seen = seen
    return classify


def run_scenario(rows, replies, labels, metadata=None, classifier_model="clf"):
    import pipeline.stage3_simulation.solvers as solvers_module

    state = make_state(metadata or {})
    captured: list = []
    with unittest.mock.patch.object(solvers_module, "classify", scripted_classifier(labels)) as clf:
        asyncio.run(
            scenario({"s1": rows}, classifier_model)(state, scripted_generate(replies, captured))
        )
    return state, captured, clf


class TestRenderTranscript(unittest.TestCase):
    def test_roles_are_bracketed_in_order(self):
        messages = [
            ChatMessageSystem(content="S"), ChatMessageUser(content="U1"),
            ChatMessageAssistant(content="A1"), ChatMessageUser(content="U2"),
        ]
        self.assertEqual(
            render_transcript(messages), "[system] S\n\n[user] U1\n\n[assistant] A1\n\n[user] U2"
        )


class TestScenarioWalk(unittest.TestCase):
    def records(self, state):
        return state.metadata["simulations"]["scenario"]

    def test_refuse_then_hedge_walks_three_turns(self):
        state, captured, clf = run_scenario(
            [tree_row()], ["r1", "r2", "r3"], ["refuse", "hedge"]
        )
        recs = self.records(state)
        self.assertEqual([r["condition"] for r in recs],
                         ["scenario_variant_1_t1", "scenario_variant_1_t2", "scenario_variant_1_t3"])
        self.assertEqual([r["path"] for r in recs], ["", "refuse", "refuse/hedge"])
        self.assertEqual([r["label"] for r in recs], ["refuse", "hedge", None])
        self.assertEqual([r["label_source"] for r in recs], ["judge", "judge", None])
        self.assertEqual([r["completion"] for r in recs], ["r1", "r2", "r3"])
        self.assertTrue(all(r["variant"] == 1 for r in recs))
        self.assertEqual([r["turn"] for r in recs], [1, 2, 3])
        # the depth-3 reply is never classified: nothing to branch into
        self.assertEqual(clf.seen, ["r1", "r2"])
        # third target call saw the full history: system, U, A, U, A, U
        self.assertEqual([m.role for m in captured[2]],
                         ["system", "user", "assistant", "user", "assistant", "user"])
        self.assertEqual(captured[2][0].text, "Deployment 1.")
        self.assertEqual(captured[2][-1].text, "After refuse/hedge.")
        self.assertEqual(captured[2][2].text, "r1")

    def test_recorded_query_is_the_transcript_up_to_that_user_turn(self):
        state, captured, _ = run_scenario([tree_row()], ["r1", "r2", "r3"], ["hedge", "refuse"])
        recs = self.records(state)
        self.assertEqual(recs[0]["query"], "[system] Deployment 1.\n\n[user] Opening ask.")
        self.assertEqual(recs[1]["query"], render_transcript(captured[1]))
        self.assertNotIn("r2", recs[1]["query"])
        self.assertIn("r2", recs[2]["query"])

    def test_comply_stops_after_one_turn(self):
        state, captured, _ = run_scenario([tree_row()], ["r1"], ["comply"])
        recs = self.records(state)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["label"], "comply")
        self.assertEqual(len(captured), 1)

    def test_two_variants_run_and_are_labelled_separately(self):
        rows = [tree_row(1), tree_row(2)]
        state, captured, _ = run_scenario(rows, ["a", "b", "c", "d"], ["comply", "refuse", "comply"])
        conditions = sorted(r["condition"] for r in self.records(state))
        self.assertEqual(len(conditions), 3)
        self.assertEqual({c.rsplit("_t", 1)[0] for c in conditions},
                         {"scenario_variant_1", "scenario_variant_2"})
        systems = {m[0].text for m in captured}
        self.assertEqual(systems, {"Deployment 1.", "Deployment 2."})

    def test_shared_state_is_untouched(self):
        state, _, _ = run_scenario([tree_row()], ["r1"], ["comply"], {"system_prompt": SYSTEM_PROMPT})
        self.assertEqual(state.messages, [])
        self.assertIsNone(state.output.completion or None)

    # ---- review-focus failure modes ----

    def test_empty_completion_is_recorded_and_stops_the_walk(self):
        state, captured, clf = run_scenario([tree_row()], ["", "never"], ["refuse"])
        recs = self.records(state)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["completion"], "")
        self.assertEqual((recs[0]["label"], recs[0]["label_source"]), (None, "empty"))
        self.assertEqual(clf.seen, [])
        self.assertEqual(len(captured), 1)

    def test_label_with_no_stored_turn_stops_the_walk(self):
        turns = {k: v for k, v in TURNS.items() if k != "hedge"}
        state, captured, _ = run_scenario([tree_row(turns=turns)], ["r1", "never"], ["hedge"])
        recs = self.records(state)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["label"], "hedge")
        self.assertEqual(len(captured), 1)

    def test_fallback_row_is_skipped(self):
        rows = [tree_row(1, fallback=True), tree_row(2)]
        state, captured, _ = run_scenario(rows, ["r1"], ["comply"])
        recs = self.records(state)
        self.assertEqual([r["variant"] for r in recs], [2])
        self.assertEqual(len(captured), 1)

    def test_all_fallback_rows_record_an_empty_family(self):
        state, captured, _ = run_scenario([tree_row(1, fallback=True)], [], [])
        self.assertEqual(self.records(state), [])
        self.assertEqual(captured, [])

    def test_sim_k_1_truncation_walks_variant_one_only(self):
        rows = truncated({"s1": sorted([tree_row(2), tree_row(1)], key=lambda r: r["variant"])}, 1)["s1"]
        state, _, _ = run_scenario(rows, ["r1"], ["comply"])
        self.assertEqual([r["condition"] for r in self.records(state)], ["scenario_variant_1_t1"])

    def test_target_failure_drops_the_rest_of_the_walk(self):
        import pipeline.stage3_simulation.solvers as solvers_module

        async def failing_generate_variant(generate, test, label, attempts=3):
            return None

        state = make_state({})
        with unittest.mock.patch.object(solvers_module, "generate_variant", failing_generate_variant), \
             unittest.mock.patch.object(solvers_module, "classify", scripted_classifier([])):
            asyncio.run(scenario({"s1": [tree_row()]}, "clf")(state, stub_generate([])))
        self.assertEqual(self.records(state), [])

    def test_families_gate(self):
        state, captured, _ = run_scenario([tree_row()], [], [], {"families": ["paraphrase"]})
        self.assertNotIn("simulations", state.metadata)
        self.assertEqual(captured, [])

    def test_families_including_scenario_runs(self):
        state, captured, _ = run_scenario([tree_row()], ["r1"], ["comply"], {"families": ["scenario"]})
        self.assertEqual(len(self.records(state)), 1)

    def test_no_stored_rows_leaves_state_alone(self):
        state = make_state({})
        asyncio.run(scenario({}, "clf")(state, stub_generate([])))
        self.assertNotIn("simulations", state.metadata)
