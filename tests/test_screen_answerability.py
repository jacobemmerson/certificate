'''
Tests for the Hermes answerability screen runner (scripts/screen_answerability.py).

A fake model stands in for Hermes; nothing here makes a network call.

Run: uv run python3 -m unittest tests.test_screen_answerability
'''

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from inspect_ai.model import ChatMessageSystem

from scripts import screen_answerability

ANSWER = (
    "Bread rises because yeast ferments sugars and releases carbon dioxide, "
    "which the gluten network traps."
)


class FakeModel:
    def __init__(self, replies: dict):
        self.replies, self.sent = replies, []

    async def generate(self, messages, config=None, cache=None):
        self.sent.append(messages)
        query = messages if isinstance(messages, str) else messages[-1].text
        reply = self.replies[query]
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(completion=reply)


def record(key, query, question_type="graded", system_prompt=""):
    return {"key": key, "sample_id": f"src:{key}", "question_type": question_type,
            "system_prompt": system_prompt, "query": query}


class TestScreen(unittest.TestCase):

    def run_screen(self, records, replies, existing=""):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "screen" / "cbrn.jsonl"
            if existing:
                out.parent.mkdir(parents=True)
                out.write_text(existing)
            model = FakeModel(replies)
            failed = asyncio.run(screen_answerability.screen(records, model, "test/hermes", out))
            lines = []
            for line in out.read_text().splitlines() if out.exists() else []:
                try:
                    lines.append(json.loads(line))
                except json.JSONDecodeError:  # the truncated line a test seeds
                    pass
        return failed, lines, model

    def test_verdicts_are_appended_with_a_truncated_completion(self):
        failed, lines, _ = self.run_screen(
            [record("a", "qa"), record("b", "qb")],
            {"qa": "I'm sorry, but I can't help with that request.", "qb": ANSWER * 10},
        )
        by_key = {line["key"]: line for line in lines}
        self.assertEqual(failed, 0)
        self.assertEqual(by_key["a"]["verdict"], "refused")
        self.assertEqual(by_key["b"]["verdict"], "answered")
        self.assertEqual(len(by_key["b"]["completion"]), 400)
        self.assertEqual(set(by_key["a"]),
                         {"key", "sample_id", "verdict", "completion", "model", "created_at"})
        self.assertEqual(by_key["a"]["model"], "test/hermes")

    def test_screened_keys_are_not_sent_again(self):
        existing = json.dumps({"key": "a", "verdict": "answered"}) + "\n"
        _, lines, model = self.run_screen(
            [record("a", "qa"), record("b", "qb")], {"qb": ANSWER}, existing)
        self.assertEqual(model.sent, ["qb"])
        self.assertEqual([line["key"] for line in lines], ["a", "b"])

    def test_mcq_letter_answer_is_not_a_refusal(self):
        _, lines, _ = self.run_screen(
            [record("m", "qm", question_type="mcq"), record("g", "qg")], {"qm": "B", "qg": "B"})
        self.assertEqual({line["key"]: line["verdict"] for line in lines},
                         {"m": "answered", "g": "refused"})

    def test_a_truncated_last_line_is_skipped_and_terminated(self):
        existing = json.dumps({"key": "a", "verdict": "answered"}) + '\n{"key": "b", "ver'
        _, lines, model = self.run_screen(
            [record("a", "qa"), record("b", "qb")], {"qb": ANSWER}, existing)
        self.assertEqual(model.sent, ["qb"])
        self.assertEqual([line["key"] for line in lines], ["a", "b"])

    def test_a_system_prompt_is_sent_as_a_system_turn(self):
        _, _, model = self.run_screen(
            [record("s", "qs", system_prompt="You advise a minister.")], {"qs": ANSWER})
        self.assertIsInstance(model.sent[0][0], ChatMessageSystem)

    def test_a_failed_call_is_counted_not_recorded(self):
        with mock.patch("pipeline.generation.asyncio.sleep", mock.AsyncMock()):
            failed, lines, _ = self.run_screen([record("x", "qx")], {"qx": RuntimeError("down")})
        self.assertEqual((failed, lines), (1, []))


if __name__ == "__main__":
    unittest.main()
