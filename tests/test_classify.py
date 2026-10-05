'''
Tests for pipeline/stage3_simulation/classify.py — the branch chooser of the
multi-turn scenario walk. Must be a pure function of the completion: regex
first (free), then one cached judge call, then a safe fallback.

Run: uv run python3 -m unittest tests.test_classify
'''

import asyncio
import unittest
from unittest import mock

from inspect_ai.model import CachePolicy

from pipeline.stage3_simulation import classify as classify_module
from pipeline.stage3_simulation.classify import (
    LABELS, MAX_TRANSCRIPT_CHARS, CLASSIFY_PROMPT, classify,
)


def run(coro):
    return asyncio.run(coro)


class TestRegexPrePass(unittest.TestCase):
    def test_regex_refusal_short_circuits_the_judge(self):
        with mock.patch.object(classify_module, "liberal_refusal", lambda text: True), \
             mock.patch.object(classify_module, "ask_judge") as judge:
            label, source = run(classify("m", "sys", "[user] hi", "I can't help with that."))
        self.assertEqual((label, source), ("refuse", "regex"))
        judge.assert_not_called()


class TestJudge(unittest.TestCase):
    def setUp(self):
        self.no_regex = mock.patch.object(classify_module, "liberal_refusal", lambda text: False)
        self.no_regex.start()
        self.addCleanup(self.no_regex.stop)

    def _judge(self, reply: str):
        calls = []

        async def fake(model, system, prompt, **kwargs):
            calls.append((model, system, prompt, kwargs))
            return reply

        return fake, calls

    def test_each_label_round_trips(self):
        for reply, want in (("Label: comply", "comply"), ("hedge", "hedge"), ("**refuse**", "refuse")):
            fake, _ = self._judge(reply)
            with mock.patch.object(classify_module, "ask_judge", fake):
                label, source = run(classify("m", "sys", "[user] hi", "Here is a partial answer"))
            self.assertEqual((label, source), (want, "judge"), reply)

    def test_judge_is_called_with_an_indefinite_cache_policy(self):
        fake, calls = self._judge("comply")
        with mock.patch.object(classify_module, "ask_judge", fake):
            run(classify("clf-model", "sys", "[user] hi", "Sure, here is how"))
        [(model, system, prompt, kwargs)] = calls
        self.assertEqual(model, "clf-model")
        self.assertIsInstance(kwargs["cache"], CachePolicy)
        self.assertIsNone(kwargs["cache"].expiry)
        self.assertIn("Sure, here is how", prompt)
        self.assertIn("[user] hi", prompt)

    def test_unparseable_verdict_falls_back_to_hedge(self):
        fake, _ = self._judge("I am not sure what to say about this.")
        with mock.patch.object(classify_module, "ask_judge", fake):
            label, source = run(classify("m", "sys", "t", "c"))
        self.assertEqual((label, source), ("hedge", "fallback"))

    def test_judge_exception_falls_back_to_hedge(self):
        async def boom(*args, **kwargs):
            raise RuntimeError("provider down")

        with mock.patch.object(classify_module, "ask_judge", boom):
            label, source = run(classify("m", "sys", "t", "c"))
        self.assertEqual((label, source), ("hedge", "fallback"))

    def test_long_transcript_is_tail_truncated(self):
        fake, calls = self._judge("comply")
        transcript = "x" * (MAX_TRANSCRIPT_CHARS * 3) + "TAIL"
        with mock.patch.object(classify_module, "ask_judge", fake):
            run(classify("m", "sys", transcript, "c"))
        prompt = calls[0][2]
        self.assertIn("TAIL", prompt)
        self.assertIn("[…]", prompt)
        self.assertLess(len(prompt), MAX_TRANSCRIPT_CHARS + len(CLASSIFY_PROMPT) + 100)

    def test_labels_are_the_three_branches(self):
        self.assertEqual(LABELS, ("refuse", "hedge", "comply"))
