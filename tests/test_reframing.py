'''
Tests for pipeline/stage3_simulation/prompts.py::parse_reframing — the contract
between the attacker model's JSON and the scenario replay solver.

The solver feeds `system` straight into ChatMessageSystem(content=...) and
joins `context`/`scenario` into the user turn, so all three must come out as
non-empty text. Models do not reliably answer a string contract with a string:
Hermes-4-405B returns nested objects for these fields and had 10 of 12
reframings silently dropped on a real run. Flattening is what keeps a
schema-noncompliant-but-usable answer, while genuinely unusable output is still
rejected with a reason.

Run: uv run python3 -m unittest discover tests
'''

import json
import unittest

from pipeline.stage3_simulation.prompts import parse_reframing

FLAT = {
    "context": "It is Tuesday at a logistics firm.",
    "system": "You are Ops Assistant.",
    "scenario": "Draft the memo.",
}


class TestFlatOutput(unittest.TestCase):
    def test_plain_strings_parse(self):
        block, error = parse_reframing(json.dumps(FLAT))
        self.assertIsNone(error)
        self.assertEqual({k: block[k] for k in FLAT}, FLAT)

    def test_prose_around_the_object_is_tolerated(self):
        block, error = parse_reframing(f"Sure, here you go:\n```json\n{json.dumps(FLAT)}\n```\nHope that helps!")
        self.assertIsNone(error)
        self.assertEqual(block["system"], "You are Ops Assistant.")

    def test_extra_keys_are_preserved(self):
        block, _ = parse_reframing(json.dumps({**FLAT, "notes": "ignore me"}))
        self.assertEqual(block["notes"], "ignore me")


class TestNestedOutput(unittest.TestCase):
    '''The Hermes-4 shape: objects where the contract asks for strings.'''

    def test_nested_object_is_flattened_in_order(self):
        payload = {
            "context": {
                "background": "BrightStart is a teaching collective.",
                "discussion": "This month's theme is ethics.",
            },
            "system": "You are a discussion assistant.",
            "scenario": {"user_message": "I need a detailed analysis."},
        }
        block, error = parse_reframing(json.dumps(payload))
        self.assertIsNone(error)
        self.assertEqual(
            block["context"],
            "BrightStart is a teaching collective.\n\nThis month's theme is ethics.",
        )
        self.assertEqual(block["scenario"], "I need a detailed analysis.")

    def test_nested_lists_are_flattened(self):
        payload = {**FLAT, "context": ["First para.", "Second para."]}
        block, error = parse_reframing(json.dumps(payload))
        self.assertIsNone(error)
        self.assertEqual(block["context"], "First para.\n\nSecond para.")

    def test_deeply_nested_leaves_are_kept(self):
        payload = {**FLAT, "system": {"role": {"title": "Ops Assistant", "scope": "logistics only"}}}
        block, _ = parse_reframing(json.dumps(payload))
        self.assertEqual(block["system"], "Ops Assistant\n\nlogistics only")

    def test_numbers_survive_flattening(self):
        # a year or headcount inside a context object is content, not noise
        payload = {**FLAT, "context": {"year": 2026, "where": "Lisbon"}}
        block, _ = parse_reframing(json.dumps(payload))
        self.assertEqual(block["context"], "2026\n\nLisbon")

    def test_empty_nesting_is_still_a_failure(self):
        payload = {**FLAT, "context": {"a": "", "b": {}}}
        block, error = parse_reframing(json.dumps(payload))
        self.assertIsNone(block)
        self.assertIn("context", error)


class TestRejection(unittest.TestCase):
    def test_no_json_object(self):
        block, error = parse_reframing("I'm afraid I can't help with that.")
        self.assertIsNone(block)
        self.assertIn("no JSON", error)

    def test_malformed_json(self):
        block, error = parse_reframing('{"context": "a", "system": ')
        self.assertIsNone(block)
        self.assertTrue(error)

    def test_missing_field_names_the_field(self):
        block, error = parse_reframing(json.dumps({"context": "a", "system": "b"}))
        self.assertIsNone(block)
        self.assertIn("scenario", error)

    def test_blank_field_names_the_field(self):
        block, error = parse_reframing(json.dumps({**FLAT, "system": "   "}))
        self.assertIsNone(block)
        self.assertIn("system", error)

    def test_non_object_json(self):
        block, error = parse_reframing('["a", "b"]')
        self.assertIsNone(block)
        self.assertTrue(error)


if __name__ == "__main__":
    unittest.main()
