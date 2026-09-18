import unittest
from datasets.prepare.cluster.schema import (
    DETECTION, GRADED, MCQ, Row, SchemaError, Source, validate,
)
from datasets.prepare.cluster.prepare import rows_from_frame
import pandas as pd


def graded_row(**over):
    base = dict(sample_id="x:1", source="x", risk="cbrn", question_type=GRADED,
                query="Do the thing?", criterion="did it", categories=["yes", "no"],
                scale_map={"yes": 0.0, "no": 1.0}, item_text="Do the thing?")
    base.update(over)
    return Row(**base)


class ContractDefaults(unittest.TestCase):
    def test_graded_defaults(self):
        row = graded_row()
        self.assertEqual(row.judge_style, "reasoned")
        self.assertEqual(row.role, "pooled")
        self.assertEqual(row.pool, "")
        self.assertEqual(row.summary, "mean")
        self.assertTrue(row.rewrite)

    def test_detection_defaults_to_no_rewrite(self):
        src = Source(name="d", risk="cyber", question_type=DETECTION, path="-",
                     detector="refusal_regex", criterion=lambda r: "refused")
        self.assertFalse(src.rewrite_default())

    def test_csv_columns(self):
        cols = graded_row().to_csv_row()
        for c in ("judge_style", "role", "pool", "summary", "rewrite"):
            self.assertIn(c, cols)


class ContractValidation(unittest.TestCase):
    def test_unknown_judge_style_fails(self):
        with self.assertRaises(SchemaError):
            validate(graded_row(judge_style="vibes"))

    def test_must_survive_must_be_in_query(self):
        with self.assertRaises(SchemaError):
            validate(graded_row(metadata={"must_survive": ["[ACCEPT]"]}))

    def test_placeholder_target_fails(self):
        row = graded_row(question_type=MCQ, choices=["<answer choice>", "B"],
                         target="<answer choice>")
        with self.assertRaises(SchemaError):
            validate(row)


class AskField(unittest.TestCase):
    ASK = "Given this context, is this action acceptable?"

    def make_source(self, **over):
        base = dict(name="hr", risk="manipulation", question_type=GRADED, path="-",
                    reader="csv", query="q", id_col="id", criterion=lambda r: "endorses",
                    categories=["yes", "no"], scale_map={"yes": 0.0, "no": 1.0},
                    ask=self.ASK)
        base.update(over)
        return Source(**base)

    def test_ask_moves_into_template(self):
        frame = pd.DataFrame([{"q": f"Scenario text.\n\n{self.ASK}", "id": "7"}])
        src = self.make_source()
        rows = rows_from_frame(src, frame)
        row = rows[0]
        self.assertEqual(row.item_text, "Scenario text.")
        self.assertIn("is this action acceptable?", row.prompt_template)
        self.assertEqual(row.prompt_template.replace("{{ITEM}}", row.item_text), row.query)

    def test_ask_already_in_query_is_stripped_and_reconstructs(self):
        frame = pd.DataFrame([{"q": f"Scenario text.\n\n{self.ASK}", "id": "7"}])
        src = self.make_source()
        row = rows_from_frame(src, frame)[0]
        self.assertEqual(row.item_text, "Scenario text.")
        self.assertIn(self.ASK, row.prompt_template)
        self.assertEqual(row.prompt_template.replace("{{ITEM}}", row.item_text), row.query)

    def test_ask_not_at_query_tail_raises(self):
        frame = pd.DataFrame([{"q": "Scenario text with no ask appended.", "id": "7"}])
        src = self.make_source()
        with self.assertRaises(SchemaError):
            rows_from_frame(src, frame)


if __name__ == "__main__":
    unittest.main()
