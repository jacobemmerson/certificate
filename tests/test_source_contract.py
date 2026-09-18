import ast
import unittest
from pathlib import Path
from datasets.prepare.cluster.schema import (
    DETECTION, GRADED, MCQ, Row, SchemaError, Source, validate,
)
from datasets.prepare.cluster.prepare import rows_from_frame
from datasets.prepare.cluster.sources import SOURCES
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

    def test_angle_bracket_payload_target_is_allowed(self):
        payload = "<scr<script>ipt>alert(1)</scr<script>ipt>"
        row = graded_row(question_type=MCQ, choices=[payload, "B"], target=payload)
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


class LiftsContract(unittest.TestCase):
    def test_to_sample_carries_contract(self):
        from pipeline.stage1_evaluation.evals.clusters import _to_sample
        row = graded_row(judge_style="classifier", role="diagnostic", pool="p",
                         summary="mean", rewrite=False).to_csv_row()
        md = _to_sample(row).metadata
        self.assertEqual(md["judge_style"], "classifier")
        self.assertEqual(md["role"], "diagnostic")
        self.assertEqual(md["pool"], "p")
        self.assertIs(md["rewrite"], False)


def source_name_lines(source_text: str, names: set[str]) -> list[int]:
    """Lines where `source_text` hard-codes one of `names` as a string literal.

    Parsed rather than scanned line by line: comments never reach the AST, so a
    `#` inside a string literal cannot hide the rest of its line.
    """
    return sorted({
        node.lineno
        for node in ast.walk(ast.parse(source_text))
        if isinstance(node, ast.Constant) and node.value in names
    })


class PipelineIsStatic(unittest.TestCase):
    def test_no_source_names_in_pipeline(self):
        names = {s.name for s in SOURCES}
        offenders = []
        repo_root = Path(__file__).resolve().parent.parent
        # stage4_aggregation is dead pre-cluster code (reads a scores_meta shape
        # nothing writes, names tasks the cluster rename removed; see
        # analysis/pipeline_audit.md B-section item 3) pending removal on approval.
        for path in (repo_root / "pipeline").rglob("*.py"):
            if "stage4_aggregation" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            offenders += [f"{path}:{n}" for n in source_name_lines(text, names)]
        self.assertEqual(offenders, [])

    def test_a_name_is_not_hidden_by_a_hash_in_a_string(self):
        text = 'PREFIXES = {"#": "harmbench"}\n'
        self.assertEqual(source_name_lines(text, {"harmbench"}), [1])

    def test_a_commented_out_name_is_not_an_offender(self):
        self.assertEqual(source_name_lines('# "harmbench"\n', {"harmbench"}), [])


if __name__ == "__main__":
    unittest.main()
