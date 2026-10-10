import json
import tempfile
import unittest
from pathlib import Path

from datasets.prepare.cluster import annotations
from datasets.prepare.cluster.leaves import Leaf, leaf_by_label
from datasets.prepare.cluster.sources import SOURCES

LEAVES = {
    "a": Leaf(id="a", title="A", cop_ref="x", legal_text="a", label="label a"),
    "b": Leaf(id="b", title="B", cop_ref="x", legal_text="b", label="label b"),
}


def task(task_id, *annotations_, benchmark="Bench"):
    return {
        "id": task_id, "data": {"benchmark": benchmark},
        "annotations": [
            {"was_cancelled": cancelled, "result": [
                {"from_name": "risk_categories", "value": {"choices": list(labels)}},
                {"from_name": "confidence", "value": {"number": confidence}},
            ]}
            for labels, confidence, cancelled in annotations_
        ],
    }


class TaskRelevanceTest(unittest.TestCase):
    def test_single_annotator_is_their_confidence(self):
        self.assertEqual(annotations.task_relevance(task(1, (["label a"], 0.8, False)), leaf_by_label(LEAVES)),
                         {"a": 0.8})

    def test_union_weighted_by_agreement(self):
        t = task(1, (["label a", "label b"], 0.9, False), (["label a"], 0.7, False))
        self.assertEqual(annotations.task_relevance(t, leaf_by_label(LEAVES)), {"a": 0.8, "b": 0.45})

    def test_cancelled_annotations_are_ignored(self):
        t = task(1, (["label a"], 0.9, False), (["label b"], 0.1, True))
        self.assertEqual(annotations.task_relevance(t, leaf_by_label(LEAVES)), {"a": 0.9})

    def test_unknown_label_raises(self):
        with self.assertRaises(ValueError):
            annotations.task_relevance(task(1, (["nope"], 0.5, False)), leaf_by_label(LEAVES))


class LoadAnnotationsTest(unittest.TestCase):
    def test_joins_tasks_to_manifest_by_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.toml"
            manifest.write_text('[[benchmark]]\nname = "bench"\nid = "7"\n', encoding="utf-8")
            export = Path(tmp) / "annotations.json"
            export.write_text(json.dumps([task(7, (["label a"], 0.5, False))]), encoding="utf-8")
            self.assertEqual(annotations.load_annotations(export, manifest, LEAVES), {"bench": {"a": 0.5}})
            export.write_text(json.dumps([task(8, (["label a"], 0.5, False))]), encoding="utf-8")
            with self.assertRaises(ValueError):
                annotations.load_annotations(export, manifest, LEAVES)

    def test_missing_file_is_empty(self):
        self.assertEqual(annotations.load_annotations(Path("/nonexistent/annotations.json")), {})

    def test_manifest_name_from_source_path(self):
        self.assertEqual(annotations.manifest_name("raw/bbq/data/*.jsonl"), "bbq")
        with self.assertRaises(ValueError):
            annotations.manifest_name("public/x.csv")


@unittest.skipUnless(annotations.ANNOTATIONS_PATH.exists(), "annotations.json not in checkout")
class CheckedInExportTest(unittest.TestCase):
    def test_every_task_loads_and_every_label_resolves(self):
        loaded = annotations.load_annotations()
        self.assertGreater(len(loaded), 50)
        for name, leaves in loaded.items():
            for leaf_id, r in leaves.items():
                self.assertTrue(0 < r <= 1, (name, leaf_id, r))

    def test_registered_sources_are_annotated_or_hand_assigned(self):
        loaded = annotations.load_annotations()
        for source in SOURCES:
            with self.subTest(source=source.name):
                self.assertTrue(annotations.manifest_name(source.path) in loaded or source.leaf,
                                f"{source.name}: no annotation and no leaf")
