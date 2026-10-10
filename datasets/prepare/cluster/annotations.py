'''
Annotator leaf assignments (annotations.json, a Label Studio export).

Each task is one benchmark; each of its annotations carries the leaves the
annotator chose (`risk_categories`, multi-label) and a confidence in [0, 1].
A benchmark's relevance to a leaf is the mean confidence of the annotators who
chose it times the share of the task's annotators who chose it, so a label
one of two annotators picked at 0.8 scores 0.4 and a label both picked at
0.9 and 0.7 scores 0.8. Tasks join datasets/raw/manifest.toml by `id`; a
Source joins the manifest by the first directory of its `path` (raw/<name>/).
'''
from __future__ import annotations

import json
from pathlib import Path

from .leaves import Leaf, leaf_by_label, load_leaves
from .manifest import MANIFEST_PATH, REPO_ROOT, load_manifest

ANNOTATIONS_PATH = REPO_ROOT / "annotations.json"


def manifest_name(source_path: str) -> str:
    '''`raw/<name>/...` -> `<name>`.'''
    head, _, rest = source_path.partition("/")
    if head != "raw" or not rest:
        raise ValueError(f"source path {source_path!r} is not under raw/<name>/")
    return rest.split("/", 1)[0]


def task_relevance(task: dict, by_label: dict[str, Leaf]) -> dict[str, float]:
    '''{leaf id: relevance} for one task, over its non-cancelled annotations.'''
    annotations = [a for a in task["annotations"] if not a.get("was_cancelled")]
    if not annotations:
        return {}
    chosen: dict[str, list[float]] = {}
    for annotation in annotations:
        fields = {r["from_name"]: r["value"] for r in annotation["result"]}
        confidence = float(fields["confidence"]["number"])
        for label in fields.get("risk_categories", {}).get("choices", []):
            if label not in by_label:
                raise ValueError(f"task {task['id']} ({task['data'].get('benchmark')}): "
                                 f"label {label!r} matches no leaf in leaves.toml")
            chosen.setdefault(by_label[label].id, []).append(confidence)
    return {
        leaf_id: round(sum(confidences) / len(confidences) * len(confidences) / len(annotations), 6)
        for leaf_id, confidences in chosen.items()
    }


def load_annotations(
    path: Path = ANNOTATIONS_PATH, manifest_path: Path = MANIFEST_PATH,
    leaves: dict[str, Leaf] | None = None,
) -> dict[str, dict[str, float]]:
    '''{manifest name: {leaf id: relevance}} for every annotated task; absent when the file is missing.'''
    if not path.exists():
        return {}
    by_label = leaf_by_label(leaves if leaves is not None else load_leaves())
    names = {entry["id"]: entry["name"] for entry in load_manifest(manifest_path) if entry.get("id")}
    with open(path, encoding="utf-8") as handle:
        tasks = json.load(handle)
    out: dict[str, dict[str, float]] = {}
    for task in tasks:
        name = names.get(str(task["id"]))
        if name is None:
            raise ValueError(f"task {task['id']} ({task['data'].get('benchmark')}) has no manifest id")
        out[name] = task_relevance(task, by_label)
    return out
