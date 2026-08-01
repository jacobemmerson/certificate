'''
author: @tae

Risk-cluster tasks — one @task per EU AI Act systemic risk, each evaluating a
filtered union of several benchmarks.

  democracy         democratic process, human rights, historical truth, bias
  persuasion        manipulation, dark patterns, unethical influence
  cyber             cyber and scientific misuse capability
  loss_of_control   scheming and oversight subversion (no data yet)

The datasets are built by `datasets/prepare/cluster/` into
`datasets/public/<risk>.csv`, one canonical schema across every source. This
module does no per-benchmark work: it lifts the schema's columns into sample
metadata and hands scoring to `scorers/cluster.py`, which dispatches on the
sample's own `question_type`. Adding a benchmark is a registry entry in
`datasets/prepare/cluster/sources/<risk>.py` — nothing here changes.

Data preparation (run once before evaluating):
  uv run python3 -m datasets.prepare.cluster.prepare

Run one cluster:
  inspect eval pipeline/stage1_evaluation/evals/clusters.py@cyber --model openai/gpt-4o
'''

import json
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample

from pipeline.stage1_evaluation.evals.common import DATASETS_DIR, csv_samples
from pipeline.stage1_evaluation.scorers.cluster import cluster_scorer
from inspect_ai.solver import generate

RISKS = ["democracy", "persuasion", "cyber", "loss_of_control"]

# Columns the prepare step JSON-encodes on the way out.
_JSON_COLUMNS = ("categories", "scale_map", "choices", "metadata")


def _to_sample(row: dict) -> Sample:
    '''
    One cluster CSV row becomes one Sample, with the whole schema in metadata.

    `target` is carried in metadata as well as on the Sample: the scorer reads
    everything it needs from one place, and stage 2/3 replay the sample through
    scratch states where only the output differs.
    '''
    decoded = {column: json.loads(row[column] or "null") for column in _JSON_COLUMNS}
    native = decoded["metadata"] or {}

    return Sample(
        input=row["query"],
        id=row["sample_id"],
        target=row["target"],
        metadata={
            # native fields first so the canonical schema always wins a clash
            **native,
            "source": row["source"],
            "risk": row["risk"],
            "question_type": row["question_type"],
            "criterion": row["criterion"],
            "rubric": row.get("rubric", ""),
            "categories": decoded["categories"] or [],
            "scale_map": decoded["scale_map"] or {},
            "choices": decoded["choices"] or [],
            "target": row["target"],
            "item_text": row["item_text"],
            "prompt_template": row["prompt_template"],
            "elicitation_family": row["elicitation_family"],
        },
    )


def cluster_dataset(risk: str):
    return csv_samples(f"{risk}.csv", to_sample=_to_sample)


def available(risk: str) -> bool:
    '''Whether this risk has been built yet — loss_of_control has no sources.'''
    return (DATASETS_DIR / f"{risk}.csv").exists()


def _cluster_task(risk: str, grader: str | list[str]) -> Task:
    return Task(
        dataset=cluster_dataset(risk),
        solver=generate(),
        scorer=cluster_scorer(model=grader),
        name=risk,
    )


@task
def democracy(grader: str | list[str] = "openai/gpt-4o"):
    '''Democratic process, human rights, historical truth, political bias.'''
    return _cluster_task("democracy", grader)


@task
def persuasion(grader: str | list[str] = "openai/gpt-4o"):
    '''Manipulation, dark patterns, and unethical influence.'''
    return _cluster_task("persuasion", grader)


@task
def cyber(grader: str | list[str] = "openai/gpt-4o"):
    '''Cyber and scientific misuse capability.'''
    return _cluster_task("cyber", grader)


@task
def loss_of_control(grader: str | list[str] = "openai/gpt-4o"):
    '''Scheming, instrumental convergence, oversight subversion.'''
    return _cluster_task("loss_of_control", grader)


CLUSTER_TASKS = {
    "democracy": democracy,
    "persuasion": persuasion,
    "cyber": cyber,
    "loss_of_control": loss_of_control,
}
