'''
author: @tae

Risk-cluster tasks — one @task per EU AI Act systemic risk, each evaluating a
filtered union of several benchmarks.

  cbrn              chemical, biological, radiological and nuclear misuse
  cyber             offensive cyber capability
  loss_of_control   scheming and oversight subversion
  manipulation      dark patterns, unethical influence, political manipulation

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
from inspect_ai.model import ChatMessageSystem, ChatMessageUser

from pipeline.stage1_evaluation.evals.common import DATASETS_DIR, csv_samples
from pipeline.stage1_evaluation.scorers.cluster import cluster_scorer
from inspect_ai.solver import generate

RISKS = ["cbrn", "cyber", "loss_of_control", "manipulation"]

# Columns the prepare step JSON-encodes on the way out.
_JSON_COLUMNS = (
    "categories", "scale_map", "choices", "metadata",
    "fallback_categories", "fallback_scale_map",
)


def _to_sample(row: dict) -> Sample:
    '''
    One cluster CSV row becomes one Sample, with the whole schema in metadata.

    `target` is carried in metadata as well as on the Sample: the scorer reads
    everything it needs from one place, and stage 2/3 replay the sample through
    scratch states where only the output differs.
    '''
    decoded = {column: json.loads(row[column] or "null") for column in _JSON_COLUMNS}
    native = decoded["metadata"] or {}

    # A row with a system_prompt is sent as two turns. Sources that steer the
    # model deliberately need this — human_rights runs each scenario under a
    # neutral, an individual-rights and a government-authority persona, and the
    # spread between those arms is what the benchmark measures.
    system_prompt = row.get("system_prompt") or ""
    prompt = (
        [ChatMessageSystem(content=system_prompt), ChatMessageUser(content=row["query"])]
        if system_prompt else row["query"]
    )

    return Sample(
        input=prompt,
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
            "detector": row.get("detector", ""),
            "system_prompt": system_prompt,
            "categories": decoded["categories"] or [],
            "scale_map": decoded["scale_map"] or {},
            "choices": decoded["choices"] or [],
            # What the judge picks from when this row's own scorer has no
            # contract left to read — see scorers/cluster.py::_judged_fallback.
            "fallback_categories": decoded["fallback_categories"] or [],
            "fallback_scale_map": decoded["fallback_scale_map"] or {},
            "target": row["target"],
            "item_text": row["item_text"],
            "prompt_template": row["prompt_template"],
            "elicitation_family": row["elicitation_family"],
        },
    )


def cluster_dataset(risk: str):
    return csv_samples(f"{risk}.csv", to_sample=_to_sample)


def available(risk: str) -> bool:
    '''Whether this risk's cluster dataset has been built yet.'''
    return (DATASETS_DIR / f"{risk}.csv").exists()


def _cluster_task(risk: str, grader: str | list[str]) -> Task:
    return Task(
        dataset=cluster_dataset(risk),
        solver=generate(),
        scorer=cluster_scorer(model=grader),
        name=risk,
    )


@task
def cbrn(grader: str | list[str] = "openai/gpt-4o"):
    '''Chemical, biological, radiological and nuclear misuse.'''
    return _cluster_task("cbrn", grader)


@task
def cyber(grader: str | list[str] = "openai/gpt-4o"):
    '''Offensive cyber capability, and refusal of benign security work.'''
    return _cluster_task("cyber", grader)


@task
def loss_of_control(grader: str | list[str] = "openai/gpt-4o"):
    '''Scheming, instrumental convergence, oversight subversion.'''
    return _cluster_task("loss_of_control", grader)


@task
def manipulation(grader: str | list[str] = "openai/gpt-4o"):
    '''Dark patterns, unethical influence, and political manipulation.'''
    return _cluster_task("manipulation", grader)


CLUSTER_TASKS = {
    "cbrn": cbrn,
    "cyber": cyber,
    "loss_of_control": loss_of_control,
    "manipulation": manipulation,
}
