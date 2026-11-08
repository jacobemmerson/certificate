'''
author: @tae

Leaf task — one parameterised @task over the CoP taxonomy leaves, each
evaluating a filtered union of the benchmarks annotated with that leaf; see
LEAVES in `datasets/prepare/cluster/sources/`.

The datasets are built by `datasets/prepare/cluster/` into
`datasets/public/<leaf>.csv`, one canonical schema across every source. This
module does no per-benchmark work: it lifts the schema's columns into sample
metadata and hands scoring to `scorers/cluster.py`, which dispatches on the
sample's own `question_type`. Adding a benchmark is a registry entry in
`datasets/prepare/cluster/sources/` — nothing here changes.

Data preparation (run once before evaluating):
  uv run python3 -m datasets.prepare.cluster.prepare

Run one leaf:
  inspect eval pipeline/stage1_evaluation/evals/clusters.py@cluster -T leaf=cyber_operational_use --model openai/gpt-4o
'''

import json

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser

from pipeline.stage1_evaluation.evals.common import DATASETS_DIR, csv_samples
from pipeline.stage1_evaluation.screen import attribution_screen
from pipeline.stage1_evaluation.scorers.cluster import cluster_scorer
from inspect_ai.solver import chain, generate

from datasets.prepare.cluster.sources import LEAVES

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
            "leaf": row["leaf"],
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
            "judge_style": row.get("judge_style") or "reasoned",
            "role": row.get("role") or "pooled",
            "pool": row.get("pool") or "",
            "summary": row.get("summary") or "mean",
            # The single stage-2/3 applicability gate (pipeline/utils/replay.py).
            # None for a CSV built before the column existed: every family applies.
            "families": json.loads(row["families"]) if row.get("families") else None,
        },
    )


def cluster_dataset(leaf: str):
    return csv_samples(f"{leaf}.csv", to_sample=_to_sample)


def available(leaf: str) -> bool:
    '''Whether this leaf's dataset has been built yet.'''
    return (DATASETS_DIR / f"{leaf}.csv").exists()


def _cluster_task(leaf: str, grader: str | list[str]) -> Task:
    return Task(
        dataset=cluster_dataset(leaf),
        # A source whose benchmark screens on a prerequisite gets that screen
        # first; every other row passes straight through (screen.py).
        solver=chain(attribution_screen(), generate()),
        scorer=cluster_scorer(model=grader),
        name=leaf,
    )


@task
def cluster(leaf: str, grader: str | list[str] = "openai/gpt-4o") -> Task:
    '''One leaf dataset; `leaf` is any entry of LEAVES.'''
    if leaf not in LEAVES:
        raise ValueError(f"unknown leaf {leaf!r}; expected one of {LEAVES}")
    return _cluster_task(leaf, grader)
