"""Builds a perturbation-augmented Task from any existing `@task`-produced
Task object (from any file in pipeline/stage1_evaluation/evals/, present or future).

The augmented Task's solver array starts with base_task's own solver (the
control condition — the actual generate() call, plus any persona
system_message step), followed by one additional solver per requested,
applicable perturbation family (pipeline/stage2_perturbation/solvers.py), followed by one
labeled scoring step per condition family
(pipeline/stage2_perturbation/scoring.py::scoring_step) — control first (`generate_scoring`),
then `paraphrase_scoring`, `reconsideration_scoring`, etc. Each condition family
and each scoring family shows up as its own labeled step in the sample
transcript, all within one episode, no epochs involved.

The pregenerated families (paraphrase, register, identity_strip, framing)
replay fixed variants loaded here from datasets/generated/<task>/<family>.jsonl
(pipeline/artifacts.py::load_family — produced by generate.py), truncated to
the first `k` variants per sample for the rewrite families. Only
`reconsideration` still builds its prompt live (it needs the target's own
control completion). Every perturbation solver stores its generation(s) into
state.metadata rather than touching the shared state.output; every scoring
step reads those recorded variants back and judges them with the benchmark's
own unchanged judge function(s), so there is no separate condition-label
bookkeeping to keep in sync with the solvers' naming.

The Task's registered scorer list stays one entry per base judge
(pipeline/stage2_perturbation/scoring.py::wrap_scorers): a thin wrapper, named the same as
the base scorer, that reports the control condition's own judgment (so the
benchmark's certification score keeps meaning exactly what it means without
--perturb) plus the per-condition breakdown in its metadata.

This produces one Task per benchmark, not a separate one — the caller
(pipeline/registry.py::apply_stages) keeps the exact same benchmark
name/log path as the unperturbed run, so no separate "_perturb" Task or log
is created.
"""
from __future__ import annotations

from inspect_ai import Task

from pipeline.artifacts import framing_applies, load_family, task_name
from pipeline.stage2_perturbation.adapters import PerturbAdapter, get_adapter
from pipeline.stage2_perturbation.scoring import SCENARIO, scoring_step, wrap_scorers
from pipeline.stage2_perturbation.solvers import (
    framing,
    identity_strip,
    paraphrase,
    reconsideration,
    register,
    scenario,
)

REPLAY_SOLVERS = {
    "paraphrase": paraphrase,
    "register": register,
    "identity_strip": identity_strip,
    "framing": framing,
}


def adapter_for(base_task: Task) -> PerturbAdapter:
    return get_adapter(task_name(base_task))


def truncated(variants_by_id: dict[str, list[dict]], k: int) -> dict[str, list[dict]]:
    """The first k stored variants per sample — --perturb-k/--sim-k now mean
    "use up to k of the frozen variants", so every model gets the same subset.
    """
    return {sample_id: rows[:k] for sample_id, rows in variants_by_id.items()}


def build_perturbed_task(
    base_task: Task,
    families: list[str],
    k: int,
    sim_k: int | None = None,
) -> Task:
    """Return base_task with one solver appended per requested, applicable
    perturbation family, one labeled scoring step per condition family
    (control included), and its scorer list wrapped one-per-base-judge.

    When `sim_k` is set, stage 3's `scenario` replay solver (+ scoring step)
    is appended after the stage-2 families, so a combined --perturb
    --simulate certification is ONE task run and ONE log: the control is
    generated and judged once, and the wrapped scorers report the two stages
    under separate metric pools (lvr/consistency vs.
    lvr_scenario/consistency_scenario — see scoring.py::wrap_scorers).

    Returns base_task unchanged if nothing applies (e.g. only "framing" was
    requested against a benchmark whose elicitation_family has no registered
    framing templates, and sim_k is None).
    """
    name = task_name(base_task)

    solver_chain = [base_task.solver]
    applied: list[str] = []
    for family in ("paraphrase", "register", "identity_strip"):
        if family in families:
            solver_chain.append(REPLAY_SOLVERS[family](truncated(load_family(name, family), k)))
            applied.append(family)
    if "framing" in families and framing_applies(name):
        solver_chain.append(framing(load_family(name, "framing")))
        applied.append("framing")
    if "reconsideration" in families:
        solver_chain.append(reconsideration())
        applied.append("reconsideration")
    if sim_k is not None:
        solver_chain.append(scenario(truncated(load_family(name, SCENARIO), sim_k)))
        applied.append(SCENARIO)

    if not applied:
        return base_task  # no requested family applies to this benchmark

    solver_chain.append(scoring_step("generate", base_task.scorer))
    solver_chain += [scoring_step(family, base_task.scorer) for family in applied]

    return Task(
        dataset=base_task.dataset,
        solver=solver_chain,
        scorer=wrap_scorers(base_task.scorer, applied),
        name=name,
    )
