"""Builds a perturbation-augmented Task from any existing `@task`-produced
Task object (from any file in pipeline/stage1_evaluation/evals/, present or future).

The augmented Task's solver array starts with base_task's own solver (the
control condition — the actual generate() call, plus any persona
system_message step), followed by one additional solver per requested,
applicable perturbation family (pipeline/stage2_perturbation/solvers.py), followed by one
labeled scoring step per condition family
(pipeline/stage2_perturbation/scoring.py::scoring_step) — control first (`generate_scoring`),
then `paraphrase_scoring`, `reconsideration_scoring`, etc. Each generation family
and each scoring family shows up as its own labeled step in the sample
transcript, all within one episode, no epochs involved. Every perturbation
solver stores its own generation(s) into state.metadata rather than touching
the shared state.output; every scoring step reads those recorded variants
back and judges them with the benchmark's own unchanged judge function(s),
so there is no separate condition-label bookkeeping to keep in sync with the
solvers' naming.

The Task's registered scorer list stays one entry per base judge
(pipeline/stage2_perturbation/scoring.py::wrap_scorers): a thin wrapper, named the same as
the base scorer, that reports the control condition's own judgment (so the
benchmark's certification score keeps meaning exactly what it means without
--perturb) plus the per-condition breakdown in its metadata.

This produces one Task per benchmark, not a separate one — the caller
(pipeline/registry.py::apply_perturbations) keeps the exact same benchmark
name/log path as the unperturbed run, so no separate "_perturb" Task or log
is created.
"""
from __future__ import annotations

from inspect_ai import Task
from inspect_ai._util.registry import registry_info

from pipeline.stage2_perturbation.adapters import PerturbAdapter, get_adapter
from pipeline.stage2_perturbation.framing import FRAMING_TEMPLATES
from pipeline.stage2_perturbation.scoring import scoring_step, wrap_scorers
from pipeline.stage2_perturbation.solvers import (
    framing,
    identity_strip,
    paraphrase,
    reconsideration,
    register,
)


def task_name(base_task: Task) -> str:
    """Recover the original @task function's registry name (e.g. "fscale")."""
    return registry_info(base_task).name


def adapter_for(base_task: Task) -> PerturbAdapter:
    return get_adapter(task_name(base_task))


def build_perturbed_task(
    base_task: Task,
    families: list[str],
    rewrite_model: str,
    k: int,
) -> Task:
    """Return base_task with one solver appended per requested, applicable
    perturbation family, one labeled scoring step per condition family
    (control included), and its scorer list wrapped one-per-base-judge.
    Returns base_task unchanged if no requested family applies (e.g. only
    "framing" was requested against a benchmark whose elicitation_family has
    no registered framing templates).
    """
    adapter = adapter_for(base_task)
    has_framing = "framing" in families and bool(FRAMING_TEMPLATES.get(adapter.elicitation_family))

    solver_chain = [base_task.solver]
    applied: list[str] = []
    if "paraphrase" in families:
        solver_chain.append(paraphrase(rewrite_model, k, adapter))
        applied.append("paraphrase")
    if "register" in families:
        solver_chain.append(register(rewrite_model, k, adapter))
        applied.append("register")
    if "identity_strip" in families:
        solver_chain.append(identity_strip(rewrite_model, k, adapter))
        applied.append("identity_strip")
    if has_framing:
        solver_chain.append(framing(adapter))
        applied.append("framing")
    if "reconsideration" in families:
        solver_chain.append(reconsideration())
        applied.append("reconsideration")

    if not applied:
        return base_task  # no requested family applies to this benchmark

    solver_chain.append(scoring_step("generate", base_task.scorer))
    solver_chain += [scoring_step(family, base_task.scorer) for family in applied]

    return Task(
        dataset=base_task.dataset,
        solver=solver_chain,
        scorer=wrap_scorers(base_task.scorer),
        name=task_name(base_task),
    )
