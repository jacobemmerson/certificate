"""Safely adapt existing Certificate tasks to C0-C4 protocols."""

from __future__ import annotations

from copy import copy

from inspect_ai import Task, task_with

from pipeline.agentic.config import AgenticConfig, Condition
from pipeline.agentic.solvers import protocol_solver
from pipeline.utils.task_transforms import (
    AGENTIC_METADATA_KEY,
    TaskTransform,
    task_name,
    validate_transform_base,
)


def with_role_models(
    base_task: Task,
    config: AgenticConfig,
    *,
    analyst_model: str | None,
    critic_model: str | None,
) -> Task:
    """Copy a task and apply shared analyst/critic role resolution."""
    staged = copy(base_task)
    staged.model_roles = dict(base_task.model_roles or {})
    if analyst_model is not None:
        staged.model_roles.setdefault(config.roles.analyst, analyst_model)
    if critic_model is not None:
        staged.model_roles.setdefault(config.roles.critic, critic_model)
    return staged


def make_agentic_task(base_task: Task, config: AgenticConfig) -> Task:
    """Return a non-mutating condition adaptation of ``base_task``.

    C0 is deliberately the untouched benchmark task. C1-C4 shallow-copy the
    task, independently copy mutable containers, and use public ``task_with``
    overrides. Dataset, setup, cleanup, scorer objects/order, metrics,
    lifecycle hooks, model settings, sandbox, and benchmark limits remain
    those of the base task.
    """
    if config.condition is Condition.C0:
        return base_task

    validate_transform_base(base_task, TaskTransform.AGENTIC)
    original_name = task_name(base_task)
    config = config.for_benchmark(original_name)
    adapted = copy(base_task)
    metadata = dict(base_task.metadata or {})
    metadata[AGENTIC_METADATA_KEY] = config.metadata(original_name)
    roles = dict(base_task.model_roles) if base_task.model_roles is not None else None
    tags = [*(base_task.tags or []), "agentic", config.condition.value]

    return task_with(
        adapted,
        solver=protocol_solver(config),
        model_roles=roles,
        metadata=metadata,
        tags=tags,
        name=f"{original_name}{config.canonical_suffix}",
    )
