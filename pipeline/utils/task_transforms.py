"""Canonical task identity and mutually exclusive transform guards."""

from __future__ import annotations

from enum import Enum

from inspect_ai import Task
from inspect_ai._util.registry import registry_info


class TaskTransform(str, Enum):
    """Task-level condition layers supported by the current pipeline."""

    AGENTIC = "agentic"
    PERTURBATION = "perturbation"
    SIMULATION = "simulation"


AGENTIC_METADATA_KEY = "agentic_protocol"
PERTURBATION_METADATA_KEY = "certificate_perturbation"
SIMULATION_METADATA_KEY = "certificate_simulation"

_TRANSFORM_KEYS = {
    TaskTransform.AGENTIC: AGENTIC_METADATA_KEY,
    TaskTransform.PERTURBATION: PERTURBATION_METADATA_KEY,
    TaskTransform.SIMULATION: SIMULATION_METADATA_KEY,
}


def task_name(task: Task) -> str:
    """Return the stable registered identity of a task or filtered task view."""
    protocol = (task.metadata or {}).get(AGENTIC_METADATA_KEY) or {}
    canonical = protocol.get("base_task") if isinstance(protocol, dict) else None
    if isinstance(canonical, str) and canonical:
        return canonical
    try:
        return registry_info(task).name
    except ValueError:
        return task.name or "task"


def applied_transforms(task: Task) -> tuple[TaskTransform, ...]:
    """Return every condition marker present on ``task``."""
    metadata = task.metadata or {}
    return tuple(
        transform
        for transform, key in _TRANSFORM_KEYS.items()
        if metadata.get(key) is not None
    )


def validate_transform_base(task: Task, requested: TaskTransform) -> None:
    """Reject repeated or cross-layer transformation at every public builder."""
    existing = applied_transforms(task)
    if not existing:
        return
    if requested in existing:
        raise ValueError(
            f"a {requested.value} task cannot be transformed by {requested.value} again"
        )
    labels = ", ".join(transform.value for transform in existing)
    raise ValueError(
        f"{requested.value} cannot wrap a task already transformed by {labels}"
    )
