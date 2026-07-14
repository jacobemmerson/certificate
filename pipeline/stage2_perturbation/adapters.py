"""Per-benchmark adapters for the surface-perturbation module.

A `PerturbAdapter` separates a benchmark item's substantive content (the part
safe to reword/reframe) from the elicitation wrapper around it (fixed
instructions a perturbation must not corrupt, e.g. a strict JSON+scale-option
contract). Perturbation solvers operate on `item_text(state)` and re-render
the full prompt via `render(state, new_text)` — never touching `state.metadata`
or the benchmark's own scorer, so the judgment function stays fixed across
conditions.

`DEFAULT_ADAPTER` treats the whole rendered prompt as reword-able content,
which is correct for any benchmark whose target-facing prompt is free text
graded by an LLM judge/LlamaGuard (the common case — covers socialharmbench,
preserving_historical_truth, llm_human_rights, and democratic_authoritarian_bias's
role_model_bias task today). `ADAPTERS` registers the exceptions: benchmarks where
the target must reply in a rigid, directly machine-parsed format (fscale,
leader_favorability), so only the raw statement/question gets reworded and the
format instructions are re-injected verbatim via the benchmark's own prompt builder.

Adding a new benchmark later needs an ADAPTERS entry only if it has this kind
of rigid, directly-parsed target output contract — otherwise nothing at all.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from inspect_ai.solver import TaskState

from pipeline.stage1_evaluation.evals.democratic_authoritarian_bias import (
    _build_fscale_prompt,
    _build_favscore_prompt,  # leader_favorability's rigid prompt builder
)

ElicitationFamily = str  # "compliance" | "judgment" | "opinion" | "generic"


@dataclass
class PerturbAdapter:
    item_text: Callable[[TaskState], str]
    render: Callable[[TaskState, str], str]
    elicitation_family: ElicitationFamily = "generic"


def _default_item_text(state: TaskState) -> str:
    return state.input_text


def _default_render(state: TaskState, new_text: str) -> str:
    return new_text


DEFAULT_ADAPTER = PerturbAdapter(
    item_text=_default_item_text,
    render=_default_render,
    elicitation_family="generic",
)


def _fscale_item_text(state: TaskState) -> str:
    return state.metadata.get("statement", state.input_text)


def _fscale_render(state: TaskState, new_text: str) -> str:
    return _build_fscale_prompt(new_text)


def _leader_favorability_item_text(state: TaskState) -> str:
    return state.metadata.get("question", state.input_text)


def _leader_favorability_render(state: TaskState, new_text: str) -> str:
    return _build_favscore_prompt(
        state.metadata.get("leader_name", ""),
        new_text,
        state.metadata.get("output_format", "json_approve_disapprove"),
    )


# Keyed by the original @task function's registry name — see
# pipeline/stage2_perturbation/build.py for how this is looked up via
# inspect_ai._util.registry.registry_info(task_instance).name.
ADAPTERS: dict[str, PerturbAdapter] = {
    "fscale": PerturbAdapter(
        item_text=_fscale_item_text,
        render=_fscale_render,
        elicitation_family="opinion",
    ),
    "leader_favorability": PerturbAdapter(
        item_text=_leader_favorability_item_text,
        render=_leader_favorability_render,
        elicitation_family="opinion",
    ),
    "udhr": PerturbAdapter(item_text=_default_item_text, render=_default_render, elicitation_family="judgment"),
    "udhr_individual": PerturbAdapter(item_text=_default_item_text, render=_default_render, elicitation_family="judgment"),
    "udhr_government": PerturbAdapter(item_text=_default_item_text, render=_default_render, elicitation_family="judgment"),
    "echr": PerturbAdapter(item_text=_default_item_text, render=_default_render, elicitation_family="judgment"),
    "echr_individual": PerturbAdapter(item_text=_default_item_text, render=_default_render, elicitation_family="judgment"),
    "echr_government": PerturbAdapter(item_text=_default_item_text, render=_default_render, elicitation_family="judgment"),
    "social_harm_bench": PerturbAdapter(item_text=_default_item_text, render=_default_render, elicitation_family="compliance"),
    "history_no_push": PerturbAdapter(item_text=_default_item_text, render=_default_render, elicitation_family="compliance"),
    "history_explicit_push": PerturbAdapter(item_text=_default_item_text, render=_default_render, elicitation_family="compliance"),
    # role_model_bias intentionally unregistered: open-ended elicitation doesn't
    # fit any of the fixed framing families, so it falls back to DEFAULT_ADAPTER
    # (elicitation_family="generic") and simply gets skipped by the framing
    # family (see pipeline/stage2_perturbation/framing.py).
}


def get_adapter(task_name: str) -> PerturbAdapter:
    return ADAPTERS.get(task_name, DEFAULT_ADAPTER)
