"""Generic direct Inspect entry point for controlled agentic conditions.

Examples:

    inspect eval pipeline/agentic/eval.py@agentic \
      --model openai/gpt-4o-mini \
      -T base_task=social_harm_bench -T condition=c3

    inspect eval pipeline/agentic/eval.py@agentic \
      --model openai/gpt-4o-mini \
      -T base_task=fscale -T condition=c1

One factory backs all 12 standard tasks and C1-C4. Clean C0 intentionally uses
its original stage-1 task entry rather than this solver-replacement entry.
"""

from __future__ import annotations

from pathlib import Path
import sys

from inspect_ai import task

# Inspect loads file entries with this directory at sys.path[0]. Add the repo
# root explicitly because this project intentionally uses ``package = false``.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.agentic import AgenticConfig, Condition  # noqa: E402
from pipeline.registry import init_condition_benchmarks  # noqa: E402
from pipeline.utils.task_transforms import (  # noqa: E402
    AGENTIC_METADATA_KEY,
    task_name,
)

DEFAULT_REVIEW_MODEL = "openrouter/meta-llama/llama-3.1-8b-instruct"


def _configured_condition(
    condition: str,
    *,
    aggregate_output_tokens: int | None,
    final_output_tokens: int | None,
    draft_output_tokens: int | None,
    review_output_tokens: int | None,
    consultation_output_tokens: int | None,
    helper_output_tokens: int | None,
    max_consultation_turns: int | None,
    max_final_turns: int | None,
    max_helper_calls: int | None,
    time_seconds: float | None,
    working_seconds: float | None,
) -> AgenticConfig:
    parsed = Condition(condition)
    if parsed is Condition.C0:
        raise ValueError(
            "Use the original benchmark task directly for clean C0; the generic "
            "agentic entry point supports C1-C4"
        )
    config = AgenticConfig.default(parsed)
    overrides = {
        key: value
        for key, value in {
            "aggregate_output_tokens": aggregate_output_tokens,
            "final_output_tokens": final_output_tokens,
            "draft_output_tokens": draft_output_tokens,
            "review_output_tokens": review_output_tokens,
            "consultation_output_tokens": consultation_output_tokens,
            "helper_output_tokens": helper_output_tokens,
            "max_consultation_turns": max_consultation_turns,
            "max_final_turns": max_final_turns,
            "max_helper_calls": max_helper_calls,
            "time_seconds": time_seconds,
            "working_seconds": working_seconds,
        }.items()
        if value is not None
    }
    return config.with_budget_overrides(**overrides)


@task
def agentic(
    base_task: str,
    condition: str = "c3",
    grader: str = "openai/gpt-4o",
    analyst_model: str = DEFAULT_REVIEW_MODEL,
    critic_model: str = DEFAULT_REVIEW_MODEL,
    llamaguard_model: str = "openrouter/meta-llama/llama-guard-4-12b",
    aggregate_output_tokens: int | None = None,
    final_output_tokens: int | None = None,
    draft_output_tokens: int | None = None,
    review_output_tokens: int | None = None,
    consultation_output_tokens: int | None = None,
    helper_output_tokens: int | None = None,
    max_consultation_turns: int | None = None,
    max_final_turns: int | None = None,
    max_helper_calls: int | None = None,
    time_seconds: float | None = None,
    working_seconds: float | None = None,
):
    """Build one standard benchmark under C1-C4 with explicit finite limits."""
    config = _configured_condition(
        condition,
        aggregate_output_tokens=aggregate_output_tokens,
        final_output_tokens=final_output_tokens,
        draft_output_tokens=draft_output_tokens,
        review_output_tokens=review_output_tokens,
        consultation_output_tokens=consultation_output_tokens,
        helper_output_tokens=helper_output_tokens,
        max_consultation_turns=max_consultation_turns,
        max_final_turns=max_final_turns,
        max_helper_calls=max_helper_calls,
        time_seconds=time_seconds,
        working_seconds=working_seconds,
    )
    benchmarks = init_condition_benchmarks(
        grader,
        config,
        llamaguard_model=llamaguard_model,
        analyst_model=analyst_model,
        critic_model=critic_model,
    )
    matches = [
        candidate
        for entry in benchmarks.values()
        for candidate in entry["tasks"]
        if (
            (candidate.metadata or {}).get(AGENTIC_METADATA_KEY, {}).get("base_task")
            == base_task
            or task_name(candidate) == base_task
        )
    ]
    if len(matches) != 1:
        available = sorted(
            (candidate.metadata or {})
            .get(AGENTIC_METADATA_KEY, {})
            .get("base_task", task_name(candidate))
            for entry in benchmarks.values()
            for candidate in entry["tasks"]
        )
        raise ValueError(
            f"unknown base_task {base_task!r}; choose one of: {', '.join(available)}"
        )
    return matches[0]
