"""
author: @tae

Utilities for grader and model loading
TODO: rename file to something more fitting since this is general utilties
"""

import json
from pathlib import Path

from inspect_ai._util.registry import registry_create
from inspect_ai.log import EvalLog
from inspect_ai.scorer import SampleScore

from pipeline.utils.scoring import is_failing
from pipeline.utils.task_transforms import AGENTIC_METADATA_KEY

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def load_graders(path: str | Path | None = None) -> list[str]:
    """Load grader model names from a text file (one per line, # comments ignored)."""
    if path is None:
        path = REPO_ROOT / "GRADERS.md"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Graders file not found: {path}")
    models = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not models:
        raise ValueError(f"No grader models found in {path}")
    return models


def load_models_with_check(model_id: str | None = None) -> tuple[list[dict], int]:
    """
    Return the models list and the index of `model_id` within it (-1 if not
    found, or if no model_id is given).
    """
    path = REPO_ROOT / "models" / "models.json"
    if not path.exists():
        raise FileNotFoundError(f"Model results file not found: {path}")

    try:
        with open(path, "r") as f:
            models = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        models = []

    if model_id:
        for i, m in enumerate(models):
            if m["id"] == model_id:
                return models, i

    return models, -1


def _canonical_task_name(task: EvalLog) -> str:
    """Resolve transformed task names back to their benchmark identity."""
    metadata = getattr(task.eval, "metadata", None) or {}
    protocol = metadata.get(AGENTIC_METADATA_KEY) or {}
    base_task = protocol.get("base_task") if isinstance(protocol, dict) else None
    return (
        base_task if isinstance(base_task, str) and base_task else str(task.eval.task)
    )


def _valid_process_metric(task: EvalLog, result_score) -> float | None:
    """Recompute the primary metric using only protocol-valid samples."""
    valid_scores: list[SampleScore] = []
    samples = getattr(task, "samples", None) or []
    for sample in samples:
        audit = (getattr(sample, "metadata", None) or {}).get("agentic_protocol_audit")
        if not isinstance(audit, dict) or audit.get("valid") is not True:
            continue
        sample_score = (getattr(sample, "scores", None) or {}).get(result_score.name)
        if sample_score is None:
            continue
        valid_scores.append(
            SampleScore(
                score=sample_score,
                sample_id=getattr(sample, "id", None),
                sample_metadata=getattr(sample, "metadata", None),
                scorer=result_score.scorer,
            )
        )

    if not valid_scores:
        return None
    metric_record = next(iter(result_score.metrics.values()))
    metric = registry_create("metric", metric_record.name, **metric_record.params)
    return float(metric(valid_scores))


def aggregate_score(
    task_results: list[EvalLog],
    *,
    valid_process_only: bool = False,
) -> tuple[float | None, dict[str, float | None]]:
    """Aggregate primary metrics, optionally excluding protocol-invalid samples."""
    by_task: dict[str, float | None] = {}
    average_values: list[float] = []
    for task in task_results:
        canonical_task = _canonical_task_name(task)
        try:
            result_score = task.results.scores[0]
            if valid_process_only:
                value = _valid_process_metric(task, result_score)
            else:
                metric_record = next(iter(result_score.metrics.values()))
                value = float(metric_record.value)

            # SocialHarmBench accuracy is a fraction; all reports use 0-100.
            if value is not None and canonical_task == "social_harm_bench":
                value *= 100
            if value is not None:
                average_values.append(value)
        except Exception as error:
            print(f"[ERROR] occurred on benchmark for task: {task}: {error}")
            value = None

        by_task[canonical_task] = value

    reported = sum(average_values) / len(average_values) if average_values else None
    return reported, by_task


def consistency_rate(
    task_results: list[EvalLog],
    families: set[str] | None = None,
) -> dict[str, dict]:
    """
    Per-family consistency rate and LVR for a perturbation-auditing run.

    ``families`` can split stage 2 and stage 3 from one composed replay log;
    the shared control is retained in either report.

    pipeline/utils/scoring.py::wrap_scorers builds one Score entry per base
    judge whose metadata carries a "conditions" breakdown — one entry per
    condition (control + every perturbation variant recorded by
    pipeline/stage2_perturbation/solvers.py), each with its "family", "value", and a
    "stable" flag (does its value match the control's). This scans every
    sample's scores and reports:

    - "consistency_rate": per (non-control) family, the fraction of conditions
      whose "stable" flag is true — "how often the model's output is stable
      across the transform" — not a pass/fail accuracy metric.
    - "lvr": per family (control included, as the unperturbed baseline), the
      fraction of conditions judged harmful/failing per each scorer's
      polarity (pipeline/utils/scoring.py::is_failing/SCORER_POLARITY).
    - "by_task": the same per-family rates broken down per task and per
      scorer. The log's own results panel only carries the pooled
      `lvr_control`/`lvr`/`consistency` metrics (kept compact on purpose) —
      this is where the full per-family detail is preserved.
    """

    def _tally(counters: dict, family: str, failing: bool, stable) -> None:
        c = counters.setdefault(
            family, {"total": 0, "harmful": 0, "stable_total": 0, "stable": 0}
        )
        c["total"] += 1
        c["harmful"] += failing
        if family != "control":
            c["stable_total"] += 1
            c["stable"] += bool(stable)

    def _rates(counters: dict) -> dict:
        return {
            "consistency_rate": {
                family: 100.0 * c["stable"] / c["stable_total"]
                for family, c in counters.items()
                if c["stable_total"]
            },
            "lvr": {
                family: 100.0 * c["harmful"] / c["total"]
                for family, c in counters.items()
            },
        }

    overall: dict = {}
    per_task_scorer: dict = {}

    for task in task_results:
        for s in task.samples or []:
            if not s.scores:
                continue
            for scorer_name, score in s.scores.items():
                conditions = (score.metadata or {}).get("conditions") or {}
                for condition in conditions.values():
                    family = condition.get("family")
                    if not family:
                        continue
                    if (
                        families is not None
                        and family != "control"
                        and family not in families
                    ):
                        continue
                    failing = is_failing(scorer_name, condition.get("value"))
                    stable = condition.get("stable")
                    _tally(overall, family, failing, stable)
                    _tally(
                        per_task_scorer.setdefault(
                            _canonical_task_name(task), {}
                        ).setdefault(scorer_name, {}),
                        family,
                        failing,
                        stable,
                    )

    return {
        **_rates(overall),
        "by_task": {
            task_name: {
                scorer_name: _rates(counters)
                for scorer_name, counters in scorers.items()
            }
            for task_name, scorers in per_task_scorer.items()
        },
    }
