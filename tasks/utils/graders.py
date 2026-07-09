'''
author: @tae

Utilities for grader and model loading
TODO: rename file to something more fitting since this is general utilties
'''

from inspect_ai.log import EvalLog
from pathlib import Path
import json

from tasks.perturb.scoring import is_failing

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

def load_models_with_check(model_id: str = None) -> tuple[list[dict], int]:
    '''
    Return a JSON array of the models and index where the model lies if model_id is given; returns index = -1 if model not found
    '''
    
    model_path = REPO_ROOT / "models" / "models.json"
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(f"Model results file not found: {path}")
    
    try:
        with open('models/models.json', 'r') as f:
            models = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        models = []

    if not model_id:
        return models, -1
    
    found = False
    for i, m in enumerate(models):
        if m['id'] == model_id:
            # Update file
            found = True
            break

    return models, i if found else -1


def aggregate_score(task_results: list[EvalLog]) -> tuple[float, dict]:

    scores = {
        'reported': -1,
        'by_task': {}
    }
    avg_score = [] # average score TODO: allow for weighted averages
    for task in task_results:

        score = task.results.scores[0] # primary metric must go first, TODO: Add support for multiple scores / score selection / score reducers

        try:
            value = next(iter(score.metrics.values())) # assumes first metric is the primary/reported metric. Other metrics reported in logs and be accessed at a later date.
            '''
            if 'custom' in score.metrics:
                value = score.metrics['custom'] # for custom metrics, like the democratic authoritarian bias score

            elif 'accuracy' in score.metrics:
                value = score.metrics['accuracy'] 

            elif 'mean' in score.metrics:
                value = score.metrics['mean']
            '''
            
            value = value.value

            # accuracy() returns fraction correct (safe); scale to 0-100
            if str(task.eval.task) in ('social_harm_bench', 'social_harm_bench_adversarial'):
                value = value * 100

            avg_score.append(value) # only keep valid values for the average

        except Exception as e:
            print(f"[ERROR] occured on benchmark for task: {task}: {e}")
            value = -1

        scores['by_task'][task.eval.task] = value

    scores['reported']= sum(avg_score) / len(avg_score)

    return scores['reported'], scores['by_task']


def alignment_rate(task_results: list[EvalLog]) -> dict[str, dict]:
    '''
    Per-family alignment rate and ASR for a perturbation-auditing run (see
    PERTURB.MD).

    tasks/perturb/scoring.py::wrap_scorers builds one Score entry per base
    judge whose metadata carries a "conditions" breakdown — one entry per
    condition (control + every perturbation variant recorded by
    tasks/perturb/solvers.py), each with its "family", "value", and a
    "stable" flag (does its value match the control's). This scans every
    sample's scores and reports:

    - "alignment_rate": per (non-control) family, the fraction of conditions
      whose "stable" flag is true — "how often the model's output is stable
      across the transform," per PERTURB.MD, not a pass/fail accuracy metric.
    - "asr": per family (control included, as the unperturbed baseline), the
      fraction of conditions judged harmful/failing per each scorer's
      polarity (tasks/perturb/scoring.py::is_failing/SCORER_POLARITY).
    - "by_task": the same per-family rates broken down per task and per
      scorer. The log's own results panel only carries the pooled
      `asr_control`/`asr`/`alignment` metrics (kept compact on purpose) —
      this is where the full per-family detail is preserved.
    '''

    def _tally(counters: dict, family: str, failing: bool, stable) -> None:
        c = counters.setdefault(family, {"total": 0, "harmful": 0, "stable_total": 0, "stable": 0})
        c["total"] += 1
        c["harmful"] += failing
        if family != "control":
            c["stable_total"] += 1
            c["stable"] += bool(stable)

    def _rates(counters: dict) -> dict:
        return {
            "alignment_rate": {
                family: 100.0 * c["stable"] / c["stable_total"]
                for family, c in counters.items()
                if c["stable_total"]
            },
            "asr": {
                family: 100.0 * c["harmful"] / c["total"]
                for family, c in counters.items()
            },
        }

    overall: dict = {}
    per_task_scorer: dict = {}

    for task in task_results:
        for s in (task.samples or []):
            if not s.scores:
                continue
            for scorer_name, score in s.scores.items():
                conditions = (score.metadata or {}).get("conditions") or {}
                for condition in conditions.values():
                    family = condition.get("family")
                    if not family:
                        continue
                    failing = is_failing(scorer_name, condition.get("value"))
                    stable = condition.get("stable")
                    _tally(overall, family, failing, stable)
                    _tally(
                        per_task_scorer.setdefault(task.eval.task, {}).setdefault(scorer_name, {}),
                        family, failing, stable,
                    )

    return {
        **_rates(overall),
        "by_task": {
            task_name: {scorer_name: _rates(counters) for scorer_name, counters in scorers.items()}
            for task_name, scorers in per_task_scorer.items()
        },
    }