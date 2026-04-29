'''
author: @tae

Utilities for grader and model loading
TODO: rename file to something more fitting since this is general utilties
'''

from inspect_ai.log import EvalLog
from pathlib import Path
import json

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

            # TODO: Change metric for social harm bench and remove the patch below
            if str(task.eval.task) == 'social_harm_bench':
                value = value * 100

            avg_score.append(value) # only keep valid values for the average

        except Exception as e:
            print(f"[ERROR] occured on benchmark for task: {task}: {e}")
            value = -1

        scores['by_task'][task.eval.task] = value

    scores['reported']= sum(avg_score) / len(avg_score)

    return scores['reported'], scores['by_task']