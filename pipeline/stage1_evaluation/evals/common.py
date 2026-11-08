'''
Shared helpers for CSV-backed benchmark tasks (see CONTRIBUTE.md).

Every eval file in pipeline/stage1_evaluation/evals/ follows the same shape: load a CSV from
datasets/ and map each row to an inspect_ai Sample. These helpers hold the
shared plumbing so a new benchmark only has to declare its columns and @task
function.
'''

from pathlib import Path
from typing import Callable

from inspect_ai.dataset import Dataset, Sample, csv_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATASETS_DIR = REPO_ROOT / "datasets" / "public"


def csv_samples(csv_name: str, to_sample: Callable[[dict], Sample]) -> Dataset:
    '''Load datasets/public/{csv_name} as an Inspect dataset, one Sample per row.'''
    return csv_dataset(str(DATASETS_DIR / csv_name), to_sample)
