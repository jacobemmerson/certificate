'''
Shared helpers for CSV-backed benchmark tasks (see CONTRIBUTE.md).

Every eval file in pipeline/stage1_evaluation/evals/ follows the same shape: load a CSV from
datasets/, map each row to an inspect_ai Sample, and score with a
primary judge plus (by default) LlamaGuard. These helpers hold the shared
plumbing so a new benchmark only has to declare its columns, judge template,
and @task function.
'''

from pathlib import Path
from typing import Callable, Sequence

from inspect_ai.dataset import Dataset, Sample, csv_dataset
from inspect_ai.scorer import Scorer

from pipeline.stage1_evaluation.scorers.harm import llamaguard_scorer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATASETS_DIR = REPO_ROOT / "datasets" / "public"


def csv_samples(
    csv_name: str,
    *,
    input_col: str | None = None,
    id_col: str | None = None,
    metadata_cols: Sequence[str] | None = None,
    to_sample: Callable[[dict], Sample] | None = None,
) -> Dataset:
    '''
    Load datasets/public/{csv_name} as an Inspect dataset.

    The common case needs only column names: `input_col` becomes the sample
    input, `id_col` the (stringified) sample id, and `metadata_cols` the
    metadata dict. Pass a custom `to_sample(row)` instead when the mapping
    isn't a straight column pick (templated input, computed ids, ...).
    '''
    if to_sample is None:
        if input_col is None or id_col is None:
            raise ValueError("csv_samples needs input_col and id_col (or a custom to_sample)")
        cols = list(metadata_cols or [])

        def to_sample(row: dict) -> Sample:
            return Sample(
                input=row[input_col],
                id=str(row[id_col]),
                metadata={col: row[col] for col in cols},
            )

    return csv_dataset(str(DATASETS_DIR / csv_name), to_sample)


def with_llamaguard(scorers: Scorer | list[Scorer], llamaguard_model: str | None) -> list[Scorer]:
    '''
    The standard scorer assembly: the benchmark's own scorer(s) first (the
    first scorer's first metric is the reported certification score — see
    pipeline/utils/graders.py::aggregate_score), plus LlamaGuard when a model
    is configured (every benchmark carries it by default).
    '''
    scorers = scorers if isinstance(scorers, list) else [scorers]
    if llamaguard_model:
        scorers = scorers + [llamaguard_scorer(model=llamaguard_model)]
    return scorers
