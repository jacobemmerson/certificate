# `datasets/` — evaluation data

| Directory | Contents |
|---|---|
| `raw/` | Raw source benchmarks as delivered (nested repos, dumps, original CSVs). Never loaded by the pipeline directly. |
| `prepare/` | One `prepare_*.py` script per source: flattens `raw/<source>/` into evaluation-ready CSV(s) under `public/`. Run once before evaluating. |
| `public/` | The processed CSVs the stage-1 evals actually load (via `pipeline/stage1_evaluation/evals/common.py::csv_samples`). One row per item. Use a `private/` sibling for non-redistributable data. |
| `generated/` | The frozen stage-2/3 artifacts (perturbed variants + scenario reframings) that `certify.py` replays against every model, produced once by `generate.py`. Committed like `public/`. See [`generated/README.md`](generated/README.md). |

## Adding a new raw CSV to the evaluation suite

1. **Drop the raw data** in `datasets/raw/<your_benchmark>/`.
2. **Write `datasets/prepare/prepare_<name>_data.py`** that reads from
   `datasets/raw/<your_benchmark>/` and writes a flat CSV to
   `datasets/public/<name>.csv`. Skip this step if the raw file is already a
   clean one-row-per-item CSV — put it straight in `public/`.
   Convention: anchor paths with
   `REPO_ROOT = Path(__file__).resolve().parent.parent.parent` (see the
   existing prepare scripts).
3. **Write the eval + register it** — one `@task` file in
   `pipeline/stage1_evaluation/evals/` and an entry in
   `pipeline/registry.py::init_benchmarks`. Full walkthrough in
   [CONTRIBUTE.md](../CONTRIBUTE.md).
