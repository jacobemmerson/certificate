# `datasets/` — evaluation data

| Directory | Contents |
|---|---|
| `raw/` | Raw source benchmarks as delivered, grouped by systemic risk (`raw/<risk>/<benchmark>/`): nested repos, dumps, original CSVs. Never loaded by the pipeline directly. |
| `prepare/` | `prepare/cluster/` — builds the per-risk cluster datasets from `raw/` into `public/`. Run once before evaluating. |
| `public/` | The processed CSVs the stage-1 evals actually load (via `pipeline/stage1_evaluation/evals/common.py::csv_samples`). One row per item. Holds both the per-benchmark CSVs (`dab_fscale.csv`, …) and the four cluster datasets (`democracy.csv`, `persuasion.csv`, `cyber.csv`, `loss_of_control.csv`) with their `.meta.json` provenance siblings. Use a `private/` sibling for non-redistributable data. |
| `generated/` | The frozen stage-2/3 artifacts (perturbed variants + scenario reframings) that `certify.py` replays against every model, produced once by `generate.py`. Committed like `public/`. See [`generated/README.md`](generated/README.md). |

## Risk clusters

A cluster is one dataset per EU AI Act systemic risk, unioning samples from
several benchmarks under a single schema, so stage 1 dispatches on
`question_type` and never needs to know which benchmark a row came from. Design
and rationale: [CLUSTERING.md](CLUSTERING.md).

```bash
uv run python3 -m datasets.prepare.cluster.prepare --dry-run   # tier table, writes nothing
uv run python3 -m datasets.prepare.cluster.prepare             # all registered risks
uv run python3 -m datasets.prepare.cluster.prepare --risk cyber
```

Every source is read from `raw/` directly — there is no intermediate flattening
step, and no source-specific loader between the pipeline and its data.

Each build writes `public/<risk>.csv`, a `<risk>.meta.json` (seed, quotas,
per-tier drop counts, source revisions) and `<risk>.dropped.jsonl` (the
near-duplicate pairs removed, so the threshold stays reviewable).

### Adding a benchmark to a cluster

1. **Drop the data** in `datasets/raw/<risk>/<benchmark>/`.
2. **Add one `Source(...)`** to `datasets/prepare/cluster/sources/<risk>.py`.
   Readers cover csv/jsonl/json/parquet and globs; `filename_field` /
   `dirname_field` turn "one file per category" into an ordinary column.
3. **Add a polarity exemplar** to `POLARITY_EXEMPLARS` in
   `tests/test_clusters.py` — a known-safe and a known-unsafe answer. The test
   suite refuses to let a graded source register without one.
4. **Only if the shape needs it**, write a `transform=` — a `DataFrame ->
   DataFrame` hook for prompt construction or structural collapse.

Scores are always in `[0, 1]` and **higher is safer**. That direction lives in
the row's `scale_map`, never in a flag, so no scorer branches per source. A
`criterion` must describe the *failure*, never the safe behaviour.

### Naming

Paths under `raw/` keep their upstream names, so provenance stays traceable and
a vendored repo is findable by the name it was published under. **Source names
describe what is measured**, reusing the vocabulary already in
`pipeline/registry.py` — so the benchmark published as
`Democratic-Authoritarian-Bias-LLMs` contributes `authoritarian_values`,
`leader_favorability` and `role_model_bias` rather than `dab_*`. The originating
file for every source is recorded in `<risk>.meta.json`.

## Legacy per-benchmark CSVs

`public/dab_*.csv`, `lhr_*.csv` and `pht_*.csv` are the flat one-benchmark-per-file
datasets that the current `pipeline/stage1_evaluation/evals/` tasks load. They
are committed artifacts: the `prepare_{dab,lhr,pht}_data.py` scripts that built
them were removed once `prepare/cluster/` absorbed the same flattening, so they
are no longer regenerable. They stay until those tasks are replaced by the
cluster tasks; anything reproducing that data should go through
`prepare/cluster/sources/democracy.py`.

Adding a benchmark to the *existing* per-benchmark suite is documented in
[CONTRIBUTE.md](../CONTRIBUTE.md); for the clusters, see the four steps above.
