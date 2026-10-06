# `datasets/` — evaluation data

| Directory | Contents |
|---|---|
| `raw/` | Raw source benchmarks as delivered, grouped by systemic risk (`raw/<risk>/<benchmark>/`): nested repos, dumps, original CSVs. Never loaded by the pipeline directly. Almost all are git submodules — `git submodule update --init` is the whole bootstrap. |
| `prepare/` | `prepare/cluster/` — builds the per-risk cluster datasets from `raw/` into `public/`. Run once before evaluating. |
| `public/` | The processed CSVs the stage-1 evals actually load (via `pipeline/stage1_evaluation/evals/common.py::csv_samples`). One row per item. Holds the four cluster datasets (`cbrn.csv`, `cyber.csv`, `loss_of_control.csv`, `manipulation.csv`) with their `.meta.json` provenance siblings. Use a `private/` sibling for non-redistributable data. |
| `generated/` | The frozen stage-2/3 artifacts (perturbed variants + scenario reframings) that `certify.py` replays against every model, produced once by `generate.py`. Committed like `public/`. See [`generated/README.md`](generated/README.md). |

`datasets/raw/manipulation/sycophancy-eval/datasets/mimicry.jsonl` is the one exception to the
submodule bootstrap: the GitHub submodule pin lacks it, so it is fetched separately from the
HuggingFace dataset `meg-tong/sycophancy-eval`.

## Risk clusters

A cluster is one dataset per EU AI Act systemic risk, unioning samples from
several benchmarks under a single schema, so stage 1 dispatches on
`question_type` and never needs to know which benchmark a row came from. Design
and rationale: [BENCHMARKS.md](BENCHMARKS.md). How rows are chosen
(embeddings, the answerability screen, strata): [BENCHMARKS.md § Sampling](BENCHMARKS.md#sampling).

```bash
uv run python3 -m datasets.prepare.cluster.prepare --dry-run   # tier table, writes nothing
uv run python3 -m datasets.prepare.cluster.prepare             # all registered risks
uv run python3 -m datasets.prepare.cluster.prepare --risk cyber
```

Every source is read from `raw/` directly — there is no intermediate flattening
step, and no source-specific loader between the pipeline and its data.

Each build writes `public/<risk>.csv`, a `<risk>.meta.json` (seed, quotas,
per-tier drop counts, source revisions) and `<risk>.dropped.jsonl` (every pair or item
dropped by cross-source dedup, cosine dedup or the screen, each tagged `near`,
`exact_cross_source` or `screen`, so thresholds stay reviewable).

### Adding a benchmark to a cluster

Full guide: **[CONTRIBUTE.md](../CONTRIBUTE.md)**. In short:

1. **Add the data** under `datasets/raw/<risk>/<benchmark>/`.
2. **Add one `Source(...)`** to `datasets/prepare/cluster/sources/<risk>.py`. Beyond
   the column mapping, declare what the pipeline cannot infer:
   - `judge_style="classifier"` only if the original judge emits a bare label.
   - `role="diagnostic"` if the benchmark measures something other than the
     cluster construct; `pool="<name>"` if several sources measure one construct.
   - `summary=` a name from `source_metrics.SUMMARIES` when a plain mean is wrong.
   - `ask=` the closing instruction, so stage 2 can never reword it away.
   - `must_survive=` any string a rewrite must keep (an output token, a name).
   - `families=` to restrict which perturbation families apply (default derives
     from `question_type`; `detection` rows get `reconsideration` and `scenario` only).
   - `screen=False` to exempt a source from the answerability screen (default:
     only compliance and generic graded/mcq sources are screened).
3. **Add a polarity exemplar** to `POLARITY_EXEMPLARS` in `tests/test_clusters.py`.
4. **Verify against the original** and record it in `BENCHMARKS.md`.
5. **Only if the shape needs it**, write a `transform=`.

Nothing in `pipeline/` changes, and the test suite fails if it would have to:
`tests/test_source_contract.py` greps `pipeline/` for source names. The two
exceptions are additions rather than per-source branches: a *new* summary name
has to be implemented in `source_metrics.SUMMARIES`, and a new pool's derived
metric (a figure computed across the pool, like the human-rights persona gap)
in `source_metrics.POOL_DERIVED`. Reusing an existing name needs neither.

Scores are always in `[0, 1]` and **higher is safer**. That direction lives in
the row's `scale_map`, never in a flag, so no scorer branches per source. A
`criterion` must describe the *failure*, never the safe behaviour.

### Naming

Paths under `raw/` keep their upstream names, so provenance stays traceable and
a vendored repo is findable by the name it was published under. **Source names
describe what is measured**, reusing the vocabulary already in
`pipeline/registry.py` — so the benchmark published as
`Democratic-Authoritarian-Bias-LLMs` contributes `authoritarian_values`,
`leader_favorability` and `role_model_bias` rather than the upstream `dab_*` file names. The originating
file for every source is recorded in `<risk>.meta.json`.
