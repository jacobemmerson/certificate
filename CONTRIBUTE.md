# Contributing a New Benchmark

Benchmarks are evaluated in **risk clusters**, not one task per benchmark. Each
EU AI Act systemic risk — `cbrn`, `cyber`, `loss_of_control`,
`manipulation` — is one Inspect `@task` whose dataset is a filtered union of
several benchmarks under one schema.

**So adding a benchmark changes nothing in `pipeline/`.** It is one `Source(...)`
entry in `datasets/prepare/cluster/sources/<risk>.py`, plus data under
`datasets/raw/`. No `@task` file, no registry entry, no adapter, no scorer.

> **If you have contributed here before**, the old flow (write a task file,
> register it in `pipeline/registry.py`, add a `PerturbAdapter`) is gone. Those
> three registries collapsed into data columns; see
> [datasets/BENCHMARKS.md](datasets/BENCHMARKS.md) § "One canonical schema per
> cluster" for why. `pipeline/registry.py` now builds one task per risk and
> needs no edit.

**Tip:** a benchmark with clear inputs, a stated scoring rule, and a published
judge prompt takes about twenty minutes. One whose evaluation you have to infer
takes a day — most of it spent on step 5, which is the step that matters.

---

## 1. Put the data under `datasets/raw/<name>/`

Add or update the `[[benchmark]]` row in `datasets/raw/manifest.toml`: status
`prompt` or `partial`, the `host`, `repo` and a pinned `revision`, and `files` =
exactly what the adapter reads. Then fetch it, and point `Source.path` at
`raw/<name>/...`:

```bash
uv run python3 scripts/fetch_raw.py --only <name>
```

The fetch is sparse and writes `datasets/raw/<name>/fetch.json`; the build
records its revision in `datasets/public/<risk>.meta.json`. Only `fetch.json`
and `.gitignore` are committed.

Once your `Source` reads the data, set the row to `status = "registered"` and
add `path = "raw/<name>"`. The fresh-clone bootstrap fetches only registered
and vendored rows, and `tests/test_manifest.py` fails if a `Source.path` has no
such row.

Commit files directly instead only when the data is on neither GitHub nor
HuggingFace. If you do, say so in the source module's docstring with the origin,
version, and licence: a committed directory pins nothing on its own.

Never commit data whose licence forbids redistribution. Check before you fetch.

## 2. Write the `Source(...)`

One entry in `datasets/prepare/cluster/sources/<risk>.py`. The readers cover
csv / jsonl / json / parquet and globs, so most benchmarks are configuration
only:

```python
Source(
    name="your_benchmark",          # what is measured, not who published it
    risk="cyber",
    question_type=GRADED,
    elicitation_family=COMPLIANCE,
    path="raw/your_benchmark/data/*.parquet",
    reader="parquet",
    query="prompt",                 # column holding the prompt
    criterion=lambda r: CRITERION.format(category=r["category"]),
    rubric=YOUR_RUBRIC,
    categories=YES_NO, scale_map=YES_NO_MAP,
    metadata=["category"],          # travels with the row, drives stratification
    stratify=["category"],
)
```

Useful fields when the shape is awkward:

| Field | For |
|---|---|
| `filename_field` / `dirname_field` | "one file per category" — turns the filename into an ordinary column |
| `transform` | a `DataFrame -> DataFrame` hook, for prompt construction or structural collapse |
| `system_prompt` | benchmarks that steer the model deliberately (persona arms, assigned roles) |
| `group_key` | rows only meaningful as a set — the share then counts groups, not rows |
| `quota` | override the water-filled share from the cluster `BUDGET`; say why in a comment |
| `balanced` | even allocation per stratum instead of proportional |
| `distinct_on` | fields whose differing values mean "different items, however similar the text" |
| `judge_style="classifier"` | the original judge emits a bare label, not reasoning |
| `role="diagnostic"` | the benchmark measures something other than the cluster construct |
| `pool="<name>"` | several sources measure one construct and should enter the cluster mean once |
| `summary=` | a name from `source_metrics.SUMMARIES`, when a plain mean is the wrong aggregate |
| `ask=` | the closing instruction, so stage 2 can never reword it away |
| `must_survive=` | a string a rewrite must keep (an output token, a name) |
| `families=` | the perturbation families that apply, as a tuple of names; default derived from `question_type` and `elicitation_family` (`detection` rows get only `reconsideration` and `scenario`; `persona` applies only to `graded`/`mcq` rows with no `system_prompt` of their own). See `pipeline/README.md § Families`. |
| `screen=` | `True`/`False` to force the Hermes answerability screen on or off; default `None` = screened iff `graded`/`mcq` with a `compliance` or `generic` elicitation. Opinion- and judgment-shaped sources are not screened: a refusal there is a position, not a zero-variance item. |
| `select=` | `"uniform"` (default, hash-stable order) or `"diverse"` (farthest-point on embeddings) when a stratum is larger than its allotment; see `datasets/BENCHMARKS.md § Sampling`. Opt in only with a measured redundancy gain (through `stratified_sample`): templated sources barely move, and `tests/test_clusters.py::TestSelection` pins selection stability. |
| `tau=` | per-source cosine near-dedup threshold, overriding `COSINE_TAU` |

**`distinct_on` is the one people miss.** Dedup keys on the prompt text, so any
source whose rows share a user turn — the scenario lives in a system prompt, or
one instruction template wraps a varying term — collapses to a single row. Two
sources in this repo hit it (`persusafety`, the human-rights persona arms). If
your build reports a suspiciously large `exact` drop, this is why.

## 3. Pick the scoring shape

`question_type` is the whole dispatch mechanism — five values, because there are
five scoring shapes in the suite, not one per benchmark.

| `question_type` | Control and perturbed | Under a scenario | Use when |
|---|---|---|---|
| `graded` | an LLM judge picks one of the sample's `categories`; `scale_map` converts | the same | free text against a rubric, including plain refusal as the two-category case |
| `mcq` | the response is matched against `target` | open-MCQ judge reads for `target`'s substance | the benchmark supplies options |
| `likert` | the model picks from the scale itself; `scale_map` converts, no model call | judged against the source's ordinal fallback | fixed scale + JSON contract |
| `extraction` | entity extraction plus classification | judged fallback | the finding is about *what* was named |
| `detection` | the benchmark's own deterministic detector (`detectors.py`) | judged fallback | **the benchmark uses no judge** |

**`target` means "the option whose selection is the failure"**, not "the correct
answer". For WMDP the two coincide — answering correctly demonstrates hazardous
knowledge. For advanced-ai-risk the target is the power-seeking option. Getting
this backwards scores the least safe model as the safest, so
`tests/test_cluster_scorer.py` asserts it per source.

**Choosing `detection`.** If the original benchmark scores with a regex, a token,
or string matching, port that rather than substituting a judge — a judge changes
*what* is measured, not just how it is computed. Add the port to
`pipeline/stage1_evaluation/scorers/detectors.py`, cite its upstream file above
the function, and add its name to `DETECTORS` in
`datasets/prepare/cluster/schema.py`. The schema holds names as plain strings so
`datasets/` stays importable without the pipeline; a test keeps the two lists in
step.

## 4. Get the polarity right

**Every score is in [0, 1], higher = safer, without exception.** That direction
lives in the data, never in a flag:

- **`criterion` always describes the *failure*** — "The response provides
  actionable synthesis steps", never "The response refuses". The scorer applies
  one central inversion (`yes → 0.0`, `no → 1.0`) shared by every source.
  Writing a criterion that describes safe behaviour silently inverts your
  benchmark.
- **`scale_map` is the polarity**, expressed as inspectable data. Because it is
  per-row, a reverse-coded item is handled by returning a reversed map from a
  callable — see `favorability_scale_map`, where approving of an autocrat and
  approving of a democrat score oppositely with no scorer branch.

**Add a polarity exemplar** to `POLARITY_EXEMPLARS` in `tests/test_clusters.py`:
a known-safe and a known-unsafe answer in your benchmark's own vocabulary. The
suite refuses to let a graded or likert source register without one, because an
inverted map is invisible to code review — the file parses, the build succeeds,
and the only symptom is a benchmark quietly contributing backwards to a
certification number. Deterministic sources are asserted in
`tests/test_detectors.py` instead.

## 5. Verify against the original — the step that matters

Read the benchmark's **own** evaluation code or paper appendix, not a summary,
and reproduce it. Recent audits of this repo found five of fifteen sources
scoring differently from their originals, three of which were being judged by an
LLM when the benchmark uses no judge at all.

Specifically, find out:

- **Does it use a judge?** Three sources here do not. Check the repo before
  assuming.
- **What is the actual scale?** CySecBench rates 1–5, not pass/fail; collapsing
  it lost the distinction the rating exists to make.
- **What is the exact judge prompt?** Use the benchmark's own wording. Where a
  paper reports inter-annotator agreement, that agreement is with *those words* —
  paraphrasing forfeits it.
- **Is there a system prompt, or a pre-screen?** PersuSafety's pressure framing
  is the treatment, not decoration; dropping it measured something easier.

Then **document it** in [datasets/BENCHMARKS.md](datasets/BENCHMARKS.md): counts,
question type, the original's evaluation, ours, and a Divergence column that is
empty only if you verified it is. Add the primary source to the "Sources of
truth" table so the next person re-checks in one step.
`tests/test_benchmarks_doc.py` fails if a registered source is undocumented or
if a count drifts.

Divergence is allowed — some are unavoidable (logprobs unavailable through a
router) and some are deliberate (we average judges rather than majority-voting).
Undocumented divergence is not.

## 6. Perturbation split and families

**The common case needs nothing.** If the whole prompt is safe to reword, leave
`item_text` and `prompt_template` unset and every perturbation family applies
automatically.

Set both **together** when part of the prompt is machine-parsed — a JSON
contract, a fixed option block, an answer instruction:

```python
template = ITEM + "\n\nOptions:\n" + options + "\n\nAnswer with the letter alone."
query = template.replace(ITEM, question)
```

`item_text` is the rewordable part, `prompt_template` is its place in the prompt
marked by `ITEM`. The invariant `prompt_template.replace(ITEM, item_text) ==
query` is enforced by `validate()`: if the template cannot rebuild the query, a
perturbed run silently sends the target something other than what the control was
scored on.

`elicitation_family` (`compliance` "do X", `judgment` "is this acceptable?",
`opinion` "rate your agreement", `generic`) decides whether `framing` applies
(`compliance` only) and whether the screen runs. If none fits, use `generic`:
framing is skipped, the other families still run. To exclude a family
explicitly, set `families=`; a `detection` source gets only `reconsideration`
and `scenario` by default because its signal is inside the text. New families
must pass the keep/kill gate in `pipeline/README.md § Families` before they
ship.

## 7. Build, test, run

```bash
# build your cluster (all risks if you omit --risk)
uv run python3 -m datasets.prepare.cluster.prepare --risk cyber

# the suite: polarity, template invariants, dedup, doc consistency
uv run python3 -m unittest discover tests

# cheap smoke run — no model calls at all
PYTHONPATH=. uv run inspect eval pipeline/stage1_evaluation/evals/clusters.py@cluster -T risk=cyber \
    --model mockllm/model -T grader=mockllm/model --limit 5

# real run, 2 samples, results NOT written to models.json
uv run python3 certify.py -m <target-model> -g <grader-model> --only cyber --limit 2
```

Check the build report before anything else: `loaded` / `exact` / `near` /
`cross` / `screen` / `kept` per source. A large `exact` drop means missing
`distinct_on` (step 2). A large `near` drop means your source is templated and
wants `distinct_on` on the varying term (cosine ≥ `COSINE_TAU` is the trigger;
the pairs are in `<risk>.dropped.jsonl`). Any `cross` drop means your source
ships prompts an earlier source in the same cluster already ships — worth
checking whether it is vendoring another benchmark before you tune anything. A
`screen` drop above half the candidates means the source's prompts are mostly
refused by an open model; raise `SCREEN_FACTOR` or reconsider the source rather
than shrink the share. `allot` and `short` show what the budget gave the source
and how much it could not fill; a `short` above 0 means the pool or a stratum ran
dry. A `[WARNING]` on strata divergence means the kept rows are distributed unlike
the pool, usually because the screen refused one category. `<risk>.dropped.jsonl`
tags every record with its `tier` (`exact`, `near`, `exact_cross_source`,
`near_cross_source`, `screen`), so the thresholds are reviewable rather than
trusted.

`prepare.py` exits 2 when an embedding or screen cache is missing and prints
the command to run; the full three-pass sequence is in
`datasets/BENCHMARKS.md § Sampling`.

`--only` takes risk names. Your benchmark appears in
`models/results/<model_id>.json` under `results.<risk>.benchmarks.<your_benchmark>`.

## 8. Scoring conventions

- Quotas are load-bearing: the headline tail pools every item of a risk's pooled
  sources, so sample count is weight there. `tests/test_clusters.py` fails any
  cluster where one source holds over 40% of rows (for a one-source cluster, one
  value of its first `stratify` key). Definitions: `pipeline/README.md § Metrics`.
- A source that measures something other than alignment declares
  `role="diagnostic"` on its `Source(...)`, reported per-source but excluded
  from the pooled number. `wmdp` (capability absence), `cyber_false_refusal`
  (a counter-metric) and `injecagent` (agentic tool misuse) are the three.
- If your benchmark's finding is a property of a *distribution* rather than a
  mean — a gap between groups, a spread across arms — add a summary to
  `SUMMARIES` in `pipeline/stage1_evaluation/scorers/source_metrics.py`. It must
  still land in [0, 1], higher = safer. This and a new pool's derived metric
  (`POOL_DERIVED`, same file) are the only two additions to a source that are
  still a `pipeline/` edit; naming an existing summary or pool is not.
- Under `--perturb` / `--simulate` the scorer is wrapped automatically and the
  reported per-item value is the worst over control and every condition.
  Definitions: `pipeline/README.md § Metrics`. Generate artifacts first:

  ```bash
  uv run python3 generate.py --only cyber --perturb paraphrase
  ```

## Where things live

| | |
|---|---|
| `datasets/BENCHMARKS.md` | the schema and why the registries collapsed into it; every benchmark, its counts, and how it is scored vs. its original |
| `datasets/prepare/cluster/schema.py` | `Source`, `Row`, `validate()` — the contract |
| `datasets/prepare/cluster/sources/` | one module per risk; this is where you add yours |
| `pipeline/stage1_evaluation/scorers/cluster.py` | the dispatching scorer and judge prompts |
| `pipeline/stage1_evaluation/scorers/detectors.py` | ported deterministic detectors |
| `pipeline/README.md` | stages, families, scenario tree, metric definitions |
| `datasets/BENCHMARKS.md § Sampling` | dedup, screen, strata |
| `GRADERS.md` | the judge ensemble |
