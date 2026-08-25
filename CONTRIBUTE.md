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
> [datasets/CLUSTERING.md](datasets/CLUSTERING.md) for why. `pipeline/registry.py`
> now builds one task per risk and needs no edit.

**Tip:** a benchmark with clear inputs, a stated scoring rule, and a published
judge prompt takes about twenty minutes. One whose evaluation you have to infer
takes a day — most of it spent on step 5, which is the step that matters.

---

## 1. Put the data under `datasets/raw/<risk>/`

Prefer a **submodule**, so the checkout pins itself:

```bash
git submodule add git@github.com:owner/repo.git datasets/raw/<risk>/<Repo-Name>
```

Keep the upstream directory name — provenance stays traceable, and the build
records each submodule's SHA in `datasets/public/<risk>.meta.json`.

Copy files in instead only when a submodule is disproportionate (a 100 MB repo
for 3 MB of data). If you copy, say so in the source module's docstring with the
repo, commit, and licence: a copied directory pins nothing on its own.

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
    path="raw/cyber/YourRepo/data/*.parquet",
    reader="parquet",
    query="prompt",                 # column holding the prompt
    criterion=lambda r: CRITERION.format(category=r["category"]),
    rubric=YOUR_RUBRIC,
    categories=YES_NO, scale_map=YES_NO_MAP,
    metadata=["category"],          # travels with the row, drives stratification
    stratify=["category"], quota=90,
)
```

Useful fields when the shape is awkward:

| Field | For |
|---|---|
| `filename_field` / `dirname_field` | "one file per category" — turns the filename into an ordinary column |
| `transform` | a `DataFrame -> DataFrame` hook, for prompt construction or structural collapse |
| `system_prompt` | benchmarks that steer the model deliberately (persona arms, assigned roles) |
| `group_key` | rows only meaningful as a set — the quota then counts groups, not rows |
| `balanced` | even allocation per stratum instead of proportional |
| `distinct_on` | fields whose differing values mean "different items, however similar the text" |

**`distinct_on` is the one people miss.** Dedup keys on the prompt text, so any
source whose rows share a user turn — the scenario lives in a system prompt, or
one instruction template wraps a varying term — collapses to a single row. Two
sources in this repo hit it (`persusafety`, the human-rights persona arms). If
your build reports a suspiciously large `exact` drop, this is why.

## 3. Pick the scoring shape

`question_type` is the whole dispatch mechanism — five values, because there are
five scoring shapes in the suite, not one per benchmark.

| `question_type` | Scored by | Use when |
|---|---|---|
| `graded` | judge picks one of `categories`; `scale_map` converts | free-text response judged against a rubric |
| `mcq` | response matched against `target` | the benchmark supplies options |
| `likert` | deterministic parse of the model's own scale answer | fixed scale + JSON contract |
| `extraction` | entity extraction, then classification | the finding is about *what* the response named |
| `detection` | the benchmark's own deterministic detector, no judge | **the benchmark doesn't use a judge** |

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

## 6. Perturbation split — only if the prompt has a rigid wrapper

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

`elicitation_family` picks which framing templates apply — `compliance` ("do X"),
`judgment` ("is this acceptable?"), `opinion` ("rate your agreement"), or
`generic`. **If none fits, use `generic`**, which skips the framing family and
leaves the others running. That is the fail-safe: a wrong wrapper breaks
content-equivalence, so the default is skip, not guess.

## 7. Build, test, run

```bash
# build your cluster (all four if you omit --risk)
uv run python3 -m datasets.prepare.cluster.prepare --risk cyber

# the suite: polarity, template invariants, dedup, doc consistency
uv run python3 -m unittest discover tests

# cheap smoke run — no model calls at all
PYTHONPATH=. uv run inspect eval pipeline/stage1_evaluation/evals/clusters.py@cyber \
    --model mockllm/model -T grader=mockllm/model --limit 5

# real run, 2 samples, results NOT written to models.json
uv run python3 certify.py -m <target-model> -g <grader-model> --only cyber --limit 2
```

Check the build report before anything else: `loaded` / `exact` / `near` /
`cross` / `kept` per source. A large `exact` drop means missing `distinct_on`
(step 2); a large `near` drop means your source is templated and wants a higher
`tau` or a `distinct_on` on the varying term. Any `cross` drop means your source
ships prompts an earlier source in the same cluster already ships — worth
checking whether it is vendoring another benchmark before you tune anything. Dropped pairs are written to
`datasets/public/<risk>.dropped.jsonl` so the threshold is reviewable rather
than trusted.

`--only` takes **risk names** now (`--only cyber`), not benchmark keys. Your
benchmark appears in the results panel as `source_<your_benchmark>` and in
`models/models.json` under `by_source`.

## 8. Scoring conventions

- The cluster score is the **mean of its per-source scores**, so sample count is
  weight. Quotas are load-bearing; `tests/test_clusters.py` fails any cluster
  where one slice exceeds 40%.
- A source that measures something other than alignment belongs in
  `DIAGNOSTIC_SOURCES` (`pipeline/utils/graders.py`) — reported per-source but
  excluded from the pooled number. `wmdp` (capability absence) and
  `cyber_false_refusal` (a counter-metric) are the two.
- If your benchmark's finding is a property of a *distribution* rather than a
  mean — a gap between groups, a spread across arms — add a summary to
  `SUMMARIES` in `pipeline/stage1_evaluation/scorers/source_metrics.py`. It must
  still land in [0, 1], higher = safer.
- Under `--perturb` / `--simulate`, scorers are wrapped automatically: the
  reported per-sample value becomes the worst outcome across control and every
  condition, with `harm_propensity` / `stability` added to the results panel.
  Generate the artifacts first:

  ```bash
  uv run python3 generate.py --only cyber --perturb paraphrase --perturb-k 1
  ```

## Where things live

| | |
|---|---|
| `datasets/CLUSTERING.md` | the schema, and why the registries collapsed into it |
| `datasets/BENCHMARKS.md` | every benchmark, its counts, and how it is scored vs. its original |
| `datasets/prepare/cluster/schema.py` | `Source`, `Row`, `validate()` — the contract |
| `datasets/prepare/cluster/sources/` | one module per risk; this is where you add yours |
| `pipeline/stage1_evaluation/scorers/cluster.py` | the dispatching scorer and judge prompts |
| `pipeline/stage1_evaluation/scorers/detectors.py` | ported deterministic detectors |
| `GRADERS.md` | the judge ensemble |
