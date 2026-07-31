# Contributing a New Benchmark

How to add a new benchmark (from a CSV) to the certification pipeline using
[Inspect AI](https://inspect.aisi.org.uk/)'s interface. The short version:
drop a CSV in `datasets/public/`, write one `@task` file in
`pipeline/stage1_evaluation/evals/`, register it in `pipeline/registry.py`, and (optionally) tell
the perturbation module how your benchmark elicits its judgment.

**Tip:** `Well-defined and formatted datasets with clear inputs/outputs or scoring templates are SIGNIFICANTLY easier to add. If you're struggling, I recommend adding a data-formatting script to datasets/prepare/ before adding any Inspect AI tasks.`

## 1. Add the dataset

Put your CSV under `datasets/public/` (or `datasets/private/` for
non-redistributable data). One row per item. Any columns are fine — you
control the mapping to Inspect `Sample`s.

## 2. Write the task file (`pipeline/stage1_evaluation/evals/<your_benchmark>.py`)

Follow the existing files (`socialharmbench.py` is the simplest model). The
pattern:

```python
from inspect_ai import Task, task
from inspect_ai.solver import generate

from pipeline.stage1_evaluation.evals.common import csv_samples, with_llamaguard
from pipeline.stage1_evaluation.scorers.harm import llm_judge_scorer

dataset = csv_samples(
    "your_benchmark.csv",              # under datasets/public/
    input_col="prompt_text",           # what the target model sees
    id_col="prompt_id",                # stable per-item id
    metadata_cols=["category", ...],   # anything scorers/analysis need
)


@task
def your_benchmark(
    grader: str | list[str] = "openai/gpt-4o",
    llamaguard_model: str = "openrouter/meta-llama/llama-guard-4-12b",
):
    return Task(
        dataset=dataset,
        solver=generate(),
        scorer=with_llamaguard(
            llm_judge_scorer(model=grader, template=YOUR_JUDGE_TEMPLATE),
            llamaguard_model,
        ),
    )
```

`pipeline/stage1_evaluation/evals/common.py` supplies the shared plumbing:
`csv_samples(...)` loads a CSV from `datasets/public/` and maps
each row to a `Sample` (pass `to_sample=<fn>` instead of column names when
the mapping isn't a straight column pick — templated input, computed ids);
`with_llamaguard(scorers, llamaguard_model)` puts your scorer(s) first and
appends LlamaGuard when configured.

Conventions that matter:

- **`@task` function name is the benchmark's identity.** The registry name
  is used for log filenames and for perturbation adapter lookup (step 4).
- **Accept `grader` and `llamaguard_model` parameters** (with defaults) so
  `certify.py`'s `--grader`/`--llamaguard` flags reach your task. Append
  `llamaguard_scorer` when `llamaguard_model` is set — every benchmark
  carries it by default.
- **Reuse existing scorers** from `pipeline/stage1_evaluation/scorers/harm.py`
  (`llm_judge_scorer`, `llamaguard_scorer`) before writing a new
  `@scorer`. If you do write one, give it `@scorer(metrics=[...])` with the
  **primary metric first** (see step 5).
- **Put your primary scorer first** in the `scorer=[...]` list. The first
  scorer's first metric becomes the benchmark's reported certification
  score.

## 3. Register it (`pipeline/registry.py::init_benchmarks`)

Import your `@task` function and add an entry to the `BENCHMARKS` dict:

```python
'yourkey': {
    'tasks': [your_benchmark(grader=grader, llamaguard_model=llamaguard_model)],
    'name': 'your_benchmark_name',   # log directory: logs/{model}/{name}/
},
```

- `'yourkey'` is the benchmark key used by `--only yourkey` and as the key
  under `scores`/`scores_meta`/`perturbations`/`status` in
  `models/models.json`.
- A benchmark entry can hold multiple related tasks (see `'hr'` or
  `'auth'`); their scores are averaged into one benchmark score.

That's all the wiring — `certify.py` iterates `init_benchmarks` and
`apply_stages` covers every registered Task automatically. One extra
step for the new pipeline: stages 2/3 replay **pregenerated** perturbed prompts,
so once your task is registered, generate its artifacts before running
`--perturb`/`--simulate`:

```bash
uv run python3 generate.py --only yourkey            # all pregenerated families
uv run python3 generate.py --only yourkey --simulate # + scenario reframings
```

The artifacts land in `datasets/generated/<task_name>/` (committed to the repo);
`certify.py` validates they exist and fails fast with the exact command
otherwise. `reconsideration` needs no artifact (it runs live). See
[`datasets/generated/README.md`](datasets/generated/README.md).

## 4. Perturbation integration (`pipeline/stage2_perturbation/`)

**The common case needs nothing.** If your target-facing prompt is free
text graded by an LLM judge/LlamaGuard, every perturbation family except
framing applies automatically, and framing is safely skipped (see below).
Two situations need a decision:

### a. Framing: pick an `elicitation_family`, or opt out

The elicitation-format framing family applies a fixed set of
content-equivalent wrapper templates. Which template set is used is a
**manual, design-time classification** — you declare it, nothing infers it.
Add an entry to `pipeline/stage2_perturbation/adapters.py::ADAPTERS` keyed by your
`@task` function's name, choosing the family that matches what your
benchmark elicits (template sets live in `pipeline/stage2_perturbation/framing.py`):

| `elicitation_family` | The item is... | Templates applied |
|---|---|---|
| `"compliance"` | a request/instruction | imperative / interrogative ("how would one...") / enumerative ("list the steps...") |
| `"judgment"` | a scenario or claim to evaluate | direct ("is this true?") / predictive ("will this be judged true?") / normative ("should this be judged true?") |
| `"opinion"` | something eliciting agreement/position | first-person / third-person / forced-choice |

```python
"your_benchmark": PerturbAdapter(
    item_text=_default_item_text,
    render=_default_render,
    elicitation_family="compliance",
),
```

**If none fits, register nothing.** Unregistered tasks fall back to
`DEFAULT_ADAPTER` (`elicitation_family="generic"`), which has no template
entry, so the framing family is skipped for that task while every other
family still runs. That's the deliberate fail-safe: a wrong wrapper would
violate the content-equivalence requirement (only surface form may change,
never content), so the default is skip, not guess.
`role_model_bias` is the in-repo example — its open-ended "list role models"
elicitation fits no family, so it's intentionally unregistered.

### b. Rigid output formats: custom `item_text`/`render`

If the target must reply in a rigid, directly machine-parsed format (a JSON
contract, fixed scale options — like `fscale`/`leader_favorability`), a naive rewrite
of the whole prompt would corrupt the format instructions and break the
parser. Register an adapter whose `item_text(state)` extracts only the raw
statement/question (usually from `state.metadata`) and whose
`render(state, new_text)` re-injects the format wrapper verbatim via your
benchmark's own prompt-builder function. See the `fscale`/`leader_favorability`
entries in `adapters.py`.

### c. Scorer polarity

Perturbation reporting picks each sample's **worst** condition and counts
**failing** outcomes (LVR), assuming higher score value = better/safer
(`value_to_float` lower = worse, `< 1.0` = failing). If your scorer is
inverted — a higher value means a *worse* outcome — register it in
`pipeline/utils/scoring.py::safety` — every scorer reports [0, 1] higher-safer
functions, as `role_model_bias_scorer` does. Otherwise worst-case scores and LVR
for your benchmark will be backwards.

## 5. Scoring conventions

- `pipeline/utils/graders.py::aggregate_score` reports the **first metric of
  the first scorer** of each task, averaged across a benchmark's tasks, as
  the certification score. Scores are expected on a 0–100 scale — if your
  primary metric is a 0–1 fraction (e.g. `accuracy()`), either rescale in
  the scorer/metric or add your task to the scaling special-case in
  `aggregate_score` (as `social_harm_bench` does).
- Under `--perturb`/`--simulate`, your scorers are wrapped automatically
  (`pipeline/utils/scoring.py::wrap_scorers`): the reported per-sample value
  becomes the worst outcome across the control and every enabled condition
  (scenario included), and pooled metrics are added to the log's results
  panel — `harm_propensity_control` always, `harm_propensity` + `stability` for the stage-2
  families, `harm_propensity_scenario` + `stability_scenario` for stage 3. Per-family
  detail lands in `models/models.json` under `perturbations.{yourkey}.by_task`
  and `simulations.{yourkey}.by_task`.

## 6. Test it

```bash
# unit tests (scoring/aggregation logic — no model calls)
uv run python3 -m unittest discover tests

# cheap end-to-end smoke run: 2 samples, results NOT saved to models.json
uv run python3 certify.py -m <target-model> -g <grader-model> \
    --only yourkey --limit 2

# with perturbations: generate the artifacts once, then replay them (k=1 keeps
# it cheap). --limit produces partial artifacts, fine for a smoke test.
uv run python3 generate.py --only yourkey --perturb paraphrase --perturb-k 1 --limit 2
uv run python3 certify.py -m <target-model> -g <grader-model> \
    --only yourkey --limit 2 --perturb paraphrase reconsideration --perturb-k 1
```

Then open the log (`inspect view --log-dir logs/<model>/<name>/`) and check:

- each sample's transcript shows your solver steps (plus, under
  `--perturb`, one tab per perturbation family and one `{family}_scoring`
  tab per condition family);
- the results panel reports your primary metric first, with
  `harm_propensity_control`/`harm_propensity`/`stability` alongside under `--perturb` (plus
  `harm_propensity_scenario`/`stability_scenario` under `--simulate`);
- a full run (no `--limit`) writes your benchmark key into
  `models/models.json` under `scores`, `scores_meta`, `perturbations`, and
  `status`.
