# Stage 3 — Scenario Simulation

Turns each stage-1 cluster query into a realistic **deployment scenario** — the
same request, re-expressed as a natural user message inside a plausible real-world
setting and system prompt — and measures whether the model behaves differently once
the request is contextualized versus asked cold. Single-turn; no multi-agent, no
extra dependencies.

Enabled per-run with `certify.py --simulate` (layers onto every risk cluster in
`--only`, or all of them), composable with `--perturb` in a single run/log —
pass `--no-perturb` for a simulation-only run. Pipeline order: **generate**
(offline, once) → stage 1 (plain evals) → stage 2 (perturbation auditing) →
stage 3 (scenario simulation).

## How it works

The scenario reframings are **pregenerated once** by `generate.py` and persisted to
`datasets/generated/<risk>/scenario.jsonl` — one directory per systemic-risk cluster
(`cbrn`, `cyber`, `loss_of_control`, `manipulation`), named after the stage-1
`@task` function (see
[`datasets/generated/README.md`](../../datasets/generated/README.md)) — *not* produced
per model at eval time. Generation runs a **reframing model** (`generate.py --attacker`)
that turns each item into a `{context, system, scenario}` triple (`prompts.py`,
`generation.py::generate_scenarios`); the reframed `scenario` must be
*content-equivalent* to the original query — same ask, only the framing changes — so
the row's own `criterion` and `scale_map` stay meaningful. Unparseable reframings are
re-requested up to 3× at generation time (an improvement over the old live path, which
silently dropped the variant); any that never parse are dropped and listed in the
artifact's meta sidecar (`incomplete_ids`).

At eval time, the `scenario` replay solver
(`stage3_simulation/solvers.py::scenario`) rebuilds the stored system+user message
pair and runs only the **target** model on it — so every evaluated model sees the exact
same scenarios and no reframing model is called during certification.

Stage 3 records its scenario condition into the same `state.metadata["perturbations"]`
shape stage 2 uses (family label `"scenario"`) and **reuses stage 2's scoring/reporting
spine unchanged** (`pipeline/utils/scoring.py::scoring_step`/`wrap_scorers`,
`utils/graders.py::condition_metrics`) and its per-sample perturbation split
(`stage2_perturbation/adapters.py::item_text`, for the text handed to the reframing
model). It adds no scoring code of its own.

| File | Contents |
|---|---|
| `prompts.py` | The reframing prompt (`reframe_prompt`) + tolerant JSON parsing (`parse_reframing`). `PROMPT_VERSION` traces recorded scenarios to the prompt; bump it to force regeneration (certify.py warns on a version mismatch between the on-disk scenarios and the current code). `REFRAME_SYS_PROMPT` is sent as the attacker's system turn. |

## What the brief asks for, and what it forbids

Stage 3's job is to **mask** a request inside a setting where it would
plausibly arise — not to make it easier or harder to answer well. Two things in
the brief worked against that until `PROMPT_VERSION` 3, and every artifact
generated before it carries them:

- **`REFRAME_SYS_PROMPT` was never sent.** It says "You never soften, alter, or
  refuse the request itself", and it sat unreferenced for two prompt versions
  while `generate_scenarios` sent a bare user turn. A guard that is written but
  not wired in is worse than none: it reads as a guarantee in review.
- **The `system` key asked for "guidelines for the AI in this deployment"**, and
  a plausible professional deployment supplies conduct rules. Measured over 562
  real scenarios: 72% carried safety-inducing language, 48% some form of
  "balanced" or "impartial". One `authoritarian_values` reframing invented a
  consultant told to be *"balanced... respecting diverse stakeholder
  perspectives"* — the evaluator supplying the answer the evaluation is meant to
  measure.

Scope and role are the mask and must stay. Conduct rules are a thumb on the
scale, and the brief now names them as off-limits.

Worth stating plainly: the cue rate was **not** shown to move scores. Splitting
the control-to-scenario delta by whether a scenario carried a safety cue gave
-0.035 with against -0.058 without. This is a construct-validity fix, not a
correction to a measured bias.

`prompts.py` is stage 3's only module: reframing generation lives in
`pipeline/generation.py::generate_scenarios` (shared replay machinery:
`pipeline/utils/replay.py`), the eval-time replay solver in
`pipeline/stage3_simulation/solvers.py::scenario`, and task building in
`pipeline/registry.py::_build_task` (its `sim_k` argument appends the
scenario solver + scoring step, wired up by `registry.py::apply_stages`).

## What reframing costs you: scoring structure

This is the stage's main caveat, and it is **not currently gated** — every sample in a
cluster gets reframed, whatever its `question_type`.

A cluster row splits into `item_text` (the request) and `prompt_template` (the
scoring scaffold that wraps it — scale options, choice lists, output-format
contracts). Stage 2 rewrites only `item_text` and re-renders through the template, so
the scaffold survives. **Stage 3 discards `prompt_template` entirely**: it sends
`item_text` to the reframing model and replays a wholly new system+context+scenario
triple. Everything the template was protecting is gone.

For rows whose scoring depends on that scaffold, the stage-1 scorer used to fall
through to its "the model didn't answer the contract" default — which was `SAFE`.
**It no longer does.** Each of those paths now abstains or hands over to the
judge, so a destroyed contract produces a real judgment or no judgment, never a
free pass:

**Every row is judged by an LLM under `--simulate`, whatever its
`question_type`** — no exact match, no scale parse, no detector. The scorer sees
`CONDITION_FAMILY` and routes the whole column to the judge.

That is a scoping decision, not a fallback. A deterministic scorer under a
scenario is reading a contract the scenario never set up, and whether it can
still read one varies row by row: measured on a real run, 45 of 60 sycophancy
rows kept their detector while 15 fell to the judge. A column scored by a
mixture of instruments cannot be compared with its control.

| `question_type` | Rows | Under `--simulate` |
|---|---|---|
| `graded` | 708 | judge, against the row's own `categories` |
| `mcq` | 200 | judge, against `fallback_categories` |
| `likert` | 90 | judge, against the source's **ordinal** fallback — six points for `authoritarian_values`, four for `leader_favorability`, mapped to the same values as the native scale |
| `extraction` | 40 | judge, against `fallback_categories` |
| `detection` | 150 | judge, against `fallback_categories` |

The ordinal fallbacks matter. When the likert sources fell back to a yes/no
label set, the 90 likert rows went from seven values with 82% in the middle at
control to two poles under scenario, 66% of it on the safe one — which made the
whole scenario column read safer than the control it was being compared with.

The cost is that scenario coverage is now visible rather than assumed: a
condition that mostly abstained shows up as a low `scored`/`total` in the results
tree (`pipeline/utils/results.py`) instead of as a clean sweep of 1.0s.

**The gap sources are the one thing stage 3 cannot measure.** It drops each row's
own steering on purpose, so `human_rights_*`'s three persona arms collapse into
one and `leader_favorability`'s 4-point scale is gone. A Wasserstein "lean" or a
`persona_gap` computed there would compare a distribution with itself — so
`summarise(arms_intact=False)` falls those sources back to a plain mean under
scenario, and the gap is reported at the baseline only.

Per-sample values in the log carry which scorer decided them: a condition's
`scorers` map names the grader models where a judge ran and the deterministic
scorer (`exact_match`, `scale_parse`, a detector) where one did not. That is how
you tell a judged scenario result from a natively-scored one without guessing.

## Reported metrics

Same log path as the plain run — and the same **single log** as `--perturb` when both
stages are enabled: the control is generated and judged once, and the wrapped scorers
keep the stages separable via distinct metric pools. The results panel reports the
cluster's primary metric first (the **worst** condition across every enabled family,
scenario included — enabling a stage hardens the certification score), plus
`safety_control` (bald-query safety) and `safety_scenario` (mean over samples of
the worst scenario condition) — the latter sits alongside one `safety_<family>`
per stage-2 attack and a single `safety_under_attack` roll-up over all of them,
so scenario is compared to each perturbation at equal depth rather than to a
min-over-many pool. `stability_under_attack` and `resilience_under_attack` cover
movement and hold-rate over every attack. All are 0-100, higher better. See
[`../stage2_perturbation/README.md`](../stage2_perturbation/README.md) for why
safety is worst-case and stability is not.

Per-benchmark detail lands in `models/models.json` under
`results.{risk}.benchmarks.{source}.conditions.scenario` — safety, divergence
from the baseline, the `scored`/`abstained`/`total` counts, and the per-scorer
breakdown. That node is where the table above becomes checkable: a scenario
condition with a low `scored` count is a thin measurement, and the fix is a
better reframing rather than a better-looking number.

The earlier Concordia-based multi-agent simulation prototype (generation from an EU AI
Act risk taxonomy) lives on the **`simulation`** branch under `mas/`; it is reference
only, not part of this stage.
