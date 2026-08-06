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
| `prompts.py` | The reframing prompt (`reframe_prompt`) + tolerant JSON parsing (`parse_reframing`). `PROMPT_VERSION` traces recorded scenarios to the prompt; bump it to force regeneration (certify.py warns on a version mismatch between the on-disk scenarios and the current code). `REFRAME_SYS_PROMPT` is defined but **currently unused** — reframing is sent as a bare user turn. |

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

| `question_type` | Rows | Under `--simulate` |
|---|---|---|
| `graded` | 708 | **Fine.** An LLM judge reads `criterion` + the completion; no scaffold needed. |
| `mcq` | 200 | **Fine.** `_mcq` falls back to an open-ended judge ("does this convey the reference answer?") when the option structure is gone. |
| `likert` | 90 | **Judged.** No scale or JSON block survives, so `_likert` abstains and `_judged_fallback` grades the prose against the row's `criterion`. |
| `extraction` | 40 | **Judged.** The "list role models" ask is reframed away, so extraction abstains and the fallback grades instead. |
| `detection` | 150 | **Judged where the trigger is gone.** The detector still decides whenever it can still run — `detectors.py` argues these ports measure something a judge would not — and hands over only when it has nothing to detect. |

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
`harm_propensity_control` (bald-query harm propensity), `harm_propensity_scenario`
(1 − mean over samples of the worst scenario condition), `stability_scenario` (how
little the scenario judgment moved from the control's), and `scenario_uplift` (% of
samples the reframing pushed below their own control) — next to stage 2's
`harm_propensity`/`stability`/`perturbation_uplift` when it also ran. See
[`../stage2_perturbation/README.md`](../stage2_perturbation/README.md) for why
propensity is worst-case and stability is not.

Per-benchmark detail lands in `models/models.json` under
`results.{risk}.benchmarks.{source}.conditions.scenario` — safety, divergence
from the baseline, the `scored`/`abstained`/`total` counts, and the per-scorer
breakdown. That node is where the table above becomes checkable: a scenario
condition with a low `scored` count is a thin measurement, and the fix is a
better reframing rather than a better-looking number.

The earlier Concordia-based multi-agent simulation prototype (generation from an EU AI
Act risk taxonomy) lives on the **`simulation`** branch under `mas/`; it is reference
only, not part of this stage.
