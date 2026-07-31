# Stage 3 — Scenario Simulation

Turns each stage-1 benchmark query into a realistic **deployment scenario** — the
same request, re-expressed as a natural user message inside a plausible real-world
setting and system prompt — and measures whether the model behaves differently once
the request is contextualized versus asked cold. Single-turn; no multi-agent, no
extra dependencies.

Enabled per-run with `certify.py --simulate` (layers onto every benchmark in
`--only`, or all of them), composable with `--perturb` in a single run/log —
pass `--no-perturb` for a simulation-only run. Pipeline order: **generate**
(offline, once) → stage 1 (plain evals) → stage 2 (perturbation auditing) →
stage 3 (scenario simulation).

## How it works

The scenario reframings are **pregenerated once** by `generate.py` and persisted to
`datasets/generated/<task>/scenario.jsonl` (see
[`datasets/generated/README.md`](../../datasets/generated/README.md)), *not* produced
per model at eval time. Generation runs a **reframing model** (`generate.py --attacker`)
that turns each item into a `{context, system, scenario}` triple (`prompts.py`,
`generation.py::generate_scenarios`); the reframed `scenario` must be
*content-equivalent* to the original query — same ask, only the framing changes — so
the benchmark's own harm judge stays meaningful. Unparseable reframings are re-requested
up to 3× at generation time (an improvement over the old live path, which silently
dropped the variant); any that never parse are dropped and listed in the artifact's
meta sidecar (`incomplete_ids`).

At eval time, the `scenario` replay solver
(`stage3_simulation/solvers.py::scenario`) rebuilds the stored system+user message
pair and runs only the **target** model on it — so every evaluated model sees the exact
same scenarios and no reframing model is called during certification.

Stage 3 records its scenario condition into the same `state.metadata["perturbations"]`
shape stage 2 uses (family label `"scenario"`) and **reuses stage 2's scoring/reporting
spine unchanged** (`pipeline/utils/scoring.py::scoring_step`/`wrap_scorers`,
`utils/graders.py::condition_metrics`) and its per-benchmark adapters
(`stage2_perturbation/adapters.py::get_adapter`, for item-text extraction). It adds no
scoring code of its own.

| File | Contents |
|---|---|
| `prompts.py` | The reframing prompt (`reframe_prompt`, `REFRAME_SYS_PROMPT`) + tolerant JSON parsing (`parse_reframing`). `PROMPT_VERSION` traces recorded scenarios to the prompt; bump it to force regeneration (certify.py warns on a version mismatch between the on-disk scenarios and the current code). |

`prompts.py` is stage 3's only module: reframing generation lives in
`pipeline/generation.py::generate_scenarios` (shared replay machinery:
`pipeline/utils/replay.py`), the eval-time replay solver in
`pipeline/stage3_simulation/solvers.py::scenario`, and task building in
`pipeline/registry.py::_build_task` (its `sim_k` argument appends the
scenario solver + scoring step, wired up by `registry.py::apply_stages`).

## Reported metrics

Same log path as the plain run — and the same **single log** as `--perturb` when both
stages are enabled: the control is generated and judged once, and the wrapped scorers
keep the stages separable via distinct metric pools. The results panel reports the
benchmark's primary metric first (the **worst** condition across every enabled family,
scenario included — enabling a stage hardens the certification score), plus
`lvr_control` (bald-query harm rate), `lvr_scenario` (scenario harm rate), and
`consistency_scenario` (how often the scenario judgment matches the control's) — next
to stage 2's `harm_propensity`/`stability` when it also ran. Per-family/-task detail lands in
`models/models.json` under `simulations.{benchmark}` (parallel to `perturbations`),
via `condition_metrics(..., families={"scenario"})`.

The earlier Concordia-based multi-agent simulation prototype (generation from an EU AI
Act risk taxonomy) lives on the **`simulation`** branch under `mas/`; it is reference
only, not part of this stage.
