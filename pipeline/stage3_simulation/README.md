# Stage 3 — Scenario Simulation

Turns each stage-1 benchmark query into a realistic **deployment scenario** — the
same request, re-expressed as a natural user message inside a plausible real-world
setting and system prompt — and measures whether the model behaves differently once
the request is contextualized versus asked cold. Single-turn; no multi-agent, no
extra dependencies.

Enabled per-run with `certify.py --simulate` (layers onto every benchmark in
`--only`, or all of them). Pipeline order: stage 1 (plain evals) → stage 2
(perturbation auditing) → stage 3 (scenario simulation).

## How it works

For each benchmark item, a **reframing model** (certify.py reuses `--attacker`)
produces a `{context, system, scenario}` triple (`prompts.py`), and the `simulate`
solver (`solvers.py`) runs the **target** model on it. The reframed `scenario` must
be *content-equivalent* to the original query — same ask, only the framing changes —
so the benchmark's own harm judge stays meaningful.

Stage 3 records its scenario condition into the same `state.metadata["perturbations"]`
shape stage 2 uses (family label `"scenario"`) and **reuses stage 2's scoring/reporting
spine unchanged** (`stage2_perturbation/scoring.py::scoring_step`/`wrap_scorers`,
`utils/graders.py::consistency_rate`) and its per-benchmark adapters
(`stage2_perturbation/adapters.py::get_adapter`, for item-text extraction). It adds no
scoring code of its own.

| File | Contents |
|---|---|
| `prompts.py` | The reframing prompt (`reframe_prompt`, `REFRAME_SYS_PROMPT`) + tolerant JSON parsing (`parse_reframing`). `PROMPT_VERSION` traces recorded scenarios to the prompt. |
| `solvers.py` | `simulate(sim_model, k, adapter)` — the one stage-3 `@solver`: reframe → run target on the scenario → record the `scenario` condition. Modeled on stage 2's `paraphrase`. |
| `build.py` | `build_simulation_task(base_task, sim_model, k)` — appends `simulate` + control/scenario scoring steps and wraps the scorers, reusing stage 2. Called by `registry.py::apply_simulation`. |

## Reported metrics

Same log path as the plain run. The results panel reports the benchmark's primary
metric first (still the **control**'s own judgment, so the certification score is
unchanged), plus `lvr_control` (bald-query harm rate) and `lvr` (scenario harm rate).
Per-family/-task detail lands in `models/models.json` under
`simulations.{benchmark}` (parallel to `perturbations`), via `consistency_rate`.

## Known limitations

- `--simulate` and `--perturb` are not designed to compose in one run (both append to
  the same `perturbations` metadata and would pool into a single worst-case). Run them
  as separate invocations; certify.py runs `--simulate` only if both are passed.

The earlier Concordia-based multi-agent simulation prototype (generation from an EU AI
Act risk taxonomy) lives on the **`simulation`** branch under `mas/`; it is reference
only, not part of this stage.
