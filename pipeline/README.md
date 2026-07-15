# `pipeline/` — the certification pipeline

Everything the certification run (`certify.py`) executes lives here,
organized by pipeline stage. To add a new benchmark, see
**[CONTRIBUTE.md](../CONTRIBUTE.md)** at the repo root; datasets live in
**[`datasets/`](../datasets/README.md)**.

## Generate once, evaluate many

The perturbed variants and reframed scenarios are **generated once**, offline, by
[`generate.py`](../generate.py) (running an attacker/reframing model) and persisted to
[`datasets/generated/`](../datasets/generated/README.md). `certify.py` then **replays**
those fixed artifacts against each target model — so every evaluated model sees the
*exact same* variants (fair, apples-to-apples) and the attacker runs once total instead
of once per model (cheaper). The only models `certify.py` calls are the target and the
judges.

The one exception is the `reconsideration` family, which challenges the target's *own*
control completion — it has no artifact and still runs live during the eval.

```
# once per artifact refresh:
uv run python generate.py [--attacker MODEL] [--perturb …] [--simulate] [--only …]
# then per candidate model (cheap, no attacker calls):
uv run python certify.py -m <target> [--perturb …] [--simulate] [--only …]
```

`--perturb` and `--simulate` compose in `certify.py` exactly as they do in
`generate.py`: one task run and **one eval log per benchmark**, in which the
control (bald query) is generated and judged once and both stages' conditions
are scored. The two stages stay legible inside that log via separate metric
pools — `lvr`/`consistency` for the stage-2 families, `lvr_scenario`/
`consistency_scenario` for stage 3 — and land in separate `models.json`
sections (`perturbations` / `simulations`). The certification score is the
worst condition across every enabled family. Use `--no-perturb` for a
simulation-only run (stage 2 is on by default).

`certify.py` validates the artifacts before any eval starts and fails fast with the
exact `generate.py` command if any are missing (`pipeline/artifacts.py::validate_artifacts`).

## Stages

| Directory | Stage | Contents |
|---|---|---|
| `stage1_evaluation/` | 1 — plain evals | `evals/` has one file per benchmark: a `csv_dataset` + `@task` definition(s), with shared CSV→`Sample`/scorer-assembly helpers in `evals/common.py`. `scorers/harm.py` holds the reusable harm classifiers (LLM judge, LlamaGuard). |
| `stage2_perturbation/` | 2 — perturbation auditing | Surface-perturbation reliability auditing: one solver per family — replaying pregenerated perturbed variants (paraphrase, register, identity_strip, framing) plus the live `reconsideration` family — with the per-benchmark adapters, framing templates, and rewrite prompts. Layered onto stage-1 tasks by `registry.py::apply_stages` (`--perturb`); reports consistency/LVR (legal violation rate) via the shared scoring spine in `utils/scoring.py`. |
| `stage3_simulation/` | 3 — scenario simulation | Single-turn scenario reframing: the `scenario` solver replays pregenerated deployment-scenario reframings (system prompt + context + the request as a natural user message), plus the reframing prompt/parsing (`prompts.py`). Layered onto stage-1 tasks by `registry.py::apply_stages` (`--simulate`, composable with `--perturb`); reports the scenario harm rate (`lvr_scenario`) and stability (`consistency_scenario`) vs. the bald-query baseline (`lvr_control`) via the same shared scoring spine. |
| `stage4_aggregation/` | 4 — cross-benchmark aggregation | Bradley–Terry synthesis of the per-benchmark results in `models.json`: pressure-resistance and steering-robustness constructs (plus a coverage-gated stage-2/3 conditional-robustness construct), DAB guardrail percentiles, and a 24-specification robustness analysis. Each model gets a 0–100 score, a 0–4 GPA, and a cohort rank. Run manually via [`aggregate.py`](../aggregate.py) (like `generate.py`, not part of `certify.py`); writes `analysis/benchmark_aggregation/` and the per-model `bt` block back into `models.json`. See its [README](stage4_aggregation/README.md). |

## Shared

| File | Contents |
|---|---|
| `generation.py` | Offline generators (the attacker-model half of stages 2/3): `generate_rewrites`, `generate_framing`, `generate_scenarios`, and `SampleView` (renders per-benchmark prompts offline via the stage-2 adapters). Driven by `generate.py`. |
| `artifacts.py` | The `datasets/generated/` store: `write_family`/`load_family`, prompt-version tracking, and `validate_artifacts` (certify.py's fail-fast pre-run check). |
| `registry.py` | `init_benchmarks()` — the master list mapping benchmark keys to their `@task`s; `apply_stages()` layers the stage-2 perturbation families and/or stage-3 scenario simulation on top in a single pass (its `_build_task` assembles each Task's solver chain from the stages' family solvers). `PREGENERATED_FAMILIES` names the families with artifacts. |
| `utils/scoring.py` | The condition-family scoring spine both stages share: `scoring_step`/`wrap_scorers`, the `lvr*`/`consistency*` metric pools, and scorer polarity (`SCORER_POLARITY`, `badness`/`is_failing`). |
| `utils/replay.py` | The shared replay machinery behind both stages' solvers: `replay` (run the target on stored artifact rows, cache-off, on scratch state copies), `generate_variant` retries, `truncated`. |
| `utils/graders.py` | Grader/model loading (`GRADERS.md`, `models/models.json`) and score aggregation (`aggregate_score`, `consistency_rate`). |
| `utils/text.py` | Shared text heuristics (refusal detection). |

The adversarial attack suite (jailbreak attacks, attack-retry solver,
multi-classifier harm scoring) lives on the **`adversarial-attacks`** branch.

## Data flow

`datasets/raw/<source>/` → `datasets/prepare/prepare_*.py` →
`datasets/public/*.csv` → `pipeline/stage1_evaluation/evals/*.py` (`@task`) →
`pipeline/registry.py` (`init_benchmarks`) → `certify.py` → `models/models.json` →
`aggregate.py` (`pipeline/stage4_aggregation`) → `analysis/benchmark_aggregation/` +
per-model `bt` blocks back in `models/models.json`.

For stages 2/3 the perturbed prompts branch off once, offline:
`datasets/public/*.csv` → `generate.py` (attacker/reframing model) →
`datasets/generated/<task>/<family>.jsonl` → replayed by `certify.py` against
every target model.

> **Semantic note.** Because variants are now fixed across runs, the consistency/LVR
> metrics isolate *target* variance: every model (and every epoch) is judged on the
> identical set of perturbed prompts, so differences reflect the models, not the luck
> of the attacker's draw.
