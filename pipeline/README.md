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
| `stage2_perturbation/` | 2 — perturbation auditing | Surface-perturbation reliability auditing: replay solvers that run the target on pregenerated perturbed variants (paraphrase, register, identity_strip, framing) plus the live `reconsideration` family, per-benchmark adapters, and the scoring wrappers that report consistency/LVR (legal violation rate). Layered onto stage-1 tasks by `registry.py::apply_stages` (`--perturb`). |
| `stage3_simulation/` | 3 — scenario simulation | Single-turn scenario reframing: the target is re-run on pregenerated deployment-scenario reframings (system prompt + context + the request as a natural user message). Layered onto stage-1 tasks by `registry.py::apply_stages` (`--simulate`, composable with `--perturb`); reuses stage 2's scoring spine and its `scenario` replay solver, reporting the scenario harm rate (`lvr_scenario`) and stability (`consistency_scenario`) vs. the bald-query baseline (`lvr_control`). |

## Shared

| File | Contents |
|---|---|
| `generation.py` | Offline generators (the attacker-model half of stages 2/3): `generate_rewrites`, `generate_framing`, `generate_scenarios`, and `SampleView` (renders per-benchmark prompts offline via the stage-2 adapters). Driven by `generate.py`. |
| `artifacts.py` | The `datasets/generated/` store: `write_family`/`load_family`, prompt-version tracking, and `validate_artifacts` (certify.py's fail-fast pre-run check). |
| `registry.py` | `init_benchmarks()` — the master list mapping benchmark keys to their `@task`s; `apply_stages()` layers the stage-2 perturbation families and/or stage-3 scenario simulation on top in a single pass. `PREGENERATED_FAMILIES` names the families with artifacts. |
| `utils/graders.py` | Grader/model loading (`GRADERS.md`, `models/models.json`) and score aggregation. |
| `utils/text.py` | Shared text heuristics (refusal detection). |

The adversarial attack suite (jailbreak attacks, attack-retry solver,
multi-classifier harm scoring) lives on the **`adversarial-attacks`** branch.

## Data flow

`datasets/raw/<source>/` → `datasets/prepare/prepare_*.py` →
`datasets/public/*.csv` → `pipeline/stage1_evaluation/evals/*.py` (`@task`) →
`pipeline/registry.py` (`init_benchmarks`) → `certify.py` → `models/models.json`.

For stages 2/3 the perturbed prompts branch off once, offline:
`datasets/public/*.csv` → `generate.py` (attacker/reframing model) →
`datasets/generated/<task>/<family>.jsonl` → replayed by `certify.py` against
every target model.

> **Semantic note.** Because variants are now fixed across runs, the consistency/LVR
> metrics isolate *target* variance: every model (and every epoch) is judged on the
> identical set of perturbed prompts, so differences reflect the models, not the luck
> of the attacker's draw.
