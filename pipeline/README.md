# `pipeline/` — the certification pipeline

Everything the certification run (`certify.py`) executes lives here,
organized by pipeline stage. To add a new benchmark, see
**[CONTRIBUTE.md](../CONTRIBUTE.md)** at the repo root; datasets live in
**[`datasets/`](../datasets/README.md)**.

## Stages

| Directory | Stage | Contents |
|---|---|---|
| `stage1_evaluation/` | 1 — plain evals | `evals/` has one file per benchmark: a `csv_dataset` + `@task` definition(s), with shared CSV→`Sample`/scorer-assembly helpers in `evals/common.py`. `scorers/harm.py` holds the reusable harm classifiers (LLM judge, LlamaGuard). |
| `stage2_perturbation/` | 2 — perturbation auditing | Surface-perturbation reliability auditing: solvers that generate perturbed variants (paraphrase, register, identity_strip, framing, reconsideration), per-benchmark adapters, and the scoring wrappers that report consistency/LVR (legal violation rate). Layered onto stage-1 tasks by `registry.py::apply_perturbations` (`--perturb`). |
| `stage3_simulation/` | 3 — scenario simulation | Single-turn scenario reframing: a `simulate` solver reframes each benchmark item into a realistic deployment scenario (system prompt + context + the request as a natural user message) and re-runs the target on it. Layered onto stage-1 tasks by `registry.py::apply_simulation` (`--simulate`); reuses stage 2's scoring spine, reporting the `scenario` family's harm rate (`lvr`) vs. the bald-query baseline (`lvr_control`). |

## Shared

| File | Contents |
|---|---|
| `registry.py` | `init_benchmarks()` — the master list mapping benchmark keys to their `@task`s; `apply_perturbations()` layers the stage-2 perturbation families on top, `apply_simulation()` layers stage-3 scenario simulation on top. |
| `utils/graders.py` | Grader/model loading (`GRADERS.md`, `models/models.json`) and score aggregation. |
| `utils/text.py` | Shared text heuristics (refusal detection). |

The adversarial attack suite (jailbreak attacks, attack-retry solver,
multi-classifier harm scoring) lives on the **`adversarial-attacks`** branch.

## Data flow

`datasets/raw/<source>/` → `datasets/prepare/prepare_*.py` →
`datasets/public/*.csv` → `pipeline/stage1_evaluation/evals/*.py` (`@task`) →
`pipeline/registry.py` (`init_benchmarks`, optionally `apply_perturbations`) →
`certify.py` → `models/models.json`.
