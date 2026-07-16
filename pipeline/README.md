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
| `stage3_simulation/` | 3 — scenario simulation | Single-turn scenario reframing: a `simulate` solver reframes each benchmark item into a realistic deployment scenario and re-runs the target. Layered onto direct stage-1 tasks by `registry.py::apply_simulation` (`--simulate`); reuses stage 2's scoring spine. |
| `agentic/` | Separate C0–C4 condition layer | Typed finite protocol config, isolated analyst/critic roles, exact-submit solvers, sanitized process audits, field-preserving task construction, and the direct `eval.py@agentic` entry. C0 is untouched; C1–C4 replace only the solver and preserve original scorer objects/order. |

## Shared

| File | Contents |
|---|---|
| `registry.py` | The 12-task master registry; C0–C4 construction with non-overwriting `_agentic_cN` keys; deterministic canonical-ID pairing and exact coverage checks; stage-2/stage-3 condition construction. |
| `utils/graders.py` | Grader/model loading (`GRADERS.md`, `models/models.json`) and score/condition aggregation. |
| `utils/errors.py` | Safe exception rendering for persisted/logged summaries. |
| `utils/task_transforms.py` | Canonical task identity plus agentic/perturbation/simulation markers and composition guards. |
| `utils/text.py` | Shared text heuristics (refusal detection). |

The adversarial attack suite (jailbreak attacks, attack-retry solver,
multi-classifier harm scoring) lives on the **`adversarial-attacks`** branch.

## Data flow

`datasets/raw/<source>/` → `datasets/prepare/prepare_*.py` →
`datasets/public/*.csv` → `pipeline/stage1_evaluation/evals/*.py` (`@task`) →
`pipeline/registry.py` (`init_benchmarks`, optionally one of C1–C4,
perturbation, or simulation) → `certify.py` → `models/models.json`.

## Condition boundaries

With no condition flags, `certify.py` runs clean C0. `--perturb` and
`--simulate` are separate C0 layers. C1–C4 are selected with `--condition` and
use the 12 standard tasks only. Agentic, perturbation, and simulation builders
mark transformed tasks and reject every cross-layer order and repeated
application through both registry and direct APIs.

C1–C4 expose no external tools. Their internal analyst, critic, consultation,
and submit actions are protocol controls. Raw Inspect logs contain sensitive
full interactions; persisted agentic audits contain sanitized structure and
usage/cost summaries only. See [`agentic/README.md`](agentic/README.md).
