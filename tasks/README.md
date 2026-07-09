# `tasks/` — evaluation pipeline

Everything the certification run (`certify.py`) executes lives here. To add a
new benchmark, see **[CONTRIBUTE.md](../CONTRIBUTE.md)** at the repo root.

## Layout

| Directory | Contents |
|---|---|
| `evals/` | One file per benchmark: a `csv_dataset` + `@task` definition(s). `common.py` holds the shared CSV→`Sample` and scorer-assembly helpers. |
| `data/` | One-off `prepare_*.py` scripts that flatten each source benchmark into a CSV under `benchmarks/datasets/public/`. Run once before evaluating. |
| `scorers/` | Reusable `@scorer`s — `harm.py` has the harm classifiers (LLM judge, LlamaGuard, StrongREJECT, HarmBench, multi-classifier majority vote). |
| `solvers/` | Custom solvers, e.g. `adversarial.py`'s attack-retry loop. |
| `attacks/` | Individual jailbreak attacks used by the adversarial solver. |
| `perturb/` | Surface-perturbation reliability auditing (see [PERTURB.MD](../PERTURB.MD)). |
| `utils/` | `graders.py` — grader/model loading and score aggregation. |
| `benchmarks.py` | `init_benchmarks()` — the master list mapping benchmark keys to their `@task`s; `apply_perturbations()` layers the perturbation families on top. |

## Data flow

`benchmarks/<source>/` (raw) → `tasks/data/prepare_*.py` →
`benchmarks/datasets/public/*.csv` → `tasks/evals/*.py` (`@task`) →
`tasks/benchmarks.py` (`init_benchmarks`) → `certify.py` →
`models/models.json`.
