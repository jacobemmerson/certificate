# Controlled agentic conditions (C0–C4)

This package evaluates the same 12 stage-1 benchmark tasks under controlled
solver conditions. The dataset and original scorer objects/order remain
unchanged. Agentic conditions are separate from stage-2 perturbation and
stage-3 scenario simulation; those layers cannot be composed or repeated.

## Conditions

| Condition | Behavior |
|---|---|
| **C0 — direct** | Return the original task and solver untouched. |
| **C1 — submission control** | One bounded submit-only final stage. |
| **C2 — self-review** | Target draft, two independent uncached target self-reviews, then a separate submit-only final stage. |
| **C3 — fixed reviewers** | Target draft, one isolated `agentic_analyst` review and one isolated `agentic_critic` review, then the final stage. |
| **C4 — dynamic consultation** | The target chooses zero to two serial analyst/critic calls, finishes consultation with a draft, then enters the separate final stage. |

C1–C4 enforce finite aggregate and per-phase output-token limits, turn limits,
wall time, and working time. The final stage accepts exactly one
`submit(answer)` call. Plain assistant text, drafts, helper output, and the most
recent message are never fallback answers. C4 helper calls are serial; no
external web, shell, file, sandbox, or computer tools are exposed.

## Files

| File | Purpose |
|---|---|
| `config.py` | Immutable conditions, role names, benchmark-aware private metadata policy, and validated finite budgets. |
| `helpers.py` | Isolated analyst/critic agents and authoritative per-phase model configuration. |
| `solvers.py` | Bounded C1–C4 protocol execution, exact submission, structured cancellation, and deterministic empty output on invalid execution. |
| `audit.py` | Content-free pre-scoring process audit: phases, roles, tools, C4 event causality, privacy sentinels, and usage/cost summaries. |
| `build.py` | Field-preserving task adaptation; replaces only the solver and adds copied role/metadata/tag containers. |
| `eval.py` | Generic direct Inspect task entry for any of the 12 tasks under C1–C4. |

The audit is separate from benchmark scoring. It attests protocol structure and
selected leakage boundaries; it does not establish that an answer is correct,
safe, or in a benchmark-required format. The original scorer remains
authoritative.

## Certificate CLI

Clean C0 is the default:

```bash
uv run python certify.py \
  --model openai/gpt-4o-mini \
  --grader openai/gpt-4o \
  --only harm --limit 5
```

Run C4 with explicit helper models:

```bash
uv run python certify.py \
  --model openai/gpt-4o-mini \
  --grader openai/gpt-4o \
  --condition c4 --only harm --limit 5 \
  --analyst-model openrouter/meta-llama/llama-3.1-8b-instruct \
  --critic-model openrouter/google/gemini-3-flash-preview
```

C0 result keys remain `auth`, `harm`, `hist`, and `hr`. C1–C4 use
non-overwriting keys such as `harm_agentic_c1` … `harm_agentic_c4`.

Use `--sample-seed`, `--sample-ids-out`, and `--sample-ids-in` to reuse exact
canonical IDs across conditions. Incomplete ID coverage stays log-only. Full
coverage persists `agentic_config`, aggregate process validity, and available
per-phase/per-role usage and cost summaries in `models/models.json`. Benchmark
scorers still run for every sample and remain visible in Inspect logs, but
`certify.py` recomputes C1-C4 reports from protocol-valid samples only. Invalid
and missing-audit counts are persisted rather than mixed into comparable
condition aggregates.

## Direct Inspect entry

```bash
uv run inspect eval pipeline/agentic/eval.py@agentic \
  --model openai/gpt-4o-mini \
  --model-role agentic_analyst=openrouter/meta-llama/llama-3.1-8b-instruct \
  --model-role agentic_critic=openrouter/google/gemini-3-flash-preview \
  -T base_task=fscale \
  -T condition=c3 \
  -T grader=openai/gpt-4o \
  --limit 1
```

Supported `base_task` values:

```text
fscale
leader_favorability
role_model_bias
social_harm_bench
history_no_push
history_explicit_push
udhr
udhr_individual
udhr_government
echr
echr_individual
echr_government
```

For C0, invoke the corresponding task under
`pipeline/stage1_evaluation/evals/` or use `certify.py`.

## Logs and privacy

The persisted audit is sanitized and excludes prompt/helper/submission text.
Raw Inspect logs still contain full target prompts, drafts, reviews, helper
arguments/results, submissions, provider events, and scorer interactions. Treat
those logs as sensitive and do not publish them without review.
