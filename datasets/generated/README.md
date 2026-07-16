# `datasets/generated/` — the frozen perturbation/simulation test set

These are the pregenerated stage-2 perturbation variants and stage-3 scenario
reframings that [`certify.py`](../../certify.py) replays against every target
model. They are produced **once** by [`generate.py`](../../generate.py) (running
an attacker/reframing model) instead of per-model at eval time, so every
evaluated model sees the *exact same* variants and the attacker cost is paid
once — see [`pipeline/README.md`](../../pipeline/README.md).

**Commit these files.** Like `datasets/public/`, they are the fixed evaluation
input; regenerating them changes what every model is scored against.

## Layout

```
datasets/generated/
  <task_name>/                 # the @task registry name, e.g. social_harm_bench, fscale
    paraphrase.jsonl           # rewrite family — attacker-reworded item
    register.jsonl             # rewrite family — register/style shift
    identity_strip.jsonl       # rewrite family — benchmark-fingerprint removal
    framing.jsonl              # deterministic elicitation-format wrappers (no model)
    scenario.jsonl             # stage-3 deployment-scenario reframings
    <family>.meta.json         # provenance sidecar per family file
```

`<task_name>` is the `@task` function's registry name (what
`pipeline/artifacts.py::task_name` returns), **not** the benchmark key. Renaming
a `@task` orphans its artifact directory — regenerate under the new name.

`framing.jsonl` is only written for benchmarks whose elicitation format has
framing templates (compliance/judgment/opinion); benchmarks with a generic,
open-ended elicitation (e.g. `role_model_bias`) have no framing file, by design.

There is no `reconsideration.jsonl`: that family challenges the target's own
answer, so it can only run live inside `certify.py`.

## Row schema (one JSON object per line)

Rewrite families (`paraphrase`, `register`, `identity_strip`):

```json
{"id": "<sample_id>", "variant": 1, "condition": "paraphrase_variant_1",
 "text": "<reworded item text>", "query": "<full prompt sent to the target>",
 "fallback": false}
```

`fallback: true` means the attacker refused or returned nothing and the original
text was kept — persisted anyway so the set stays complete and every model still
sees the same fixed prompt.

Framing (`framing.jsonl`): `{"id", "condition": "framing_<name>", "query"}` —
deterministic, no `text`/`variant`/`fallback`.

Scenario (`scenario.jsonl`):

```json
{"id": "<sample_id>", "variant": 1, "condition": "scenario_variant_1",
 "context": "...", "system": "...", "scenario": "...",
 "query": "[system] ...\n\n[user] <context>\n\n<scenario>"}
```

`query` is the exact prompt scoring surfaces as `worst_query`; the replay solver
rebuilds the real `system` + `user` messages from the separate fields.

Each `<family>.meta.json` records provenance: `generator_model`,
`prompt_version`, `k`, `created_at`, `git_commit`, `num_samples`,
`num_variants`, `incomplete_ids` (scenario samples whose reframing never
parsed), and `partial` (true if generated with `--limit`).

## Regenerating

```bash
# everything, all benchmarks (default attacker):
uv run python generate.py

# one benchmark, specific families, k variants:
uv run python generate.py --only harm --perturb paraphrase framing --perturb-k 3
uv run python generate.py --only harm --simulate --sim-k 1

# fill gaps only (e.g. scenario reframings that failed to parse), keep the rest:
uv run python generate.py --missing-only

# rebuild from scratch, overwriting:
uv run python generate.py --force
```

By default `generate.py` skips a family whose file already exists; use
`--missing-only` to top up or `--force` to overwrite. Bumping a prompt version
(`rewrite.py::REWRITE_PROMPT_VERSION`, `framing.py::FRAMING_VERSION`,
`prompts.py::PROMPT_VERSION`) makes `certify.py` warn that the on-disk artifacts
are stale — regenerate with `--force`.
