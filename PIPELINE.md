# Pipeline Overview

Raw benchmarks are sub-sampled into one dataset per Code of Practice taxonomy leaf, an
attacker model (Hermes 70B) generates frozen adversarial variants of every item once, and each
target model is replayed through the same items and variants, judged by an LLM
ensemble, and rolled up to one tail-risk headline per leaf. Grouping leaves into systemic risks is a human
mapping that does not exist yet; nothing below names a risk.

## Data flow

```
datasets/raw/<name>/          fetched upstream files (scripts/fetch_raw.py, manifest.toml)
        │  Source(...) adapters in datasets/prepare/cluster/sources/*.py (one flat registry)
        ▼
datasets/public/<leaf>.csv    prepare.py: exact dedup → leaf similarity → threshold → one pass per leaf
        │  + <leaf>.meta.json (provenance, tier counts) + <leaf>.dropped.jsonl
        ▼
datasets/generated/<leaf>/    generate.py: attacker writes <family>.jsonl + .meta.json
        │                     (paraphrase, register, past_tense, multilingual, framing,
        │                      persona, scenario); committed, replayed for every model
        ▼
models/results/<model>.json   certify.py: Inspect eval_set per leaf → scores/results/status
models/models.json            rebuilt from results/; logs/<model>/<run_id>/ hold .eval logs
```

Everything above the results line is built once and committed. `certify.py` only calls
the target, the judges (`GRADERS.md`) and the stage-3 branch classifier.

## Stage A: leaf datasets (`datasets/prepare/cluster/`)

- **Schema.** A `Source` declares how one benchmark file maps onto the shared row schema
  (`schema.py`): `question_type` drives the stage-1 scorer, `scale_map` fixes polarity
  (scores are 0..1, higher is safer, the criterion names the failure), `families` lists
  which perturbation families apply. The pipeline never branches on a source name; a test
  greps for that.
- **Steps** (`prepare.py::build_leaf`, once per leaf in `sources.LEAVES`): load every source
  annotated with the leaf, exact dedup within and across them → score every row by max
  cosine of its scoring text (system prompt + query, or a declared `scoring_text`) to the
  leaf's anchors (legal text plus exemplars) → per source keep its top `P_MAX (5%) x r` share
  of units, at least 5, but nothing under the leaf's off-topic guard (95th percentile of the
  score over cached items outside the pool) → one pass over the leaf in score order, keeping
  a unit when `score - max(0, maxsim_to_kept - hinge_start)` clears its cut, `hinge_start`
  being the median nearest-neighbour similarity in the cache. Ranks and quantiles only, so
  the rule is embedder-independent. No budget, no strata, no model call. Every tie-break
  hashes `(seed, sample_id)`; the two measurements use a seeded sample. A benchmark
  annotated with several leaves is sampled once per leaf.
- **Leaves and annotations.** `leaves.toml` holds the CoP leaves (legal text, `definition`
  for the 1.3.2 propensities, exemplars, the Label Studio `label`). `annotations.json` (Label
  Studio export) gives each benchmark its leaves: the adjudicated gold annotation (the one
  without a confidence field) when present, else the union over annotators; relevance = mean
  confidence of the scored annotators who chose the leaf x share who chose it, 0.5 for a
  leaf with no confidence behind it (`annotations.py`). `Source.leaves` is the hand fallback
  (CBRN benchmarks, `cysecbench`). The registry (`sources/__init__.py`) resolves this at
  import; `LEAVES` is every leaf a source serves and is the dataset list everywhere
  downstream. Leaf sizes follow pool size x relevance (2,999 rows over 21 leaves, Oct 11).
- **Cache** (`datasets/cache/embeddings/`, gitignored): `items.npz` (every source's queries)
  and `leaves.npz` (anchors), written by `scripts/embed_items.py` in a throwaway env.
  `prepare.py` exits 2 and prints the filling command when an item or anchor is missing.
  Because the cache is not committed, a fresh clone cannot rebuild `public/` byte-identically
  until the embedder is pinned; see `docs/superpowers/plans/2026-10-09-reproducible-sampling.md`.
- **Model aggregate.** With no leaf-to-risk mapping, `models.json`'s headline is a plain mean
  over the leaf datasets (`pipeline/utils/results.py::model_aggregate`).

## Stage B: artifact generation (`generate.py`, `pipeline/generation.py`)

- The attacker for the rewrite families is Hermes-4-405B on OpenRouter (`-a
  openrouter/nousresearch/hermes-4-405b`); scenario trees since Oct 12 2026 come from Grok 4.3
  with reasoning off (`-a openrouter/x-ai/grok-4.3 -M reasoning_enabled=false`), which follows
  the tree format with no parse retries and is several times faster. Hermes-4-70B can instead
  be served by vLLM either on slurm
  (`scripts/generate_hermes_slurm.sh`) or on Modal (`scripts/hermes_modal.py` behind a
  `modal.forward` tunnel, driven by `scripts/generate_hermes_modal.sh`). vLLM is never in
  `uv.lock`.
- A rewrite family rewords only `item_text` and re-renders it through the row's
  `prompt_template`, so the stage-1 scorer keeps its output contract. A failed or
  refused rewrite is stored as `fallback=true` (the original text) so artifact sets stay
  complete. Scenario rows are branching trees (depth 3, branches `refuse`/`hedge`,
  `comply` terminal), `--sim-k` trees per item.
- `pipeline/artifacts.py::PROMPT_VERSIONS` is bumped whenever a generation prompt
  changes; stale versions fail validation. After a dataset rebuild run with `FORCE=1`,
  since `--missing-only` keeps rows for ids that no longer exist.
- Throughput measured Oct 2026 on 4×H100: ~5,000 generation tok/s; a full regeneration
  is 2-5 h; the app bills ~$16/h whether busy or idle, stop it by app id.

## Stage C: certification (`certify.py`, `pipeline/`)

- `registry.py` builds one Inspect task per risk from the CSV and layers the replay
  solvers (stage 2), the scenario walker (stage 3, `--simulate`) and the scoring spine.
  `reconsideration` is the only live perturbation (it challenges the target's own
  control answer).
- Before any eval: artifact validation (coverage per family, prompt versions, no orphan
  rows, strict on scenario), call-count estimate, and a canary call to every judge and
  the classifier. A failure prints the `generate.py` command that fixes it.
- Scoring: each condition is judged to a value in [0, 1] or abstains; abstentions never
  count as 0. Per item `worst` = min over all conditions including control. Per source
  and per risk `tail` = CVaR@10% of per-item worsts over the union of pooled items.
  `scores.<risk>` is that tail, `null` unless the run status is `success`. Definitions:
  `pipeline/README.md § Metrics`.
- Resume: re-issue the same command. `eval_set` skips finished tasks and re-runs only
  missing samples under `logs/<model>/current`; `--rerun` moves that directory aside.
  `--limit` runs never write results. Fleet runs: `scripts/certify_slurm.sh` over
  `scripts/models.txt`.

## External dependencies and secrets

`.env` (never committed): `OPENROUTER_API_KEY` (judges, classifier, API attackers;
reported expired Oct 2026), `VLLM_API_KEY` (bearer token for the Hermes server; the same
value is in the Modal secret `hermes-vllm`), AWS keys, Label Studio key. `HF_TOKEN` is
needed to fetch gated sources (`gpqa_diamond`, `sorry_bench`, `decodingtrust`, `mask`).
Modal volumes `hf-cache` (weights) and `vllm-cache` (compile artifacts) make restarts fast.

## Invariants the tests enforce (`uv run python3 -m unittest discover tests`)

- No source-specific branch in `pipeline/`; every source has a polarity exemplar.
- Cluster builds are deterministic for a fixed cache; the `families` column is present.
- Committed artifacts match the committed CSV ids and the current prompt versions
  (`tests/test_artifacts_current.py`; two checks are `expectedFailure` until the next
  full regeneration).
- Results JSON never contains `-1`; unmeasured is `null`/NaN.

## Where to look first when something breaks

| Symptom | Likely cause | Look at |
|---|---|---|
| `prepare.py` exits 2 | embedding cache miss (items or leaf anchors) | the command it prints; `datasets/cache/{items,leaves}.embed_input.jsonl` |
| importing `datasets.prepare.cluster.sources` raises | `annotations.json` has a label `leaves.toml` lacks, a task with no manifest id, or a source with neither annotation nor `leaves=` | `annotations.py`, `datasets/raw/manifest.toml` |
| `certify.py` refuses to start | stale or incomplete artifacts, CSV older than artifacts, grader canary failed | `artifacts.py::validate_artifacts`, `GRADERS.md`, `.env` keys |
| Many `fallback=true` rows after generation | attacker refusals or a dead endpoint | `datasets/generated/<risk>/<family>.meta.json::incomplete_reasons`, the vLLM log |
| A risk scores `null` | run status `partial`/`failed`, or every judge abstained | `status.<risk>` in the results file, `scripts/reaggregate_from_logs.py` |
| Judges abstain on `loss_of_control` | provider-side content filter on a judge model | `pipeline/utils/retry_policy.py`; swap the judge in `GRADERS.md` |
| Rebuilt CSV differs from the committed one | embeddings regenerated with another library version or device | `<risk>.meta.json` theta per source, `<risk>.dropped.jsonl` |

## Open work (Oct 2026)

- Reproducible sampling plan (commit the embedding cache, provenance): written, not
  implemented; the screen-cache parts are obsolete.
- Sources whose content sits in the system prompt (`persusafety`) score on a constant
  query; scoring should embed system + query for them.
- CBRN leaves have no annotation labels; the CBRN benchmarks keep hand `leaves`.
- Leaf-to-risk mapping: a human decision, not started. Until then there is no per-risk
  dataset, headline or `--only` group.
- Generation prompt fixes (past_tense, loss_of_control scenarios, follow-ups) need a
  prompt-version bump and a full regeneration.
- Sources still unregistered: biolp_bench (docx only), persuasionbench (no tweet text),
  the HF-gated four above.
