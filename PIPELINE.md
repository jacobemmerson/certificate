# Pipeline Overview

Raw benchmarks are sub-sampled into one dataset per EU AI Act systemic risk, an
attacker model (Hermes 70B) generates frozen adversarial variants of every item once, and each
target model is replayed through the same items and variants, judged by an LLM
ensemble, and rolled up to one tail-risk headline per risk.

## Data flow

```
datasets/raw/<name>/          fetched upstream files (scripts/fetch_raw.py, manifest.toml)
        │  Source(...) adapters in datasets/prepare/cluster/sources/<risk>.py
        ▼
datasets/public/<risk>.csv    prepare.py: dedup → relevance → budget → strata → screen
        │  + <risk>.meta.json (provenance, tier counts) + <risk>.dropped.jsonl
        ▼
datasets/generated/<risk>/    generate.py: attacker writes <family>.jsonl + .meta.json
        │                     (paraphrase, register, past_tense, multilingual, framing,
        │                      persona, scenario); committed, replayed for every model
        ▼
models/results/<model>.json   certify.py: Inspect eval_set per risk → scores/results/status
models/models.json            rebuilt from results/; logs/<model>/<run_id>/ hold .eval logs
```

Everything above the results line is built once and committed. `certify.py` only calls
the target, the judges (`GRADERS.md`) and the stage-3 branch classifier.

## Stage A: cluster datasets (`datasets/prepare/cluster/`)

- **Schema.** A `Source` declares how one benchmark file maps onto the shared row schema
  (`schema.py`): `question_type` drives the stage-1 scorer, `scale_map` fixes polarity
  (scores are 0..1, higher is safer, the criterion names the failure), `families` lists
  which perturbation families apply. The pipeline never branches on a source name; a test
  greps for that.
- **Tiers** (`prepare.py::build_risk`): exact dedup → near dedup (MiniLM cosine ≥ 0.92) →
  cross-source exact and near → relevance filter → water-fill budget over sources
  (`BUDGET` per risk module, `Source.quota` override) → strata → windowed pre-select
  (SCREEN_FACTOR 3.5 × share) → Hermes answerability screen (drop what an open model
  refuses) → fill. Every tie-break hashes `(seed, sample_id)`; no RNG.
- **Relevance tier** (provisional, Oct 2026): each `Source.leaf` names a legal-group leaf
  in `leaves.toml` (CoP Appendix 1.3/1.4 text plus exemplars). Score = max cosine to the
  leaf's anchors; threshold = leave-one-out p10 of the exemplars; a leaf with fewer than
  two exemplars passes everything through. `Source.relevance` r sets how far down the
  ranking selection may reach. On MiniLM the thresholds rarely bind. `definition` fields
  on the 1.3.2 leaves and `scripts/generate_leaf_exemplars.py` exist to grow the exemplar
  sets; the annotators own leaf assignment and exemplars.
- **Caches** (`datasets/cache/`, gitignored): `embeddings/<risk>.npz` and `leaves.npz`
  (written by `scripts/embed_items.py` in a throwaway env), `screen/<risk>.jsonl`
  (verdicts). `prepare.py` exits 2 and prints the filling command when either is missing;
  it never builds on a partial cache. Because the caches are not committed, a fresh clone
  cannot rebuild `public/` byte-identically; the fix is the unimplemented plan
  `docs/superpowers/plans/2026-10-09-reproducible-sampling.md`.
- **Clusters.** `cbrn`, `cyber`, `loss_of_control`, `manipulation` are headline risks;
  `alignment` is provisional (`SYSTEMIC = False`) and excluded from the model aggregate.

## Stage B: artifact generation (`generate.py`, `pipeline/generation.py`)

- The attacker is Hermes-4-70B, served by vLLM either on slurm
  (`scripts/generate_hermes_slurm.sh`) or on Modal (`scripts/hermes_modal.py` behind a
  `modal.forward` tunnel, driven by `scripts/generate_hermes_modal.sh`). Both wrappers
  run the screen loop first (screen → prepare until exit 0), then generation. vLLM is
  never in `uv.lock`.
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
| `prepare.py` exits 2 | cache miss (embeddings or screen) | the command it prints; `datasets/cache/*.{embed,screen}_input.jsonl` |
| `certify.py` refuses to start | stale or incomplete artifacts, CSV older than artifacts, grader canary failed | `artifacts.py::validate_artifacts`, `GRADERS.md`, `.env` keys |
| Many `fallback=true` rows after generation | attacker refusals or a dead endpoint | `datasets/generated/<risk>/<family>.meta.json::incomplete_reasons`, the vLLM log |
| A risk scores `null` | run status `partial`/`failed`, or every judge abstained | `status.<risk>` in the results file, `scripts/reaggregate_from_logs.py` |
| Judges abstain on `loss_of_control` | provider-side content filter on a judge model | `pipeline/utils/retry_policy.py`; swap the judge in `GRADERS.md` |
| Rebuilt CSV differs from the committed one | caches regenerated (embeddings or screen are not reproducible) | `<risk>.meta.json`, `<risk>.dropped.jsonl` tier counts |

## Open work (Oct 2026)

- Reproducible sampling plan (commit caches, float64 cosine, provenance): written, not
  implemented, two decisions pending (plain git vs LFS for 78 MB of embeddings; keep
  existing screen verdicts).
- Relevance tier calibration: exemplars from several benchmarks per leaf, a stronger
  embedder behind an AUC test, then rebuild and re-screen newly surfaced candidates.
- Generation prompt fixes (past_tense, loss_of_control scenarios, follow-ups) need a
  prompt-version bump and a full regeneration.
- Sources still unregistered: biolp_bench (docx only), persuasionbench (no tweet text),
  the HF-gated four above.
