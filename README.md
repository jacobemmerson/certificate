<div align="center">
  <img style="height: 168" src="media/logo.png" alt="EuroSafeAI Logo">

  <h4>[<a href="https://eurosafe.ai.toronto.edu">Homepage</a>]</h4>
</div>

# AI Safety Benchmarks & Certification

The pipeline and datasets for certifying frontier models against the EU AI Act Code of
Practice taxonomy, one dataset per taxonomy leaf (`datasets/prepare/cluster/leaves.toml`): each
leaf dataset is a subset of the benchmarks annotated with that leaf, hardened by frozen surface
perturbations (stage 2) and a multi-turn deployment scenario (stage 3), judged by a two-model
ensemble, and rolled up to one headline per leaf. Grouping leaves into systemic risks is a
human mapping still to come. One-page orientation on the conditions and metrics:
[`docs/conditions-and-metrics.md`](docs/conditions-and-metrics.md). How it is built and what the
numbers mean: [`pipeline/README.md`](pipeline/README.md). What is in the data:
[`datasets/BENCHMARKS.md`](datasets/BENCHMARKS.md). Adding a benchmark:
[CONTRIBUTE.md](CONTRIBUTE.md).

## Setup

Built on the AISI [Inspect](https://inspect.aisi.org.uk) framework. You need an API key for
a [supported provider](https://inspect.aisi.org.uk/providers.html) in the environment, and
[`uv`](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv sync                                     # locked deps; ML libraries are deliberately not in here
# raw benchmarks, only needed to rebuild datasets: see datasets/README.md for the fetch_raw.py call
```

Judges are listed in [`GRADERS.md`](GRADERS.md). Hermes-4-70B (the attacker) is served
by vLLM through `uvx` or on Modal, never installed into the project env.

## Three scripts

| Script | Runs | Writes |
|---|---|---|
| `datasets/prepare/cluster/prepare.py` | once per dataset change; two passes (below) | `datasets/public/<leaf>.csv` + `.meta.json` + `.dropped.jsonl`, committed |
| `generate.py` | once per artifact refresh, attacker model on slurm or Modal | `datasets/generated/<leaf>/<family>.jsonl` + `.meta.json`, committed |
| `certify.py` | once per target model; target + judges + classifier only | `models/results/<model_id>.json`, `models/models.json` (rebuilt), `logs/<model_id>/<run_id>/` |

## Recipes

### Build the leaf datasets (two passes)

`prepare.py` exits 2 whenever the embedding cache lacks an item or a leaf anchor and prints
the exact command that fills it. A CSV is never built on a partial cache. No model is called:
rows are chosen by similarity to the leaf exemplars in `datasets/prepare/cluster/leaves.toml`.

```bash
uv run python3 -m datasets.prepare.cluster.prepare        # exit 2: embeddings missing
uv run --no-project --with sentence-transformers --with numpy python scripts/embed_items.py
uv run python3 -m datasets.prepare.cluster.prepare        # every leaf a source serves
```

Review `datasets/public/<leaf>.dropped.jsonl` (`threshold` and `redundant` tiers) and the
per-source `theta` / `floor_used` in `<leaf>.meta.json` before committing.
`--dry-run` prints the per-source table without writing; `--leaf <id>` builds one leaf. Detail: [`datasets/BENCHMARKS.md § Sampling`](datasets/BENCHMARKS.md#sampling).

### Generate the artifacts

```bash
sbatch scripts/generate_hermes_slurm.sh      # vLLM Hermes-4-70B: rewrites → scenario trees, all risks
scripts/generate_hermes_modal.sh             # same job, Hermes on Modal via modal run --detach (scripts/hermes_modal.py); FORCE=1 after a rebuild
uv run python3 generate.py --only cyber --perturb paraphrase multilingual   # API attacker, subset
uv run python3 generate.py --missing-only    # fill gaps after a preempted job
```

Artifacts are committed. `certify.py` refuses to run on stale ones and prints the
`generate.py` command that fixes it.

### Certify

```bash
# one model, every stage (stage 2 is on by default; --simulate adds stage 3)
uv run python3 certify.py -m openrouter/<slug> --name "<Display>" --provider <Co> --region <R> --simulate

# resume: re-issue the same command. eval_set skips finished tasks and re-runs only
# failed or unfinished samples from logs/<model_id>/current.
uv run python3 certify.py -m openrouter/<slug> --simulate

# start over for one risk (moves the whole logs/<model_id>/current aside, so other
# risks' unfinished logs stop resuming)
uv run python3 certify.py -m openrouter/<slug> --simulate --only cbrn --rerun

# fleet: one slurm array task per line of scripts/models.txt (CPU only)
sbatch scripts/certify_slurm.sh
bash scripts/batch_certify.sh                # same file, serial, no slurm

# re-derive risks that never certified cleanly from their logs, offline
uv run python3 scripts/reaggregate_from_logs.py <model_id>

# smoke test: results are NOT written, logs are
uv run python3 certify.py -m openrouter/<slug> --only cbrn --limit 3 --simulate

# offline smoke test, no API keys: both CLIs end to end against inspect's mock model (~10 s)
uv run python3 -m unittest tests.test_smoke
```

Before any eval starts `certify.py` validates artifacts, prints the estimated target /
judge / classifier call counts per task, and pings every grader and the classifier.

### Running through the AI gateway

The Bifrost gateway serves GPT, Claude, Grok etc. behind one OpenAI-compatible endpoint.
Inspect's stock `openai-api` provider reaches it with no code: a model id
`openai-api/bifrost/<model>` reads `BIFROST_BASE_URL` and `BIFROST_API_KEY` and sends the
plain `<model>` name. Every model role takes a full model id, so prefix each one.

Setup for someone with gateway access, from a fresh clone:

1. Install: `uv sync` (needs `uv`; the lockfile pins inspect-ai 0.3.246).
2. Put the credentials in `.env` at the repo root (inspect loads it; nothing else reads it):
   ```
   BIFROST_BASE_URL=https://bifrost.is.localnet/openai
   BIFROST_API_KEY=sk-bf-...
   SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt   # Debian/Ubuntu; RedHat: /etc/pki/tls/certs/ca-bundle.crt
   ```
   `SSL_CERT_FILE` makes Python trust the internal CA that signs the gateway certificate.
   If the CA is not in the system bundle, point it at the CA file instead.
3. Check the connection with one leaf, 2 items, every model role through the gateway:
   ```bash
   uv run python3 certify.py -m openai-api/bifrost/gpt-4.1 --name "GPT-4.1" --provider OpenAI --region US \
       -g openai-api/bifrost/gpt-4.1 openai-api/bifrost/claude-sonnet-5 \
       --sim-classifier openai-api/bifrost/gpt-4.1 --simulate --only loc_deception --limit 2
   ```
   A wrong model name fails at the preflight model check; a certificate problem fails
   with an SSL error on the first request. `--limit` runs never write `models/results`.
4. Full run: the same command without `--only` and `--limit`, plus `--max-tasks 21` so the
   leaves run in parallel (`--max-connections` per leaf defaults to 20; raise it if the
   gateway allows). Expect a few hours and the token totals printed in the preflight table.
5. Results land in `models/results/<model_id>.json`; `logs/<model_id>/<run_id>/` holds the
   inspect logs (`inspect view`). A killed run resumes finished leaves with the same `--run-id`.

Regenerating artifacts through the gateway is the same prefix on `generate.py` (`-a
openai-api/bifrost/<model>`), only needed when the prompts or datasets change: certify
replays the committed artifacts.

Caveats:

- Judges other than `GRADERS.md` make the certificate incomparable with runs graded by
  the standard panel. Drop `-g` to keep it (that panel needs `OPENROUTER_API_KEY`); the
  stage-3 classifier likewise defaults to an OpenRouter model, so pass `--sim-classifier`.
- The gateway's model names and content filters are unverified from this side: the stub
  test (`tests/test_gateway.py`) covers the request shape and the CA variable only.
- A commercial attacker refuses more rewrites than Hermes, so expect more `fallback` rows.
- `--max-price` and `--cheapest` are OpenRouter-only and refuse a gateway target.

## Where results land

`models/results/<model_id>.json` is the source of truth, written atomically;
`models/models.json` is rebuilt from those files. `scores.<risk>` is the headline (the
CVaR@10% tail of per-item worst values, see `pipeline/README.md § Metrics`), `results.<risk>`
the full tree, `status.<risk>` completion and token usage. Logs are in
`logs/<model_id>/<run_id>/`; `inspect view` opens them.

## CLI

<!-- certify-help -->
```
usage: certify.py [-h] --model MODEL [--name NAME] [--provider PROVIDER] [--region REGION]
                  [--grader MODEL [MODEL ...]] [--perturb FAMILY [FAMILY ...]] [--no-perturb]
                  [--perturb-k PERTURB_K] [--simulate] [--sim-k SIM_K]
                  [--sim-classifier SIM_CLASSIFIER] [--only LEAF [LEAF ...]] [--limit LIMIT]
                  [--epochs EPOCHS] [--rerun] [--run-id RUN_ID] [--max-tasks MAX_TASKS]
                  [--max-connections MAX_CONNECTIONS] [--max-retries MAX_RETRIES]
                  [--attempt-timeout ATTEMPT_TIMEOUT] [--timeout TIMEOUT]
                  [--working-limit WORKING_LIMIT] [--max-price KEY=VALUE [KEY=VALUE ...]]
                  [--cheapest]

Certify one target model: stage 1 (control) on every leaf dataset, stage 2 (surface perturbations,
replayed from datasets/generated/) by default, stage 3 (scenario trees) under --simulate. Writes
models/results/<model_id>.json unless --limit is set.

options:
  -h, --help            show this help message and exit

target and identity:
  --model MODEL, -m MODEL
                        Inspect model id of the target, e.g. openrouter/<slug> or openai-
                        api/bifrost/<model>.
  --name NAME, -n NAME  Display name for the certificate table.
  --provider PROVIDER, -p PROVIDER
                        Company behind the model.
  --region REGION, -r REGION
                        Region the model is developed in (US, EU, China).
  --grader MODEL [MODEL ...], -g MODEL [MODEL ...]
                        Judge model(s) overriding GRADERS.md, e.g. -g openai-api/bifrost/gpt-4.1
                        openai-api/bifrost/claude-sonnet-5.

stages:
  --perturb FAMILY [FAMILY ...]
                        Stage-2 families to run (default: all of framing, multilingual,
                        paraphrase, past_tense, persona, reconsideration, register). Every family
                        but reconsideration replays datasets/generated/ (run generate.py first).
  --no-perturb          Skip stage 2 entirely.
  --perturb-k PERTURB_K
                        Stored variants per item for paraphrase/register/past_tense (default: 1;
                        at most the k generated).
  --simulate            Run stage 3: the target walks the pregenerated deployment-scenario trees
                        (up to 3 turns); composes with stage 2 in one run and one log per leaf.
  --sim-k SIM_K         Stored scenario trees per item (default: 2).
  --sim-classifier SIM_CLASSIFIER
                        Model labelling each target reply refuse/hedge/comply to pick the next
                        stored turn (regex-prefiltered, disk-cached, canaried with the graders;
                        default: openrouter/google/gemini-3-flash-preview).

run control:
  --only LEAF [LEAF ...], -o LEAF [LEAF ...]
                        Run only these leaf datasets (e.g. --only cbrn cyber_operational_use);
                        other results are kept.
  --limit LIMIT, -l LIMIT
                        Sample this many items per leaf. Smoke-test mode: models/results is NOT
                        written, logs are.
  --epochs EPOCHS, -e EPOCHS
                        Generations per sample to average over (default: 1).
  --rerun               Re-run every requested leaf even if certified, moving logs/MODEL/RUN_ID
                        aside under a timestamp.
  --run-id RUN_ID       Log directory under logs/MODEL/ to write to and resume from (default:
                        current). A finished leaf in it is skipped, an unfinished one re-runs its
                        missing samples; --limit appends -limitN.
  --max-tasks MAX_TASKS
                        Leaf tasks to run in parallel (default: inspect's 10).

inspect limits:
  Bounds so an unattended batch fails a stuck model instead of hanging. Inspect's own defaults
  retry forever and count backoff as waiting, which --working-limit never ends.

  --max-connections MAX_CONNECTIONS
                        Concurrent model connections (default: 100).
  --max-retries MAX_RETRIES
                        Retries per model request (default: 10).
  --attempt-timeout ATTEMPT_TIMEOUT
                        Seconds before one request attempt is abandoned and retried (default:
                        600).
  --timeout TIMEOUT     Seconds after the first attempt before a request is given up regardless of
                        retries left (default: 1800).
  --working-limit WORKING_LIMIT
                        Working seconds per sample before it fails and retries, excluding time
                        waiting on retries (default: 900).

openrouter routing:
  openrouter/ targets only; the graders and classifier route unconstrained.

  --max-price KEY=VALUE [KEY=VALUE ...]
                        Price ceiling per token class in USD per million tokens, KEY one of
                        prompt/completion/request/image (e.g. --max-price prompt=1.25
                        completion=4.25). Endpoints above the cap are dropped; a cap under the
                        model's floor errors every sample.
  --cheapest            Pin the cheapest endpoints that still serve the model's most capable
                        configuration (full precision, longest context; see
                        pipeline/utils/routing.py). Endpoints considered are recorded in
                        models.json.
```
<!-- /certify-help -->

<!-- generate-help -->
```
usage: generate.py [-h] [--attacker ATTACKER] [-M KEY=VALUE] [--model-base-url MODEL_BASE_URL]
                   [--max-connections MAX_CONNECTIONS] [--perturb FAMILY [FAMILY ...]]
                   [--no-perturb] [--perturb-k PERTURB_K] [--simulate] [--sim-k SIM_K]
                   [--reasoning] [--only LEAF [LEAF ...]] [--missing-only] [--force]
                   [--limit LIMIT]

Generate the frozen perturbation/simulation artifacts certify.py replays.

options:
  -h, --help            show this help message and exit
  --attacker ATTACKER, -a ATTACKER
                        Rewrite/reframing model for the generative families (default:
                        openrouter/nousresearch/hermes-4-70b). Any inspect API provider, or
                        vllm/<repo> pointed at a running vLLM server via --model-base-url.
  -M KEY=VALUE          Model argument forwarded to inspect's get_model() (repeatable), e.g. -M
                        tensor_parallel_size=8 -M device=cuda. Values are YAML-parsed.
  --model-base-url MODEL_BASE_URL
                        Base URL of an already-running inference server (e.g. a vLLM server
                        launched in a separate slurm job).
  --max-connections MAX_CONNECTIONS
                        Concurrent attacker generations (default: 20). Tune down for a self-hosted
                        server, up for a large API quota.
  --perturb FAMILY [FAMILY ...]
                        Stage-2 families to generate (default: all pregenerated families).
                        reconsideration has no artifacts — it runs live in certify.py.
  --no-perturb          Skip stage-2 families entirely (e.g. to generate only --simulate
                        artifacts).
  --perturb-k PERTURB_K
                        Variants per item for the repeat rewrite families (paraphrase, register,
                        past_tense); default=1.
  --simulate            Also generate stage-3 scenario trees (scenario.jsonl, prompt v4).
  --sim-k SIM_K         Scenario trees per item under --simulate; default=2. Variant 2 is prompted
                        to use a different deployment (role, sector, asker) from variant 1.
  --reasoning           Request reasoning mode from the attacker for scenario reframings
                        (thinking=True via vLLM's chat_template_kwargs, e.g. Hermes-4); the
                        <think> block is stripped before parsing. Models/servers without the flag
                        ignore it, but plain API providers (e.g. OpenRouter) may reject the extra
                        body — leave it off for those.
  --only LEAF [LEAF ...], -o LEAF [LEAF ...]
                        Generate only for these leaf datasets (e.g. --only cbrn
                        cyber_operational_use).
  --missing-only        Fill gaps in existing artifact files (missing samples/variants, e.g.
                        failed reframings) and merge, instead of skipping files that already
                        exist.
  --force               Regenerate every requested family from scratch, overwriting existing
                        artifacts.
  --limit LIMIT, -l LIMIT
                        Generate for only the first N samples per task. WARNING: produces partial
                        artifacts (marked partial in the meta sidecar) that fail certify.py's
                        full-run validation — smoke-testing only.
```
<!-- /generate-help -->

<!-- prepare-help -->
```
usage: prepare.py [-h]
                  [--leaf {cbrn,cbrn_biological,collusion,cyber_attack_scaling,cyber_exploit_generation,cyber_operational_use,cyber_vulnerability_discovery,discriminatory_bias,hallucination_misinformation,harmful_deployment,lawlessness,loc_deception,loc_oversight_evasion,loc_self_replication,loc_shutdown_resistance,manipulation_persuasion,manipulation_propensity,misalignment_intent,misalignment_values,power_seeking,unreliability}]
                  [--seed SEED] [--dry-run]
                  [--embedding-model EMBEDDING_MODEL]

Build one dataset per CoP leaf.

    uv run python3 -m datasets.prepare.cluster.prepare --leaf cyber_operational_use
    uv run python3 -m datasets.prepare.cluster.prepare --dry-run

Selection, per leaf (datasets/BENCHMARKS.md § Sampling):

1. Load every source annotated with the leaf (annotations.json, or the
   Source's hand `leaves`); drop exact repeats inside a source and across them.
2. Score every row by its max cosine to the leaf's anchors (legal text and
   exemplars).
3. Per source, keep its top P_MAX x r share of units (r = the annotators'
   relevance of the source to the leaf), never fewer than FLOOR_MIN, and never a
   unit that looks less like the leaf than 95% of the items the annotators did
   not assign to it (the off-topic guard). Ranks and quantiles, not cosines, so
   any embedder works.
4. One pass over the whole leaf in score order: a row is kept when its score,
   less a penalty for resembling a row already kept (from any source), still
   clears its source's cut. "Resembling" is measured against the embedder: the
   hinge starts at the median nearest-neighbour similarity in the item cache.

Writes datasets/public/<leaf>.csv plus <leaf>.meta.json (seed, constants,
embedding model, per-source relevance, thresholds and counts, source revisions)
and <leaf>.dropped.jsonl (every drop, tagged exact, exact_cross_source,
threshold or redundant, so the constants are reviewable rather than trusted).

Reads one gitignored cache, datasets/cache/embeddings/ (items.npz for queries,
leaves.npz for anchors). On a miss it appends what is missing to the embed
input, prints the command that fills it and exits 2.

options:
  -h, --help            show this help message and exit
  --leaf {cbrn,cbrn_biological,collusion,cyber_attack_scaling,cyber_exploit_generation,cyber_operational_use,cyber_vulnerability_discovery,discriminatory_bias,hallucination_misinformation,harmful_deployment,lawlessness,loc_deception,loc_oversight_evasion,loc_self_replication,loc_shutdown_resistance,manipulation_persuasion,manipulation_propensity,misalignment_intent,misalignment_values,power_seeking,unreliability}
                        build one leaf (repeatable); default is every leaf a
                        source serves
  --seed SEED
  --dry-run             print the per-source table without writing
  --embedding-model EMBEDDING_MODEL
                        sentence-transformers model whose cache to read (a
                        comparison; the default is the committed one)
```
<!-- /prepare-help -->

## Repository structure

`pipeline/` ([README](pipeline/README.md)): `stage1_evaluation/`, `stage2_perturbation/`,
`stage3_simulation/`, shared `utils/`. `datasets/` ([README](datasets/README.md)): `raw/`
fetches → `prepare/` → `public/` CSVs → `generated/` artifacts
([schemas](datasets/generated/README.md)). `scripts/`: slurm jobs, audits, one-off
migrations. `tests/`: `uv run python3 -m unittest discover tests`. The adversarial attack
suite lives on the `adversarial-attacks` branch.
