<div align="center">
  <img style="height: 168" src="media/logo.png" alt="EuroSafeAI Logo">

  <h4>[<a href="https://eurosafe.ai.toronto.edu">Homepage</a>]</h4>
</div>

# AI Safety Benchmarks & Certification

The pipeline and datasets for certifying frontier models against the four EU AI Act
systemic-risk clusters (`cbrn`, `cyber`, `loss_of_control`, `manipulation`): each cluster is
a subset of several benchmarks, hardened by frozen surface perturbations (stage 2) and a
multi-turn deployment scenario (stage 3), judged by a two-model ensemble, and rolled up to
one headline per risk. How it is built and what the numbers mean:
[`pipeline/README.md`](pipeline/README.md). What is in the data:
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

Judges are listed in [`GRADERS.md`](GRADERS.md). Hermes-4-70B (attacker, answerability
screen) is served locally by vLLM through `uvx`, never installed into the project env.

## Three scripts

| Script | Runs | Writes |
|---|---|---|
| `datasets/prepare/cluster/prepare.py` | once per dataset change; three passes (below) | `datasets/public/<risk>.csv` + `.meta.json` + `.dropped.jsonl`, committed |
| `generate.py` | once per artifact refresh, attacker model on slurm | `datasets/generated/<risk>/<family>.jsonl` + `.meta.json`, committed |
| `certify.py` | once per target model; target + judges + classifier only | `models/results/<model_id>.json`, `models/models.json` (rebuilt), `logs/<model_id>/<run_id>/` |

## Recipes

### Build the cluster datasets (three passes)

`prepare.py` exits 2 whenever a cache it needs is missing and prints the exact command
that fills it. A CSV is never built on a partial cache.

```bash
uv run python3 -m datasets.prepare.cluster.prepare --risk cbrn        # exit 2: embeddings missing
uv run --no-project --with sentence-transformers --with numpy python scripts/embed_items.py --risk cbrn
uv run python3 -m datasets.prepare.cluster.prepare --risk cbrn        # exit 2: screen verdicts missing
sbatch --export=ALL,SCREEN_ONLY=1 scripts/generate_hermes_slurm.sh    # screens pending inputs, rebuilds the CSV, stops before generation
uv run python3 -m datasets.prepare.cluster.prepare --risk cbrn        # clean build if the slurm job did not already
```

Review `datasets/public/cbrn.dropped.jsonl` (new `screen` tier) and `refused_dropped` in
`cbrn.meta.json` before committing: the screen drops what an open model refuses, by design.
`--dry-run` prints the tier table without writing. Omit `--risk` for all four. Detail: [`datasets/BENCHMARKS.md § Sampling`](datasets/BENCHMARKS.md#sampling).

### Generate the artifacts

```bash
sbatch scripts/generate_hermes_slurm.sh      # vLLM Hermes-4-70B: screen → rewrites → scenario trees, all risks
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
```

Before any eval starts `certify.py` validates artifacts, prints the estimated target /
judge / classifier call counts per task, and pings every grader and the classifier.

## Where results land

`models/results/<model_id>.json` is the source of truth, written atomically;
`models/models.json` is rebuilt from those files. `scores.<risk>` is the headline (the
CVaR@10% tail of per-item worst values, see `pipeline/README.md § Metrics`), `results.<risk>`
the full tree, `status.<risk>` completion and token usage. Logs are in
`logs/<model_id>/<run_id>/`; `inspect view` opens them.

## CLI

<!-- certify-help -->
```
usage: certify.py [-h] --model MODEL [--grader GRADER] [--name NAME]
                  [--provider PROVIDER] [--region REGION]
                  [--specialty SPECIALTY] [--epochs EPOCHS] [--rerun]
                  [--run-id RUN_ID] [--llamaguard LLAMAGUARD] [--limit LIMIT]
                  [--only RISK [RISK ...]] [--perturb FAMILY [FAMILY ...]]
                  [--no-perturb] [--perturb-k PERTURB_K] [--simulate]
                  [--sim-k SIM_K] [--sim-classifier SIM_CLASSIFIER]
                  [--max-connections MAX_CONNECTIONS]
                  [--max-retries MAX_RETRIES]
                  [--attempt-timeout ATTEMPT_TIMEOUT] [--timeout TIMEOUT]
                  [--working-limit WORKING_LIMIT]
                  [--max-price KEY=VALUE [KEY=VALUE ...]] [--cheapest]

options:
  -h, --help            show this help message and exit
  --model MODEL, -m MODEL
                        The model to be evaluated using AISI inspect.
  --grader GRADER, -g GRADER
                        Grader model override (single model). If omitted,
                        loads from GRADERS.md
  --name NAME, -n NAME  The name of the model for formatting the certificate
                        table.
  --provider PROVIDER, -p PROVIDER
                        The provider of the model for formatting the
                        certificate table.
  --region REGION, -r REGION
                        The region of the world where the model is developed
                        and data is sourced.
  --specialty SPECIALTY, -s SPECIALTY
                        What the model has been tuned or designated to do
                        (i.e. coding, math, etc).
  --epochs EPOCHS, -e EPOCHS
                        The number of turns to generate a response per sample
                        and average over.
  --rerun               Rerun every requested risk even if it already has
                        results, and move the run's log directory
                        (logs/MODEL/RUN_ID) aside under a timestamp first, so
                        the resume logic cannot pick up its old logs.
  --run-id RUN_ID       Name of the log directory under logs/MODEL/ this run
                        writes to and resumes from (default: current). A
                        finished task in it is skipped and an unfinished one
                        re-runs only its missing samples; a --limit run uses
                        RUN_ID-limitN.
  --llamaguard LLAMAGUARD
                        LlamaGuard model for adversarial harm scoring
                        (default: openrouter/meta-llama/llama-guard-4-12b).
  --limit LIMIT, -l LIMIT
                        Randomly sample this many examples per task (default:
                        run the full dataset). WARNING: if limit is present,
                        results will not be saved. They can still be accessed
                        in logs/MODEL_NAME/
  --only RISK [RISK ...], -o RISK [RISK ...]
                        Run only these systemic-risk clusters (e.g. --only
                        cyber manipulation). Other existing results are
                        preserved. A finished log in the run dir is reused,
                        not re-run.
  --perturb FAMILY [FAMILY ...]
                        Run surface-perturbation auditing for these families
                        (paraphrase register past_tense multilingual framing
                        persona reconsideration; default: all). Replays
                        datasets/generated/ (run generate.py first);
                        reconsideration runs live.
  --no-perturb          Skip stage-2 perturbation auditing entirely (e.g. for
                        a --simulate-only run — --perturb is on by default).
  --perturb-k PERTURB_K
                        Use up to this many stored variants per item for the
                        rewrite families (paraphrase, register, past_tense);
                        default=1. Must not exceed the k the artifacts were
                        generated with.
  --simulate            Run stage-3 scenario simulation (see
                        pipeline/stage3_simulation/) on top of every benchmark
                        in --only (or all benchmarks if --only is omitted):
                        the target walks the pregenerated deployment-scenario
                        trees (up to 3 turns; branch chosen by --sim-
                        classifier) from datasets/generated/ (run generate.py
                        --simulate first). Composes with --perturb in one
                        run/one log: the panel reports safety_scenario next to
                        stage 2's safety_<family>/stability_under_attack.
  --sim-k SIM_K         Use up to this many stored scenario trees per item
                        under --simulate; default=2.
  --sim-classifier SIM_CLASSIFIER
                        Model that labels each target reply
                        refuse/hedge/comply to choose the next stored turn
                        under --simulate
                        (pipeline/stage3_simulation/classify.py). Regex-
                        prefiltered, disk-cached with no expiry, canaried with
                        the graders. Default:
                        openrouter/google/gemini-3-flash-preview.
  --max-connections MAX_CONNECTIONS
                        Max concurrent model connections Inspect opens
                        (default: 100).
  --max-retries MAX_RETRIES
                        Give up on a model request after this many retries
                        (default: 10). Inspect's own default is to retry
                        transient provider errors forever, and the backoff
                        sleep is reported as waiting time so --working-limit
                        never ends it — which turns a provider outage into an
                        unattended batch that hangs instead of failing the
                        model and moving on to the next one. Raise it for a
                        run against a heavily rate-limited provider.
  --attempt-timeout ATTEMPT_TIMEOUT
                        Abandon a single model request after this many seconds
                        and retry it (default: 600). Bounds a connection that
                        has stopped responding; unset, one wedged request
                        waits indefinitely because --working-limit excludes
                        time spent waiting on retries.
  --timeout TIMEOUT     Give up on a model request this many seconds after its
                        first attempt, however many retries are left (default:
                        1800). --max-retries bounds the number of attempts but
                        not their duration: backoff caps at 30 minutes a wait,
                        so a handful of retries can hold a sample for hours.
                        Whichever bound is hit first ends the request.
  --working-limit WORKING_LIMIT
                        Max working seconds per sample before it fails and
                        retries; bounds a hung provider connection so one
                        stuck request can't wedge the whole run (default:
                        900). Excludes time spent waiting on rate
                        limits/retries.
  --max-price KEY=VALUE [KEY=VALUE ...]
                        Cap what OpenRouter is allowed to pay for the target
                        model, as one or more of
                        prompt/completion/request/image in USD per million
                        tokens (e.g. --max-price prompt=1.25 completion=4.25).
                        Endpoints above the cap are dropped from routing, and
                        a request with no qualifying endpoint fails outright
                        rather than falling back to a pricier one — so a cap
                        under the model's floor price errors every sample.
                        openrouter/ models only, and the target only: the
                        graders and --llamaguard route unconstrained.
  --cheapest            Route every target call to the cheapest OpenRouter
                        endpoint that still serves the model's most capable
                        configuration — full weight precision, longest
                        context, widest parameter support (see
                        pipeline/utils/routing.py). Falls back only to other
                        equally capable endpoints, never onto a degraded one,
                        and fails the run rather than guess when the endpoint
                        list cannot be read. Every endpoint considered is
                        recorded per benchmark in models.json. Composes with
                        --max-price. openrouter/ models only, and the target
                        only: the graders and --llamaguard route
                        unconstrained.
```
<!-- /certify-help -->

<!-- generate-help -->
```
usage: generate.py [-h] [--attacker ATTACKER] [-M KEY=VALUE]
                   [--model-base-url MODEL_BASE_URL]
                   [--max-connections MAX_CONNECTIONS]
                   [--perturb FAMILY [FAMILY ...]] [--no-perturb]
                   [--perturb-k PERTURB_K] [--simulate] [--sim-k SIM_K]
                   [--reasoning] [--only RISK [RISK ...]] [--missing-only]
                   [--force] [--limit LIMIT]

Generate the frozen perturbation/simulation artifacts certify.py replays.

options:
  -h, --help            show this help message and exit
  --attacker ATTACKER, -a ATTACKER
                        Rewrite/reframing model for the generative families
                        (default: openrouter/nousresearch/hermes-4-70b). Any
                        inspect API provider, or vllm/<repo> pointed at a
                        running vLLM server via --model-base-url.
  -M KEY=VALUE          Model argument forwarded to inspect's get_model()
                        (repeatable), e.g. -M tensor_parallel_size=8 -M
                        device=cuda. Values are YAML-parsed.
  --model-base-url MODEL_BASE_URL
                        Base URL of an already-running inference server (e.g.
                        a vLLM server launched in a separate slurm job).
  --max-connections MAX_CONNECTIONS
                        Concurrent attacker generations (default: 20). Tune
                        down for a self-hosted server, up for a large API
                        quota.
  --perturb FAMILY [FAMILY ...]
                        Stage-2 families to generate (default: all
                        pregenerated families). reconsideration has no
                        artifacts — it runs live in certify.py.
  --no-perturb          Skip stage-2 families entirely (e.g. to generate only
                        --simulate artifacts).
  --perturb-k PERTURB_K
                        Variants per item for the repeat rewrite families
                        (paraphrase, register, past_tense); default=1.
  --simulate            Also generate stage-3 scenario trees (scenario.jsonl,
                        prompt v4).
  --sim-k SIM_K         Scenario trees per item under --simulate; default=2.
                        Variant 2 is prompted to use a different deployment
                        (role, sector, asker) from variant 1.
  --reasoning           Request reasoning mode from the attacker for scenario
                        reframings (thinking=True via vLLM's
                        chat_template_kwargs, e.g. Hermes-4); the <think>
                        block is stripped before parsing. Models/servers
                        without the flag ignore it, but plain API providers
                        (e.g. OpenRouter) may reject the extra body — leave it
                        off for those.
  --only RISK [RISK ...], -o RISK [RISK ...]
                        Generate only for these systemic-risk clusters (e.g.
                        --only cyber manipulation).
  --missing-only        Fill gaps in existing artifact files (missing
                        samples/variants, e.g. failed reframings) and merge,
                        instead of skipping files that already exist.
  --force               Regenerate every requested family from scratch,
                        overwriting existing artifacts.
  --limit LIMIT, -l LIMIT
                        Generate for only the first N samples per task.
                        WARNING: produces partial artifacts (marked partial in
                        the meta sidecar) that fail certify.py's full-run
                        validation — smoke-testing only.
```
<!-- /generate-help -->

<!-- prepare-help -->
```
usage: prepare.py [-h] [--risk {cbrn,cyber,loss_of_control,manipulation}]
                  [--seed SEED] [--dry-run]

Build the risk-cluster datasets. uv run python3 -m
datasets.prepare.cluster.prepare --risk cyber uv run python3 -m
datasets.prepare.cluster.prepare --dry-run Writes datasets/public/<risk>.csv
plus a <risk>.meta.json sibling (provenance: seed, quotas, per-tier drop
counts, embedding model and threshold, screen model and refusals, source
revisions) and <risk>.dropped.jsonl (every pair tiers 1b and 2 removed and
every candidate the screen dropped, each tagged with its `tier`, so thresholds
are reviewable rather than trusted). Reads two gitignored caches under
datasets/cache/. On a miss it writes what is missing, prints the command that
fills it and exits 2: embeddings first, then screen verdicts. The sequence is
in datasets/BENCHMARKS.md § Sampling.

options:
  -h, --help            show this help message and exit
  --risk {cbrn,cyber,loss_of_control,manipulation}
                        build one risk (repeatable); default is all registered
  --seed SEED
  --dry-run             print the tier table without writing
```
<!-- /prepare-help -->

## Repository structure

`pipeline/` ([README](pipeline/README.md)): `stage1_evaluation/`, `stage2_perturbation/`,
`stage3_simulation/`, shared `utils/`. `datasets/` ([README](datasets/README.md)): `raw/`
fetches → `prepare/` → `public/` CSVs → `generated/` artifacts
([schemas](datasets/generated/README.md)). `scripts/`: slurm jobs, audits, one-off
migrations. `tests/`: `uv run python3 -m unittest discover tests`. The adversarial attack
suite lives on the `adversarial-attacks` branch.
