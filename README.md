<div align="center">
  <img style="height: 168" src="media/logo.png" alt="EuroSafeAI Logo">

  <h4>[<a href="https://eurosafe.ai.toronto.edu">Homepage</a>]</h4>
</div>

# AI Safety Benchmarks & Certification

This repository contains the pipeline and datasets required for evaluating frontier models on benchmarks developed by researchers at EuroSafeAI. By rigorously evaluating frontier models on a variety of benchmarks, we aim to reduce the systemic risk posed by AI and its applications. 

### Getting Started

This pipeline relies heavily on the AISI Inspect framework for tracking model performance, grading, and logging. You'll need an API key from a supported provider, a full list can be found [here](https://inspect.aisi.org.uk/providers.html). Store this key as an environment variable.

The packages are managed by `uv`; the instructions for installing it can be found [here](https://docs.astral.sh/uv/getting-started/installation/). After installing according to the latest documentation, create the virtual environment and install the (locked) dependencies with:

```
uv sync
```

The certification runs in two steps. Stages 2 (`--perturb`) and 3 (`--simulate`) replay **pregenerated** perturbed prompts and scenarios so every evaluated model sees the exact same variants and the attacker model runs once total instead of once per model. Generate those artifacts first (only needed once per artifact refresh, or when adding a benchmark):
```
uv run generate.py \
  --attacker    {OPTIONAL: rewrite/reframing model, default=openrouter/deepseek/deepseek-v4-flash} \
  --perturb     {OPTIONAL: stage-2 families to generate, default=all; e.g. --perturb paraphrase framing} \
  --perturb-k   {OPTIONAL: variants per item for the rewrite families, default=1} \
  --simulate    {OPTIONAL: also generate stage-3 scenario reframings} \
  --sim-k       {OPTIONAL: scenarios per item under --simulate, default=1} \
  --only        {OPTIONAL: generate only for these benchmark keys, i.e. --only harm hr} \
  --missing-only {OPTIONAL: fill gaps in existing artifacts instead of skipping complete files} \
  --force       {OPTIONAL: regenerate everything from scratch}
```
The artifacts land in [`datasets/generated/`](datasets/generated/README.md) and are committed to the repo. See its README for the schema. `reconsideration` is the one perturbation family with no artifact — it challenges the target's own answer, so it always runs live.

Then evaluate each model (cheap — no attacker calls; only the target and judges run):
```
uv run certify.py \
  --model       {api-formatted model name for testing, i.e. openrouter/google/gemini-3-flash-preview} \
  --grader      {OPTIONAL: api-formatted model name for grading, i.e. openai/gpt-4o} \
  --name        {OPTIONAL: the name stored in models/models.json} \
  --provider    {OPTIONAL: the model provider, stored in models/models.json} \
  --region      {OPTIONAL: a description of the model's origin (i.e. US, Asia)} \
  --specialty   {OPTIONAL: the model's primary task (i.e. coding, math)} \
  --epochs      {OPTIONAL: the number of epochs to run, default=1} \
  --rerun       {OPTIONAL: rerun results that are already present for the model} \
  --only        {OPTIONAL: run only these benchmark keys, i.e. --only harm hr, see pipeline/registry.py for the keys} \
  --perturb     {OPTIONAL: stage-2 families to replay, default=all; reconsideration runs live} \
  --no-perturb  {OPTIONAL: skip stage 2 entirely, e.g. for a --simulate-only run} \
  --perturb-k   {OPTIONAL: use up to this many stored variants per item, default=1} \
  --llamaguard  {OPTIONAL: LlamaGuard model used for harm scoring across all tasks, default=openrouter/meta-llama/llama-guard-4-12b; pass an empty string to disable it (i.e. "")} \
  --limit       {OPTIONAL: randomly sample this many examples per task instead of running the full dataset} \
  --simulate    {OPTIONAL: run stage-3 scenario simulation by replaying the pregenerated reframings} \
  --sim-k       {OPTIONAL: use up to this many stored scenarios per item under --simulate, default=1}
```
`--perturb` and `--simulate` compose in one run: each benchmark gets a single eval log in which the control is generated and judged once, stage 2 reports under the `lvr`/`consistency` metrics and stage 3 under `lvr_scenario`/`consistency_scenario` (the certification score is the worst condition across every enabled family). `certify.py` checks that the artifacts for the requested `--perturb`/`--simulate` families exist before spending anything, and fails fast with the exact `generate.py` command if they don't. If a grader model is not specified with `--grader`, a group of models is used for LLM-as-a-judge grading as specified in `GRADERS.md`.

All results are stored in `models/models.json` which will automatically be updated with new models or replace previously run models. By default, the script will skip benchmarks that have already been processed; however, you can override this with by adding `--rerun` argument to `certify.py`. All logs will be in `logs/{benchmark_name}`; these can be accessed to use unreported metrics or other metadata about the samples.

Runs using `--limit` are treated as smoke tests: since they only cover a random subset of each benchmark, results are **not** written to `models/models.json`, though they remain available in `logs/{model_name}`.

**You can also use any package manager of your choice** (i.e. anaconda); install the requirements by omitting `uv` and execute the pipeline using `python certify.py` with the appropriate arguments.

To evaluate on individual benchmarks, you can use AISI Inspect's CLI `uv run inspect eval pipeline/stage1_evaluation/evals/{file}.py@{task}`. Note that you wil have to set certain parameters, like the model to be evaluated, which can be found [here](https://inspect.aisi.org.uk/reference/inspect_eval.html).

### Repository structure

The source lives in [`pipeline/`](pipeline/README.md), organized into three stages: `stage1_evaluation/` (plain benchmark evals), `stage2_perturbation/` (surface-perturbation reliability auditing, enabled via `--perturb`), and `stage3_simulation/` (single-turn scenario simulation, enabled via `--simulate`). Stages 2 and 3 replay artifacts pregenerated once by [`generate.py`](generate.py) into [`datasets/generated/`](datasets/generated/README.md), so every model is evaluated on identical variants. Evaluation data lives in [`datasets/`](datasets/README.md) (`raw/` sources → `prepare/` scripts → `public/` CSVs → `generated/` artifacts); see [CONTRIBUTE.md](CONTRIBUTE.md) for adding a new benchmark. The adversarial attack suite (jailbreaks, attack-retry solver, multi-classifier harm scoring) lives on the **`adversarial-attacks`** branch.


### Benchmarks


---

| arXiv | Benchmark | File | Tasks |
| - | - | -: | - |
| [2506.12758](https://arxiv.org/abs/2506.12758) | Democratic or Authoritarian? Probing a New Dimension of Political Biases in Large Language Models | democratic_authoritarian_bias.py | @fscale <br> @favscore <br> @rolemodel|
| [2602.17433](https://arxiv.org/abs/2602.17433) | Preserving Historical Truth: Detecting Historical Revisionism in Large Language Models | preserving_historical_truth.py | @no_push <br> @explicit_push | 
| [2510.04891](https://arxiv.org/abs/2510.04891) | SocialHarmBench: Revealing LLM Vulnerabilities to Socially Harmful Requests | socialharmbench.py | @social_harm_bench |
| [2603.04217](https://arxiv.org/abs/2603.04217) | When Do Language Models Endorse Limitations on Universal Human Rights Principles? | llm_human_rights.py | @udhr <br> @udhr_individual <br> @udhr_government <br> @echr <br> @echr_individual <br> @echr_government |

---
