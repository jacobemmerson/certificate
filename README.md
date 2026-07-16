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

To run the certification pipeline using `uv`, use the following:
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
  --attacker    {OPTIONAL: rewrite/paraphrase model for the perturbation stage, default=openrouter/deepseek/deepseek-v4-flash} \
  --llamaguard  {OPTIONAL: LlamaGuard model used for harm scoring across all tasks, default=openrouter/meta-llama/llama-guard-4-12b; pass an empty string to disable it (i.e. "")} \
  --limit       {OPTIONAL: deterministically select this many canonical IDs per task} \
  --condition   {OPTIONAL: c0, c1, c2, c3, or c4; default is clean c0} \
  --analyst-model {OPTIONAL: C3/C4 analyst role model; defaults to --attacker} \
  --critic-model  {OPTIONAL: C3/C4 critic role model; defaults to the first grader} \
  --sample-seed {OPTIONAL: deterministic sample-pairing seed, default=0} \
  --sample-ids-out {OPTIONAL: write the selected canonical-ID map} \
  --sample-ids-in  {OPTIONAL: reuse an exact canonical-ID map} \
  --perturb     {OPTIONAL: explicit stage-2 families; incompatible with --simulate and C1-C4} \
  --simulate    {OPTIONAL: stage-3 scenario simulation; incompatible with --perturb and C1-C4} \
  --sim-k       {OPTIONAL: reframed scenarios per item under --simulate, default=1}
```
If a grader model is not specified with `--grader`, a group of models is used for LLM-as-a-judge grading as specified in `GRADERS.md`.

All results are stored in `models/models.json` which will automatically be updated with new models or replace previously run models. By default, the script will skip benchmarks that have already been processed; however, you can override this with by adding `--rerun` argument to `certify.py`. All logs will be in `logs/{benchmark_name}`; these can be accessed to use unreported metrics or other metadata about the samples.

Default invocation is clean C0: it runs the untouched stage-1 tasks with no perturbation or simulation. C1–C4 replace only each task solver, retain the original scorer objects/order, enforce finite code-level limits, and accept exactly one `submit(answer)` payload. See [`pipeline/agentic/README.md`](pipeline/agentic/README.md).

Runs without exact canonical-ID coverage (including ordinary `--limit` smoke runs) are **not** written to `models/models.json`, though they remain available in `logs/{model_name}`. Reuse `--sample-ids-out`/`--sample-ids-in` for paired C0–C4 comparisons. Modes use non-overwriting result keys: clean C0 keeps `harm`, perturbation and simulation use `harm_perturbation` and `harm_simulation`, and C1–C4 use `harm_agentic_c1` through `harm_agentic_c4`. C1–C4 scorer outputs remain in logs, while reported aggregates exclude protocol-invalid samples and persist invalid counts.

**You can also use any package manager of your choice** (i.e. anaconda); install the requirements by omitting `uv` and execute the pipeline using `python certify.py` with the appropriate arguments.

To evaluate an individual C0 benchmark, use `uv run inspect eval pipeline/stage1_evaluation/evals/{file}.py@{task}`. The generic C1–C4 entry is `uv run inspect eval pipeline/agentic/eval.py@agentic` with `-T base_task=... -T condition=...`; see the agentic README for role-model arguments. Inspect CLI parameters are documented [here](https://inspect.aisi.org.uk/reference/inspect_eval.html).

### Repository structure

The source lives in [`pipeline/`](pipeline/README.md): `stage1_evaluation/` contains the 12 plain tasks, `stage2_perturbation/` provides explicit surface perturbations, `stage3_simulation/` provides single-turn scenario simulation, and `agentic/` provides the separate C0–C4 condition layer. Agentic, perturbation, and simulation transforms cannot compose or repeat. Evaluation data lives in [`datasets/`](datasets/README.md) (`raw/` sources → `prepare/` scripts → `public/` CSVs); see [CONTRIBUTE.md](CONTRIBUTE.md) for adding a benchmark. The adversarial attack suite lives on the **`adversarial-attacks`** branch.


### Benchmarks


---

| arXiv | Benchmark | File | Tasks |
| - | - | -: | - |
| [2506.12758](https://arxiv.org/abs/2506.12758) | Democratic or Authoritarian? Probing a New Dimension of Political Biases in Large Language Models | democratic_authoritarian_bias.py | @fscale <br> @favscore <br> @rolemodel|
| [2602.17433](https://arxiv.org/abs/2602.17433) | Preserving Historical Truth: Detecting Historical Revisionism in Large Language Models | preserving_historical_truth.py | @no_push <br> @explicit_push | 
| [2510.04891](https://arxiv.org/abs/2510.04891) | SocialHarmBench: Revealing LLM Vulnerabilities to Socially Harmful Requests | socialharmbench.py | @social_harm_bench |
| [2603.04217](https://arxiv.org/abs/2603.04217) | When Do Language Models Endorse Limitations on Universal Human Rights Principles? | llm_human_rights.py | @udhr <br> @udhr_individual <br> @udhr_government <br> @echr <br> @echr_individual <br> @echr_government |

---
