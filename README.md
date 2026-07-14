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
  --limit       {OPTIONAL: randomly sample this many examples per task instead of running the full dataset} \
  --simulate    {OPTIONAL: run stage-3 scenario simulation — reframe each item into a realistic deployment scenario and re-run the target on it; reuses --attacker as the reframing model} \
  --sim-k       {OPTIONAL: number of reframed scenarios generated per item under --simulate, default=1}
```
If a grader model is not specified with `--grader`, a group of models is used for LLM-as-a-judge grading as specified in `GRADERS.md`.

All results are stored in `models/models.json` which will automatically be updated with new models or replace previously run models. By default, the script will skip benchmarks that have already been processed; however, you can override this with by adding `--rerun` argument to `certify.py`. All logs will be in `logs/{benchmark_name}`; these can be accessed to use unreported metrics or other metadata about the samples.

Runs using `--limit` are treated as smoke tests: since they only cover a random subset of each benchmark, results are **not** written to `models/models.json`, though they remain available in `logs/{model_name}`.

**You can also use any package manager of your choice** (i.e. anaconda); install the requirements by omitting `uv` and execute the pipeline using `python certify.py` with the appropriate arguments.

To evaluate on individual benchmarks, you can use AISI Inspect's CLI `uv run inspect eval pipeline/stage1_evaluation/evals/{file}.py@{task}`. Note that you wil have to set certain parameters, like the model to be evaluated, which can be found [here](https://inspect.aisi.org.uk/reference/inspect_eval.html).

### Repository structure

The source lives in [`pipeline/`](pipeline/README.md), organized into three stages: `stage1_evaluation/` (plain benchmark evals), `stage2_perturbation/` (surface-perturbation reliability auditing, enabled via `--perturb`), and `stage3_simulation/` (single-turn scenario simulation, enabled via `--simulate`). Evaluation data lives in [`datasets/`](datasets/README.md) (`raw/` sources → `prepare/` scripts → `public/` CSVs); see [CONTRIBUTE.md](CONTRIBUTE.md) for adding a new benchmark. The adversarial attack suite (jailbreaks, attack-retry solver, multi-classifier harm scoring) lives on the **`adversarial-attacks`** branch.


### Benchmarks


---

| arXiv | Benchmark | File | Tasks |
| - | - | -: | - |
| [2506.12758](https://arxiv.org/abs/2506.12758) | Democratic or Authoritarian? Probing a New Dimension of Political Biases in Large Language Models | democratic_authoritarian_bias.py | @fscale <br> @favscore <br> @rolemodel|
| [2602.17433](https://arxiv.org/abs/2602.17433) | Preserving Historical Truth: Detecting Historical Revisionism in Large Language Models | preserving_historical_truth.py | @no_push <br> @explicit_push | 
| [2510.04891](https://arxiv.org/abs/2510.04891) | SocialHarmBench: Revealing LLM Vulnerabilities to Socially Harmful Requests | socialharmbench.py | @social_harm_bench |
| [2603.04217](https://arxiv.org/abs/2603.04217) | When Do Language Models Endorse Limitations on Universal Human Rights Principles? | llm_human_rights.py | @udhr <br> @udhr_individual <br> @udhr_government <br> @echr <br> @echr_individual <br> @echr_government |

---
