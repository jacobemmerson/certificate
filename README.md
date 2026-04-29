<div align="center">
  <img style="height: 168" src="media/logo.png" alt="EuroSafeAI Logo">

  <h4>[<a href="https://eurosafe.ai.toronto.edu">Homepage</a>]</h4>
</div>

# AI Safety Benchmarks & Certification

This repository contains the pipeline and datasets required for evaluating frontier models on benchmarks developed by researchers at EuroSafeAI. By rigorously evaluating frontier models on a variety of benchmarks, we aim to reduce the systemic risk posed by AI and its applications. 

---

| arXiv | Benchmark | File | Tasks |
| - | - | -: | - |
| [2506.12758](https://arxiv.org/abs/2506.12758) | Democratic or Authoritarian? Probing a New Dimension of Political Biases in Large Language Models | democratic_authoritarian_bias.py | @fscale <br> @favscore <br> @rolemodel|
| [2602.17433](https://arxiv.org/abs/2602.17433) | Preserving Historical Truth: Detecting Historical Revisionism in Large Language Models | preserving_historical_truth.py | @no_push <br> @explicit_push | 
| [2510.04891](https://arxiv.org/abs/2510.04891) | SocialHarmBench: Revealing LLM Vulnerabilities to Socially Harmful Requests | socialharmbench.py | @social_harm_bench |
| [2603.04217](https://arxiv.org/abs/2603.04217) | When Do Language Models Endorse Limitations on Universal Human Rights Principles? | llm_human_rights.py | @udhr <br> @udhr_individual <br> @udhr_government <br> @echr <br> @echr_individual <br> @echr_government |

---

### Getting Started

This pipeline relies heavily on the AISI Inspect framework for tracking model performance, grading, and logging. You'll need an API key from a supported provider, a full list can be found [here](https://inspect.aisi.org.uk/providers.html). Store this key as an environment variable.

The packages are managed by `uv`; the instructions for installing it can be found [here](https://docs.astral.sh/uv/getting-started/installation/). After installing according to the latest documentation, create your virtual environment with `python=3.10` and download the required packages.

```
uv venv --python 3.10
uv pip install -r requirements.txt
```

To run the certification pipeline using `uv`, use the following:
```
uv run certify.py \
  --model       {api-formatted model name for testing, i.e. openrouter/google/gemini-3-flash-preview} \
  --grader      {OPTIONAL: api-formatted model name for grading, i.e. openai/gpt-4o} \
  --name        {OPTIONAL: the name stored in models/models.json} \
  --proivder    {OPTIONAL: the model provider, stored in models/models.json} \
  --region      {OPTIONAL: a description of the model's origin (i.e. US, Asia)} \
  --speciality  {OPTIONAL: the model's primary task (i.e. coding, math)} \
  --epochs      {OPTIONAL: the number of epochs to run, default=1} \
  --rerun       {OPTIONAL: rerun results that are already present for the model}
```
If a grader model is not specified with `--grader`, a group of models is used for LLM-as-a-judge grading as specified in `GRADERS.md`.

All results are stored in `models/models.json` which will automatically be updated with new models or replace previously run models. By default, the script will skip benchmarks that have already been processed; however, you can override this with by adding `--rerun` argument to `certify.py`. All logs will be in `logs/{benchmark_name}`; these can be accessed to use unreported metrics or other metadata about the samples. 

**You can also use any package manager of your choice** (i.e. anaconda); install the requirements by omitting `uv` and execute the pipeline using `python certify.py` with the appropriate arguments.

To evaluate on individual benchmarks, you can use AISI Inspect's CLI `inspect eval scripts/evals/{file}@{task}.py`. Note that you wil have to set certain parameters, like the model to be evaluated, which can be found [here](https://inspect.aisi.org.uk/reference/inspect_eval.html).

## Future Tasks / TODOs

In order of urgency:
1. ~~Write the summarization/metric scripts to calculate overall model performance on benchmarks with multiple tasks (i.e. Democratic vs. Authoritarian Bias).~~
2. ~~Modify `certify.py` to allow the specification of individual tasks.~~ Not Implementing
3. Update `scripts/README.md` and `benchmarks/README.md` to outline how to incorporate new benchmarks and define the repository's structure.
4. Support locally run models as well as api-models.
5. Generate private/held out datasets.
6. ~~Rename `scripts` directory to something more fitting~~ 
7. Connect repo to ESAI's certificate page to automatically flag for updates when new models are run.
  - Will need a personal access token
8. Impement Majority Voting for LLM-as-a-judge grading
9. Combine UDHR and ECHR datasets for the human rights limitations; currently benchmarking on UDHR.
10. Use the UDHR and ECHR individual and government steering in evaluations.
11. Add file to process results from logs incase of a crash --- avoiding crashes with try, except, but logs are always stored and can be used as a fallback