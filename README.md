<div align="center">
  <img style="height: 168" src="media/logo.png" alt="EuroSafeAI Logo">

  <h4>[<a href="https://eurosafe.ai.toronto.edu">Homepage</a>]</h4>
</div>

# AI Safety Benchmarks & Certification

This repository contains the pipeline and datasets required for evaluating frontier models on benchmarks developed by researchers at EuroSafeAI. By rigorously evaluating frontier models on a variety of benchmarks, we aim to reduce the systemic risk posed by AI and its applications. 

This pipeline relies heavily on the AISI Inspect framework for tracking model performance, grading, and logging; you can read more [here](https://inspect.aisi.org.uk).

### Getting Started

The packages are managed by `uv`; the instructions for installing it can be found [here](https://docs.astral.sh/uv/getting-started/installation/). After installing according to the latest documentation, create your virtual environment with `python=3.10` and download the required packages.

```
uv venv --python 3.10
uv pip install -r requirements.txt
```

To run the certification pipeline using `uv`, use the following:
```
uv run certify.py --model {MODEL_NAME} --grader {GRADER_NAME}
```
`{MODEL_NAME}` may be something like `openai/gpt-4o` with the grader following the same format.


**You can also use any package manager of your choice** and install the requirements by omitting `uv` and execute the pipeline with `python certify.py`.

To evaluate on individual benchmarks, you can use AISI Inspect's CLI `inspect eval scripts/evals/{file}.py`. Note that you wil have to set certain parameters, like the model to be evaluated, which can be found [here](https://inspect.aisi.org.uk/reference/inspect_eval.html).

## Benchmarks

### [[Abstract](https://arxiv.org/abs/2506.12758) | [Paper](https://arxiv.org/pdf/2506.12758)] Democratic or Authoritarian? Probing a New Dimension of Political Biases in Large Language Models 

### [[Abstract](https://arxiv.org/abs/2602.17433) | [Paper](https://arxiv.org/pdf/2602.17433)] Preserving Historical Truth: Detecting Historical Revisionism in Large Language Models 

### [[Abstract](https://arxiv.org/abs/2510.04891) | [Paper](https://arxiv.org/pdf/2510.04891)] SocialHarmBench: Revealing LLM Vulnerabilities to Socially Harmful Requests 

### [[Abstract](https://arxiv.org/abs/2603.04217) | [Paper](https://arxiv.org/pdf/2603.04217)] When Do Language Models Endorse Limitations on Universal Human Rights Principles? 

## Future Tasks / TODOs

1. Write the summarization/metric scripts to calculate overall model performance on benchmarks with multiple tasks (i.e. Democratic vs. Authoritarian Bias).
2. Modify `certify.py` to allow the specification of individual tasks.
3. Update `scripts/README.md` and `benchmarks/README.md` to outline how to incorporate new benchmarks and define the repository's structure.
4. Support locally run models as well as api-models.
5. Generate private/held out datasets.