<div align="center">
  <img style="height: 168;" src="media/logo.png" alt="EuroSafeAI Logo">
</div>

# AI Safety Benchmarks & Certification

This repository holds the pipeline and datasets required for evaluating frontier models on the public and private datasets used for awarding AI safety certificates, described in https://eurosafe.ai.toronto.edu/certificate.

## Getting Started

```
uv venv --python 3.10
uv pip install -r requirements.txt
uv run certify.py --model {MODEL_NAME} --grader {GRADER_NAME}
```

## Datasets

We evaluate models on four primary datasets with both publically available and private data.

### Democratic or Authoritarian? Probing a New Dimension of Political Biases in Large Language Models

[Paper](https://arxiv.org/abs/2506.12758)

### Preserving Historical Truth: Detecting Historical Revisionism in Large Language Models

[Paper](https://arxiv.org/abs/2602.17433)

### SocialHarmBench: Revealing LLM Vulnerabilities to Socially Harmful Requests

[Paper](https://arxiv.org/abs/2510.04891)

### When Do Language Models Endorse Limitations on Universal Human Rights Principles?

[Paper](https://arxiv.org/abs/2603.04217)