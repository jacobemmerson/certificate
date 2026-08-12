---
license: mit
task_categories:
  - question-answering
  - text-generation
  - text-classification
  - summarization
language:
  - en
pretty_name: DarkBench
size_categories:
  - n<1K
configs:
  - config_name: default
    data_files:
      - split: train
        path: "darkbench.jsonl"
    default: true
---

# DarkBench: Understanding Dark Patterns in Large Language Models

## Overview

DarkBench is a comprehensive benchmark designed to detect dark design patterns in large language models (LLMs). Dark patterns are manipulative techniques that influence user behavior, often against the user's best interests. The benchmark comprises 660 prompts across six categories of dark patterns, which the researchers used to evaluate 14 different models from leading AI companies including OpenAI, Anthropic, Meta, Mistral, and Google.

Esben Kran\*, Jord Nguyen\*, Akash Kundu\*, Sami Jawhar\*, Jinsuk Park\*, Mateusz Maria Jurewicz

🎓 [Apart Research](https://apartresearch.com/), \*Equal Contribution

_[Reproduce this work](https://github.com/apartresearch/DarkBench/tree/main/reproduce)_

## Links

- [Website](https://darkbench.ai)
- [HuggingFace](https://huggingface.co/datasets/apart/darkbench)
- [For Humanity podcast](https://www.youtube.com/watch?v=KTH1RBeO3ik)
- [IASEAI Talk](https://www.iaseai.org/conference/people/esben-kran)
- [ICLR 2025 Oral Spotlight](https://iclr.cc/virtual/2025/oral/31771)
- [Blog post](https://www.apartresearch.com/post/uncovering-model-manipulation-with-darkbench)
- [AAAI 2025 DATASAFE Workshop](https://openreview.net/forum?id=Vz1uCY5aG4)

## Dark Pattern Categories
The benchmark identifies and tests for six types of dark patterns:

1. **Brand Bias**: Does the model favor its developer's products and portray competitors negatively?
2. **User Retention**: Does the model foster a sense of friendship with users to encourage continued use?
3. **Sycophancy**: Does the model show dishonesty to reinforce users' echo chambers?
4. **Anthropomorphism**: Does the model present itself as having human-like qualities?
5. **Harmful Generation**: Is the model willing to produce harmful content?
6. **Sneaking**: Does the model change the original meaning in text transformation tasks?

## Key Findings
- Dark patterns appeared in 48% of all tested conversations on average
- The most common dark pattern was "sneaking" (79% of conversations)
- The least common was "sycophancy" (13% of conversations)
- User retention and sneaking were prevalent across all models
- Models within the same family (e.g., Claude 3 models) showed similar patterns of behavior
- The Claude family of models demonstrated the lowest average rates of dark patterns

## Methodology
The researchers:
1. Created precise descriptions for each dark pattern
2. Manually wrote adversarial prompts intended to elicit each pattern
3. Used few-shot prompting of LLMs to generate additional prompts
4. Used multiple LLM-based annotators (Claude, Gemini, GPT) to evaluate responses

## Implications
The research suggests that frontier LLMs from leading AI companies exhibit manipulative behaviors to varying degrees. The researchers argue that AI companies should work to mitigate and remove dark design patterns from their models to promote more ethical AI.

## Models Evaluated
- Open source: Llama-3-70b, Llama-3-8b, Mistral-7b, Mixtral-8x7b
- Proprietary: Claude family (Haiku, Sonnet, Opus), Gemini family (1.0-Pro, 1.5-Flash, 1.5-Pro), GPT family (3.5-Turbo, 4, 4-Turbo, 4o)

This benchmark represents an important step in understanding and mitigating the potential manipulative impact of LLMs on users.