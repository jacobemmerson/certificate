---
language:
- en
size_categories:
- n<1K
task_categories:
- text-generation
- question-answering
pretty_name: SocialHarmBench
tags:
- safety
- politics
- society
- adversarial
- jailbreak
- vulnerability
license: apache-2.0
---

# SocialHarmBench: Revealing LLM Vulnerabilities to Socially Harmful Requests
📄 [Arxiv Link](https://arxiv.org/abs/2510.04891) | 😊 [HF Link](https://huggingface.co/papers/2510.04891) | 💻 [Code](https://github.com/psyonp/SocialHarmBench)

**Description:**  
SocialHarmBench is a curated benchmark of adversarial prompts targeting sociopolitical vulnerabilities in language models. Each prompt is annotated with context to analyze potential harms across historical, geographic, and sociocultural dimensions.
 
<p align="center">
  <img src="socialharmbench_overview.png" alt="SocialHarmBench Overview" width="1200"/>
</p>

**Columns:**  

| Column       | Description |
|--------------|-------------|
| `prompt_id`  | Unique identifier for each prompt |
| `category`   | Broad category of harm (e.g., discrimination, misinformation) |
| `sub_topic`  | More specific topic within the category |
| `type`       | Type of prompt (e.g., question, statement) |
| `ethnicity`  | Ethnic group referenced in the prompt (if applicable) |
| `religion`   | Religion referenced in the prompt (if applicable) |
| `location`   | Country or region referenced |
| `prompt_text`| The text of the prompt |
| `year`       | Year or century the prompt refers to |

**Usage:**  
Load the dataset as a CSV or pandas DataFrame and use it to benchmark LLM behavior under sociopolitical stressors.

**Citation:**
If you find SocialHarmBench useful, please feel free to cite our work as such:

```bibtex
@misc{socialharmbench,
  title        = {SocialHarmBench: Revealing LLM Vulnerabilities to Socially Harmful Requests},
  author       = {Punya Syon Pandey and Hai Son Le and Devansh Bhardwaj and Rada Mihalcea and Zhijing Jin},
  year         = {2025},
  url          = {https://arxiv.org/abs/2510.04891},
}
```