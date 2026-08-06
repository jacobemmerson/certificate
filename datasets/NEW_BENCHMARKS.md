> **Superseded.** Everything shortlisted here has been acted on: the clusters
> were renamed onto these four risks, and all six missing benchmarks are now
> vendored under `datasets/raw/`. Four are registered (`harmbench`,
> `cyberseceval_mitre`, `agentharm`, `injecagent`); SAD and Anthropic/persuasion
> are vendored but deliberately unregistered — see CLUSTERING.md § "Vendored but
> not registered" for why.
>
> Current state lives in [BENCHMARKS.md](BENCHMARKS.md) (what each source is and
> how it is scored) and [CLUSTERING.md](CLUSTERING.md) (roster, download
> manifest, on-disk status). **This file is kept only as the record of the
> selection decision and can be removed.**
>
> The sycophancy overlap flagged at the bottom was resolved by keeping both
> sources in `manipulation` and documenting the double-count, so the cluster
> reads as nine distinct constructs across ten sources.

## CBRN

| Benchmark | Coverage | URL |
|---|---|---|
| **WMDP** | 3,668 MCQs, bio/chem/cyber proxy hazard knowledge | https://huggingface.co/datasets/cais/wmdp |
| **SoSBench** | 3,000 regulation-grounded prompts, 500 each across chemistry, biology, medicine, pharmacology, physics, psychology; plus a 300-sample Lite subset. Implemented in Inspect Evals, which is a real adoption signal | https://github.com/SOSBench/SOSBenchEval · https://sosbench.github.io/ |
| **HarmBench** | 400 behaviours incl. chemical/biological and indiscriminate-weapons classes, plus 18 attack methods | https://github.com/centerforaisafety/HarmBench |

SoSBench is the strongest coverage pick here — open-ended generation grounded in actual regulatory text, which maps to CoP language more directly than WMDP's multiple-choice proxy knowledge.

## Cyber offence

| Benchmark | Coverage | URL |
|---|---|---|
| **CyberSecEval (MITRE + FRR)** | ATT&CK-mapped compliance across reconnaissance, evasion, execution; FRR catches the safety-utility tradeoff. Multilingual prompts included (machine-translated — the repo cautions about accuracy) | https://github.com/meta-llama/PurpleLlama/tree/main/CybersecurityBenchmarks |
| **AgentHarm** | 110 malicious agent tasks (440 augmented), 11 harm categories | https://huggingface.co/datasets/ai-safety-institute/AgentHarm |
| **InjecAgent** | 1,054 indirect prompt-injection cases over 17 tools | https://github.com/uiuc-kang-lab/InjecAgent |

The FRR component is the one I'd emphasize for a certificate: it's the only artifact in your whole map that measures over-refusal alongside the hazardous capability, which is what stops a "safe" score from just meaning "refuses everything."

## Loss of control

| Benchmark | Coverage | URL |
|---|---|---|
| **Model-Written Evaluations** (advanced-ai-risk) | Power-seeking, self-preservation, survival instinct, corrigibility, awareness — broadest static set for this risk | https://github.com/anthropics/evals |
| **SAD** (Situational Awareness Dataset) | 7 categories, 13 subtasks, 13k+ questions, own harness | https://github.com/LRudL/sad |
| **sycophancy-eval** | Feedback, answer, and mimicry sycophancy | https://github.com/meg-tong/sycophancy-eval |

## Harmful manipulation

| Benchmark | Coverage | URL |
|---|---|---|
| **DarkBench** | 660 prompts across six categories: brand bias, user retention, sycophancy, anthropomorphism, harmful generation, and sneaking; ICLR 2025 oral spotlight, runs as an Inspect eval | https://huggingface.co/datasets/apart/darkbench · https://github.com/apartresearch/DarkBench |
| **SocialHarmBench** | 585 prompts, 7 sociopolitical categories, 34 countries — propaganda, revisionism, surveillance, censorship | https://huggingface.co/datasets/psyonp/SocialHarmBench |
| **Anthropic persuasion** | Claims + model/human arguments with human persuasiveness ratings | https://huggingface.co/datasets/Anthropic/persuasion |

DarkBench is the better of the three on the multi-turn/undetectable-influence clause specifically — sneaking and user-retention are exactly the "individuals unaware of or unable to detect such influence" construct, and nothing else in your map operationalizes it.

One consistency note: DarkBench's sycophancy category overlaps sycophancy-eval, which I put under loss of control. Same construct, two risks. Pick one home for it in the certificate or the coverage numbers inflate.