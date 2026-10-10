---
dataset_info:
  features:
  - name: goal
    dtype: string
  - name: original_term
    dtype: string
  - name: subject
    dtype: string
  splits:
  - name: train
    num_bytes: 568630
    num_examples: 3000
  download_size: 212743
  dataset_size: 568630
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
pretty_name: SOSBench
license: other
tags:
- ai-safety
- biosecurity
- chemistry
- biology
- medical
- physical
- phycology
library_name: datasets
extra_gated_prompt: >
  By requesting access you confirm that you will **NOT** use this dataset to
  design, facilitate, or evaluate activities that may cause harm to human
  subjects—including (but not limited to) the creation or deployment of
  biological, chemical, or other threats.   You also agree to comply with all
  applicable laws, institutional‐review requirements, and Hugging Face’s Terms
  of Service.
extra_gated_fields:
  Full name: text
  Affiliation / Company / Institution: text
  Email address: text
  Country of residence: country
  Intended use:
    type: select
    options:
    - label: Research
      value: research
    - label: Education
      value: education
    - label: Other (please specify in the request message)
      value: other
  Expected start date: date_picker
  I agree to use this dataset for non-commercial purposes ONLY: checkbox
  I agree to delete local copies if my access is later revoked: checkbox
---

# SOSBench


## 1  Intended Use

| Category              | Details                                                                                                                                                                                                                                                                                         |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Primary purpose**   | Academic research on LLM safety, alignment, oversight, red‑teaming and policy analysis.                                                                                                                                                                                                         |
| **Secondary purpose** | Classroom teaching and demonstration of safety attacks/defences (graduate level or above).                                                                                                                                                                                                      |
| **Forbidden uses**    | \* Designing, facilitating or evaluating real‑world harmful activities (e.g. biothreat creation, chemical weapon design). <br>\* Commercial exploitation or hosting the dataset behind a paywall. <br>\* Generating or disseminating disallowed content under the HF ToS or any applicable law. |

### 1.1  User responsibilities

By downloading or using SOSBench you **affirm that you will**:

1. **Comply with all applicable laws and/or institutional review‑board (IRB) requirements.**
2. **Refrain from using the data to cause harm**, including but not limited to the creation or deployment of chemical, biological, radiological or nuclear threats.
3. **Cite the dataset** in any disseminated work that uses or builds upon it.
4. **Delete local copies** if your access is later revoked or if you cease to comply with the RUA.

---

## 2  Dataset Overview

| Split | # examples | 
| ----- | ---------- | 
| Train | 3 000      | 


### 2.1  Collection methodology

Prompts were evolved from regulation‑derived seed terms using coverage‑driven mutation and expert review. Final questions were validated by domain experts for realism and potential misuse impact. No personal or proprietary data are included.

### 2.2  Limitations & potential risks

* **Coverage** – The six domains do not exhaust all possible high‑risk areas.
* **Prompt realism** – While expert‑curated, prompts may not capture every adversarial tactic.
* **Model misalignment** – Evaluating with SOSBench may surface previously unseen unsafe behaviour; publishing generations could itself spread harmful knowledge.

Researchers must perform separate *output safety filtering* when releasing model completions.

---

## 3  Responsible‑Use Agreement (RUA)

The RUA supplements the CC BY‑NC‑SA 4.0 license and **overrides it in case of conflict.** Key clauses:

1. **Non‑commercial research or education only.**
2. **No redistribution** outside your immediate research team **without written permission.** Forking inside HF with identical gating is allowed.
3. **No weaponisation or facilitation of harm.**
4. **Mandatory citation** of the dataset and RUA.

A full copy of the RUA is included in `LICENSE_RUA.txt`.

---

## 4  Licensing & Access

* **License:** [`CC BY‑NC‑SA 4.0`](https://creativecommons.org/licenses/by‑nc‑sa/4.0/) **+ Responsible‑Use Agreement** (see above).
* **Access:** Gated on Hugging Face; applicants must fill the form shown below and agree to the RUA.
* **Redistribution:** Permitted *only* if (a) non‑commercial, (b) the README, RUA and original license are retained *unaltered*, and (c) equivalent access controls are applied.

---


## Citation

```bibtex
@article{jiang2025sosbench,
  title={SOSBENCH: Benchmarking Safety Alignment on Scientific Knowledge},
  author={Jiang, Fengqing and Ma, Fengbo and Xu, Zhangchen and Li, Yuetai and Ramasubramanian, Bhaskar and Niu, Luyao and Li, Bo and Chen, Xianyan and Xiang, Zhen and Poovendran, Radha},
  journal={arXiv preprint arXiv:2505.21605},
  year={2025}
}
```