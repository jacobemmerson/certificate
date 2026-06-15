# ESAI Harm-Bench-Legal Map — Ontology

This document describes the data model underlying `mapping/maps/*.csv` and
`mapping/utils/dag.py`. Reference it when working on the front-end so that
field names, relationships, and coding conventions stay consistent with how
the maps are annotated.

## Purpose

Map benchmarks to laws so we can answer two questions:

1. Which benchmarks are relevant to which legal provisions?
2. Where are the coverage gaps (i.e. harms a legal provision covers but no
   benchmark measures)?

## Scope

- **Jurisdictions:** UK and EU.
- **EU:** GPAI / foundation models, i.e. AI Act Chapter V (Arts 53–55), the
  GPAI Code of Practice, and GDPR where it applies concurrently. (Not the
  Annex III high-risk use regime.)
- **UK:** sectoral / principles-based; regulator guidance and existing laws
  applied to AI; no horizontal AI statute.

## Architecture

**Harms are the "hub"** (think of it as the join key). Benchmarks and
provisions each point into it. The benchmark↔law edge is obtained by joining
on the harm.

This simplifies labeling from `O(#Benchmarks x #Laws)` to `O(#Harms)`, and
gives the whole project a tangible shared language.

## Nodes

### Harm

A harm caused by AI. The taxonomy is based on the **MIT Risk Taxonomy** for
compatibility and comparability with their mapping efforts.

| Field | Description |
| --- | --- |
| `entity` | `individual` / `ai` / `other` |
| `intent` | `intentional` / `unintentional` / `other` |
| `timing` | `pre-deployment` / `post-deployment` / `other` |
| `mechanism` | Free-form theory of how the harm manifests itself |

**Coding instructions**

- There is no need to add the `entity`, `intent`, and `timing` fields
  yourself — they will be imported via the `ev_id`.
- The `mechanism` field is a causal theory of how we believe the harm to
  manifest. This forms the basis of the EU AI Office Risk Scenario template
  that we have to fill out when we submit benchmarks to them. So, when coming
  up with the mechanism, think about how this harm could manifest in a
  concrete application scenario.

  **Sycophancy example:**
  > approval-optimized affirmation → confidence inflation → suppressed
  > self-correction → harmful action in a high-stakes domain.

- Don't add new top-level categories. If you think we should add a new
  category, let's discuss.
- Before adding a new harm, check whether there is one already.
  - **If there is none:** add to the end of its subdomain.
    1. Add a row.
    2. Add the `harm_id` by following the pattern (i.e. increment the
       right-most number).
    3. Add the `ev_id` from the AI risk repository. Everything else is
       filled out already.
  - **If there is one:** add the row as described above, then add the
    `ev_id` to the "also known as" column.

### Benchmark

A benchmark, organized hierarchically because some papers propose entire
suites: **Suite → Task → Metric**.

| Field | Description |
| --- | --- |
| `quick ref` | Reference for the paper: last name of the first author + publication year (e.g. `Jin2026`). If a paper with this exact name and year already exists, append `b` (`c`, `d`, …) to the new one, and append `a` to the quick ref of the first paper (i.e. `Jin2026a`, `Jin2026b`). |
| `modality` | `intrinsic` (representational probe — e.g. asking a model about the gender of "nurse" vs "doctor") or `extrinsic` (behavioural probe — e.g. asking a model to rank CVs that are identical except for the gender of the applicant) |
| `interaction horizon` | `single-shot` / `multi-turn` / `agentic` |
| `aggregation scale` | At what scale does the benchmark test the harm? `instance-level` / `population-level` |
| `metric` | What it measures concretely |
| `communicated metric` | How the score is reported |

**Coding instructions**

- Each paper gets its own ID in the benchmarks table (`BX.YY.ZZ`). `X` is the
  paper ID. Each paper corresponds to a "suite".
- For each task in the paper, add a new row. Increment the `YY` part of the
  ID, starting at `01`. Add the name/label of the task to the `task` column.
- For each metric in a task, add a new row. Increment the `ZZ` part of the
  ID, starting at `01`. Add the name/label of the metric to the `metric`
  column. Same for the `communicated metric`.
- This means that for each paper added to the benchmarks table, there should
  be a row with an ID that looks like `BX.01.01` and has `task` and `metric`
  filled.
- Document your reasoning for assigning the properties above in the `notes`
  column.
- Set the `version` to `1`.
- You currently do not have to code the `modality`, `interaction horizon` and
  `aggregation scale` columns — pending further work on the ontology for
  technical evaluation approaches (see TODO below). If something is obviously
  intrinsic (e.g. it is a gender representation benchmark that probes the
  model to tell you whether a "doctor" is male or female) you may of course
  add this, but it is not required.

> **TODO (@Tae):** flesh out the *Method* section — ontology for technical
> evaluation approaches (`modality`, `interaction horizon`,
> `aggregation scale`).

### Legal provision

The unit is a **provision / article / law**, not the whole law.

| Field | Description |
| --- | --- |
| `jurisdiction` | `EU` / `UK` |
| `instrument` | The high-level instrument, e.g. "EU AI Act" |
| `instrument type` | `article` / `annex` / `code-of-practice` / `harmonised-standard` / `regulator-guidance` / `case-law` |
| `binding force` | `hard law` / `soft law` / `guidance` |
| `enforceability date` | Bound vs. enforceable differ — e.g. GPAI obligations binding since 2 Aug 2025; Commission enforcement from 2 Aug 2026 |
| `obligation type` | `evaluate` / `document-report` / `mitigate` / `disclose` / `prohibit` / `grant-right` |
| `citation` | Where exactly the provision occurs; as precise as possible |

## Additional attributes

Some additional attributes are not part of the ontology but help us humans:

- **Label:** Free-form description of what this object is, e.g. the paper
  title for a Benchmark Suite.
- **Version:** Version of this object. We might find that a benchmark has
  been saturated and a new version is released, or a law has been updated.
- **Notes:** Free-form description for the annotator.
- **`*_id`:** The ID field of this object. Just database stuff.

## Edges

### Benchmark — measures → Harm

Connects benchmarks and harms. Should be read as **"Benchmark B measures
harm H"**.

| Field | Description |
| --- | --- |
| `benchmark_id` / `harm_id` | References to the respective benchmark and harm |
| `strength` | How strong the correlation between the benchmark and the harm is, e.g. `direct` / `strong-proxy` / `weak-proxy` / `contested` |
| `basis` | Reason for why the annotator assigned the above strength level, e.g. `validated-against-downstream` / `face-validity-only` / `known-non-correlation` |
| `confidence` | How confident the annotator is in their assessment |

**Coding instructions**

- The `benchmark_id` is the one from the benchmarks table above.
- The `harm_id` is the `harm_id` in the harms table above.
- The table will automagically pull in the benchmark paper title, the harm
  category and subcategory, the harm domain, and the harm subdomain.
- `strength` is your best guess for how strongly the benchmark metric
  correlates with the harm. Use `notes` to give a quick reason for why you
  chose a particular level of strength.
- Use `confidence` to record how confident you are in your assessment.

### Legal provision — addresses → Harm

Connects legal provisions to harms. Should be read as **"Legal provision L
addresses Harm H"**.

| Field | Description |
| --- | --- |
| `coverage` | How this provision addresses the harm: `explicit` (named in text/annex) / `interpretive` (reading-in required) / `contested` |
| `justification` | Why this provision applies |

**Coding instructions (EU AI Act and CoP specific)**

We are mostly interested in **systemic risks**, so we have to check whether
the relevant provisions apply. If you can complete the coding process below
(and justify your decisions), we have a (strong candidate for a) systemic
risk.

1. Decide whether your harm is one of the four in CoP App. 1.4. If yes, code
   the relevant point. Also code Art. 56(1) and Recital 110.
2. Code the relevant points in CoP App. 1.1.
   - If no point applies, it's not a systemic risk.
   - Multiple points can apply.
3. Code the relevant points in CoP App. 1.2.1.
   - If CoP 1.2.1 (1) or (2) applies, also code EU AIA Art. 3(64).
4. Code the relevant points in CoP App. 1.2.2.
5. Code the relevant points in CoP App. 1.3.*.
6. Finally, if everything above worked out, code Art. 3(65).

**A note on coding harms in categories 7.2 and 7.3:** some of the harms
there are not systemic risks on their own, but contribute to systemic risks
in the sense of Art. 3(65). We believe those are still very valuable and we
should collect them and also develop and publish benchmarks. While they might
not enter as systemic risk evaluations on their own, they still provide
evidence for systemic risks.

### Benchmark — Harm — Legal Provision (three-way join)

Connect all three. Should be read as **"Benchmark B measures Harm H, which
Legal Provision L addresses"**.

Using the fields in the three nodes, we can enrich this statement, e.g.:

> "Benchmark B is a strong proxy for Harm H, which Legal Provision L
> regulates and puts an evaluation obligation on."

With this two- (three-?) way join, we can answer most questions we are
interested in:

- Is a given benchmark a good candidate for a certain provision?
- Is a provision covered by benchmarks?
- Are our benchmarks addressing something regulators care about?
- …

Importantly, **this can be computed automatically.**
