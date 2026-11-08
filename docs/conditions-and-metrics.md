# Conditions and metrics: a one-page orientation

What a certification run actually measures. Written for an engineer joining the project;
full definitions live in [`pipeline/README.md § Metrics`](../pipeline/README.md#metrics) and
the code paths named below. Nothing here is normative; if this page and the code disagree,
the code wins and this page is wrong.

## The unit of measurement

One **item** is a row of a leaf dataset (`datasets/public/<leaf>.csv`): a request
(`item_text`) inside a scoring scaffold (`prompt_template`), plus the judge's `criterion`
and the `question_type` that picks the scorer. Every item is run under several
**conditions**. The control is the published wording; every other condition is the same
ask, transformed. Each condition's completion is judged to a safety value in **[0, 1],
higher is safer**, or the judge abstains. The certificate asks one question per item: *how
bad does the model get on this ask, under any of the ways we asked it?*

Transformations are generated once by an attacker model (`generate.py`) and frozen under
`datasets/generated/<leaf>/<family>.jsonl`. Every certified model replays the identical
variants, so differences between models are never attacker noise.

## Stage 2: surface perturbations

Same ask, different surface. A family is applicable per item through the CSV's `families`
column (`pipeline/utils/replay.py::family_applies`); detection-type rows get no rewrites
because the rewrite would move the thing being detected.

| Family | What changes | How many conditions per item | Produced by |
|---|---|---|---|
| `paraphrase` | wording and sentence structure of `item_text` | `--perturb-k` (default 1) | attacker rewrite |
| `register` | formal ↔ casual register | `--perturb-k` | attacker rewrite |
| `past_tense` | the request recast as historical | `--perturb-k` | attacker rewrite |
| `multilingual` | translation into `de`, `zh`, `ar`; the judge reads the English criterion against the translated exchange | 3, one per language | attacker translation |
| `framing` | the request wrapped as imperative / interrogative / enumerative | 3 templates | deterministic string templates, `compliance` rows only |
| `persona` | a benign professional or researcher system prompt replaces the system turn | 2 templates | deterministic, rows without their own system prompt |
| `reconsideration` | the control answer is challenged ("Are you sure? …") and the model answers again | 1 | live, at certification time; the only family with no artifact |

Rewrites only touch `item_text` and are re-rendered through the row's own `prompt_template`,
so the stage-1 scorer's output contract is unchanged. A rewrite the attacker refused or
botched is stored as `fallback=true` and replays as an abstention, never as the original.

Code: `pipeline/stage2_perturbation/` (`rewrite.py` prompts and defect filter, `framing.py`
templates, `solvers.py` replay), prompt versions in `pipeline/artifacts.py::PROMPT_VERSIONS`.

## Stage 3: agentic scenario (`--simulate`)

Same ask, inside a deployment. Each item has `--sim-k` (default 2) frozen **scenario
trees**: a deployment system prompt plus user turns keyed by the path of the model's prior
replies. The opening turn poses the original ask in context; follow-ups push back or narrow
but may never add harmful detail, because the harm judgment reuses the row's own criterion.

```
turn 1: system + opening ask           → reply → classify: refuse | hedge | comply
turn 2: turns["refuse"] or turns["hedge"] → reply → classify
turn 3: turns["refuse/hedge"] etc.     → reply       (depth 3, no further branch)
```

`comply` is terminal: a complied turn already sets the item's floor. The branch is chosen by
a classifier of the model's reply (`--sim-classifier`; regex pre-pass for plain refusals,
otherwise one cached judge call), never by an attacker, so every model walks the same tree.
Each judged turn is a condition `scenario_variant_<k>_t<depth>`; the judge sees the whole
rendered transcript as "the request".

Code: `pipeline/stage3_simulation/` (`prompts.py` tree prompt and parser, `solvers.py`
walk, `classify.py`).

## How conditions collapse to numbers

All published numbers are **0–100, higher is safer**. Abstentions are dropped from every
statistic; a pool with nothing scored is `null` (tree) or `NaN` (panel), never 0 or 100.

**Per item** (`pipeline/utils/scoring.py`):

- `worst` = min over every scored condition, **control included**. The item's Score.
- family value = mean of the family's conditions (k variants are repeats of one test),
  except `scenario` = min over turns (turns escalate, the deepest reached is the finding).
- `average` = mean over families of their family values, control counted as a family.

**Per leaf, per model** (`pipeline/utils/results.py`, written to `models/results/<model>.json`):

| Number | Definition | Question it answers |
|---|---|---|
| `baseline` | mean of control values over items | how safe is the published wording |
| `aggregate.average` | mean of per-item `average` | typical behaviour across conditions |
| `aggregate.worst` | mean of per-item `worst` | typical worst case |
| **`aggregate.tail`** | **CVaR@10%**: mean of the lowest ⌈0.1·n⌉ per-item `worst` values (n ≤ 10: the min) | how bad the worst tenth of items get; **the headline `scores.<leaf>`** |
| `by_family.<family>` | mean over sources of the family value | which transformation moves the model |
| `conditions.<family>.stability` | 100 · (1 − mean \|drift from control\|) | how far the judgment moved, regardless of direction |

Per-leaf figures are unweighted means over the leaf's pooled sources (a source marked
`diagnostic` is reported but not pooled), and `aggregate.tail` is taken over the union of
per-item worsts across those sources, not a mean of per-source tails.

**Inspect eval panel** (per task, in `logs/`): `safety_control`, `safety_<family>`,
`safety_worst`, `safety_average`, `stability_under_attack`, and `resilience_under_attack` =
% of items whose worst attack condition held at or above their own control. Resilience is
the perturbation audit's own question (did any transform find harm the bald query missed);
stability is a mean because "how far does it move" differs from "how bad does it get".

**Model aggregate** (`models/models.json`): a plain mean of each leaf's `average`/`worst`/`tail`
over leaves with a value. Grouping leaves into systemic risks is a human mapping that does not
exist yet.

## Reading a result

- A leaf is certified iff `scores.<leaf>` is a number. A `partial` or `failed` run publishes
  its tree but no headline, and never overwrites a leaf that certified cleanly.
- `status.<leaf>.refusals` counts samples the *provider* refused on content policy. Those are
  the hardest end of the dataset, so a partial figure is a score on an easier benchmark; the
  coverage counts (`scored`, `abstained`, `total`) ride next to every number for that reason.
- More variants mean a lower `worst`. That is what worst case means. Compare families through
  `by_family`, which gives each family one value per item.
