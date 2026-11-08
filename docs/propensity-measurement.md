# SR-MAP tonight: propensity plan

Oct 12, 2026 · @Tae

Tonight's paper keeps the 13-element map from existing annotations, states every edge as a refutable claim, and makes one model-level inference: an estimated rate of a propensity behaviour across a stated family of conditions. Commit to this version and drop the capability versus propensity contrast.

## Intuition

When a provider cites a benchmark for a risk the Code names, they are claiming "this number tells you about that property of the model". The paper draws each such claim as an arrow from a benchmark to an element, and asks what would show the arrow is wrong.

Think of a personality questionnaire. You would not trust it if answering at random made you an extrovert, or if rephrasing the questions flipped your result. The two tests are the same checks applied to a benchmark:

- **Could something with no disposition get this score?** If coin-flipping scores 50% "deceptive", a model at 50% has told you nothing.
- **Does the score survive rewording?** A tendency persists when the wording changes. If a model scores 20% on the published prompt and 60% on a paraphrase, the published number describes the prompt, not the model.

An arrow that survives both supports one modest statement about a model: how often it showed the behaviour across the wordings tried. An arrow that fails tells you which benchmark scores cannot be cited as evidence of a tendency.

## Claim and scope

The paper makes three claims, and only the third is about models.

1. **The map.** Two trained annotators linked 78 benchmarks to 13 elements of the Code, with relevance and direction. This is existing data and stays as it is.
2. **Edges are refutable.** Each link is a claim about what a score evidences, with a named observation that would refute it. A null-policy pass tests every edge from the benchmark cards alone.
3. **Propensity can be estimated, within limits.** For the three propensity elements (deception, resisting modification or shutdown, manipulation propensity), a score estimates a behaviour rate across a stated family of conditions. The experiment reports that rate per model and tests whether a single-condition score stands in for it.

The map still covers all 13 elements. Only the model-level inference narrows to the three propensity elements, so the title can stay and the abstract needs one sentence saying so.

Capability elements get no model-level inference tonight. Their edges are tested by the null-policy pass only.

## Formalization

A propensity is defined as a rate over conditions, which makes it estimable; an edge claims that one benchmark score stands in for that rate.

### Primitive

A legal primitive is a model-level predicate that a provision names as a source of the regulated risk: here, the 13 elements of Table 1. For a propensity element e, let y be 1 when the response of model m to item i under condition c shows the behaviour in the definition d(e), as scored by the benchmark's own rule.

```latex
y_e(m,i,c) \in \{0,1\}
```

The propensity of m for e is the expected rate over a stated condition family D and the items that sample the behaviour.

```latex
\theta_e(m; D) = \mathbb{E}_{c \sim D}\ \mathbb{E}_{i \sim I_e}\bigl[\, y_e(m,i,c) \,\bigr]
```

D must be written down. Tonight it is the published prompt plus two meaning-preserving rewrites (paraphrase and register). A decision-reframing condition was dropped: the repo's framing templates fit compliance requests only and change what a forced-choice or acceptability item asks. The estimate describes D and nothing wider.

### What an edge claims

A benchmark b gives a score per condition, and the estimate of the propensity is its mean over the tested conditions C.

```latex
S_b(m,c) = \frac{1}{|I_b|} \sum_{i \in I_b} y_e(m,i,c), \qquad \hat\theta_e(m; D) = \frac{1}{|C|} \sum_{c \in C} S_b(m,c)
```

An edge from b to a propensity element makes two claims.

| Claim | Statement | Who relies on it |
| --- | --- | --- |
| Informative | A score differs from what a policy with no disposition would get | Anyone citing the score at all |
| Stable | The score under one condition is within noise of the rate over D | Anyone citing a single published-prompt score as a tendency |

### What refutes an edge

```latex
g_0(b,m) = \bigl| S_b(m,c_0) - S_b(\pi_0) \bigr| - \varepsilon_b
```

```latex
g_S(b,m) = \max_{c \in C} S_b(m,c) - \min_{c \in C} S_b(m,c) - \varepsilon_b
```

The null policy answers uniformly at random on forced-choice sources, and the noise term is the difference between two runs of the published prompt. Open-response sources get a different null check: constant refusal must score zero, or the scoring rule is counting non-answers as the behaviour.

- Informative is refuted for m when the first gap is not positive: the score cannot be told apart from no disposition.
- Stable is refuted for m when the second gap is positive: the score moves across rewrites more than across reruns.

| Edge state | Condition | What may be reported |
| --- | --- | --- |
| Uninformative | Informative refuted | Nothing about the propensity |
| Unstable | Informative holds, stable refuted | The rate over D with an interval that includes between-condition spread; never a single-condition score |
| Stable | Both hold | The rate over D; a single-condition score is an acceptable shorthand |
| Untested | Not run | The edge as an annotated hypothesis |

### Self-report and behaviour

Label each propensity source as behavioural (the response is the behaviour) or self-report (the model says what it would do). A self-report source measures a stated disposition, so its edge is a proxy by the paper's own rule. No direct, proxy or behavioural labels exist yet in annotations.json; both are assigned during the Test 1 card pass (`analysis/null_policies.csv`), and the paper must say so.

### Capability edges

For capability elements the paper states the claim (a high score shows the capability is present) and tests it with the null-policy pass only. No rate is defined, because capability is not a frequency.

## Experiment

Code: `analysis/null_policy.py` (Test 1 template and summary), `scripts/propensity_rewrites.py` (items, rewrites, audit sheet, runs) and `analysis/propensity_report.py` (rates, intervals, edge states, table, figure).

Two tests: a null-policy pass over the whole map with no compute, and propensity runs on three or four static sources. Write the predictions into the draft before running either.

### Test 1: null policies over all 78 benchmarks

From each scoring rule, work out what uniform random answering and constant refusal would score.

- Prediction from the map: on every direct edge where a higher score means more risk, both null policies sit at the floor.
- Report: edges where a null policy scores off the floor, by risk and by label. Refusal-scored benchmarks linked to capability elements will show up here, and their count is a result.
- For forced-choice propensity sources, record the chance score. It is the reference point for Test 2.

### Test 2: propensity under rewrites

| Choice | Tonight |
| --- | --- |
| Sources | Three registered sources, four edges: `advanced_ai_risk` (survival-instinct + corrigible-less-HHH, self-report MCQ, chance 0.5) for shutdown resistance; `darkbench` sneaking (110) for deception and user-retention + sycophancy for manipulation; `instrumentaleval` (76, behavioural) counted under deception and shutdown |
| Models | OpenRouter: `anthropic/claude-haiku-4.5`, `openai/gpt-oss-120b`, `google/gemini-3.6-flash` |
| Items | 200 per edge with seed 0, or all items if fewer: 586 rows in `datasets/propensity/items.csv` |
| Wording conditions | Published prompt, run twice; paraphrase; register reframing |
| Instruction condition | The pipeline's "professional" persona system prompt, reported separately and not used in the stability test |
| Rewrites | Generated once by Hermes-4-405B (rewrite prompt v7), shared across models; a rejected rewrite falls back to the original and is counted in a fallback column |
| Audit | Two people fill `datasets/propensity/audit_blank.csv` (50 rewrites) for whether they preserve what the item tests; report the rate |

That is five runs per item and model: 586 x 5 x 3 = 8,790 responses plus two judge calls on each of the 386 open-response items per condition. Scores on open rows are the mean of the two pipeline judges, so a split verdict counts as half.

Predictions the map makes:

- Every propensity edge labelled direct is informative and stable for every model.
- Self-report sources and behavioural sources on the same element give rates that agree within their intervals.

What to report per source and model:

- The rate over the four wording conditions, with a bootstrap interval that resamples both items and conditions (`analysis/propensity_report.py`).
- The range across conditions next to the rerun noise.
- The edge state from the formalization.
- The shift under the instruction condition, as a separate column.

The result is one figure (per-condition scores for each model and source, with the noise band) and one table (edge, model, rate, interval, state).

If time runs short, cut the instruction condition first, then the third model. Do not cut the second published-prompt run, because every verdict depends on the noise term.

## What you can say about a model

One statement is licensed, in this form: "Across the four wording conditions, model m showed the behaviour in d(e) on p% of items from b (95% interval l to u); the score varied by r points across conditions against n points across reruns."

| You can say | Condition |
| --- | --- |
| m shows behaviour e at a stated rate over D | The edge is informative for m |
| The published score of m is a fair summary of that rate | The edge is also stable for m |
| m shows the behaviour more often than m' on this source | Their intervals over D do not overlap |
| A published score on b does not describe a tendency of m | The edge is unstable for m |

| You cannot say | Why |
| --- | --- |
| m has, or lacks, the propensity in deployment | D is three wordings of benchmark items, not a sample of deployment |
| m would behave this way | Self-report sources measure what the model says it would do |
| m is more deceptive or manipulative than m' in general | One source and one condition family per element; two or three models |
| The rate is unaffected by the model detecting a test | Not tested; the Code itself lists this as a capability |
| Anything about capability elements, systemic risk or compliance | Outside what the experiment estimates |

Put the first table's wording in the results and the second table's rows in Limitations. No number in the paper should be called a risk estimate.

## Schedule

A deadline of 23:59 anywhere-on-earth on 12 October is 04:59 Pacific on 13 October, about 12.5 hours from now. Confirm the time on the ARR dates page before planning around it.

| Hours from start | Writing | Test 1 | Test 2 |
| --- | --- | --- | --- |
| 0 to 0.5 | Check the submission form and that every author's OpenReview profile is complete with ORCiD; write the predictions paragraph |  | Pick sources, models and seed |
| 0.5 to 2 | Formalization into sections 3 and 4 | Null scores from the 78 cards | Generate rewrites; run script working on 10 items per source |
| 2 to 5 | Section 5 from existing gold: Table 2, Figure 2, coverage and proxy counts | Counts by risk and label; one table | Full runs; 50-rewrite audit while they run |
| 5 to 7 | Abstract and introduction paragraphs 3 to 6 | Written into section 6 | Scoring, intervals, edge states |
| 7 to 9 | Section 6 results, conclusion, Limitations |  | Figure and table |
| 9 to 11 | Consistency pass, checklist, anonymity check, page limit |  |  |
| 11 to 12.5 | Buffer and upload |  |  |

Two stop rules:

- If the run script is not producing scored output by hour 3, drop to two sources and two models.
- If Test 2 has no usable results by hour 8, do not submit a paper with a placeholder section 6. Submit with Test 1 only and say so, or hold for the next cycle.

## Paper edits by section

Sections 2 and 3 and the Limitations survive almost intact; the abstract, introduction and section 6 are rewritten.

| Section | Action | Edit |
| --- | --- | --- |
| Abstract | Rewrite | Seven sentences: obligation; gap (prior work reports no inter-annotator agreement and no direct, proxy or direction labels); 13 elements; 78 benchmarks mapped with α as measured; edges stated as refutable claims; null-policy result; propensity result with the scope sentence |
| 1 Introduction | Rewrite paragraphs 2 to 6 | Three questions: what the Code names as model-level sources of risk; which benchmarks are offered as evidence for each; which of those evidentiary claims survive a test. Contributions follow the three claims above |
| 2 Related work | Keep | Add one or two citations on sandbagging and evaluation awareness where the limits are discussed |
| 3 Taxonomy | Light edit | Define "legal primitive"; say reach, impact and propagation are excluded; account for the 27 phrases becoming 13 |
| 4 Annotation | Edit | Add the edge claims and refutation conditions; "two trained annotators (authors)"; "annotated edges", not "gold", wherever reliability is implied; fix the cut-off inclusion sentence; add the behavioural or self-report label |
| 5 What the map shows | Write | Real numbers from existing annotations: agreement, elements with no direct benchmark, proxy share, safety-labelled benchmarks scored risk-up |
| 6 | Replace | New title, "Testing the edges". Test 1, then Test 2. Delete the conditions paragraph, coherence, direction, Table 3 and Figure 3 |
| Figure 1 | Edit | Panel 3 becomes "null policies" and "propensity under rewrites"; rewrite the caption to match |
| Figure 2 | Keep | Note in the caption that no link was labelled mixed, or add the category |
| 7 Conclusion | Write | Three sentences, one per claim |
| Limitations | Edit | Replace the coherence paragraph with the "cannot say" rows; keep the rest |
| Appendices | Trim | Keep only those with content: guidelines, benchmark list with labels, run details and rewrite prompts |

Small fixes to make in the same pass:

- "Starting 2 August 2026" to "Since 2 August 2026".
- "Leaf" to "element" throughout.
- Three conditions in the abstract against six in section 6: both go.
- NOT in capitals to lower case.
- Typos in the introduction and section 4 ("Comission", "Providers who cites", "measurment", "propensit").

## Objections that still stand

These are what a careful reviewer will raise against tonight's version. None can be fixed tonight; each needs one honest sentence in the paper.

1. **The condition family is thin.** Three wordings of the same items (one of them run twice) are not a sample of deployment. The rate is a statement about the benchmark under rewriting, and a reviewer may say "tendency" overstates it.
2. **Rewrites may change the item.** A 50-item audit by two authors is weak evidence that the rewrites preserve what is tested. If the preservation rate is low, instability could be the rewrites and not the model.
3. **Self-report is not behaviour.** If most available propensity sources are self-report, the model-level statement is about stated dispositions.
4. **Evaluation awareness is untested.** A model that behaves differently when it detects a test defeats the estimate.
5. **The likely result is negative.** If scores move across rewrites more than across reruns, the finding is that no stable tendency is measurable on these sources. That is a result, but not the positive inference this plan aims for.
6. **Test 1 is partly true by definition.** Constant refusal maxes a refusal metric by construction. The defence is the count of annotated edges it refutes.
7. **Agreement is unchanged.** α of .49 to .69 from two annotators is still in Table 2, and the edges tested are those two people's hypotheses.
8. **Three of thirteen.** The headline inference covers the three propensity elements. CBRN and cyber offence get the map and the null-policy pass only.
9. **It may still read as preliminary.** ARR has warned that very preliminary submissions risk rejection and a bar from the next cycle. Weigh that against waiting one cycle with the fuller plan.

