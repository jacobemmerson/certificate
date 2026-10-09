# Relevance-scored sub-sampling against legal-group anchors

## Context

New sampling design: each benchmark is assigned to a legal group (a Code of Practice 1.3/1.4 leaf); every item in it is embedded and scored by cosine similarity to that group's reference text (legal description plus hand-picked exemplar items); selection keeps items above a threshold with a percentile fallback, and a per-benchmark relevance value controls how sharply selection concentrates on the top scores. Draw is deterministic (rank, tie-break by hash). The per-benchmark relevance values and the leaf assignments do not exist yet; the design must run with defaults until they arrive.

What exists and is reused: the MiniLM embedding cache and `scripts/embed_items.py::embed` (any `{key, text}` record embeds), `embed_key`/`load_embeddings`/`_vectors` in `prepare.py`, `blake2b(seed:sample_id)` tie-breaking, `_select`'s dispatch on `source.select`, water-filling over `len(rows)` (the filter's floor keeps every pool at least 3.5× its provisional share, so shares do not change), `Row.metadata` passthrough and `dropped.jsonl`/`meta.json` reporting. `Source.leaf` exists but drives nothing.

## Assessment of the proposal (what I would change, and why)

1. **A 0.9 cosine constant on MiniLM is near-duplicate territory.** The near-dedup tier already drops pairs at 0.92. Items scoring ≥ 0.9 against an exemplar are rewordings of that exemplar, so the constant would keep almost nothing and the percentile fallback would fire every time, making the design "top 1% by similarity to the exemplars" in disguise. Legal prose scores lower still (0.2-0.5 against user prompts on this model). **Change:** calibrate the threshold per leaf from the anchors themselves: θ_leaf = the 10th percentile of each exemplar's leave-one-out max-similarity to the other anchors of that leaf. It is deterministic, data-driven, and adapts when a stronger embedder replaces MiniLM. `0.9` stays available as an explicit override.
2. **The 99th-percentile fallback is budget-blind.** 1% of a 76-row pool is one item; 1% of scisafeeval is 257. The screen needs `SCREEN_FACTOR × share` candidates per source, and strata need coverage. **Change:** the fallback floor is `n_floor = max(ceil(SCREEN_FACTOR × provisional share), ceil(0.01 × pool))`, where the provisional share comes from water-filling the unfiltered pools. Threshold_eff = min(θ_leaf, score at rank n_floor). This is the user's rule ("use the percentile if it keeps more than the constant") with the percentile sized to what the pipeline needs.
3. **"Temperature" has no effect on a deterministic rank** (a monotone transform does not change order). The meaningful deterministic knob is *how far down the ranking selection may reach*. **Change:** relevance r ∈ (0, 1] sets the candidate window: `n_cand = n_needed + round(r × (stratum − n_needed))` clipped to `[n_needed, stratum]`, where `n_needed` is the existing screen pre-selection size. r = 1 (fully representative) lets the existing uniform/diverse selection roam the whole passing stratum; r → 0 (proxy) restricts it to the top n_needed by score. (Final review: the earlier `ceil(n_needed / r)` was inverted, since r = 1 then meant the narrowest window.) Existing strata, diversity anchors and the screen operate unchanged inside the window. `Source.relevance = None` (the default until per-benchmark values are supplied) applies no window.
4. **Exemplar-driven scoring selects items that look like the exemplars.** That is the intent for proxies, but for representative benchmarks it narrows coverage. The window in (3) plus the anchored farthest-point selection already in place is the mitigation; the report should show per-source score distributions and which anchor each kept item matched, so a leaf whose selection collapses onto one exemplar is visible.
5. **Legal text alone is a weak anchor; keep it, but do not rely on it.** Score = max cosine over all anchors of the leaf (legal paragraph and exemplars alike), and record which anchor hit. If the legal paragraph never wins, the report says so.
6. **Embedder.** MiniLM (384-d, dedup model) is adequate to start and keeps one cache. The scoring code reads vectors through one function, so swapping to a stronger model later means a second cache file and one constant, not a redesign.

## Design

**Inputs (data, committed):**
- `datasets/prepare/cluster/leaves.toml`: one `[[leaf]]` per legal group: `id` (slug), `title`, `cop_ref` ("Appendix 1.4(b)"), `legal_text`, `exemplars = ["…", …]`, optional `threshold` override. Starts with the leaves the current 32 sources map to; grows with the annotators' work.
- `Source.leaf` becomes the slug of the leaf the benchmark serves (one per source; a benchmark serving two leaves is two `Source` entries, as `human_rights_udhr/echr` already are). `Source.relevance: float | None` (per-benchmark value; None → role default). Both recorded in `meta.sources`.
- `datasets/cache/embeddings/leaves.npz`: anchor vectors, produced by the same `embed_items.py` from `datasets/cache/leaves.embed_input.jsonl`. Committed with the other caches (see the reproducible-sampling plan).

**Scoring (`prepare.py`, pure functions):**
- `leaf_anchors(leaf, embeddings) -> (names, matrix)`; `relevance_scores(rows, anchors, embeddings) -> (scores, hit)` = `np.round(V @ A.T, 9)` max over anchors (float64, per the determinism plan).
- `leaf_threshold(anchors, override) -> float`: leave-one-out 10th percentile, or the override.
- `relevance_filter(pools, leaves, embeddings, budget) -> (pools, dropped, report)`: provisional shares via `allocate_budget(pools, budget)`; per source: scores, `n_floor`, `threshold_eff`, keep `score ≥ threshold_eff`; each dropped row recorded with tier `"relevance"`, its score, the leaf and the threshold; each kept row gets `metadata["relevance"] = score` (for a `group_key` source, the group's best arm on every arm, with the arm's own score in `relevance_own`) and `metadata["relevance_anchor"] = hit`. Sources with no `leaf` pass through untouched (reported as `relevance_status: "unscored"`).
- Window in `_take`: `n_cand = n_needed + round(r × (len(indices) − n_needed))`, clipped to `[n_needed, len(indices)]`; the top `n_cand` by `(−score, key_bytes)` are kept in input order, then the existing `_select` runs. No window when `Source.relevance` is None, the source has no leaf, a row lacks a score, or the leaf is uncalibrated (< 2 exemplars, no override: rows are scored, nothing is dropped).

**Placement in `build_risk`:** exact → near → cross-exact → cross-near → **relevance filter** → water-fill → strata → windowed pre-select → screen → fill. The filter precedes water-filling. The floor guarantees every kept pool is at least 3.5× its provisional share, so water-filling is unaffected by the filter in practice; it only removes off-topic candidates.

**Reporting:** `meta.sources[name]` gains `leaf, relevance, relevance_threshold, relevance_floor_used (bool), score_quantiles {min, p10, p50, p90, max}, anchor_hits {anchor: count}`; `meta.leaves` lists each leaf's threshold and anchor count; `print_report` warns when a source's floor was used (threshold did not bite) or when > 80% of kept rows hit one anchor.

**Clusters.** Output stays per risk module for now; leaves are a per-source attribute. When the annotators' mapping lands, regrouping modules by leaf is a mechanical move of `Source` entries, and `BUDGET` moves with them.

## Tasks

1. **Schema and data.** `Source.relevance`; `leaves.toml` with the leaves implied by today's sources (fill `legal_text` from the CoP; exemplars: 2-5 per leaf from `annotations.csv` `sample_item_*` where a benchmark is unambiguous, else left for the annotators); loader `datasets/prepare/cluster/leaves.py` (toml → dict, validated: unique ids, non-empty legal_text, every `Source.leaf` resolves). Tests: loader validation; registry test that every `leaf` string resolves.
2. **Anchor embeddings.** `require_embeddings` also collects anchor texts into `leaves.embed_input.jsonl`; `EMBED_COMMAND` gains `--risk leaves`; `load_embeddings("leaves")`. Test with the existing `embedded()` helper.
3. **Scoring + filter.** The four functions above, inserted into `build_risk`; `dropped.jsonl` tier `relevance`. Tests on synthetic pools: threshold calibration (leave-one-out), floor sizing on a tiny and a huge pool, pass-through for unscored sources, determinism (two runs identical), the floor keeps shares unchanged after filtering.
4. **Window in selection.** `_take` truncation by `r`; role defaults; tests: r = 1 reproduces today's selection on a pool with uniform scores; r = 0.25 picks from the top window; strata and anchors still honoured.
5. **Reporting + docs.** meta/report fields, warnings; BENCHMARKS.md "Sampling" gains the relevance tier and the threshold rule; CONTRIBUTE.md `leaf`/`relevance` fields; the reproducible-sampling plan's cache list gains `leaves.npz`.
6. **Dry run (no screen, no generation).** `prepare --risk <r> --dry-run` per risk against the committed caches: print per-source threshold, floor-used, kept/pool, anchor hits. Review the numbers before any rebuild; a rebuild re-uses the existing screen verdicts (keys unchanged) and only re-screens newly surfaced candidates, which needs your go.

Order 1 → 2 → 3 → 4 → 5 → 6. Commits per task.

## Verification
- Full suite green after each task (`uv run python3 -m unittest discover tests`).
- Task 3: on a synthetic pool with planted exemplar look-alikes, the filter keeps exactly the look-alikes plus the floor; the same run twice is byte-identical.
- Task 6: dry-run tables for all five risks; `relevance_floor_used` should be rare once thresholds are calibrated; no source with > 80% hits on a single anchor without a warning.

## Open for you
- The leaf list and legal texts (CoP Appendix 1.3/1.4 excerpts) and the per-benchmark leaf assignment: I can draft `leaves.toml` from the 32 sources' current cluster and the annotation `task_description`, marked provisional, for the annotators to correct.
- Whether proxies (`role="diagnostic"`) should default to r = 0.5 or stay at 1.0 until real relevance values exist.
