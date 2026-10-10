# Sampling: cluster budget, cluster-level dedup and diversity, distribution report

Date: 2026-10-07. Branch: `refactor/source-contract`. Status: draft for review.
Builds on `2026-10-05-pipeline-refactor-design.md` §1 (embedding dedup + Hermes screen); the
tier order there stands. All code lives in `datasets/prepare/cluster/prepare.py` unless named.

## Problem

Each `Source` carries a hand-set `quota`. With 8+ clusters and ~60 new benchmarks the quotas
would be guessed per source and never add up to a cluster size anyone chose. Two sources can
also fill their quotas from near-identical regions of embedding space, and paraphrased items
shared across benchmarks pass the exact-only cross-source tier. Finally, the build can land
under its target without saying so: `_allocate` stops when no stratum can move, the screen
accepts a shortfall once the pre-selection covers a stratum, and neither number reaches
`meta.json`.

## Decisions

| | Decision |
|---|---|
| Budget | One integer per cluster, `BUDGET` in `sources/<risk>.py`. Water-filled across sources after dedup. `Source.quota` becomes an override. |
| Filter order | Unchanged: exact → near (per source) → cross-source exact → **cross-source near (new)** → water-fill → stratify → pre-select 3.5× → screen → fill. The screen stays on candidates only, so budget decides screen cost. |
| Diversity | Still per source, but farthest-point starts from what the cluster has already selected. |
| Reporting | Per-source `allotted`/`kept`/`shortfall`, per-stratum pool vs kept counts and a divergence number, plus exact drops in `dropped.jsonl`. Shortfalls warn; they do not fail the build. |

## 1. Cluster budget (water-filling)

**Config.** Each risk module declares `BUDGET: int` beside `SOURCES`. `sources/__init__.py`
exposes `budget_for(risk) -> int`; a module with sources and no `BUDGET` is a test failure.
Initial values equal today's row counts (cbrn 186, cyber 300, loss_of_control 140,
manipulation 562) so the first rebuild changes shares, not cluster size. Raising to the 2× target
is an operator edit, one line per cluster.

**Rule** (`allocate_budget(pools, budget) -> dict[name, int]`, pure function, runs in
`build_risk` after cross-source dedup):

1. Sources with `quota` set take `min(quota, pool)` and are removed from the pool list.
   `quota` keeps its current meaning (rows, or groups when `group_key` is set).
2. `remaining = budget - fixed`. Free sources are sorted by `(pool_rows, name)` ascending.
   The i-th takes `share = remaining // (free_left)`, then `take = min(pool_rows, share)`,
   `remaining -= take`. Small sources keep everything; unused share flows to larger ones.
3. Any integer remainder is handed out +1 to the largest free sources, largest first, capped
   at pool size.
4. For a `group_key` source, `pool_rows` is its row count and `take` is converted to groups:
   `take_groups = take // group_size`, `group_size = round(rows / groups)`.

Sorting by pool size before name makes the result a function of the data, not of registry
order; ties resolve on the name so it stays deterministic.

**Plumbing.** `stratified_sample`, `_grouped_sample` and `_row_sample` take `quota: int | None`
as a parameter instead of reading `source.quota`. `build_risk` passes the allocation. The
`quota is None` branch stays for callers (tests) that want everything.

**Not done.** No `weight` field, no per-source minimum. If a source needs more than its share,
set `quota`. No automatic cap on pool size before dedup (the "no cap" decision stands; see §6).

## 2. Cross-source near-dedup (new tier 2b)

`cross_source_near_dedup(pools, embeddings, tau=COSINE_TAU) -> (pools, dropped)`, after
`cross_source_dedup`:

- Rows from sources with `dedup=False` are excluded, respecting the per-source opt-out.
- Payload is `row.query` for every row: `dedup_on` names a metadata field that has no
  counterpart in another source, so the delivered prompt is the only comparable text.
- Vectors are built with the existing `_vectors`; similarity is the same blockwise rounded
  matrix product as `near_dedup`. A pair is a candidate when `row.source` differs and
  `_distinguishable(left, right, ())` holds (the MCQ target guard still applies).
- Greedy drop order is identical to `near_dedup` (highest similarity first, drop the higher
  index). Pools are concatenated in registry order, so the later source loses, matching the
  exact cross-source rule.
- Records use tier `"near_cross_source"`, same fields as `"near"`.

The within-source and cross-source passes stay separate rather than one pass over the union
because `tau`, `dedup_on` and `distinct_on` are per-source declarations. Cost is one more
`N × N` blocked product per cluster, the same order as the per-source pass.

## 3. Cluster-level diversity (anchored farthest-point)

`_diverse_order` gains `anchors: np.ndarray | None` (unit vectors of rows already selected in
this cluster). When given, `nearest` is initialised to each candidate's maximum similarity to
any anchor and the first pick is `argmin(nearest)` instead of the stable-order head. With no
anchors the behaviour is unchanged.

`Caches` gains `selected: list[np.ndarray]`. After each source is sampled, `build_risk`
appends the query-payload vectors of its kept rows (uniform sources included: their picks are
anchors even though they do not spread). Anchors are passed only to sources whose payload is
the query (`dedup_on is None`), because a source spread on a metadata field has nothing
comparable to anchor against.

Result: shares are decided by §1, items by farthest-point against the whole cluster so far.
Registry order determines who anchors whom; it is already the tie-break everywhere else.

## 4. Reporting

**Per source** (in `report[name]`, written to `meta["sources"]`):

- `allotted` (the §1 allocation), `kept`, `shortfall = allotted - kept`. This captures both
  `_allocate` stopping early and the accepted screen shortfall.
- `strata`: `{stratum_key: {"pool": n, "kept": k}}` replacing the bare count. Keys are the
  joined `stratify` values.
- `divergence`: total variation distance between the pool and kept stratum distributions
  (`0.5 × Σ |p_pool − p_kept|`), rounded to 3 decimals; `null` when unstratified.

**Per cluster** (`meta` top level): `budget`, `shortfall`. (`allotted` is not summed at cluster level: group_key sources allot groups, the rest rows.)

**Warnings** in `print_report`: any source with `shortfall > 0`; any non-`balanced` source
with `divergence > 0.10` (balanced sources skew by design). A report row gains `allot` and
`short` columns.

**dropped.jsonl.** `exact_dedup` returns `(rows, dropped: list[dict])` with tier `"exact"`,
`similarity: 1.0`, `kept`/`dropped` ids and 300-char texts, like every other tier.
`report["exact_dropped"]` becomes `len(dropped)`.

## 5. Incidental fixes (same functions, no new behaviour)

- Empty-payload rows have zero vectors and so are picked first by farthest-point as "far".
  `_diverse_order` places them last instead (`nearest = +inf` from the start, ties by hash).
- `_distinguishable` is renamed `_mergeable` (it returns True when a pair may merge).

## 6. Out of scope

- Default `stratify` from a native category column, and round-two pruning from fleet
  variance (after the first fleet run).
- Bounding `near_dedup`'s candidate list. It is quadratic on a templated source; the 2048-row
  block keeps the matrix bounded but not the pair list. Feasible today on a slurm node up to a
  few hundred thousand rows; revisit when a source exceeds that.
- Changing `SCREEN_FACTOR`, `COSINE_TAU` or the screen scope.

## 7. Verification

- Unit tests (`tests/test_clusters.py`): `allocate_budget` on small, equal and oversubscribed
  pools, `quota` override, group conversion, remainder handling, determinism under reorder;
  cross-source near drops a paraphrase and leaves same-source and differing-MCQ-target pairs;
  anchored diversity avoids an anchor's neighbourhood; shortfall and divergence values on a
  synthetic pool; exact drops appear in `dropped`; `BUDGET` present for every risk.
- `uv run python3 -m datasets.prepare.cluster.prepare --risk <r>` with the existing caches
  rebuilds each cluster at its current size (exit 2 if the Hermes cache is still missing, as
  today); `meta.json` shows `budget == rows + shortfall`.
- Suite stays green (652 OK, 1 skip, 2 expectedFailure).
