# Sampling Budget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace hand-set per-source quotas with one water-filled cluster budget, add cross-source near-dedup and cluster-anchored diversity, and make every shortfall and skew visible in `meta.json`.

**Architecture:** All logic stays in `datasets/prepare/cluster/prepare.py::build_risk` and its helpers. New pure functions (`allocate_budget`, `cross_source_near_dedup`, `_divergence`) slot into the existing tier order; `stratified_sample` takes its quota as a parameter; `Caches` carries the cluster's selected vectors as anchors. Config is one `BUDGET` constant per `sources/<risk>.py`.

**Tech Stack:** Python 3.12, numpy, stdlib `unittest` (`uv run python3 -m unittest`). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-07-sampling-budget-design.md`

## Global Constraints

- Run everything with `uv run python3`. Never add torch/vllm/sentence-transformers to the project; `scripts/embed_items.py` stays ephemeral.
- Never open, grep or print data under `datasets/raw/**` (other than `fetch.json`/`manifest.toml`), `datasets/public/{cbrn,cyber}.csv`, `datasets/generated/{cbrn,cyber}/*`, `logs/**`, `analysis/**`.
- Do not rebuild `datasets/public/*.csv` or `.meta.json` in this plan (the Hermes screen cache is pending; `prepare.py` exits 2). Unit tests only.
- Tier order is fixed by the spec: exact → near (per source) → cross-source exact → cross-source near → water-fill → stratify → pre-select `SCREEN_FACTOR` × → screen → fill. `SCREEN_FACTOR = 3.5` and `COSINE_TAU = 0.92` do not change.
- Determinism: same inputs and caches produce identical rows. Every new ordering ties on `(value, name)` or the existing `key_bytes` hash.
- Conventional commits, one per task, ending with the trailers
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01KTE97g3hnTpnAtX6KZpD7F`.
- Suite baseline: 654 OK, 1 skip, 2 expectedFailure (`tests/test_artifacts_current.py`, pending artifact regen). It must stay there after each task.
- Test helpers already in `tests/test_clusters.py`: `make_row(**overrides)`, `unit(*values)`, `embedded(rows, vectors, payload=...)`. Reuse them; do not add a fixture module.

## Review Focus

1. A source whose pool is smaller than its share must keep everything and pass the unused share on (Task 5 `test_small_pools_keep_everything_and_pass_on_their_share`).
2. A cluster where every source sets `quota` has no free sources; the allocator must not divide by zero (Task 5 `test_all_fixed_quotas_leave_no_free_sources`).
3. A `group_key` source whose share is not a multiple of its group size must round down to whole groups (Task 5 `test_grouped_sources_take_whole_groups`).
4. A `dedup_on` source must not receive anchors, since its spread payload is not the query (Task 4 `test_anchors_are_skipped_for_payload_sources`).
5. A stratum report with nothing kept must give `divergence = None`, not a ZeroDivisionError (Task 2 `test_divergence_is_none_when_nothing_is_kept`).

---

### Task 1: Exact drops reach `dropped.jsonl`

**Files:**
- Modify: `datasets/prepare/cluster/prepare.py:349-359` (`exact_dedup`), `:738-751` (`build_risk` tier 1 loop)
- Test: `tests/test_clusters.py` (`TestGroupedSelection` :437-451, `TestTiers` :765-770, new test in `TestTiers`)

**Interfaces:**
- Produces: `exact_dedup(rows, distinct_on=()) -> tuple[list[Row], list[dict]]`; each record `{"tier": "exact", "similarity": 1.0, "kept", "kept_text", "dropped", "dropped_text"}`.

- [ ] **Step 1: Update the three existing call sites and add the failing test**

In `tests/test_clusters.py` change every `(len(x), dropped)` / `dropped` comparison on `exact_dedup`'s second value to `len(dropped)`:

```python
# :438
self.assertEqual((len(collapsed), len(dropped)), (1, 2), "undeclared: arms collapse")
# :441
self.assertEqual((len(kept), len(dropped)), (3, 0), "declared: arms survive")
# :452
self.assertEqual((len(kept), len(dropped)), (1, 2))
# :769
self.assertEqual(len(dropped), 1)
```

Add to `TestTiers`:

```python
def test_exact_drops_are_recorded_like_every_other_tier(self):
    rows = self.rows("Korean War", "korean war!", "Vietnam War")
    kept, dropped = prepare.exact_dedup(rows)
    self.assertEqual(len(kept), 2)
    self.assertEqual(dropped, [{
        "tier": "exact", "similarity": 1.0,
        "kept": rows[0].sample_id, "kept_text": rows[0].query,
        "dropped": rows[1].sample_id, "dropped_text": rows[1].query,
    }])
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run python3 -m unittest tests.test_clusters.TestTiers tests.test_clusters.TestGroupedSelection -v`
Expected: FAIL (`len()` of int / list mismatch).

- [ ] **Step 3: Implement**

```python
def exact_dedup(
    rows: list[Row], distinct_on: Sequence[str] = ()
) -> tuple[list[Row], list[dict]]:
    '''Tier 1: drop repeats of the normalised query inside one source.'''
    kept, seen, dropped = [], {}, []
    for row in rows:
        key = (normalised(row.query), _identity(row, distinct_on))
        incumbent = seen.get(key)
        if incumbent is not None:
            dropped.append({
                "tier": "exact", "similarity": 1.0,
                "kept": incumbent.sample_id, "kept_text": incumbent.query[:300],
                "dropped": row.sample_id, "dropped_text": row.query[:300],
            })
            continue
        seen[key] = row
        kept.append(row)
    return kept, dropped
```

In `build_risk`:

```python
        rows, exact_dropped = exact_dedup(rows, source.distinct_on)
        all_dropped.extend(exact_dropped)
        report[source.name] = {
            "loaded": loaded,
            "exact_dropped": len(exact_dropped),
```

- [ ] **Step 4: Run the suite**

Run: `uv run python3 -m unittest discover tests 2>&1 | grep -E '^(Ran|OK|FAILED)'`
Expected: `OK (skipped=1, expected failures=2)`, 655 tests.

- [ ] **Step 5: Commit**

```bash
git add datasets/prepare/cluster/prepare.py tests/test_clusters.py
git commit -m "feat(datasets): record exact-dedup drops in dropped.jsonl"
```

---

### Task 2: Quota as a parameter; shortfall, strata and divergence in the report

**Files:**
- Modify: `datasets/prepare/cluster/prepare.py:491-533` (`stratified_sample`, `_grouped_sample`), `:642-673` (`_row_sample`), `:779-788` (`build_risk` tier 3 loop), `:850-868` (`print_report`)
- Test: `tests/test_clusters.py` (new class `TestSampleReport` after `TestAllocation`)

**Interfaces:**
- Produces: `stratified_sample(rows, source, seed, caches=None, quota=None) -> (rows, report)` where `quota=None` falls back to `source.quota`; report = `{"allotted": int, "selected": int, "strata": dict[str, {"pool": int, "kept": int}], "divergence": float | None}` plus `"groups": int` from `_grouped_sample`. `allotted`/`selected` are in the quota's unit (rows, or groups for `group_key` sources).
- Produces: `_divergence(strata: dict) -> float | None`.
- `report[name]` gains `allotted`, `shortfall`, `divergence`; `strata` becomes the dict. The old `allocated` key is removed (nothing read it).

- [ ] **Step 1: Write the failing tests**

```python
class TestSampleReport(unittest.TestCase):
    '''What the build says about each source, so a short or skewed source is seen.'''

    def pool(self, n, categories):
        return [
            make_row(sample_id=f"src:{i}", query=f"item {i}",
                     metadata={"category": categories[i % len(categories)]})
            for i in range(n)
        ]

    def source(self, **overrides):
        return Source(name="src", risk="cbrn", question_type=GRADED, path="unused",
                      stratify=["category"], **overrides)

    def test_quota_parameter_overrides_the_source(self):
        rows = self.pool(40, ["a", "b"])
        kept, report = prepare.stratified_sample(rows, self.source(quota=4), seed=0, quota=10)
        self.assertEqual((len(kept), report["allotted"], report["selected"]), (10, 10, 10))

    def test_quota_none_falls_back_to_the_source(self):
        rows = self.pool(40, ["a", "b"])
        kept, _ = prepare.stratified_sample(rows, self.source(quota=4), seed=0)
        self.assertEqual(len(kept), 4)

    def test_strata_report_pool_and_kept_per_key(self):
        rows = self.pool(30, ["a", "a", "b"])  # 20 a, 10 b
        _, report = prepare.stratified_sample(rows, self.source(), seed=0, quota=9)
        self.assertEqual(report["strata"], {"a": {"pool": 20, "kept": 6}, "b": {"pool": 10, "kept": 3}})
        self.assertEqual(report["divergence"], 0.0)

    def test_divergence_measures_skew(self):
        strata = {"a": {"pool": 50, "kept": 10}, "b": {"pool": 50, "kept": 0}}
        self.assertEqual(prepare._divergence(strata), 0.5)

    def test_divergence_is_none_when_nothing_is_kept(self):
        self.assertIsNone(prepare._divergence({"a": {"pool": 5, "kept": 0}}))
        self.assertIsNone(prepare._divergence({}))

    def test_unstratified_source_reports_no_strata(self):
        rows = self.pool(10, ["a"])
        _, report = prepare.stratified_sample(rows, self.source(stratify=()), seed=0, quota=3)
        self.assertEqual((report["strata"], report["divergence"]), ({}, None))

    def test_a_quota_above_the_pool_allots_the_pool(self):
        rows = self.pool(3, ["a", "b", "c"])
        _, report = prepare.stratified_sample(rows, self.source(), seed=0, quota=10)
        self.assertEqual((report["allotted"], report["selected"]), (3, 3))

    def test_grouped_report_counts_groups(self):
        rows = [
            make_row(sample_id=f"src:{g}:{arm}", query=f"scenario {g} arm {arm}",
                     metadata={"scenario": str(g), "arm": arm, "category": "x"})
            for g in range(6) for arm in ("p", "q", "r")
        ]
        source = self.source(group_key="scenario", distinct_on=["arm"])
        kept, report = prepare.stratified_sample(rows, source, seed=0, quota=4)
        self.assertEqual((len(kept), report["groups"], report["allotted"], report["selected"]),
                         (12, 6, 4, 4))
```

`allotted` is what the source was given (capped at its pool on the all-rows path), `selected` what it got; `build_risk` reports `shortfall = allotted - selected`, which is non-zero when `_allocate` stops early or the screen exhausts a stratum.

- [ ] **Step 2: Run to verify failure**

Run: `uv run python3 -m unittest tests.test_clusters.TestSampleReport -v`
Expected: FAIL (`unexpected keyword argument 'quota'`, missing `_divergence`).

- [ ] **Step 3: Implement**

```python
def stratified_sample(
    rows: list[Row], source: Source, seed: int, caches: Caches | None = None,
    quota: int | None = None,
) -> tuple[list[Row], dict]:
    '''`quota` is the allotment decided by the cluster budget; None reads the source's own.'''
    if quota is None:
        quota = source.quota
    if source.group_key:
        return _grouped_sample(rows, source, seed, caches, quota)
    return _row_sample(rows, source, seed, caches, quota)
```

`_grouped_sample(rows, source, seed, caches=None, quota=None)`: pass `quota` through to
`_row_sample(list(leaders.values()), source, seed, caches, quota)`; replace the two report
lines at the end with `report["groups"] = len(groups)` only (drop `report["allocated"]`).

`_row_sample(rows, source, seed, caches=None, quota=None)`: replace `quota = source.quota` with
nothing (it is now the parameter) and build the report through one helper:

```python
def _sample_report(buckets: dict, chosen: list[int], allotted: int) -> dict:
    taken = set(chosen)
    strata = {
        "|".join(key): {"pool": len(indices), "kept": sum(i in taken for i in indices)}
        for key, indices in buckets.items()
    }
    return {"allotted": allotted, "selected": len(chosen),
            "strata": strata, "divergence": _divergence(strata)}


def _divergence(strata: dict) -> float | None:
    '''Total variation distance between the pool's and the kept set's stratum shares.'''
    pool = sum(s["pool"] for s in strata.values())
    kept = sum(s["kept"] for s in strata.values())
    if not pool or not kept:
        return None
    return round(0.5 * sum(abs(s["pool"] / pool - s["kept"] / kept) for s in strata.values()), 3)
```

The three return sites in `_row_sample` become:

```python
    if quota is None or quota >= len(rows):
        ...
        return [rows[i] for i in sorted(chosen)], _sample_report({}, chosen, len(rows))

    if not source.stratify:
        chosen = _take(rows, list(range(len(rows))), quota, source, seed, caches)
        return [rows[i] for i in sorted(chosen)], _sample_report({}, chosen, quota)

    ...
    return [rows[i] for i in sorted(chosen)], _sample_report(buckets, chosen, quota)
```

In `build_risk`'s tier 3 loop replace `report[source.name]["strata"] = allocation["strata"]` with:

```python
        sample = allocation  # rename the local to `sample` for clarity
        report[source.name].update({
            "kept": len(rows),
            "allotted": sample["allotted"],
            "shortfall": sample["allotted"] - sample["selected"],
            "strata": sample["strata"],
            "divergence": sample["divergence"],
        })
```

In `print_report` add two columns and two warnings:

```python
    header = (f"  {'source':22s} {'loaded':>7s} {'exact':>6s} {'near':>6s} "
              f"{'cross':>6s} {'screen':>6s} {'allot':>6s} {'kept':>6s} {'short':>6s} {'share':>6s}")
    ...
        print(
            f"  {name:22s} {stats['loaded']:7d} {stats['exact_dropped']:6d} "
            f"{stats['near_dropped']:6d} {stats['cross_source_dropped']:6d} "
            f"{refused:6d} {stats.get('allotted', 0):6d} {stats['kept']:6d} "
            f"{stats.get('shortfall', 0):6d} {100 * stats['kept'] / total:5.1f}%"
        )
        if stats.get("shortfall", 0) > 0:
            print(f"  [WARNING] {name}: short {stats['shortfall']} of {stats['allotted']}; "
                  f"its pool or a stratum ran dry")
        divergence = stats.get("divergence")
        if divergence is not None and divergence > 0.10 and not stats.get("balanced"):
            print(f"  [WARNING] {name}: kept strata diverge from the pool (TVD {divergence:.2f})")
```

Update the TOTAL line's blank columns to match the new header (two more `{'':6s}`).

- [ ] **Step 4: Run the suite**

Run: `uv run python3 -m unittest discover tests 2>&1 | grep -E '^(Ran|OK|FAILED)'`
Expected: OK, 663 tests, 1 skip, 2 expected failures.

- [ ] **Step 5: Commit**

```bash
git add datasets/prepare/cluster/prepare.py tests/test_clusters.py
git commit -m "feat(datasets): report allotment, shortfall and strata skew"
```

---

### Task 3: Cross-source near-dedup (tier 2b)

**Files:**
- Modify: `datasets/prepare/cluster/prepare.py:174-190` (`require_embeddings`), `:415-427` (`_distinguishable` → `_mergeable`), `:442-486` (`near_dedup`, split into shared helpers), `:775-777` (`build_risk`)
- Test: `tests/test_clusters.py` (new class `TestCrossSourceNearDedup` after `TestCrossSourceDedup`; rename references to `_distinguishable` if any)

**Interfaces:**
- Produces: `cross_source_near_dedup(pools, embeddings, tau=COSINE_TAU) -> (pools, dropped)`; records tier `"near_cross_source"`.
- Produces: `_mergeable(left, right, distinct_on) -> bool` (renamed from `_distinguishable`, same body).
- Produces: `_near_candidates(rows, vectors, tau, allowed) -> list[tuple[float, int, int]]` and `_greedy_drop(candidates, rows, payload, tier) -> (set[int], list[dict])`, shared by both near passes.
- `require_embeddings` now also requires a vector for every row's `query` (anchors in Task 4 and this tier both compare queries).

- [ ] **Step 1: Write the failing tests**

```python
class TestCrossSourceNearDedup(unittest.TestCase):
    '''Tier 2b: a paraphrase shipped by two sources survives once, in the earlier source.'''

    def pools(self):
        a = [make_row(sample_id="a:1", source="a", query="how to make a bomb"),
             make_row(sample_id="a:2", source="a", query="unrelated gardening question")]
        b = [make_row(sample_id="b:1", source="b", query="how do I make a bomb"),
             make_row(sample_id="b:2", source="b", query="recipe for bread")]
        src = lambda name, **kw: Source(name=name, risk="cbrn", question_type=GRADED, path="unused", **kw)
        vectors = {
            "a:1": (1, 0, 0), "a:2": (0, 1, 0), "b:1": (0.99, 0.1, 0), "b:2": (0, 0, 1),
        }
        rows = a + b
        embeddings = embedded(rows, [vectors[r.sample_id] for r in rows])
        return [(src("a"), a), (src("b"), b)], embeddings, src

    def test_a_paraphrase_in_a_later_source_is_dropped(self):
        pools, embeddings, _ = self.pools()
        kept, dropped = prepare.cross_source_near_dedup(pools, embeddings)
        self.assertEqual([r.sample_id for _, rows in kept for r in rows], ["a:1", "a:2", "b:2"])
        self.assertEqual(dropped[0]["tier"], "near_cross_source")
        self.assertEqual((dropped[0]["kept"], dropped[0]["dropped"]), ("a:1", "b:1"))

    def test_same_source_pairs_are_left_to_tier_2(self):
        pools, embeddings, src = self.pools()
        twin = make_row(sample_id="a:3", source="a", query="how to make a bomb!")
        pools[0][1].append(twin)
        embeddings.update(embedded([twin], [(1, 0, 0)]))
        kept, _ = prepare.cross_source_near_dedup(pools, embeddings)
        self.assertIn("a:3", [r.sample_id for _, rows in kept for r in rows])

    def test_differing_mcq_targets_never_merge(self):
        pools, embeddings, _ = self.pools()
        for _, rows in pools:
            for row in rows:
                row.question_type = MCQ
        pools[0][1][0].target = "A"
        pools[1][1][0].target = "B"
        kept, dropped = prepare.cross_source_near_dedup(pools, embeddings)
        self.assertEqual(dropped, [])

    def test_a_source_that_opted_out_of_dedup_is_untouched(self):
        pools, embeddings, src = self.pools()
        pools[1] = (src("b", dedup=False), pools[1][1])
        kept, dropped = prepare.cross_source_near_dedup(pools, embeddings)
        self.assertEqual(dropped, [])
        self.assertEqual(len(kept[1][1]), 2)

    def test_queries_of_payload_sources_need_embeddings_too(self):
        row = make_row(sample_id="a:1", query="the wrapper", metadata={"event": "the event"})
        source = Source(name="a", risk="cbrn", question_type=GRADED, path="unused", dedup_on="event")
        embeddings = embedded([row], [(1, 0)], payload=lambda r: r.metadata["event"])
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(prepare, "CACHE_DIR", Path(tmp)):
            with self.assertRaises(prepare.CacheMiss):
                prepare.require_embeddings("cbrn", [(source, [row])], embeddings)
            written = [json.loads(line) for line in (Path(tmp) / "cbrn.embed_input.jsonl").read_text().splitlines()]
        self.assertEqual([w["text"] for w in written], ["the wrapper"])
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run python3 -m unittest tests.test_clusters.TestCrossSourceNearDedup -v`
Expected: FAIL (`no attribute 'cross_source_near_dedup'`).

- [ ] **Step 3: Implement**

Rename `_distinguishable` to `_mergeable` (docstring: "True when the pair may merge: same mcq target and equal `distinct_on` fields."). Update its two call sites. `grep -n _distinguishable datasets tests` must return nothing afterwards.

Split `near_dedup` into shared helpers:

```python
def _near_candidates(rows, vectors, tau, allowed) -> list[tuple[float, int, int]]:
    '''Pairs at or above tau, in row blocks; `allowed(left, right)` filters by index.'''
    candidates = []
    for start in range(0, len(rows), _BLOCK):
        # Rounded so a BLAS summing in another order cannot flip a pair across
        # tau or reorder ties between machines.
        similarity = np.round(vectors[start:start + _BLOCK] @ vectors.T, 6)
        for offset, right in zip(*np.nonzero(similarity >= tau)):
            left, right = start + int(offset), int(right)
            if left < right and allowed(left, right):
                candidates.append((float(similarity[offset, right]), left, right))
    return candidates


def _greedy_drop(candidates, rows, payload, tier) -> tuple[set[int], list[dict]]:
    '''Highest similarity first; the later row of each surviving pair is dropped.'''
    dropped_indices: set[int] = set()
    records = []
    for score, left, right in sorted(candidates, reverse=True):
        if left in dropped_indices or right in dropped_indices:
            continue
        dropped_indices.add(right)
        records.append({
            "tier": tier, "similarity": round(score, 4),
            "kept": rows[left].sample_id, "kept_text": payload(rows[left])[:300],
            "dropped": rows[right].sample_id, "dropped_text": payload(rows[right])[:300],
        })
    return dropped_indices, records
```

`near_dedup` body becomes:

```python
    payload = _payload_fn(dedup_on)
    vectors = _vectors(rows, payload, embeddings)
    candidates = _near_candidates(
        rows, vectors, tau, lambda l, r: _mergeable(rows[l], rows[r], distinct_on))
    dropped_indices, dropped_pairs = _greedy_drop(candidates, rows, payload, "near")
    survivors = [row for index, row in enumerate(rows) if index not in dropped_indices]
    return survivors, dropped_pairs
```

New tier:

```python
def cross_source_near_dedup(
    pools: list[tuple[Source, list[Row]]], embeddings: dict[str, np.ndarray],
    tau: float = COSINE_TAU,
) -> tuple[list[tuple[Source, list[Row]]], list[dict]]:
    '''
    Tier 2b: a paraphrase of an earlier source's prompt, shipped by a later one.

    Compares the delivered query for every row: `dedup_on` names a metadata
    field with no counterpart in another source. Pools are walked in registry
    order, so the later source loses, as in the exact cross-source tier.
    Sources with `dedup=False` keep their opt-out.
    '''
    rows = [row for source, pool in pools if source.dedup for row in pool]
    payload = _payload_fn(None)
    vectors = _vectors(rows, payload, embeddings)
    candidates = _near_candidates(
        rows, vectors, tau,
        lambda l, r: rows[l].source != rows[r].source and _mergeable(rows[l], rows[r], ()),
    )
    dropped_indices, dropped = _greedy_drop(candidates, rows, payload, "near_cross_source")
    gone = {rows[i].sample_id for i in dropped_indices}
    kept_pools = [(source, [row for row in pool if row.sample_id not in gone]) for source, pool in pools]
    return kept_pools, dropped
```

`require_embeddings`: inside the row loop, also require the query:

```python
        for row in rows:
            for text in {payload(row), row.query}:
                key = embed_key(text)
                if key and key not in embeddings:
                    missing[key] = normalised(text)
```

`build_risk`, after `pools, cross_dropped = cross_source_dedup(pools)` and its `all_dropped.extend`:

```python
    pools, cross_near_dropped = cross_source_near_dedup(pools, embeddings)
    all_dropped.extend(cross_near_dropped)
```

and make `report[name]["cross_source_dropped"]` count both (it already uses `sizes[name] - len(rows)` after the pools are replaced, so nothing else changes).

- [ ] **Step 4: Run the suite**

Run: `uv run python3 -m unittest discover tests 2>&1 | grep -E '^(Ran|OK|FAILED)'`
Expected: OK, 668 tests, 1 skip, 2 expected failures.

- [ ] **Step 5: Commit**

```bash
git add datasets/prepare/cluster/prepare.py tests/test_clusters.py
git commit -m "feat(datasets): near-dedup paraphrases across sources"
```

---

### Task 4: Anchored farthest-point and the empty-payload fix

**Files:**
- Modify: `datasets/prepare/cluster/prepare.py:65-73` (`Caches`), `:555-585` (`_diverse_order`), `:626-639` (`_select`), `:779-788` (`build_risk` tier 3 loop)
- Test: `tests/test_clusters.py` (`TestSelection`, new tests)

**Interfaces:**
- Produces: `Caches.selected: list[np.ndarray]` and `Caches.anchors() -> np.ndarray | None`.
- Produces: `_diverse_order(rows, indices, take, source, seed, embeddings, anchors=None)`.
- Empty-payload rows are picked last (`nearest = 2.0`, above any cosine).

- [ ] **Step 1: Write the failing tests**

Add to `TestSelection`:

```python
    def test_anchors_push_the_walk_away_from_what_the_cluster_already_holds(self):
        rows = self.pool(3)
        embeddings = embedded(rows, [(1, 0, 0), (0, 1, 0), (0, 0, 1)])
        anchors = np.array([unit(1, 0, 0)])
        order = prepare._diverse_order(rows, [0, 1, 2], 1, self.source(1, select="diverse"),
                                       0, embeddings, anchors=anchors)
        self.assertNotEqual(order, [0], "the anchored region is picked last")

    def test_empty_payloads_are_picked_last(self):
        rows = self.pool(3)
        rows[1].query = ""
        embeddings = embedded([rows[0], rows[2]], [(1, 0), (0, 1)])
        order = prepare._diverse_order(rows, [0, 1, 2], 3, self.source(3, select="diverse"), 0, embeddings)
        self.assertEqual(order[-1], 1)

    def test_anchors_are_skipped_for_payload_sources(self):
        rows = [make_row(sample_id=f"src:{i}", query=f"wrapper {i}", metadata={"event": f"event {i}"})
                for i in range(3)]
        source = self.source(1, select="diverse", dedup_on="event")
        embeddings = embedded(rows, [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
                              payload=lambda r: r.metadata["event"])
        caches = prepare.Caches(embeddings, selected=[np.array([unit(1, 0, 0)])])
        with mock.patch.object(prepare, "_diverse_order", wraps=prepare._diverse_order) as spy:
            prepare._select(rows, [0, 1, 2], 1, source, 0, caches)
        self.assertIsNone(spy.call_args.kwargs.get("anchors"))

    def test_selected_vectors_become_anchors(self):
        caches = prepare.Caches({}, selected=[np.array([unit(1, 0)]), np.array([unit(0, 1)])])
        self.assertEqual(caches.anchors().shape, (2, 2))
        self.assertIsNone(prepare.Caches({}).anchors())
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run python3 -m unittest tests.test_clusters.TestSelection -v`
Expected: FAIL (`unexpected keyword argument 'anchors'`, `selected`).

- [ ] **Step 3: Implement**

`Caches`:

```python
    selected: list[np.ndarray] = field(default_factory=list)  # query vectors of rows already kept

    def anchors(self) -> np.ndarray | None:
        if not self.selected:
            return None
        stacked = np.vstack(self.selected)
        return stacked if len(stacked) else None  # every source so far kept nothing
```

`_diverse_order`:

```python
def _diverse_order(
    rows: list[Row], indices: list[int], take: int, source: Source, seed: int,
    embeddings: dict[str, np.ndarray], anchors: np.ndarray | None = None,
) -> list[int]:
    '''
    ... (keep the docstring) ...
    With `anchors` (rows the cluster has already kept), the walk starts from
    the candidate farthest from any anchor instead of the stable-order head, so
    two sources cannot fill the same region.
    '''
    vectors = _vectors([rows[i] for i in indices], _payload_fn(source.dedup_on), embeddings)
    ties = [key_bytes(rows[i], seed) for i in indices]
    # Each item's similarity to the closest pick so far; picked items are
    # pinned at +inf. Empty payloads have zero vectors and would otherwise read
    # as "far from everything": 2.0 is above any cosine, so they go last.
    empty = ~vectors.any(axis=1)
    if anchors is None:
        order = _stable_order(rows, indices, seed)
        first = next((indices.index(i) for i in order if not empty[indices.index(i)]),
                     indices.index(order[0]))
        picked = [first]
        nearest = np.round(vectors @ vectors[first], 6)
        nearest[first] = np.inf
    else:
        picked = []
        nearest = np.round(vectors @ anchors.T, 6).max(axis=1)
    nearest[empty & np.isfinite(nearest)] = 2.0
    while len(picked) < take:
        candidate = min(range(len(indices)), key=lambda p: (nearest[p], ties[p]))
        picked.append(candidate)
        if not empty[candidate]:
            nearest = np.maximum(nearest, np.round(vectors @ vectors[candidate], 6))
        nearest[candidate] = np.inf
    return [indices[p] for p in picked]
```

(`np.maximum` against an all-zero candidate vector is a no-op, so the `if not empty` guard
is only there to make the intent readable; keep it.)

`_select`'s diverse branch:

```python
    if source.select == DIVERSE:
        if caches is None:
            raise ValueError(f"{source.name}: diverse selection needs the embedding cache")
        anchors = caches.anchors() if source.dedup_on is None else None
        return _diverse_order(rows, indices, take, source, seed, caches.embeddings, anchors=anchors)
```

`build_risk` tier 3 loop, after `all_rows.extend(rows)`:

```python
        caches.selected.append(_vectors(rows, _payload_fn(None), embeddings))
```

- [ ] **Step 4: Run the suite**

Run: `uv run python3 -m unittest discover tests 2>&1 | grep -E '^(Ran|OK|FAILED)'`
Expected: OK, 672 tests, 1 skip, 2 expected failures. `TestSelection.test_diverse_selection_is_deterministic` and `..._covers_every_topic` still pass (no anchors in those calls).

- [ ] **Step 5: Commit**

```bash
git add datasets/prepare/cluster/prepare.py tests/test_clusters.py
git commit -m "feat(datasets): anchor diverse selection on the cluster so far"
```

---

### Task 5: Cluster `BUDGET` and water-filling

**Files:**
- Modify: `datasets/prepare/cluster/sources/__init__.py`, `sources/{cbrn,cyber,loss_of_control,manipulation}.py` (add `BUDGET`, remove `quota=`), `datasets/prepare/cluster/prepare.py` (`allocate_budget`, `build_risk`, `write_outputs`, `print_report`)
- Test: `tests/test_clusters.py` (new class `TestBudget` after `TestSampleReport`; `TestRegistry` one test; `TestMeta` update)

**Interfaces:**
- Consumes: `stratified_sample(..., quota=)` from Task 2.
- Produces: `sources.BUDGETS: dict[str, int]`, `sources.budget_for(risk) -> int`.
- Produces: `allocate_budget(pools, budget) -> dict[str, int]` (units: rows, or groups for `group_key` sources), `_group_size(source, rows) -> int`.
- `meta` top level gains `budget` and `shortfall` (`budget - rows`).

- [ ] **Step 1: Write the failing tests**

```python
class TestBudget(unittest.TestCase):
    '''One number per cluster, water-filled: small sources keep everything,
    the unused share flows to the larger ones, `quota` is an override.'''

    def src(self, name, **kw):
        return Source(name=name, risk="cbrn", question_type=GRADED, path="unused", **kw)

    def pool(self, name, n, **meta):
        return [make_row(sample_id=f"{name}:{i}", source=name, query=f"{name} {i}",
                         metadata=meta) for i in range(n)]

    def test_equal_shares_when_every_pool_is_large(self):
        pools = [(self.src("a"), self.pool("a", 100)), (self.src("b"), self.pool("b", 100))]
        self.assertEqual(prepare.allocate_budget(pools, 60), {"a": 30, "b": 30})

    def test_small_pools_keep_everything_and_pass_on_their_share(self):
        pools = [(self.src("a"), self.pool("a", 5)), (self.src("b"), self.pool("b", 100)),
                 (self.src("c"), self.pool("c", 100))]
        self.assertEqual(prepare.allocate_budget(pools, 65), {"a": 5, "b": 30, "c": 30})

    def test_remainder_goes_to_the_largest_pools(self):
        pools = [(self.src("a"), self.pool("a", 100)), (self.src("b"), self.pool("b", 100)),
                 (self.src("c"), self.pool("c", 100))]
        allocation = prepare.allocate_budget(pools, 64)
        self.assertEqual(sum(allocation.values()), 64)
        self.assertEqual(sorted(allocation.values()), [21, 21, 22])

    def test_quota_is_an_override_taken_off_the_top(self):
        pools = [(self.src("a", quota=10), self.pool("a", 100)), (self.src("b"), self.pool("b", 100))]
        self.assertEqual(prepare.allocate_budget(pools, 60), {"a": 10, "b": 50})

    def test_all_fixed_quotas_leave_no_free_sources(self):
        pools = [(self.src("a", quota=10), self.pool("a", 100)), (self.src("b", quota=5), self.pool("b", 3))]
        self.assertEqual(prepare.allocate_budget(pools, 60), {"a": 10, "b": 3})

    def test_grouped_sources_take_whole_groups(self):
        rows = [make_row(sample_id=f"g:{g}:{arm}", source="g", query=f"s {g} {arm}",
                         metadata={"scenario": str(g), "arm": arm})
                for g in range(20) for arm in ("p", "q", "r")]
        pools = [(self.src("g", group_key="scenario", distinct_on=["arm"]), rows),
                 (self.src("b"), self.pool("b", 100))]
        allocation = prepare.allocate_budget(pools, 100)
        # g's share is 50 rows -> 16 groups (48 rows); the 2 leftover rows flow to b.
        self.assertEqual(allocation, {"g": 16, "b": 52})

    def test_allocation_is_independent_of_registry_order(self):
        a, b = (self.src("a"), self.pool("a", 5)), (self.src("b"), self.pool("b", 100))
        self.assertEqual(prepare.allocate_budget([a, b], 50), prepare.allocate_budget([b, a], 50))

    def test_budget_below_fixed_quotas_gives_free_sources_nothing(self):
        pools = [(self.src("a", quota=80), self.pool("a", 100)), (self.src("b"), self.pool("b", 100))]
        self.assertEqual(prepare.allocate_budget(pools, 60), {"a": 80, "b": 0})
```

Add to `TestRegistry`:

```python
    def test_every_risk_declares_a_budget(self):
        from datasets.prepare.cluster.sources import BUDGETS, budget_for
        for risk in RISKS:
            with self.subTest(risk=risk):
                self.assertIsInstance(budget_for(risk), int)
                self.assertGreater(budget_for(risk), 0)
        self.assertEqual(set(BUDGETS), set(RISKS))

    def test_no_source_hard_codes_a_quota(self):
        '''Shares come from BUDGET; `quota` is an override that needs a comment where used.'''
        self.assertEqual([s.name for s in SOURCES if s.quota is not None], [])
```

In `TestMeta.test_meta_records_embedding_and_screen`, after the `meta["screen"]` assertion add:

```python
        self.assertEqual((meta["budget"], meta["shortfall"]), (140, 139))
```

(`write_outputs` is called with one row for `loss_of_control`, whose `BUDGET` is 140.)

- [ ] **Step 2: Run to verify failure**

Run: `uv run python3 -m unittest tests.test_clusters.TestBudget tests.test_clusters.TestRegistry tests.test_clusters.TestMeta -v`
Expected: FAIL (`allocate_budget` missing, `BUDGETS` missing, sources with quotas).

- [ ] **Step 3: Implement the config**

`sources/__init__.py`, after `RISKS`:

```python
# Rows per cluster. prepare.allocate_budget water-fills it across the risk's
# sources after dedup; Source.quota overrides a share.
BUDGETS: dict[str, int] = {risk: _MODULES[risk].BUDGET for risk in RISKS}


def budget_for(risk: str) -> int:
    return BUDGETS[risk]
```

In each risk module, directly above `SOURCES = [`:

```python
BUDGET = 186   # cbrn: today's cluster size; raise toward 2x once the screen cache is in
BUDGET = 300   # cyber
BUDGET = 140   # loss_of_control
BUDGET = 562   # manipulation
```

Remove every `quota=<n>` from the `Source(...)` calls in the four modules (19 sources). Where
a comment nearby says "quota counts scenarios" or "redundancy ... at this quota", reword it to
"share" (e.g. "the share counts scenarios, not rows: 3 persona arms each"). Do not delete the
redundancy measurements.

- [ ] **Step 4: Implement the allocator**

In `prepare.py`, before `build_risk`:

```python
def _group_size(source: Source, rows: list[Row]) -> int:
    '''Rows per selection unit: 1, or the mean group size for a group_key source.'''
    if not source.group_key or not rows:
        return 1
    groups = {str(row.metadata.get(source.group_key, i)) for i, row in enumerate(rows)}
    return max(1, round(len(rows) / len(groups)))


def allocate_budget(pools: list[tuple[Source, list[Row]]], budget: int) -> dict[str, int]:
    '''
    Water-fill `budget` rows across sources. Sources with `quota` take it off
    the top; the rest are visited smallest pool first, each taking
    min(pool, remaining / sources left), so a small source keeps everything and
    its unused share flows on. Returned values are in each source's unit: rows,
    or groups for a group_key source (a share rounds down to whole groups).
    '''
    takes: dict[str, int] = {}
    remaining = budget
    free: list[tuple[int, str, Source, int]] = []
    for source, rows in pools:
        size = _group_size(source, rows)
        units = math.ceil(len(rows) / size)
        if source.quota is not None:
            takes[source.name] = min(source.quota, units)
            remaining -= takes[source.name] * size
        else:
            free.append((len(rows), source.name, source, size))
    free.sort(key=lambda item: item[:2])
    for position, (pool_rows, name, _, size) in enumerate(free):
        share = max(remaining, 0) // (len(free) - position)
        takes[name] = min(pool_rows, share) // size
        remaining -= takes[name] * size
    # Integer remainder: one more unit to the largest pools with room, largest first.
    moved = True
    while remaining > 0 and moved:
        moved = False
        for pool_rows, name, _, size in reversed(free):
            if remaining >= size and (takes[name] + 1) * size <= pool_rows:
                takes[name] += 1
                remaining -= size
                moved = True
    return takes
```

`build_risk`, replace the tier 3 loop header:

```python
    caches = Caches(embeddings, verdicts=load_screen(risk))
    allocation = allocate_budget(pools, budget_for(risk))
    for source, rows in pools:
        ...
        rows, sample = stratified_sample(rows, source, seed, caches, quota=allocation[source.name])
```

Import `budget_for` beside `RISKS, for_risk`. `write_outputs`:

```python
    meta = {
        "risk": risk,
        "rows": len(rows),
        "budget": budget_for(risk),
        "shortfall": budget_for(risk) - len(rows),
        "seed": seed,
```

`print_report` receives `risk` already; after the TOTAL line:

```python
    budget = budget_for(risk)
    if len(rows) < budget:
        print(f"  [WARNING] {risk}: {len(rows)} rows against a budget of {budget}; "
              f"every pool is exhausted")
    elif len(rows) > budget:
        print(f"  [WARNING] {risk}: {len(rows)} rows exceed the budget of {budget}; "
              f"fixed quotas add up to more than BUDGET")
```

- [ ] **Step 5: Run the suite**

Run: `uv run python3 -m unittest discover tests 2>&1 | grep -E '^(Ran|OK|FAILED)'`
Expected: OK, 682 tests, 1 skip, 2 expected failures. If an existing test reads `quota` from a registry source (grep `.quota` in tests/), pass the value explicitly via `stratified_sample(..., quota=)` instead. `tests/test_benchmarks_doc.py` still passes (it compares the committed CSVs, which are not rebuilt here).

- [ ] **Step 6: Commit**

```bash
git add datasets/prepare/cluster/prepare.py datasets/prepare/cluster/sources tests/test_clusters.py
git commit -m "feat(datasets): water-fill one BUDGET per cluster over its sources"
```

---

### Task 6: Documentation

**Files:**
- Modify: `datasets/BENCHMARKS.md:479-500` (tier table), `:566-569` (Strata intro), `:606-615` (Provenance), `datasets/README.md:40-43`, `CONTRIBUTE.md:70` (example), `:76-90` (field table), `:228-239` (build report paragraph)
- Test: `uv run python3 -m unittest tests.test_benchmarks_doc`

- [ ] **Step 1: BENCHMARKS.md**

Replace the paragraph at :479-481 with:

> Every registered source contributes to its cluster, always. Each `sources/<risk>.py`
> declares one `BUDGET` (rows for the cluster); `prepare.py::allocate_budget` water-fills it
> across the risk's sources after dedup, smallest pool first, so a small source keeps
> everything and its unused share flows to the larger ones. `Source.quota` overrides a share.
> `_allocate` then splits each share across strata; everything below decides *which* rows fill
> each allotment.

Insert a tier row after `1b`:

```
| 2b | **cross-source near-dedup** on the delivered query at `COSINE_TAU`, later source loses; sources with `dedup=False` are skipped | one more `V @ V.T` per cluster |
```

Change tier 3's text to "water-filled share, then `_allocate` per stratum, then per stratum pre-select `SCREEN_FACTOR` (3.5) × allotment by the source's `select`; `diverse` starts from the item farthest from anything the cluster has already kept".

Change tier 1's cost column from "free" to "free; pairs in `dropped.jsonl` (`tier: exact`)".

Strata intro (:568): "The share is allocated proportionally (or evenly where `balanced` is declared) ...". In the strata table change "(quota counts scenarios, 3 arms each)" to "(the share counts scenarios, 3 arms each)".

Provenance (:608-615): list the new keys: top-level `budget`, `shortfall`; per source `quota` (the override, usually null), `allotted`, `shortfall`, `strata` (`{key: {pool, kept}}`), `divergence` (total variation distance pool vs kept, null when unstratified); `dropped.jsonl` tiers `exact`, `near`, `exact_cross_source`, `near_cross_source`, `screen`.

- [ ] **Step 2: datasets/README.md and CONTRIBUTE.md**

README :40-43: "(seed, budget and shortfall, per-source allotments, per-tier drop counts, strata skew, source revisions)" and the tier list "`exact`, `near`, `exact_cross_source`, `near_cross_source` or `screen`".

CONTRIBUTE.md :70: `stratify=["category"],` (drop `quota=90`). In the field table add a row:

```
| `quota` | override the water-filled share from the cluster `BUDGET`; say why in a comment |
```

and reword the `group_key` row to "rows only meaningful as a set — the share then counts groups, not rows". In the build-report paragraph (:228-239) add: "`allot` and `short` show what the budget gave the source and how much it could not fill; a `short` above 0 means the pool or a stratum ran dry. A `[WARNING]` on strata divergence means the kept rows are distributed unlike the pool, usually because the screen refused one category." Replace "rather than shrink the quota" with "rather than shrink the share", and extend the tier list with `exact` and `near_cross_source`.

- [ ] **Step 3: Verify and commit**

Run: `uv run python3 -m unittest tests.test_benchmarks_doc tests.test_clusters 2>&1 | grep -E '^(Ran|OK|FAILED)'` then `grep -rn 'quota' datasets/BENCHMARKS.md datasets/README.md CONTRIBUTE.md` and confirm every remaining mention describes the override.

```bash
git add datasets/BENCHMARKS.md datasets/README.md CONTRIBUTE.md
git commit -m "docs(datasets): describe the cluster budget and new tiers"
```

---

## Verification (whole plan)

1. `uv run python3 -m unittest discover tests` → OK, 1 skip, 2 expected failures.
2. `uv run python3 -m datasets.prepare.cluster.prepare --risk manipulation --dry-run` from the main checkout: expect either the tier table with `allot`/`short` columns or exit 2 naming the embed/screen command (the query embeddings for `historical_revisionism` are new, so an embed run is expected before the first rebuild).
3. `grep -rn '_distinguishable\|"allocated"' datasets tests` → nothing.
4. Operator follow-up (not in this plan): rerun `scripts/embed_items.py` per risk, the SCREEN_ONLY sbatch, then rebuild and review `shortfall`/`divergence` in each `meta.json` before raising `BUDGET`.
