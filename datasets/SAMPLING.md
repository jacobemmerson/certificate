# How a cluster's rows are chosen

Every registered source contributes to its cluster, always. Nothing here decides
*whether* a benchmark appears or *how many* rows it gets — that is its `quota`,
split across its strata by `prepare.py::_allocate`. This document is only about
which rows fill each stratum's allotment.

## The four tiers

`datasets/prepare/cluster/prepare.py`, in order:

| Tier | What it does |
|---|---|
| 0 | `transform` — the source's own reshaping hook |
| 1 | `exact_dedup` — drop repeats of the same normalised text |
| 2 | `near_dedup` — drop pairs above `tau`, Jaccard over the payload |
| 3 | **quota** — `_allocate` splits the quota across strata, then `_take` fills each |

Tiers 0–2 and `_allocate` are deterministic. Tier 3's fill is what this describes.

## Filling an allotment

`_take(rows, indices, take, source, seed)` handles one stratum. If the allotment
is at least the stratum size it keeps everything; otherwise it uses whichever
mode the source declares in `Source.select`.

### `uniform` (default)

Order the stratum by `blake2b(f"{seed}:{sample_id}")` and take the first N.

Uniform and unbiased, but an item's rank depends **only on its own hash**, so a
change elsewhere in the pool cannot reshuffle it. This replaced
`frame.sample(random_state=seed)`, which pinned a shuffle of *positions*: one
extra upstream row re-drew up to 90% of a source's selection, silently changing
which items were certified between dataset versions and invalidating every
stage-2/3 artifact (they are keyed by `sample_id`).

The guarantee is *minimal* churn, not zero. An item hashing below the cutoff
displaces exactly one incumbent, and `_allocate` can move a row between strata
when a stratum's size changes.

### `diverse` (opt-in)

Greedy farthest-point within the stratum:

1. Tokenise each item's **payload**.
2. First pick = the hash-stable first, so the walk is deterministic without
   depending on input order.
3. Track `nearest[i]` = the highest Jaccard from item *i* to anything already
   picked.
4. Repeatedly take `argmin(nearest)`, ties broken by hash. Update `nearest`.

`O(take × |stratum|)` Jaccard comparisons, offline only.

**Payload** is `dedup_on`'s metadata field where a source declares one, the
rendered query otherwise (`_payload_fn`) — the same rule `near_dedup` follows.
`historical_revisionism`'s items differ by historical event inside a shared
~100-word instruction, so spreading on the rendered prompt would spread on
boilerplate.

## Which sources are `diverse`, and why only those

Six: `wmdp`, `advanced_ai_risk`, `sosbench`, `cyber_false_refusal`, `darkbench`,
`cysecbench`. Measured gains of 38–50% on mean nearest-neighbour Jaccard *within
the selected set* — "how much of this sample is a near-duplicate of something
else in it".

Everything else stays `uniform`. Those sources are **template-shaped**: every row
renders the same wrapper around one varying field, so the query text is
near-identical by construction and lexical distance measures the wrapper.
Spreading them picks whichever rows word their boilerplate oddly — noise, and it
biases toward outliers for no coverage gain. Measured: `persusafety` 0%,
`role_model_bias` 1%, `injecagent` 1%, `leader_favorability` 3%.

`historical_revisionism` is the instructive case. It looks like a candidate and
gains **exactly 0%**: `dedup_on` already deduplicated the pool on the event
payload at `tau=0.8`, and `distinct_on` caps each event at one row per push
level, so any 60-row draw lands ~53 distinct events either way.

Measure before opting a source in, and measure through `stratified_sample`
itself — a whole-pool approximation on the rendered query over-promises for any
source that declares `dedup_on` or stratifies finely.

## Properties worth knowing

- **Diversity trades representativeness for coverage.** Farthest-point
  over-weights unusual items, so a `diverse` source's score is not an unbiased
  estimate of how the model does on the whole pool. For a coverage-oriented
  certificate that is the intent, but it is a real property.
- **`diverse` sources keep some churn.** Farthest-point runs over the whole
  stratum rather than a hash-stable shortlist, so a pool change can shift the
  greedy path. Chosen deliberately for maximum spread; the `uniform` sources are
  effectively pinned.
- **Stratification binds first.** Farthest-point runs inside a stratum, which is
  why live gains are smaller than whole-pool estimates.
- **Near-dedup still runs.** Farthest-point largely subsumes it, but tier 2 is
  unchanged — it is what produces the reviewable `<risk>.dropped.jsonl`.
- **Cross-source dedup runs before the quota.** Tier 1b removes a prompt a later
  source ships identically to an earlier one, so the copy never spends budget
  and the source backfills from its own pool.

## Changing it

`select="diverse"` on a `Source`, and a note recording the measured gain.
Selection is covered by `tests/test_clusters.py::TestSelection`: stability under
an unrelated row entering the pool, determinism in both modes, the seed still
changing the draw, diverse beating uniform on redundancy, diverse spreading on
the payload rather than the wrapper, and an unknown mode raising.

Any change here re-draws the affected sources, which invalidates their
stage-2/3 artifacts. Land it **before** paying to regenerate them.
