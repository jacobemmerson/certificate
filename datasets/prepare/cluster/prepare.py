'''
Build the risk-cluster datasets.

    uv run python3 -m datasets.prepare.cluster.prepare --risk cyber
    uv run python3 -m datasets.prepare.cluster.prepare --dry-run

Writes datasets/public/<risk>.csv plus a <risk>.meta.json sibling (provenance:
seed, quotas, per-tier drop counts, embedding model and threshold, screen model
and refusals, source revisions) and <risk>.dropped.jsonl (every pair tiers 1b
and 2 removed and every candidate the screen dropped, each tagged with its
`tier`, so thresholds are reviewable rather than trusted).

Reads two gitignored caches under datasets/cache/. On a miss it writes what is
missing, prints the command that fills it and exits 2: embeddings first, then
screen verdicts. The sequence is in datasets/BENCHMARKS.md § Sampling.
'''

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from . import readers
from .schema import (
    COLUMNS, ITEM, MCQ, Row, SchemaError, Source, normalised, validate,
)
from .sources import RISKS, for_risk

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUT_DIR = REPO_ROOT / "datasets" / "public"
CACHE_DIR = REPO_ROOT / "datasets" / "cache"

# Kept equal to scripts/embed_items.py::MODEL (tests/test_clusters.py checks).
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_COMMAND = (
    "uv run --no-project --with sentence-transformers --with numpy "
    "python scripts/embed_items.py --risk {risk}"
)
# Tier 3b: candidates per allotted row sent to the answerability screen. 3.5
# fills an allotment while Hermes refuses up to ~70% of its candidates; past
# that the build stops and names the gap (raise this, never shrink the quota).
SCREEN_FACTOR = 3.5
SCREEN_COMMANDS = (
    "sbatch --export=ALL,SCREEN_ONLY=1 scripts/generate_hermes_slurm.sh",
    "uv run python3 scripts/screen_answerability.py --risk {risk} "
    "--model openrouter/nousresearch/hermes-4-70b",
)


class CacheMiss(Exception):
    '''A cache prepare.py reads lacks entries. The message says what to run.'''


@dataclass
class Caches:
    '''What tier 3 reads besides the rows, threaded through as one argument.'''
    embeddings: dict[str, np.ndarray]
    # screen key -> cache record. None turns the screen off, which only unit
    # tests do: build_risk always loads it, so no CSV is built unscreened.
    verdicts: dict[str, dict] | None = None
    missing: dict[str, dict] = field(default_factory=dict)  # screen inputs still needed
    refused: list[dict] = field(default_factory=list)       # "screen" tier drop records
    candidates: int = 0                                      # rows sent through the screen


def screen_key(row: Row) -> str:
    '''The prompt as delivered (system + user), so a verdict survives an id change.'''
    return hashlib.blake2b(
        f"{row.system_prompt}\x00{row.query}".encode(), digest_size=16
    ).hexdigest()


def load_screen(risk: str) -> dict[str, dict]:
    '''key -> record from datasets/cache/screen/<risk>.jsonl (append-only; last wins).'''
    path = CACHE_DIR / "screen" / f"{risk}.jsonl"
    if not path.exists():
        return {}
    records = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:  # truncated by a preempted job
                continue
            records[record["key"]] = record
    return records


def require_screen(risk: str, caches: Caches) -> None:
    '''Raise CacheMiss, after writing the screen input, if any candidate lacks a verdict.'''
    path = CACHE_DIR / f"{risk}.screen_input.jsonl"
    if not caches.missing:
        path.unlink(missing_ok=True)
    else:
        raise _cache_miss(
            path,
            [caches.missing[key] for key in sorted(caches.missing)],
            *(command.format(risk=risk) for command in SCREEN_COMMANDS),
        )


def _screen(rows: list[Row], pool: list[int], caches: Caches) -> list[int]:
    '''Drop candidates Hermes refused. One with no verdict yet is kept
    provisionally and queued; build_risk raises CacheMiss before writing.'''
    caches.candidates += len(pool)
    kept = []
    for index in pool:
        row = rows[index]
        key = screen_key(row)
        record = caches.verdicts.get(key)
        verdict = record and record.get("verdict")
        if verdict == "refused":
            caches.refused.append({
                "tier": "screen", "dropped": row.sample_id,
                "dropped_text": row.query[:300], "model": record["model"],
            })
        elif verdict == "answered":
            kept.append(index)
        else:
            caches.missing[key] = {
                "key": key, "sample_id": row.sample_id, "question_type": row.question_type,
                "system_prompt": row.system_prompt, "query": row.query,
            }
            kept.append(index)
    return kept


def _cache_miss(path: Path, records: list[dict], *commands: str) -> CacheMiss:
    '''Write what the cache lacks to `path` and build the error naming the fix.'''
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return CacheMiss(f"{len(records)} missing -> {path}\n  run: " + "\n   or: ".join(commands))


def embed_key(text: str) -> str | None:
    '''
    Cache key of a payload's embedding: blake2b-16 of its normalised text, which
    is also the text embedded (all-MiniLM-L6-v2 is uncased, so folding case and
    punctuation loses nothing). Payloads that normalise identically therefore
    share one vector by construction. None for a payload with no words: there
    is nothing to embed, and it is never anyone's duplicate.
    '''
    text = normalised(text)
    return hashlib.blake2b(text.encode(), digest_size=16).hexdigest() if text else None


def load_embeddings(risk: str) -> dict[str, np.ndarray]:
    '''key -> unit float32 vector from datasets/cache/embeddings/<risk>.npz; {} if absent.'''
    path = CACHE_DIR / "embeddings" / f"{risk}.npz"
    if not path.exists():
        return {}
    with np.load(path) as data:
        if str(data["model"]) != EMBEDDING_MODEL:
            return {}
        keys = data["keys"].tolist()
        vectors = data["vectors"].astype(np.float32)
    # Stored as float16, so re-normalise rather than trust the rounding.
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return dict(zip(keys, vectors))


def require_embeddings(
    risk: str, pools: list[tuple[Source, list[Row]]], embeddings: dict
) -> None:
    '''Raise CacheMiss, after writing the embed input, if any payload lacks a vector.'''
    missing = {}
    for source, rows in pools:
        payload = _payload_fn(source.dedup_on)
        for row in rows:
            key = embed_key(payload(row))
            if key and key not in embeddings:
                missing[key] = normalised(payload(row))
    if missing:
        raise _cache_miss(
            CACHE_DIR / f"{risk}.embed_input.jsonl",
            [{"key": key, "text": text} for key, text in sorted(missing.items())],
            EMBED_COMMAND.format(risk=risk),
        )

# Tier 2: cosine similarity of all-MiniLM-L6-v2 embeddings at or above this is a
# near-duplicate. 0.92 is the spec's start value, checked against the pairs the
# retired Jaccard tier dropped (datasets/BENCHMARKS.md § Sampling). Embeddings
# made the token gate unnecessary: on long text Jaccard measured shared
# boilerplate, while an embedding of the payload does not. `distinct_on` and the
# mcq-target guard still cover templated sources whose items differ by one term.
COSINE_TAU = 0.92
# Rows of the similarity matrix computed at once: memory is _BLOCK x N float32.
_BLOCK = 2048


# ----- tier 0-1: load, map, exact dedup -----

def load_source(source: Source) -> list[Row]:
    '''Read, transform (tier 0), and map to canonical rows.'''
    frame = readers.read(
        source.path,
        source.reader,
        columns=source.columns,
        record_path=source.record_path,
        filename_field=source.filename_field,
        dirname_field=source.dirname_field,
        first_row_field=source.first_row_field,
    )
    return rows_from_frame(source, frame)


def rows_from_frame(source: Source, frame) -> list[Row]:
    if source.transform is not None:
        frame = source.transform(frame)

    rows = []
    for position, record in enumerate(frame.to_dict("records")):
        native_id = source.resolve(record, source.id_col) if source.id_col else position
        query = source.resolve(record, source.query)
        if not str(query).strip() or str(query) == "nan":
            continue

        categories = source.categories
        if callable(categories):
            categories = categories(record)
        scale_map = source.scale_map
        if callable(scale_map):
            scale_map = scale_map(record)

        choices = source.resolve(record, source.choices) if source.choices else []
        target = source.resolve(record, source.target) if source.target else ""

        fallback_categories = source.fallback_categories
        if callable(fallback_categories):
            fallback_categories = fallback_categories(record)
        fallback_scale_map = source.fallback_scale_map
        if callable(fallback_scale_map):
            fallback_scale_map = fallback_scale_map(record)

        # Default split: the whole prompt is rewordable, so the template is
        # just the slot. Sources with a rigid elicitation wrapper override both.
        item_text = (
            str(source.resolve(record, source.item_text))
            if source.item_text else str(query)
        )
        prompt_template = (
            str(source.resolve(record, source.prompt_template))
            if source.prompt_template else ITEM
        )

        ask = str(source.resolve(record, source.ask) or "") if source.ask else ""
        if ask:
            # The ask is part of the query the model sees, but never part of
            # what stage 2 may reword. A query that doesn't already end with
            # the declared ask is refused rather than appended to a second
            # time, which would silently duplicate the instruction.
            stripped_query = str(query).rstrip()
            if not stripped_query.endswith(ask):
                raise SchemaError(
                    f"{source.name}:{native_id}: query does not end with the declared ask"
                )
            item_text = stripped_query[: -len(ask)].rstrip()
            prompt_template = f"{ITEM}\n\n{ask}"
            query = f"{item_text}\n\n{ask}"

        must_survive = source.must_survive
        if callable(must_survive):
            must_survive = must_survive(record)

        system_prompt = str(
            source.resolve(record, source.system_prompt) or ""
        ) if source.system_prompt else ""

        row = Row(
            sample_id=f"{source.name}:{native_id}",
            source=source.name,
            risk=source.risk,
            question_type=source.question_type,
            query=str(query),
            criterion=str(source.resolve(record, source.criterion) or ""),
            rubric=str(source.resolve(record, source.rubric) or ""),
            categories=list(categories or []),
            scale_map={str(k): float(v) for k, v in (scale_map or {}).items()},
            choices=[str(c) for c in (choices or [])],
            target=str(target or ""),
            **({"fallback_categories": list(fallback_categories)}
               if fallback_categories else {}),
            **({"fallback_scale_map":
                {str(k): float(v) for k, v in fallback_scale_map.items()}}
               if fallback_scale_map else {}),
            detector=source.detector,
            system_prompt=system_prompt,
            item_text=item_text,
            prompt_template=prompt_template,
            elicitation_family=source.elicitation_family,
            judge_style=source.judge_style, role=source.role, pool=source.pool,
            summary=source.summary,
            families=source.families_for(system_prompt),
            metadata={**{key: _plain(record.get(key)) for key in source.metadata},
                      **({"must_survive": list(must_survive)} if must_survive else {})},
        )
        validate(row)
        rows.append(row)

    return _dedupe_ids(rows)


def _plain(value):
    '''Coerce numpy/pandas scalars so the metadata column is JSON-serialisable.'''
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _dedupe_ids(rows: list[Row]) -> list[Row]:
    '''Suffix collisions rather than dropping them; some sources reuse ids.'''
    seen: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        seen[row.sample_id] += 1
        if seen[row.sample_id] > 1:
            row.sample_id = f"{row.sample_id}#{seen[row.sample_id]}"
    return rows


def _identity(row: Row, distinct_on: Sequence[str]) -> tuple:
    """What makes this row a distinct item, beyond its text.

    `distinct_on` names fields whose differing values mean two rows are
    different items however similar they read. It is the same declaration
    near_dedup consults, applied here so one concept covers both tiers: the
    persona arms of a human-rights scenario share a user message and differ only
    in the system prompt, so keying on text alone collapsed three arms into one
    (observed as 288 of 432 rows dropped) and left nothing to compare.
    """
    return tuple(str(row.metadata.get(field, "")) for field in distinct_on)


def exact_dedup(
    rows: list[Row], distinct_on: Sequence[str] = ()
) -> tuple[list[Row], int]:
    kept, seen = [], set()
    for row in rows:
        key = (normalised(row.query), _identity(row, distinct_on))
        if key in seen:
            continue
        seen.add(key)
        kept.append(row)
    return kept, len(rows) - len(kept)



def cross_source_dedup(
    pools: list[tuple[Source, list[Row]]]
) -> tuple[list[tuple[Source, list[Row]]], list[dict]]:
    """Tier 1b: drop a prompt a later source ships identically to an earlier one.

    exact_dedup runs inside one source, so a benchmark that vendors another's
    items puts the same prompt in the cluster twice under two sample_ids —
    double-weighting it in every cluster mean.

    Two rules make this tier safe to run without review, unlike near_dedup:

    - It compares the prompt *as delivered*, user text plus system text. A
      source that wraps the same question in its own system prompt is asking
      something else, and survives.
    - A source's own texts are registered only once its whole pool has been
      walked, so identical text inside one source never collides with itself.
      That is tier 1's call to make, where `distinct_on` can declare the rows
      distinct items; across sources there is no such shared declaration, so
      identical delivered text is a copy.

    Earlier sources win, making the survivor a function of registration order
    rather than of which pool happened to be walked first.
    """
    def key(row: Row) -> tuple[str, str]:
        return normalised(row.query), normalised(row.system_prompt)

    seen: dict[tuple[str, str], Row] = {}
    kept_pools, dropped = [], []

    for source, rows in pools:
        kept = []
        for row in rows:
            incumbent = seen.get(key(row))
            if incumbent is None:
                kept.append(row)
                continue
            dropped.append({
                "tier": "exact_cross_source",
                "similarity": 1.0,
                "kept": incumbent.sample_id,
                "kept_text": incumbent.query[:300],
                "dropped": row.sample_id, "dropped_text": row.query[:300],
            })
        for row in kept:
            seen.setdefault(key(row), row)
        kept_pools.append((source, kept))

    return kept_pools, dropped


# ----- tier 2: cosine near-dedup -----

def _distinguishable(left: Row, right: Row, distinct_on: Sequence[str]) -> bool:
    '''
    Exact guard against the similarity filter's blind spot: when a benchmark
    varies one term inside a fixed template, the entire distinction is a few
    characters that barely moves a whole-text similarity. Items differing in
    ground truth, or in a field the source declares identifying, are never
    duplicates however similar the surrounding wording.
    '''
    if left.question_type == MCQ and left.target != right.target:
        return False
    return all(
        left.metadata.get(field) == right.metadata.get(field) for field in distinct_on
    )


def _vectors(rows: list[Row], payload, embeddings: dict[str, np.ndarray]) -> np.ndarray:
    '''One unit vector per row's payload. A payload with no words gets zeros, so
    it is similar to nothing: never a duplicate, never "close" in a spread.'''
    dim = len(next(iter(embeddings.values()), ()))
    matrix = np.zeros((len(rows), dim), dtype=np.float32)
    for position, row in enumerate(rows):
        key = embed_key(payload(row))
        if key:
            matrix[position] = embeddings[key]
    return matrix


def near_dedup(
    rows: list[Row],
    embeddings: dict[str, np.ndarray],
    tau: float = COSINE_TAU,
    *,
    dedup_on: str | None = None,
    distinct_on: Sequence[str] = (),
) -> tuple[list[Row], list[dict]]:
    '''
    Cosine near-dedup over cached embeddings. Returns survivors and the dropped
    pairs, which get written out so `tau` can be reviewed on real data.

    `dedup_on` compares a metadata field instead of the rendered query — the
    doc's "filter the case pool, never the rendered prompt" rule, made
    executable: PHT's payload is the historical event, not the 100-word
    instruction wrapped around it.
    '''
    payload = _payload_fn(dedup_on)
    vectors = _vectors(rows, payload, embeddings)

    candidates = []
    for start in range(0, len(rows), _BLOCK):
        # Rounded so a BLAS summing in another order cannot flip a pair across
        # tau or reorder ties between machines.
        similarity = np.round(vectors[start:start + _BLOCK] @ vectors.T, 6)
        for offset, right in zip(*np.nonzero(similarity >= tau)):
            left, right = start + int(offset), int(right)
            if left < right and _distinguishable(rows[left], rows[right], distinct_on):
                candidates.append((float(similarity[offset, right]), left, right))

    dropped_indices: set[int] = set()
    dropped_pairs = []
    for score, left, right in sorted(candidates, reverse=True):
        if left in dropped_indices or right in dropped_indices:
            continue
        dropped_indices.add(right)
        dropped_pairs.append({
            "tier": "near",
            "similarity": round(score, 4),
            "kept": rows[left].sample_id, "kept_text": payload(rows[left])[:300],
            "dropped": rows[right].sample_id, "dropped_text": payload(rows[right])[:300],
        })

    survivors = [row for index, row in enumerate(rows) if index not in dropped_indices]
    return survivors, dropped_pairs


# ----- tier 3: stratified quota -----

def stratified_sample(
    rows: list[Row], source: Source, seed: int, caches: Caches | None = None
) -> tuple[list[Row], dict]:
    if source.group_key:
        return _grouped_sample(rows, source, seed, caches)
    return _row_sample(rows, source, seed, caches)


def _grouped_sample(
    rows: list[Row], source: Source, seed: int, caches: Caches | None = None
) -> tuple[list[Row], dict]:
    '''
    Sample whole groups, so rows that are only meaningful together survive
    together.

    The persona arms of one human-rights scenario are compared against each
    other; sampling rows independently would keep a scenario's neutral arm and
    drop its government-authority arm, leaving nothing to compare and silently
    computing the gap over mismatched scenarios. The quota therefore counts
    groups, not rows — a quota of 20 over 3-arm groups yields 60 rows.
    '''
    groups: defaultdict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row.metadata.get(source.group_key, index))].append(index)

    # Select over one representative row per group, so groups are picked by the
    # same stratification the source declares, then expand back to every member.
    leaders = {key: rows[indices[0]] for key, indices in groups.items()}
    picked, report = _row_sample(list(leaders.values()), source, seed, caches)

    by_id = {id(row): key for key, row in leaders.items()}
    wanted = {by_id[id(row)] for row in picked}
    chosen = [i for key, indices in groups.items() if key in wanted for i in indices]

    report["groups"] = len(groups)
    report["allocated"] = len(chosen)
    return [rows[i] for i in sorted(chosen)], report


UNIFORM = "uniform"
DIVERSE = "diverse"


def _stable_order(rows: list[Row], indices: list[int], seed: int) -> list[int]:
    '''
    `indices` ordered by a hash of each row's id — a uniform draw that is a
    property of the *item* rather than of the pool.

    `frame.sample(random_state=seed)` pins a shuffle of positions, so one extra
    upstream row re-drew a large share of the selection (measured at 90% on
    cyber_false_refusal for a single-row change). That churn silently changed
    which items were certified between dataset versions and re-invalidated every
    stage-2/3 artifact, which are keyed by sample_id. Hashing the id instead
    makes an item's fate depend only on its own hash, so a pool change moves
    nothing else.
    '''
    def key(index: int) -> bytes:
        return hashlib.blake2b(
            f"{seed}:{rows[index].sample_id}".encode(), digest_size=16
        ).digest()

    return sorted(indices, key=key)


def _diverse_order(
    rows: list[Row], indices: list[int], take: int, source: Source, seed: int,
    embeddings: dict[str, np.ndarray],
) -> list[int]:
    '''
    Greedy farthest-point on embedding cosine: repeatedly take the item least
    similar to everything already taken.

    Near-dedup only removes pairs above tau — it never asks whether the *kept*
    set spans its stratum. This does, so a quota of 90 drawn from 12,662 buys
    coverage rather than a lottery ticket.

    Compares the same payload near_dedup does (`dedup_on` where declared, the
    query otherwise). The first pick comes from `_stable_order` and ties break
    on `key_bytes`, so the walk is deterministic without being tied to input
    order.
    '''
    vectors = _vectors([rows[i] for i in indices], _payload_fn(source.dedup_on), embeddings)
    ties = [key_bytes(rows[i], seed) for i in indices]
    first = indices.index(_stable_order(rows, indices, seed)[0])
    picked = [first]
    # Each item's similarity to the closest pick so far, taken items pinned at
    # +inf; the next pick minimises it.
    nearest = np.round(vectors @ vectors[first], 6)
    nearest[first] = np.inf
    while len(picked) < take:
        candidate = min(range(len(indices)), key=lambda p: (nearest[p], ties[p]))
        picked.append(candidate)
        nearest = np.maximum(nearest, np.round(vectors @ vectors[candidate], 6))
        nearest[candidate] = np.inf
    return [indices[p] for p in picked]


def key_bytes(row: Row, seed: int) -> bytes:
    '''Deterministic tie-break, so equally-distant candidates resolve stably.'''
    return hashlib.blake2b(
        f"{seed}:{row.sample_id}".encode(), digest_size=16
    ).digest()


def _payload_fn(dedup_on: str | None):
    '''The text that identifies an item — near_dedup's rule, reused.'''
    if dedup_on:
        return lambda row: str(row.metadata.get(dedup_on, ""))
    return lambda row: row.query


def _take(
    rows: list[Row], indices: list[int], take: int, source: Source, seed: int,
    caches: Caches | None = None,
) -> list[int]:
    '''
    Fill one stratum's allotment. With the screen on: pre-select SCREEN_FACTOR x
    the allotment by the source's own selection, drop what Hermes refused, and
    fill the allotment from the survivors by the same selection. A shortfall is
    an error while a wider pre-selection could still fill it, and accepted once
    the pre-selection already covers the whole stratum.
    '''
    if caches is None or caches.verdicts is None or not source.screened():
        return _select(rows, indices, take, source, seed, caches)
    pool = _select(rows, indices, math.ceil(SCREEN_FACTOR * take), source, seed, caches)
    kept = _screen(rows, pool, caches)
    if len(kept) < take and len(pool) < len(indices):
        raise ValueError(
            f"{source.name}: the screen kept {len(kept)} of {len(pool)} candidates for an "
            f"allotment of {take}; short by {take - len(kept)}. Raise SCREEN_FACTOR "
            f"({SCREEN_FACTOR}) rather than shrink the quota."
        )
    return _select(rows, kept, take, source, seed, caches)


def _select(
    rows: list[Row], indices: list[int], take: int, source: Source, seed: int,
    caches: Caches | None = None,
) -> list[int]:
    '''Fill one stratum's allotment, by whichever selection the source declares.'''
    if take >= len(indices):
        return list(indices)
    if source.select == UNIFORM:
        return _stable_order(rows, indices, seed)[:take]
    if source.select == DIVERSE:
        if caches is None:
            raise ValueError(f"{source.name}: diverse selection needs the embedding cache")
        return _diverse_order(rows, indices, take, source, seed, caches.embeddings)
    raise ValueError(f"{source.name}: unknown select mode {source.select!r}")


def _row_sample(
    rows: list[Row], source: Source, seed: int, caches: Caches | None = None
) -> tuple[list[Row], dict]:
    quota = source.quota
    if quota is None or quota >= len(rows):
        if source.select not in (UNIFORM, DIVERSE):
            raise ValueError(f"{source.name}: unknown select mode {source.select!r}")
        chosen = _take(rows, list(range(len(rows))), len(rows), source, seed, caches)
        return [rows[i] for i in sorted(chosen)], {"strata": 0, "allocated": len(chosen)}

    if not source.stratify:
        chosen = _take(rows, list(range(len(rows))), quota, source, seed, caches)
        return [rows[i] for i in sorted(chosen)], {"strata": 1, "allocated": len(chosen)}

    keys = [
        tuple(str(row.metadata.get(column, "")) for column in source.stratify)
        for row in rows
    ]
    buckets: defaultdict[tuple, list[int]] = defaultdict(list)
    for index, key in enumerate(keys):
        buckets[key].append(index)

    allocation = _allocate(buckets, quota, balanced=source.balanced)

    chosen: list[int] = []
    for key, take in allocation.items():
        chosen.extend(_take(rows, buckets[key], take, source, seed, caches))

    return (
        [rows[i] for i in sorted(chosen)],
        {"strata": len(buckets), "allocated": len(chosen)},
    )


def _allocate(buckets: dict, quota: int, *, balanced: bool) -> dict:
    '''
    Split `quota` across strata. `balanced` gives every stratum the same share
    regardless of its size — needed where the metric depends on group balance
    (DAB leader favourability), and wrong everywhere else, since it would
    distort the source's own distribution.
    '''
    sizes = {key: len(indices) for key, indices in buckets.items()}
    total = sum(sizes.values())

    # More strata than budget: a floor of one per stratum would overshoot the
    # quota, so cover as many strata as the budget allows instead. Largest
    # first, ties broken by key, so the choice stays deterministic.
    if len(sizes) > quota:
        order = sorted(sizes, key=lambda key: (-sizes[key], key))
        return {key: 1 for key in order[:quota]}

    if balanced:
        base = quota // len(buckets)
        allocation = {key: min(base, size) for key, size in sizes.items()}
    else:
        allocation = {
            key: max(1, round(quota * size / total)) for key, size in sizes.items()
        }
        allocation = {key: min(take, sizes[key]) for key, take in allocation.items()}

    # Hand out or claw back the rounding remainder, largest strata first, so the
    # total lands exactly on the quota and the result stays deterministic.
    order = sorted(sizes, key=lambda key: (-sizes[key], key))
    while sum(allocation.values()) != quota:
        short = quota - sum(allocation.values())
        moved = False
        for key in order if short > 0 else reversed(order):
            if short > 0 and allocation[key] < sizes[key]:
                allocation[key] += 1
                moved = True
            elif short < 0 and allocation[key] > 1:
                allocation[key] -= 1
                moved = True
            if sum(allocation.values()) == quota:
                break
        if not moved:
            break  # every stratum is exhausted or at its floor

    return allocation


# ----- driver -----

def build_risk(risk: str, seed: int) -> tuple[list[Row], dict, list[dict]]:
    sources = for_risk(risk)
    if not sources:
        raise SystemExit(f"no sources registered for risk {risk!r}")

    all_rows: list[Row] = []
    all_dropped: list[dict] = []
    report = {}

    # Tier 1 runs for every source first, so the embedding cache is checked for
    # the whole risk at once: one miss, one embed run.
    pools = []
    for source in sources:
        rows = load_source(source)
        loaded = len(rows)
        rows, exact_dropped = exact_dedup(rows, source.distinct_on)
        report[source.name] = {
            "loaded": loaded,
            "exact_dropped": exact_dropped,
            "near_dropped": 0,
            "cross_source_dropped": 0,
            "quota": source.quota,
            "stratify_on": list(source.stratify),
            "balanced": source.balanced,
            "question_type": source.question_type,
            "path": source.path,
        }
        pools.append((source, rows))

    embeddings = load_embeddings(risk)
    require_embeddings(risk, pools, embeddings)

    # Tier 2 is per-source, because tau, dedup_on and distinct_on are per-source
    # declarations. Tier 1b then runs over the assembled pools — before tier 3,
    # so a copy is removed while its source can still backfill the quota from
    # its own pool rather than leaving the cluster short.
    deduped = []
    for source, rows in pools:
        if source.dedup:
            rows, near_dropped = near_dedup(
                rows, embeddings,
                source.tau if source.tau is not None else COSINE_TAU,
                dedup_on=source.dedup_on,
                distinct_on=source.distinct_on,
            )
            all_dropped.extend(near_dropped)
            report[source.name]["near_dropped"] = len(near_dropped)
        deduped.append((source, rows))
    pools = deduped

    sizes = {source.name: len(rows) for source, rows in pools}
    pools, cross_dropped = cross_source_dedup(pools)
    all_dropped.extend(cross_dropped)
    caches = Caches(embeddings, verdicts=load_screen(risk))
    for source, rows in pools:
        report[source.name]["cross_source_dropped"] = sizes[source.name] - len(rows)
        refused, candidates = len(caches.refused), caches.candidates
        rows, allocation = stratified_sample(rows, source, seed, caches)
        report[source.name]["kept"] = len(rows)
        report[source.name]["strata"] = allocation["strata"]
        if source.screened():
            report[source.name]["screen_candidates"] = caches.candidates - candidates
            report[source.name]["screen_refused"] = len(caches.refused) - refused
        all_rows.extend(rows)

    require_screen(risk, caches)
    all_dropped.extend(caches.refused)
    return all_rows, report, all_dropped


def source_revisions() -> dict:
    '''Pin what produced this build: the repo HEAD, submodule SHAs, and the
    resolved revision of every datasets/raw/<name>/fetch.json (scripts/fetch_raw.py).'''
    def git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
            ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"

    revisions = {"repo": git("rev-parse", "HEAD")}

    # Read gitlinks (mode 160000) straight from the index rather than using
    # `git submodule status`, which aborts entirely if any path is missing from
    # .gitmodules — one stale entry would otherwise leave the provenance blank
    # without failing the build.
    for line in git("ls-files", "--stage").splitlines():
        fields = line.split(maxsplit=3)
        if len(fields) == 4 and fields[0] == "160000":
            revisions[fields[3].strip()] = fields[1]

    for record in sorted((REPO_ROOT / "datasets" / "raw").glob("*/fetch.json")):
        revisions[f"datasets/raw/{record.parent.name}"] = json.loads(record.read_text())["revision"]

    if set(revisions) == {"repo"}:
        revisions["_warning"] = "no source revisions recorded"
    return revisions


def write_outputs(risk: str, rows: list[Row], report: dict, dropped: list[dict], seed: int):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame([row.to_csv_row() for row in rows], columns=COLUMNS)
    csv_path = OUT_DIR / f"{risk}.csv"
    frame.to_csv(csv_path, index=False)

    screened = [source.name for source in for_risk(risk) if source.screened()]
    meta = {
        "risk": risk,
        "rows": len(rows),
        "seed": seed,
        "embedding": {
            "model": EMBEDDING_MODEL,
            "tau_cosine": COSINE_TAU,
            "cache": f"datasets/cache/embeddings/{risk}.npz",
        },
        "screen": {
            "model": sorted({record["model"] for record in load_screen(risk).values()}),
            "applies_to": screened,
            "candidate_factor": SCREEN_FACTOR,
            "refused_dropped": {name: report[name]["screen_refused"] for name in screened},
        },
        "sources": report,
        "revisions": source_revisions(),
    }
    (OUT_DIR / f"{risk}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    with open(OUT_DIR / f"{risk}.dropped.jsonl", "w", encoding="utf-8") as f:
        for pair in dropped:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    return csv_path


def print_report(risk: str, report: dict, rows: list[Row]):
    print(f"\n=== {risk} ===")
    header = (f"  {'source':22s} {'loaded':>7s} {'exact':>6s} {'near':>6s} "
              f"{'cross':>6s} {'screen':>6s} {'kept':>6s} {'share':>6s}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    total = len(rows) or 1
    for name, stats in report.items():
        refused = stats.get("screen_refused", 0)
        print(
            f"  {name:22s} {stats['loaded']:7d} {stats['exact_dropped']:6d} "
            f"{stats['near_dropped']:6d} {stats['cross_source_dropped']:6d} "
            f"{refused:6d} {stats['kept']:6d} {100 * stats['kept'] / total:5.1f}%"
        )
        candidates = stats.get("screen_candidates", 0)
        if candidates and 2 * refused > candidates:
            print(f"  [WARNING] {name}: the screen refused {refused} of {candidates} "
                  f"candidates; raise SCREEN_FACTOR rather than shrink the quota")
    print(f"  {'TOTAL':22s} {'':7s} {'':6s} {'':6s} {'':6s} {'':6s} {total:6d}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--risk", choices=RISKS, action="append",
                        help="build one risk (repeatable); default is all registered")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the tier table without writing")
    args = parser.parse_args()

    risks = args.risk or [risk for risk in RISKS if for_risk(risk)]

    pending = []
    for risk in risks:
        try:
            rows, report, dropped = build_risk(risk, args.seed)
        except CacheMiss as miss:
            print(f"\n=== {risk}: cache miss ===\n  {miss}")
            pending.append(risk)
            continue
        print_report(risk, report, rows)
        if args.dry_run:
            continue
        path = write_outputs(risk, rows, report, dropped, args.seed)
        print(f"  wrote {len(rows)} rows -> {path.relative_to(REPO_ROOT)}")
    if pending:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
