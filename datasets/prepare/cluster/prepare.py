'''
Build one dataset per CoP leaf.

    uv run python3 -m datasets.prepare.cluster.prepare --leaf cyber_operational_use
    uv run python3 -m datasets.prepare.cluster.prepare --dry-run

Selection, per leaf (datasets/BENCHMARKS.md § Sampling):

1. Load every source annotated with the leaf (annotations.json, or the
   Source's hand `leaves`); drop exact repeats inside a source and across them.
2. Score every row by its max cosine to the leaf's anchors (legal text and
   exemplars).
3. Per source, keep its top P_MAX x r share of units (r = the annotators'
   relevance of the source to the leaf), never fewer than FLOOR_MIN, and never a
   unit that looks less like the leaf than 95% of the items the annotators did
   not assign to it (the off-topic guard). Ranks and quantiles, not cosines, so
   any embedder works.
4. One pass over the whole leaf in score order: a row is kept when its score,
   less a penalty for resembling a row already kept (from any source), still
   clears its source's cut. "Resembling" is measured against the embedder: the
   hinge starts at the median nearest-neighbour similarity in the item cache.

Writes datasets/public/<leaf>.csv plus <leaf>.meta.json (seed, constants,
embedding model, per-source relevance, thresholds and counts, source revisions)
and <leaf>.dropped.jsonl (every drop, tagged exact, exact_cross_source,
threshold or redundant, so the constants are reviewable rather than trusted).

Reads one gitignored cache, datasets/cache/embeddings/ (items.npz for queries,
leaves.npz for anchors). On a miss it appends what is missing to the embed
input, prints the command that fills it and exits 2.
'''

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from . import readers
from .leaves import Leaf, anchor_texts, load_leaves
from .schema import (
    COLUMNS, ITEM, Row, SchemaError, Source, normalised, validate,
)
from .sources import LEAVES, annotated, for_leaf

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUT_DIR = REPO_ROOT / "datasets" / "public"
CACHE_DIR = REPO_ROOT / "datasets" / "cache"

# Kept equal to scripts/embed_items.py::MODEL (tests/test_clusters.py checks).
# --embedding-model overrides it for a comparison; the cache then lives under
# embeddings/<model slug>/ (scripts/embed_items.py::cache_path).
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_MODEL = EMBEDDING_MODEL
EMBED_COMMAND = (
    "uv run --no-project --with sentence-transformers --with numpy "
    "python scripts/embed_items.py"
)

# Selection constants. A source keeps its top P_MAX x relevance share of units
# per leaf (r = 1 means "every item is this leaf": keep 5%), at least FLOOR_MIN.
# The redundancy penalty is a hinge: resemblance to a kept row below the
# embedder's near-duplicate level (`hinge_start`, measured from the cache) costs
# nothing, above it costs PENALTY per unit of cosine, so a near-copy drops out
# only when it was within that margin of its cut anyway.
P_MAX = 0.05
FLOOR_MIN = 5
PENALTY = 1.0
# Resemblance starts to cost at the median nearest-neighbour similarity in the
# cache: closer than a typical item is to its own closest item is a near-copy.
# The 10th percentile (0.70 on MiniLM) dropped more rows than it kept.
HINGE_QUANTILE = 50
GUARD_QUANTILE = 95   # percentile of off-topic items' scores a kept unit must beat
class CacheMiss(Exception):
    '''The embedding cache lacks entries. The message says what to run.'''


def scoring_text(row: Row) -> str:
    '''What a row is embedded as: its scoring_text when the source declares one,
    else its query, with the system prompt in front when there is one, so a
    source whose content sits in the system prompt (persusafety) does not score
    every row on one constant instruction.'''
    text = row.scoring_text or row.query
    return f"{row.system_prompt}\n\n{text}" if row.system_prompt else text


def _cache_miss(path: Path, missing: dict[str, str], cache: dict) -> CacheMiss:
    '''Add what the cache lacks to the embed input at `path` (keeping earlier
    misses the cache still lacks, so a run over several leaves accumulates them)
    and build the error naming the fix.'''
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record["key"] not in cache:
                missing.setdefault(record["key"], record["text"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for key, text in sorted(missing.items()):
            f.write(json.dumps({"key": key, "text": text}, ensure_ascii=False) + "\n")
    return CacheMiss(f"{len(missing)} missing -> {path}\n  run: {embed_command()}")


def embed_key(text: str) -> str | None:
    '''
    Cache key of a text's embedding: blake2b-16 of its normalised form, which
    is also the text embedded (all-MiniLM-L6-v2 is uncased, so folding case and
    punctuation loses nothing). Texts that normalise identically share one
    vector by construction. None for a text with no words.
    '''
    text = normalised(text)
    return hashlib.blake2b(text.encode(), digest_size=16).hexdigest() if text else None


def embedding_cache(name: str) -> Path:
    '''`name` is "items" (every source's queries) or "leaves" (every leaf's anchors).'''
    base = CACHE_DIR / "embeddings"
    if EMBEDDING_MODEL != DEFAULT_EMBEDDING_MODEL:
        base = base / EMBEDDING_MODEL.replace("/", "__")
    return base / f"{name}.npz"


def embed_command() -> str:
    command = EMBED_COMMAND
    if EMBEDDING_MODEL != DEFAULT_EMBEDDING_MODEL:
        command += f" --model {EMBEDDING_MODEL}"
    return command


def load_embeddings(name: str) -> dict[str, np.ndarray]:
    '''key -> unit float32 vector from the named cache (embedding_cache); {} if absent.'''
    path = embedding_cache(name)
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
    pools: list[tuple[Source, list[Row]]], embeddings: dict, leaf: Leaf, leaf_embeddings: dict,
) -> None:
    '''Raise CacheMiss, after writing the embed input, if any query or anchor of `leaf` lacks a vector.'''
    missing = {}
    for source, rows in pools:
        for row in rows:
            key = embed_key(scoring_text(row))
            if key and key not in embeddings:
                missing[key] = normalised(scoring_text(row))
    if missing:
        raise _cache_miss(CACHE_DIR / "items.embed_input.jsonl", missing, embeddings)
    anchors = {embed_key(text): normalised(text) for text in anchor_texts(leaf)}
    missing_anchors = {key: text for key, text in anchors.items() if key not in leaf_embeddings}
    if missing_anchors:
        raise _cache_miss(CACHE_DIR / "leaves.embed_input.jsonl", missing_anchors, leaf_embeddings)


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
        scoring = str(
            source.resolve(record, source.scoring_text) or ""
        ) if source.scoring_text else ""

        row = Row(
            sample_id=f"{source.name}:{native_id}",
            source=source.name,
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
            scoring_text=scoring,
            metadata={**{key: _plain(record[key]) for key in source.metadata},
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
    different items however similar they read: the persona arms of a
    human-rights scenario share a user message and differ only in the system
    prompt, so keying on text alone collapsed three arms into one (observed as
    288 of 432 rows dropped) and left nothing to compare.
    """
    return tuple(str(row.metadata.get(field, "")) for field in distinct_on)


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



def cross_source_dedup(
    pools: list[tuple[Source, list[Row]]]
) -> tuple[list[tuple[Source, list[Row]]], list[dict]]:
    """Tier 1b: drop a prompt a later source ships identically to an earlier one.

    exact_dedup runs inside one source, so a benchmark that vendors another's
    items puts the same prompt in the cluster twice under two sample_ids —
    double-weighting it in every cluster mean.

    Two rules make this tier safe to run without review:

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


# ----- scoring -----

def _vectors(rows: list[Row], embeddings: dict[str, np.ndarray]) -> np.ndarray:
    '''One unit vector per row's query, as float64. A query with no words gets
    zeros, so it is similar to nothing: it scores 0 and never penalises anything.'''
    dim = len(next(iter(embeddings.values()), ()))
    matrix = np.zeros((len(rows), dim), dtype=np.float64)
    for position, row in enumerate(rows):
        key = embed_key(scoring_text(row))
        if key:
            matrix[position] = embeddings[key]
    return matrix


def key_bytes(row: Row, seed: int) -> bytes:
    '''Deterministic tie-break, so equally-scored rows resolve stably.'''
    return hashlib.blake2b(
        f"{seed}:{row.sample_id}".encode(), digest_size=16
    ).digest()


def leaf_anchors(leaf: Leaf, leaf_embeddings: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    '''Anchor names ("legal", then "exemplar:<i>") and their vectors as float64 rows.'''
    names = ["legal", *(f"exemplar:{i}" for i in range(len(leaf.exemplars)))]
    keys = [embed_key(text) for text in anchor_texts(leaf)]
    missing = [name for name, key in zip(names, keys) if key not in leaf_embeddings]
    if missing:
        raise CacheMiss(f"leaf {leaf.id!r}: no embedding for anchors {missing}; "
                        f"run: {embed_command()}")
    return names, np.array([leaf_embeddings[key] for key in keys], dtype=np.float64)


def relevance_scores(vectors: np.ndarray, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''Each row's max cosine over the anchors, and the index of the anchor that
    attains it (the first, on a tie; -1 for a zero vector, which matched
    nothing). Rounded so BLAS summation order cannot reorder rows across machines.'''
    similarity = np.round(vectors @ matrix.T, 9)
    hit = similarity.argmax(axis=1)
    hit[~vectors.any(axis=1)] = -1
    return similarity.max(axis=1), hit


def _group_ids(source: Source, rows: list[Row]) -> list[str]:
    '''Each row's group; a row without the group_key field is its own group.'''
    return [str(row.metadata.get(source.group_key, i)) for i, row in enumerate(rows)]


def score_rows(
    source: Source, rows: list[Row], leaf: Leaf,
    embeddings: dict[str, np.ndarray], leaf_embeddings: dict[str, np.ndarray],
) -> np.ndarray:
    '''
    Each row's selection score: its max cosine to the leaf's anchors; for a
    group_key source, the best arm of the row's group on every arm, so a group
    is kept or dropped whole. Written to metadata as relevance (the score),
    relevance_anchor and, for groups, relevance_own.
    '''
    vectors = _vectors(rows, embeddings)
    names, matrix = leaf_anchors(leaf, leaf_embeddings)
    cosine, hit = relevance_scores(vectors, matrix)
    unit = cosine
    if source.group_key:
        group_best: dict[str, float] = {}
        groups = _group_ids(source, rows)
        for group, score in zip(groups, cosine):
            group_best[group] = max(group_best.get(group, -np.inf), float(score))
        unit = np.array([group_best[group] for group in groups])
    for i, row in enumerate(rows):
        # A fresh dict: readers may share one metadata dict across rows.
        row.metadata = {**row.metadata, "relevance": round(float(unit[i]), 6),
                        "relevance_anchor": names[hit[i]] if hit[i] >= 0 else None}
        if source.group_key:
            row.metadata["relevance_own"] = round(float(cosine[i]), 6)
    return unit


def on_topic_guard(
    leaf: Leaf, pool_keys: set[str], embeddings: dict[str, np.ndarray],
    leaf_embeddings: dict[str, np.ndarray],
) -> float | None:
    '''The score a unit must beat to count as on-topic: the GUARD_QUANTILE-th
    percentile of the leaf score over every cached item outside the leaf's own
    pool, i.e. items the annotators assigned to other leaves. Measured per leaf
    and per embedder, so "chance level" needs no constant in cosine units. None
    when the cache holds nothing outside the pool.'''
    keys = sorted(set(embeddings) - pool_keys)
    if not keys:
        return None
    _, anchors = leaf_anchors(leaf, leaf_embeddings)
    others = np.array([embeddings[key] for key in keys], dtype=np.float64)
    scores = np.round(others @ anchors.T, 9).max(axis=1)
    return round(float(np.percentile(scores, GUARD_QUANTILE)), 6)


def source_cut(unit_scores: Sequence[float], relevance: float, guard: float | None) -> tuple[float, int, bool]:
    '''The score a unit of this source must reach: the score at rank
    max(ceil(P_MAX x relevance x units), FLOOR_MIN), raised to `guard` when the
    guard is higher. Returns (cut, rank, guard_used).'''
    ranked = sorted(unit_scores, reverse=True)
    if not ranked:
        return float("inf"), 0, False
    rank = min(max(math.ceil(round(P_MAX * relevance * len(ranked), 9)), FLOOR_MIN), len(ranked))
    cut = ranked[rank - 1]
    if guard is not None and guard > cut:
        return float(guard), rank, True
    return float(cut), rank, False


def hinge_start(embeddings: dict[str, np.ndarray], seed: int, sample: int = 2000) -> float:
    '''Where resemblance starts to cost: the HINGE_QUANTILE-th percentile of
    nearest-neighbour cosine among `sample` seeded picks from the item cache.
    Measured per embedder, since "near-duplicate" sits at 0.7 on MiniLM and 0.9
    on e5-large.'''
    keys = sorted(embeddings)
    if len(keys) < 2:
        return 1.0
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(keys), size=min(sample, len(keys)), replace=False)
    matrix = np.array([embeddings[key] for key in keys], dtype=np.float64)
    similarity = np.round(matrix[picks] @ matrix.T, 9)
    similarity[np.arange(len(picks)), picks] = -np.inf
    return round(float(np.percentile(similarity.max(axis=1), HINGE_QUANTILE)), 6)


@dataclass
class Unit:
    '''One selection unit: a row, or every arm of a group_key group.'''
    source: str
    rows: list[Row]
    score: float
    order: bytes  # tie-break


def units_for(source: Source, rows: list[Row], scores: np.ndarray, seed: int) -> list[Unit]:
    if not source.group_key:
        return [Unit(source.name, [row], float(score), key_bytes(row, seed))
                for row, score in zip(rows, scores)]
    groups: dict[str, Unit] = {}
    for group, row, score in zip(_group_ids(source, rows), rows, scores):
        unit = groups.get(group)
        if unit is None:
            groups[group] = Unit(source.name, [row], float(score), key_bytes(row, seed))
        else:
            unit.rows.append(row)
            unit.score = max(unit.score, float(score))  # the group's best arm
    return list(groups.values())


def cluster_select(
    units: list[Unit], thetas: dict[str, float], embeddings: dict[str, np.ndarray],
    penalty_free: float,
) -> tuple[list[Row], list[dict]]:
    '''
    The one pass over a leaf. Units are walked best score first (ties by
    `order`); each is kept when score - PENALTY x max(0, maxsim - penalty_free)
    reaches its source's cut, where maxsim is the highest cosine between any
    of its rows and any row already kept, from any source. So between two
    near-copies in different benchmarks the higher-scoring one is kept, and
    no benchmark order is involved. Drops are tagged `threshold` (below theta
    on its own) or `redundant` (only the penalty sank it; `similar_to` names
    the kept row).
    '''
    units = sorted(units, key=lambda unit: (-unit.score, unit.order))
    total_rows = sum(len(unit.rows) for unit in units)
    dim = len(next(iter(embeddings.values()), ()))
    kept_vectors = np.zeros((total_rows, dim), dtype=np.float64)
    kept_rows: list[Row] = []
    dropped: list[dict] = []
    for unit in units:
        theta = thetas[unit.source]
        if unit.score < theta:
            dropped.extend(_drop(unit, "threshold", theta))
            continue
        vectors = _vectors(unit.rows, embeddings)
        similar_to, maxsim = None, 0.0
        if kept_rows:
            similarity = np.round(vectors @ kept_vectors[:len(kept_rows)].T, 9)
            arm, kept_index = np.unravel_index(similarity.argmax(), similarity.shape)
            maxsim = float(similarity[arm, kept_index])
            similar_to = kept_rows[kept_index]
        effective = unit.score - PENALTY * max(0.0, maxsim - penalty_free)
        if effective < theta:
            dropped.extend(_drop(unit, "redundant", theta, similar_to, maxsim))
            continue
        kept_vectors[len(kept_rows):len(kept_rows) + len(unit.rows)] = vectors
        kept_rows.extend(unit.rows)
    return kept_rows, dropped


def _drop(unit: Unit, tier: str, theta: float, similar_to: Row | None = None,
          similarity: float | None = None) -> list[dict]:
    records = []
    for row in unit.rows:
        record = {"tier": tier, "dropped": row.sample_id, "dropped_text": row.query[:300],
                  "score": round(unit.score, 6), "threshold": round(theta, 6)}
        if similar_to is not None:
            record.update(similarity=round(similarity, 6), kept=similar_to.sample_id,
                          kept_text=similar_to.query[:300])
        records.append(record)
    return records


def leaf_auc(
    leaf: Leaf, pool: np.ndarray, leaf_embeddings: dict[str, np.ndarray], seed: int,
    negatives: int = 200,
) -> float | None:
    '''
    How well this embedder separates the leaf's own exemplars from the pool:
    the AUC of each exemplar's leave-one-out max cosine to the other anchors
    against the max cosine of `negatives` pool rows (a seeded, evenly spaced
    pick) to all anchors. None below two exemplars. 0.5 is chance.
    '''
    if len(leaf.exemplars) < 2 or not len(pool):
        return None
    _, anchors = leaf_anchors(leaf, leaf_embeddings)
    exemplars = anchors[1:]
    similarity = np.round(exemplars @ anchors.T, 9)
    np.fill_diagonal(similarity[:, 1:], -np.inf)
    positives = similarity.max(axis=1)
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(pool), size=min(negatives, len(pool)), replace=False)
    negative_scores = np.round(pool[picks] @ anchors.T, 9).max(axis=1)
    wins = (positives[:, None] > negative_scores[None, :]).sum() + 0.5 * (
        positives[:, None] == negative_scores[None, :]).sum()
    return round(float(wins / (len(positives) * len(negative_scores))), 3)


def _quantiles(scores: Sequence[float]) -> dict[str, float] | None:
    if not len(scores):
        return None
    points = np.percentile(np.asarray(scores, dtype=np.float64), [0, 10, 50, 90, 100])
    return dict(zip(("min", "p10", "p50", "p90", "max"), (round(float(p), 6) for p in points)))


# ----- driver -----

def build_leaf(leaf_id: str, seed: int) -> tuple[list[Row], dict, list[dict]]:
    sources = for_leaf(leaf_id)
    if not sources:
        raise SystemExit(f"no source is scored against leaf {leaf_id!r}")
    leaf = load_leaves()[leaf_id]

    all_dropped: list[dict] = []
    report = {}

    pools = []
    for source, relevance in sources:
        rows = load_source(source)
        loaded = len(rows)
        rows, exact_dropped = exact_dedup(rows, source.distinct_on)
        all_dropped.extend(exact_dropped)
        report[source.name] = {
            "loaded": loaded,
            "exact_dropped": len(exact_dropped),
            "cross_source_dropped": 0,
            "relevance": relevance,
            "annotated": annotated(source),
            "question_type": source.question_type,
            "path": source.path,
        }
        pools.append((source, rows))

    sizes = {source.name: len(rows) for source, rows in pools}
    pools, cross_dropped = cross_source_dedup(pools)
    all_dropped.extend(cross_dropped)
    for source, rows in pools:
        report[source.name]["cross_source_dropped"] = sizes[source.name] - len(rows)

    embeddings = load_embeddings("items")
    leaf_embeddings = load_embeddings("leaves")
    require_embeddings(pools, embeddings, leaf, leaf_embeddings)

    pool_keys = {embed_key(scoring_text(row)) for _, rows in pools for row in rows}
    guard = on_topic_guard(leaf, pool_keys, embeddings, leaf_embeddings)
    penalty_free = hinge_start(embeddings, seed)
    report["_guard"] = guard
    report["_penalty_free"] = penalty_free
    units: list[Unit] = []
    thetas: dict[str, float] = {}
    for source, rows in pools:
        scores = score_rows(source, rows, leaf, embeddings, leaf_embeddings)
        source_units = units_for(source, rows, scores, seed)
        cut, rank, guard_used = source_cut([unit.score for unit in source_units],
                                           report[source.name]["relevance"], guard)
        thetas[source.name] = cut
        report[source.name].update({
            "units": len(source_units), "rank": rank, "theta": round(cut, 6),
            "guard_used": guard_used,
            "score_quantiles": _quantiles([unit.score for unit in source_units]),
        })
        units.extend(source_units)

    pool_vectors = _vectors([row for _, rows in pools for row in rows], embeddings)
    report["_auc"] = leaf_auc(leaf, pool_vectors, leaf_embeddings, seed)

    kept, dropped = cluster_select(units, thetas, embeddings, penalty_free)
    for row in kept:
        row.leaf = leaf_id
    all_dropped.extend(dropped)
    counts = defaultdict(lambda: defaultdict(int))
    for record in dropped:
        counts[record["dropped"].split(":", 1)[0]][record["tier"]] += 1
    for row in kept:
        counts[row.source]["kept"] += 1
    for name in (name for name in report if not name.startswith("_")):
        report[name].update({
            "kept": counts[name]["kept"],
            "threshold_dropped": counts[name]["threshold"],
            "redundant_dropped": counts[name]["redundant"],
        })
    # Source registration order, then the build's score order inside a source.
    order = {source.name: i for i, (source, _) in enumerate(sources)}
    position = {id(row): i for i, row in enumerate(kept)}
    kept.sort(key=lambda row: (order[row.source], position[id(row)]))
    return kept, report, all_dropped


def source_revisions() -> dict:
    '''Pin what produced this build: the repo HEAD and the resolved revision of every datasets/raw/<name>/fetch.json (scripts/fetch_raw.py).'''
    def git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
            ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"

    revisions = {"repo": git("rev-parse", "HEAD")}

    for record in sorted((REPO_ROOT / "datasets" / "raw").glob("*/fetch.json")):
        revisions[f"datasets/raw/{record.parent.name}"] = json.loads(record.read_text())["revision"]

    if set(revisions) == {"repo"}:
        revisions["_warning"] = "no source revisions recorded"
    return revisions


def write_outputs(leaf_id: str, rows: list[Row], report: dict, dropped: list[dict], seed: int):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame([row.to_csv_row() for row in rows], columns=COLUMNS)
    csv_path = OUT_DIR / f"{leaf_id}.csv"
    frame.to_csv(csv_path, index=False)

    leaf = load_leaves()[leaf_id]
    auc = report.pop("_auc", None)
    guard = report.pop("_guard", None)
    penalty_free = report.pop("_penalty_free", None)
    meta = {
        "leaf": leaf_id,
        "cop_ref": leaf.cop_ref,
        "label": leaf.label,
        "rows": len(rows),
        "seed": seed,
        "embedding": {
            "model": EMBEDDING_MODEL,
            "cache": str(embedding_cache("items").relative_to(REPO_ROOT)),
        },
        "constants": {
            "p_max": P_MAX, "floor_min": FLOOR_MIN, "penalty": PENALTY,
            "hinge_quantile": HINGE_QUANTILE, "guard_quantile": GUARD_QUANTILE,
        },
        "guard": guard,
        "penalty_free": penalty_free,
        "anchors": len(anchor_texts(leaf)),
        "anchor_keys": sorted(embed_key(text) for text in anchor_texts(leaf)),
        "auc": auc,
        "sources": report,
        "revisions": source_revisions(),
    }
    (OUT_DIR / f"{leaf_id}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    with open(OUT_DIR / f"{leaf_id}.dropped.jsonl", "w", encoding="utf-8") as f:
        for record in dropped:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return csv_path


def print_report(leaf_id: str, report: dict, rows: list[Row]):
    print(f"\n=== {leaf_id} ===")
    header = (f"  {'source':22s} {'loaded':>7s} {'exact':>6s} {'cross':>6s} {'rel':>5s} "
              f"{'rank':>5s} {'cut':>6s} {'guard':>5s} {'kept':>6s} {'thr':>6s} {'red':>6s}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    total = len(rows) or 1
    leaf = load_leaves().get(leaf_id)
    if leaf is not None and not leaf.exemplars:
        print(f"  [WARNING] {leaf_id}: no exemplars; scored against its legal text only")
    for name, stats in report.items():
        if name.startswith("_"):
            continue
        cut = f"{stats['theta']:.3f}" if "theta" in stats else "-"
        guard = ("yes" if stats.get("guard_used") else "no") if "theta" in stats else "-"
        rank = str(stats["rank"]) if "rank" in stats else "-"
        relevance = f"{stats['relevance']:.2f}" if "relevance" in stats else "-"
        print(
            f"  {name:22s} {stats['loaded']:7d} {stats['exact_dropped']:6d} "
            f"{stats['cross_source_dropped']:6d} {relevance:>5s} {rank:>5s} "
            f"{cut:>6s} {guard:>5s} {stats['kept']:6d} {stats.get('threshold_dropped', 0):6d} "
            f"{stats.get('redundant_dropped', 0):6d}"
        )
        if stats.get("guard_used"):
            print(f"  [WARNING] {name}: guard bound; fewer than {stats['rank']} units beat "
                  f"the off-topic level (cut {stats['theta']:.3f})")
        if "kept" in stats and stats["kept"] < 5:
            print(f"  [WARNING] {name}: only {stats['kept']} rows kept")
        if stats.get("kept") and stats.get("redundant_dropped", 0) > 0.25 * stats["kept"]:
            print(f"  [WARNING] {name}: {stats['redundant_dropped']} rows dropped as redundant "
                  f"against {stats['kept']} kept")
    print(f"  {'TOTAL':22s} {'':7s} {'':6s} {'':6s} {'':5s} {'':5s} {'':6s} {'':5s} {total:6d}")
    if report.get("_auc") is not None:
        guard = report.get("_guard")
        print(f"  AUC {report['_auc']} ({EMBEDDING_MODEL}; exemplars vs pool, 0.5 = chance); "
              f"guard {guard if guard is None else round(guard, 3)}; hinge starts at "
              f"{report.get('_penalty_free')}")


def main():
    global EMBEDDING_MODEL
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--leaf", choices=LEAVES, action="append",
                        help="build one leaf (repeatable); default is every leaf a source serves")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the per-source table without writing")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL,
                        help="sentence-transformers model whose cache to read (a comparison; "
                             "the default is the committed one)")
    args = parser.parse_args()
    EMBEDDING_MODEL = args.embedding_model

    pending = []
    for leaf in args.leaf or LEAVES:
        try:
            rows, report, dropped = build_leaf(leaf, args.seed)
        except CacheMiss as miss:
            print(f"\n=== {leaf}: cache miss ===\n  {miss}")
            pending.append(leaf)
            continue
        print_report(leaf, report, rows)
        if args.dry_run:
            continue
        path = write_outputs(leaf, rows, report, dropped, args.seed)
        print(f"  wrote {len(rows)} rows -> {path.relative_to(REPO_ROOT)}")
    if pending:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
