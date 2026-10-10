'''
Build the risk-cluster datasets.

    uv run python3 -m datasets.prepare.cluster.prepare --risk cyber
    uv run python3 -m datasets.prepare.cluster.prepare --dry-run

Selection, per risk (datasets/BENCHMARKS.md § Sampling):

1. Load every source, drop exact repeats inside a source and across sources.
2. Score every row against the leaves its source serves (annotations.json, or
   the Source's hand `leaves`): max cosine to the leaf's anchors (legal text
   and exemplars), then the leaf's relevance r smooths the score like a
   temperature, s' = s ** (1 / (0.5 + r)). A row's score is its best leaf.
3. Per source, theta = min(THRESHOLD, the score at rank ceil(FLOOR_FRACTION x
   units)), so a source always contributes its top slice.
4. One pass over the whole risk in score order: a row is kept when its score,
   less a penalty for resembling a row already kept (from any source), still
   clears its source's theta.

Writes datasets/public/<risk>.csv plus <risk>.meta.json (seed, constants,
embedding model, per-source thresholds and counts, source revisions) and
<risk>.dropped.jsonl (every drop, tagged exact, exact_cross_source,
threshold or redundant, so the constants are reviewable rather than trusted).

Reads one gitignored cache, datasets/cache/embeddings/. On a miss it writes
what is missing, prints the command that fills it and exits 2.
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
from .annotations import load_annotations, manifest_name
from .leaves import Leaf, anchor_texts, load_leaves
from .schema import (
    COLUMNS, ITEM, Row, SchemaError, Source, normalised, validate,
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

# Selection constants. A transformed score at or above THRESHOLD is kept;
# FLOOR_FRACTION of a source's units are kept even when none reach it. The
# penalty is a hinge: resemblance to a kept row below PENALTY_FREE costs
# nothing, above it costs PENALTY per unit of cosine, so a near-copy drops out
# only when it was within that margin of its threshold anyway.
THRESHOLD = 0.9
FLOOR_FRACTION = 0.01
PENALTY = 1.0
PENALTY_FREE = 0.8
# Relevance of a hand-assigned leaf (no annotation): the identity temperature.
DEFAULT_RELEVANCE = 0.5


class CacheMiss(Exception):
    '''The embedding cache lacks entries. The message says what to run.'''


def _cache_miss(path: Path, records: list[dict], *commands: str) -> CacheMiss:
    '''Write what the cache lacks to `path` and build the error naming the fix.'''
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return CacheMiss(f"{len(records)} missing -> {path}\n  run: " + "\n   or: ".join(commands))


def embed_key(text: str) -> str | None:
    '''
    Cache key of a text's embedding: blake2b-16 of its normalised form, which
    is also the text embedded (all-MiniLM-L6-v2 is uncased, so folding case and
    punctuation loses nothing). Texts that normalise identically share one
    vector by construction. None for a text with no words.
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


def source_leaves(source: Source, annotations: dict[str, dict[str, float]]) -> dict[str, float]:
    '''{leaf id: relevance} the source is scored against: its hand `leaves` when set, else its annotation.'''
    if source.leaves:
        return {leaf: DEFAULT_RELEVANCE for leaf in source.leaves}
    annotated = annotations.get(manifest_name(source.path))
    if annotated:
        return dict(annotated)
    raise SchemaError(f"{source.name}: no annotation in annotations.json and no `leaves`; "
                      "it cannot be scored")


def require_embeddings(
    risk: str, pools: list[tuple[Source, list[Row]]], embeddings: dict, leaf_embeddings: dict,
    leaves_used: dict[str, dict[str, float]] | None = None,
) -> None:
    '''Raise CacheMiss, after writing the embed input, if any query or leaf anchor lacks a vector.'''
    missing = {}
    for source, rows in pools:
        for row in rows:
            key = embed_key(row.query)
            if key and key not in embeddings:
                missing[key] = normalised(row.query)
    if missing:
        raise _cache_miss(
            CACHE_DIR / f"{risk}.embed_input.jsonl",
            [{"key": key, "text": text} for key, text in sorted(missing.items())],
            EMBED_COMMAND.format(risk=risk),
        )
    leaves = load_leaves()
    if leaves_used is None:
        annotations = load_annotations()
        leaves_used = {source.name: source_leaves(source, annotations) for source, _ in pools}
    anchors = {
        embed_key(text): normalised(text)
        for leaf_ids in leaves_used.values() for leaf_id in leaf_ids
        for text in anchor_texts(leaves[leaf_id])
    }
    missing_anchors = {key: text for key, text in anchors.items() if key not in leaf_embeddings}
    if missing_anchors:
        # Every risk shares this input, so a run over several risks accumulates
        # their misses rather than each overwriting the last.
        path = CACHE_DIR / "leaves.embed_input.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                if record["key"] not in leaf_embeddings:
                    missing_anchors.setdefault(record["key"], record["text"])
        raise _cache_miss(
            path,
            [{"key": key, "text": text} for key, text in sorted(missing_anchors.items())],
            EMBED_COMMAND.format(risk="leaves"),
        )


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
        key = embed_key(row.query)
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
                        f"run: {EMBED_COMMAND.format(risk='leaves')}")
    return names, np.array([leaf_embeddings[key] for key in keys], dtype=np.float64)


def relevance_scores(vectors: np.ndarray, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''Each row's max cosine over the anchors, and the index of the anchor that
    attains it (the first, on a tie; -1 for a zero vector, which matched
    nothing). Rounded so BLAS summation order cannot reorder rows across machines.'''
    similarity = np.round(vectors @ matrix.T, 9)
    hit = similarity.argmax(axis=1)
    hit[~vectors.any(axis=1)] = -1
    return similarity.max(axis=1), hit


def temperature(relevance: float) -> float:
    if not 0 <= relevance <= 1:
        raise ValueError(f"relevance must be in [0, 1], got {relevance}")
    return 0.5 + relevance


def smooth(scores: np.ndarray, relevance: float) -> np.ndarray:
    '''s ** (1 / T): rank-preserving; r > 0.5 lifts scores toward 1 (more cross
    THRESHOLD), r < 0.5 pushes them down, r = 0.5 is the identity. Negative
    cosines clip to 0 first.'''
    return np.round(np.clip(scores, 0, None) ** (1 / temperature(relevance)), 9)


def _group_ids(source: Source, rows: list[Row]) -> list[str]:
    '''Each row's group; a row without the group_key field is its own group.'''
    return [str(row.metadata.get(source.group_key, i)) for i, row in enumerate(rows)]


def score_rows(
    source: Source, rows: list[Row], leaf_relevance: dict[str, float], leaves: dict[str, Leaf],
    embeddings: dict[str, np.ndarray], leaf_embeddings: dict[str, np.ndarray],
) -> np.ndarray:
    '''
    Each row's selection score: over the source's leaves, the best smoothed
    cosine to that leaf's anchors; for a group_key source, the best arm of the
    row's group on every arm, so a group is kept or dropped whole. Written to
    metadata as relevance (the score), relevance_raw, relevance_leaf,
    relevance_anchor and, for groups, relevance_own.
    '''
    vectors = _vectors(rows, embeddings)
    best = np.full(len(rows), -np.inf)
    raw = np.zeros(len(rows))
    best_leaf: list[str | None] = [None] * len(rows)
    best_anchor: list[str | None] = [None] * len(rows)
    for leaf_id in sorted(leaf_relevance):
        names, matrix = leaf_anchors(leaves[leaf_id], leaf_embeddings)
        cosine, hit = relevance_scores(vectors, matrix)
        smoothed = smooth(cosine, leaf_relevance[leaf_id])
        better = smoothed > best
        best[better] = smoothed[better]
        raw[better] = cosine[better]
        for i in np.flatnonzero(better):
            best_leaf[i] = leaf_id
            best_anchor[i] = names[hit[i]] if hit[i] >= 0 else None
    unit = best
    if source.group_key:
        group_best: dict[str, float] = {}
        groups = _group_ids(source, rows)
        for group, score in zip(groups, best):
            group_best[group] = max(group_best.get(group, -np.inf), float(score))
        unit = np.array([group_best[group] for group in groups])
    for i, row in enumerate(rows):
        # A fresh dict: readers may share one metadata dict across rows.
        row.metadata = {**row.metadata, "relevance": round(float(unit[i]), 6),
                        "relevance_raw": round(float(raw[i]), 6),
                        "relevance_leaf": best_leaf[i], "relevance_anchor": best_anchor[i]}
        if source.group_key:
            row.metadata["relevance_own"] = round(float(best[i]), 6)
    return unit


def source_threshold(unit_scores: Sequence[float]) -> tuple[float, bool]:
    '''theta for one source: THRESHOLD, or the score of its ceil(FLOOR_FRACTION x
    units)-th best unit when fewer reach THRESHOLD. (theta, floor_used).'''
    ranked = sorted(unit_scores, reverse=True)
    if not ranked:
        return THRESHOLD, False
    floor = ranked[min(math.ceil(FLOOR_FRACTION * len(ranked)), len(ranked)) - 1]
    return (THRESHOLD, False) if floor >= THRESHOLD else (float(floor), True)


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
) -> tuple[list[Row], list[dict]]:
    '''
    The one pass over a risk. Units are walked best score first (ties by
    `order`); each is kept when score - PENALTY x max(0, maxsim - PENALTY_FREE)
    reaches its source's theta, where maxsim is the highest cosine between any
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
        effective = unit.score - PENALTY * max(0.0, maxsim - PENALTY_FREE)
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


def _quantiles(scores: Sequence[float]) -> dict[str, float] | None:
    if not len(scores):
        return None
    points = np.percentile(np.asarray(scores, dtype=np.float64), [0, 10, 50, 90, 100])
    return dict(zip(("min", "p10", "p50", "p90", "max"), (round(float(p), 6) for p in points)))


# ----- driver -----

def build_risk(risk: str, seed: int) -> tuple[list[Row], dict, list[dict]]:
    sources = for_risk(risk)
    if not sources:
        raise SystemExit(f"no sources registered for risk {risk!r}")

    all_dropped: list[dict] = []
    report = {}
    annotations = load_annotations()
    leaves = load_leaves()

    pools = []
    for source in sources:
        rows = load_source(source)
        loaded = len(rows)
        rows, exact_dropped = exact_dedup(rows, source.distinct_on)
        all_dropped.extend(exact_dropped)
        leaf_relevance = source_leaves(source, annotations)
        report[source.name] = {
            "loaded": loaded,
            "exact_dropped": len(exact_dropped),
            "cross_source_dropped": 0,
            "leaves": leaf_relevance,
            "annotated": not source.leaves and manifest_name(source.path) in annotations,
            "question_type": source.question_type,
            "path": source.path,
        }
        pools.append((source, rows))

    sizes = {source.name: len(rows) for source, rows in pools}
    pools, cross_dropped = cross_source_dedup(pools)
    all_dropped.extend(cross_dropped)
    for source, rows in pools:
        report[source.name]["cross_source_dropped"] = sizes[source.name] - len(rows)

    embeddings = load_embeddings(risk)
    leaf_embeddings = load_embeddings("leaves")
    require_embeddings(risk, pools, embeddings, leaf_embeddings,
                       {name: stats["leaves"] for name, stats in report.items()})

    units: list[Unit] = []
    thetas: dict[str, float] = {}
    for source, rows in pools:
        scores = score_rows(source, rows, report[source.name]["leaves"], leaves,
                            embeddings, leaf_embeddings)
        source_units = units_for(source, rows, scores, seed)
        theta, floor_used = source_threshold([unit.score for unit in source_units])
        thetas[source.name] = theta
        report[source.name].update({
            "units": len(source_units), "theta": round(theta, 6), "floor_used": floor_used,
            "score_quantiles": _quantiles([unit.score for unit in source_units]),
        })
        units.extend(source_units)

    kept, dropped = cluster_select(units, thetas, embeddings)
    all_dropped.extend(dropped)
    counts = defaultdict(lambda: defaultdict(int))
    for record in dropped:
        counts[record["dropped"].split(":", 1)[0]][record["tier"]] += 1
    for row in kept:
        counts[row.source]["kept"] += 1
    for name in report:
        report[name].update({
            "kept": counts[name]["kept"],
            "threshold_dropped": counts[name]["threshold"],
            "redundant_dropped": counts[name]["redundant"],
        })
    # Source registration order, then the build's score order inside a source.
    order = {source.name: i for i, source in enumerate(sources)}
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


def write_outputs(risk: str, rows: list[Row], report: dict, dropped: list[dict], seed: int):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame([row.to_csv_row() for row in rows], columns=COLUMNS)
    csv_path = OUT_DIR / f"{risk}.csv"
    frame.to_csv(csv_path, index=False)

    leaves = load_leaves()
    used = sorted({leaf_id for stats in report.values() for leaf_id in stats.get("leaves", {})})
    meta = {
        "risk": risk,
        "rows": len(rows),
        "seed": seed,
        "embedding": {
            "model": EMBEDDING_MODEL,
            "cache": f"datasets/cache/embeddings/{risk}.npz",
        },
        "constants": {
            "threshold": THRESHOLD, "floor_fraction": FLOOR_FRACTION,
            "penalty": PENALTY, "penalty_free": PENALTY_FREE,
        },
        "leaves": {
            leaf_id: {
                "cop_ref": leaves[leaf_id].cop_ref,
                "label": leaves[leaf_id].label,
                "anchors": len(anchor_texts(leaves[leaf_id])),
                "anchor_keys": sorted(embed_key(text) for text in anchor_texts(leaves[leaf_id])),
            } for leaf_id in used
        },
        "sources": report,
        "revisions": source_revisions(),
    }
    (OUT_DIR / f"{risk}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    with open(OUT_DIR / f"{risk}.dropped.jsonl", "w", encoding="utf-8") as f:
        for record in dropped:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return csv_path


def print_report(risk: str, report: dict, rows: list[Row]):
    print(f"\n=== {risk} ===")
    header = (f"  {'source':22s} {'loaded':>7s} {'exact':>6s} {'cross':>6s} {'leaves':>6s} "
              f"{'theta':>6s} {'floor':>5s} {'kept':>6s} {'thr':>6s} {'red':>6s}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    total = len(rows) or 1
    leaves = load_leaves()
    warned_leaves = set()
    for name, stats in report.items():
        theta = f"{stats['theta']:.3f}" if "theta" in stats else "-"
        floor = ("yes" if stats.get("floor_used") else "no") if "theta" in stats else "-"
        print(
            f"  {name:22s} {stats['loaded']:7d} {stats['exact_dropped']:6d} "
            f"{stats['cross_source_dropped']:6d} {len(stats.get('leaves', {})):6d} "
            f"{theta:>6s} {floor:>5s} {stats['kept']:6d} {stats.get('threshold_dropped', 0):6d} "
            f"{stats.get('redundant_dropped', 0):6d}"
        )
        if stats.get("floor_used"):
            print(f"  [WARNING] {name}: floor used; fewer than {100 * FLOOR_FRACTION:g}% of "
                  f"its units reach {THRESHOLD} (theta {stats['theta']:.3f})")
        for leaf_id in stats.get("leaves", {}):
            if leaf_id in leaves and not leaves[leaf_id].exemplars and leaf_id not in warned_leaves:
                warned_leaves.add(leaf_id)
                print(f"  [WARNING] {leaf_id}: no exemplars; scored against its legal text only")
        if "kept" in stats and stats["kept"] < 5:
            print(f"  [WARNING] {name}: only {stats['kept']} rows kept")
        if stats.get("kept") and stats.get("redundant_dropped", 0) > 0.25 * stats["kept"]:
            print(f"  [WARNING] {name}: {stats['redundant_dropped']} rows dropped as redundant "
                  f"against {stats['kept']} kept")
    print(f"  {'TOTAL':22s} {'':7s} {'':6s} {'':6s} {'':6s} {'':6s} {'':5s} {total:6d}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--risk", choices=RISKS, action="append",
                        help="build one risk (repeatable); default is all registered")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the per-source table without writing")
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
