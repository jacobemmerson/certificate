'''
Build the risk-cluster datasets.

    uv run python3 -m datasets.prepare.cluster.prepare --risk cyber
    uv run python3 -m datasets.prepare.cluster.prepare --dry-run

Writes datasets/public/<risk>.csv plus a <risk>.meta.json sibling (provenance:
seed, quotas, per-tier drop counts, source revisions) and <risk>.dropped.jsonl
(the near-duplicate pairs tier 2 removed, so the threshold is reviewable rather
than trusted).

Selection is lexical only — no embeddings. The evidence for that, and for the
token gate on tier 2, is in datasets/CLUSTERING.md.
'''

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import pandas as pd

from . import readers
from .schema import (
    COLUMNS, ITEM, MCQ, Row, Source, jaccard, normalised, tokens, validate,
)
from .sources import RISKS, for_risk

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUT_DIR = REPO_ROOT / "datasets" / "public"

# Tier 2 compares only short texts. On long text, Jaccard measures shared
# boilerplate rather than shared meaning: PHT's rendered prompts peak at 0.598
# between *different* events, while genuine ECHR near-duplicates reach only
# 0.411 — no threshold separates those.
#
# The gate is per *pair*, not per source. A source-level median hides the
# problem whenever text length varies inside one source: WMDP mixes one-line
# conceptual questions with packet-capture items carrying a hex dump, and those
# long ones collide at 0.98 while differing in the only part that matters (the
# field being asked about).
# tau is high because these benchmarks are largely templated: SOSBench is one
# instruction shape over 1,628 regulated hazards, so "developing Dichloroethylene"
# and "developing Tetrachloroethane" score 0.875 while being entirely different
# items. Measured on real drops, false positives crowd 0.70-0.89 and genuine
# duplicates sit at 0.90+. Sources whose payload is short and untemplated (PHT's
# event titles) override this downward.
TOKEN_GATE = 25
JACCARD_TAU = 0.9
BLOCKING_MAX_DOCS = 60


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
            system_prompt=str(
                source.resolve(record, source.system_prompt) or ""
            ) if source.system_prompt else "",
            item_text=item_text,
            prompt_template=prompt_template,
            elicitation_family=source.elicitation_family,
            metadata={key: _plain(record.get(key)) for key in source.metadata},
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


# ----- tier 2: lexical near-dedup -----

def _distinguishable(left: Row, right: Row, distinct_on: Sequence[str]) -> bool:
    '''
    Exact guard against the lexical filter's blind spot: when a benchmark varies
    one term inside a fixed template, the entire distinction is a few characters
    that Jaccard weights at 1/N. Items differing in ground truth, or in a field
    the source declares identifying, are never duplicates however similar the
    surrounding wording.
    '''
    if left.question_type == MCQ and left.target != right.target:
        return False
    return all(
        left.metadata.get(field) == right.metadata.get(field) for field in distinct_on
    )


def near_dedup(
    rows: list[Row],
    tau: float = JACCARD_TAU,
    *,
    dedup_on: str | None = None,
    distinct_on: Sequence[str] = (),
) -> tuple[list[Row], list[dict]]:
    '''
    Token-set Jaccard with inverted-index blocking. Returns survivors and the
    dropped pairs, which get written out so `tau` can be reviewed on real data.

    `dedup_on` compares a metadata field instead of the rendered query — the
    doc's "filter the case pool, never the rendered prompt" rule, made
    executable: PHT's payload is the historical event, not the 100-word
    instruction wrapped around it.
    '''
    def payload(row: Row) -> str:
        return str(row.metadata.get(dedup_on, "")) if dedup_on else row.query

    token_sets = {index: tokens(payload(row)) for index, row in enumerate(rows)}
    short = {index for index, t in token_sets.items() if len(t) < TOKEN_GATE}
    if not short:
        return rows, []

    postings: defaultdict[str, list[int]] = defaultdict(list)
    for index in short:
        for token in token_sets[index]:
            postings[token].append(index)

    candidates = set()
    for indices in postings.values():
        if len(indices) > BLOCKING_MAX_DOCS:
            continue  # stopword-ish; blocking on it would be quadratic and useless
        for i, left in enumerate(indices):
            for right in indices[i + 1:]:
                candidates.add((left, right))

    scored = sorted(
        ((jaccard(token_sets[a], token_sets[b]), a, b) for a, b in candidates
         if _distinguishable(rows[a], rows[b], distinct_on)),
        reverse=True,
    )

    dropped_indices: set[int] = set()
    dropped_pairs = []
    for score, left, right in scored:
        if score < tau:
            break
        if left in dropped_indices or right in dropped_indices:
            continue
        dropped_indices.add(right)
        dropped_pairs.append({
            "similarity": round(score, 4),
            "kept": rows[left].sample_id, "kept_text": payload(rows[left])[:300],
            "dropped": rows[right].sample_id, "dropped_text": payload(rows[right])[:300],
        })

    survivors = [row for index, row in enumerate(rows) if index not in dropped_indices]
    return survivors, dropped_pairs


# ----- tier 3: stratified quota -----

def stratified_sample(
    rows: list[Row], source: Source, seed: int
) -> tuple[list[Row], dict]:
    if source.group_key:
        return _grouped_sample(rows, source, seed)
    return _row_sample(rows, source, seed)


def _grouped_sample(
    rows: list[Row], source: Source, seed: int
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
    picked, report = _row_sample(list(leaders.values()), source, seed)

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
    rows: list[Row], indices: list[int], take: int, source: Source, seed: int
) -> list[int]:
    '''
    Greedy farthest-point: repeatedly take the item least similar to everything
    already taken.

    Near-dedup only removes pairs above tau, and only for texts under
    TOKEN_GATE — it never asks whether the *kept* set spans its stratum. This
    does, so a quota of 90 drawn from 12,662 buys coverage rather than a lottery
    ticket.

    Compares the same payload near_dedup does (`dedup_on` where declared, the
    query otherwise): PHT's items differ by historical event inside a shared
    ~100-word instruction, and spreading on the rendered prompt would spread on
    boilerplate. The first pick comes from `_stable_order` so the whole walk is
    deterministic without being tied to input order.
    '''
    payload = _payload_fn(source)
    token_sets = {index: tokens(payload(rows[index])) for index in indices}

    first = _stable_order(rows, indices, seed)[0]
    picked = [first]
    # Each item's similarity to the closest thing already picked; the next pick
    # is whatever minimises it.
    nearest = {
        index: jaccard(token_sets[index], token_sets[first]) for index in indices
    }

    taken = {first}
    while len(picked) < take:
        candidate = min(
            (index for index in indices if index not in taken),
            key=lambda index: (nearest[index], key_bytes(rows[index], seed)),
        )
        picked.append(candidate)
        taken.add(candidate)
        for index in indices:
            nearest[index] = max(
                nearest[index], jaccard(token_sets[index], token_sets[candidate])
            )

    return picked


def key_bytes(row: Row, seed: int) -> bytes:
    '''Deterministic tie-break, so equally-distant candidates resolve stably.'''
    return hashlib.blake2b(
        f"{seed}:{row.sample_id}".encode(), digest_size=16
    ).digest()


def _payload_fn(source: Source):
    '''The text that identifies an item — near_dedup's rule, reused.'''
    if source.dedup_on:
        return lambda row: str(row.metadata.get(source.dedup_on, ""))
    return lambda row: row.query


def _take(
    rows: list[Row], indices: list[int], take: int, source: Source, seed: int
) -> list[int]:
    '''Fill one stratum's allotment, by whichever selection the source declares.'''
    if take >= len(indices):
        return list(indices)
    if source.select == UNIFORM:
        return _stable_order(rows, indices, seed)[:take]
    if source.select == DIVERSE:
        return _diverse_order(rows, indices, take, source, seed)
    raise ValueError(f"{source.name}: unknown select mode {source.select!r}")


def _row_sample(
    rows: list[Row], source: Source, seed: int
) -> tuple[list[Row], dict]:
    quota = source.quota
    if quota is None or quota >= len(rows):
        if source.select not in (UNIFORM, DIVERSE):
            raise ValueError(f"{source.name}: unknown select mode {source.select!r}")
        return rows, {"strata": 0, "allocated": len(rows)}

    if not source.stratify:
        chosen = _take(rows, list(range(len(rows))), quota, source, seed)
        return [rows[i] for i in sorted(chosen)], {"strata": 1, "allocated": quota}

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
        chosen.extend(_take(rows, buckets[key], take, source, seed))

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

    for source in sources:
        rows = load_source(source)
        loaded = len(rows)

        rows, exact_dropped = exact_dedup(rows, source.distinct_on)

        if source.dedup:
            rows, near_dropped = near_dedup(
                rows,
                source.tau if source.tau is not None else JACCARD_TAU,
                dedup_on=source.dedup_on,
                distinct_on=source.distinct_on,
            )
        else:
            near_dropped = []
        all_dropped.extend(near_dropped)

        rows, allocation = stratified_sample(rows, source, seed)

        report[source.name] = {
            "loaded": loaded,
            "exact_dropped": exact_dropped,
            "near_dropped": len(near_dropped),
            "quota": source.quota,
            "kept": len(rows),
            "strata": allocation["strata"],
            "stratify_on": list(source.stratify),
            "balanced": source.balanced,
            "question_type": source.question_type,
            "path": source.path,
        }
        all_rows.extend(rows)

    return all_rows, report, all_dropped


def source_revisions() -> dict:
    '''Pin what produced this build: submodule SHAs plus the repo HEAD.'''
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

    vendored = [key for key in revisions if key != "repo"]
    if not vendored:
        revisions["_warning"] = "no submodule revisions recorded"
    return revisions


def write_outputs(risk: str, rows: list[Row], report: dict, dropped: list[dict], seed: int):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame([row.to_csv_row() for row in rows], columns=COLUMNS)
    csv_path = OUT_DIR / f"{risk}.csv"
    frame.to_csv(csv_path, index=False)

    meta = {
        "risk": risk,
        "rows": len(rows),
        "seed": seed,
        "jaccard_tau_default": JACCARD_TAU,
        "token_gate": TOKEN_GATE,
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
    header = f"  {'source':22s} {'loaded':>7s} {'exact':>6s} {'near':>6s} {'kept':>6s} {'share':>6s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    total = len(rows) or 1
    for name, stats in report.items():
        print(
            f"  {name:22s} {stats['loaded']:7d} {stats['exact_dropped']:6d} "
            f"{stats['near_dropped']:6d} {stats['kept']:6d} "
            f"{100 * stats['kept'] / total:5.1f}%"
        )
    print(f"  {'TOTAL':22s} {'':7s} {'':6s} {'':6s} {total:6d}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--risk", choices=RISKS, action="append",
                        help="build one risk (repeatable); default is all registered")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the tier table without writing")
    args = parser.parse_args()

    risks = args.risk or [risk for risk in RISKS if for_risk(risk)]

    for risk in risks:
        rows, report, dropped = build_risk(risk, args.seed)
        print_report(risk, report, rows)
        if args.dry_run:
            continue
        path = write_outputs(risk, rows, report, dropped, args.seed)
        print(f"  wrote {len(rows)} rows -> {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
