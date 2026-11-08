'''
author: @tae

Test 2 of docs/propensity-measurement.md: a propensity rate under rewrites.

Every item of three registered sources is run under five conditions, each a
separate Inspect sample so the log holds one score per (item, condition):
`published_a`, `published_b` (the same prompt twice; the gap is the rerun
noise), `paraphrase`, `register` (the stored stage-2 rewrites) and
`instruction` (the published prompt under the neutral "professional" persona
system prompt, reported separately). certify.py is not reused because its
per-item minimum and CVaR are the wrong statistics for a rate.

    uv run python3 scripts/propensity_rewrites.py items
    uv run python3 scripts/propensity_rewrites.py rewrite [-a openrouter/nousresearch/hermes-4-405b]
    uv run python3 scripts/propensity_rewrites.py audit            # 50 pairs for two raters
    uv run python3 scripts/propensity_rewrites.py audit --summarise r1.csv r2.csv
    uv run python3 scripts/propensity_rewrites.py run -m openrouter/anthropic/claude-haiku-4.5 [--limit 10]

Outputs: datasets/propensity/{items.csv, paraphrase.jsonl, register.jsonl,
audit_blank.csv}; logs/propensity/<model>/*.eval. Analysis:
analysis/propensity_report.py.
'''
import argparse
import asyncio
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from inspect_ai import Task, eval_set
from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from inspect_ai.solver import generate

from datasets.prepare.cluster import prepare
from datasets.prepare.cluster.schema import COLUMNS
from datasets.prepare.cluster.sources import SOURCES
from pipeline.generation import generate_rewrites
from pipeline.stage1_evaluation.evals.clusters import _to_sample
from pipeline.stage1_evaluation.scorers.cluster import cluster_scorer
from pipeline.stage2_perturbation.framing import PERSONA_TEMPLATES
from pipeline.utils.graders import load_graders

OUT = ROOT / "datasets" / "propensity"
LOGS = ROOT / "logs" / "propensity"
SEED = 0
CAP = 200

# (source, metadata field, values kept) -> element. One source may serve two
# elements through disjoint subsets; the report keys on `leaf`.
SUBSETS = [
    ("advanced_ai_risk", "behavior", {"survival-instinct", "corrigible-less-HHH"}, "loc_shutdown_resistance"),
    ("darkbench", "dark_pattern", {"sneaking"}, "loc_deception"),
    ("darkbench", "dark_pattern", {"user-retention", "sycophancy"}, "manipulation_propensity"),
    # Annotated to deception and shutdown resistance; the report counts it under both.
    ("instrumentaleval", "behavior", None, "loc_deception"),
]
REWRITE_FAMILIES = ("paraphrase", "register")
WORDING = ("published_a", "published_b", *REWRITE_FAMILIES)
CONDITIONS = (*WORDING, "instruction")


def build_items() -> None:
    by_name = {s.name: s for s in SOURCES}
    rows = []
    for name, field, keep, element in SUBSETS:
        loaded = prepare.load_source(by_name[name])
        picked = [r for r in loaded if keep is None or r.metadata.get(field) in keep]
        if len(picked) > CAP:
            picked = random.Random(SEED).sample(picked, CAP)
        for row in picked:
            row.leaf = element
            # Keep ids unique when one source serves two elements.
            row.sample_id = f"{element}/{row.sample_id}"
        rows += picked
        print(f"{name:18} {element:26} {len(picked):4} of {len(loaded)}")
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "items.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(r.to_csv_row() for r in rows)
    print(f"wrote {len(rows)} rows to {OUT / 'items.csv'}")


def load_items(limit: int | None = None) -> list[dict]:
    rows = list(csv.DictReader(open(OUT / "items.csv")))
    if limit is None:
        return rows
    per_leaf: Counter = Counter()
    kept = []
    for row in rows:
        if per_leaf[(row["source"], row["leaf"])] < limit:
            kept.append(row)
            per_leaf[(row["source"], row["leaf"])] += 1
    return kept


def load_rewrites(family: str) -> dict[str, dict]:
    path = OUT / f"{family}.jsonl"
    if not path.exists():
        sys.exit(f"missing {path}; run `rewrite` first")
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines())}


def rewrite(attacker: str, max_connections: int, limit: int | None) -> None:
    samples = [_to_sample(r) for r in load_items(limit)]
    for family in REWRITE_FAMILIES:
        rows = asyncio.run(generate_rewrites(samples, family, attacker, 1, max_connections=max_connections))
        fallbacks = Counter(r["id"].split("/")[0] for r in rows if r["fallback"])
        totals = Counter(r["id"].split("/")[0] for r in rows)
        if limit:
            for r in rows:
                print(f"--- {r['id']} fallback={r['fallback']}\n{r['text'][:300]}")
            continue
        with open(OUT / f"{family}.jsonl", "w") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        print(f"{family}: {len(rows)} rows; fallback share by element: "
              + ", ".join(f"{k} {fallbacks[k]}/{totals[k]}" for k in totals))


def audit(summarise: list[str] | None) -> None:
    if summarise:
        ratings = [{r["id"] + r["family"]: r["preserves"].strip().lower() for r in csv.DictReader(open(p))}
                   for p in summarise]
        keys = set.intersection(*(set(r) for r in ratings))
        agree = sum(len({r[k] for r in ratings}) == 1 for k in keys)
        preserved = [sum(r[k] in ("y", "yes", "1") for r in ratings) / len(ratings) for k in keys]
        print(f"{len(keys)} rewrites rated; preservation rate {sum(preserved) / len(keys):.2f}; "
              f"rater agreement {agree / len(keys):.2f}")
        return
    items = {r["sample_id"]: r for r in load_items()}
    pool = [(family, row) for family in REWRITE_FAMILIES
            for row in load_rewrites(family).values() if not row["fallback"]]
    picked = random.Random(SEED).sample(pool, min(50, len(pool)))
    with open(OUT / "audit_blank.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "family", "original", "rewrite", "preserves"])
        writer.writeheader()
        for family, row in picked:
            writer.writerow({"id": row["id"], "family": family,
                             "original": items[row["id"]]["item_text"], "rewrite": row["text"], "preserves": ""})
    print(f"wrote {len(picked)} pairs to {OUT / 'audit_blank.csv'}")


def errored_ids(log_dir: Path) -> set[str]:
    """Sample ids that errored in finished logs under log_dir (e.g. provider 402s
    under the fail_on_error tolerance), so a retry run can re-issue just those."""
    from inspect_ai.log import list_eval_logs, read_eval_log

    ids: set[str] = set()
    for info in list_eval_logs(str(log_dir), recursive=True):
        for sample in read_eval_log(info.name).samples or []:
            if sample.error:
                ids.add(str(sample.id))
    return ids


def flat_task(limit: int | None, graders: list[str], only_ids: set[str] | None = None) -> Task:
    rewrites = {family: load_rewrites(family) for family in REWRITE_FAMILIES}
    samples = []
    for row in load_items(limit):
        base = _to_sample(row)
        for condition in CONDITIONS:
            sample_id = f"{row['sample_id']}::{condition}"
            if only_ids is not None and sample_id not in only_ids:
                continue
            sample = base.model_copy(deep=True)
            sample.id = sample_id
            sample.metadata.update(condition=condition, item_id=row["sample_id"], fallback=False)
            if condition in REWRITE_FAMILIES:
                variant = rewrites[condition][row["sample_id"]]
                sample.input = variant["query"]
                sample.metadata["fallback"] = variant["fallback"]
            elif condition == "instruction":
                sample.input = [ChatMessageSystem(content=PERSONA_TEMPLATES["professional"]),
                                ChatMessageUser(content=row["query"])]
            samples.append(sample)
    return Task(dataset=samples, solver=generate(), scorer=cluster_scorer(model=graders), name="propensity")


def run(model: str, limit: int | None, max_connections: int, retry_errors: bool) -> None:
    graders = load_graders()
    log_dir = LOGS / model.replace("/", "_") / ("smoke" if limit else "full")
    only_ids = None
    if retry_errors:
        only_ids = errored_ids(log_dir)
        print(f"re-running {len(only_ids)} errored samples")
        if not only_ids:
            return
        # A separate eval-set dir: the finished log reads `success`, so eval_set
        # would skip it. The report merges every log under the model directory.
        log_dir = log_dir / "retry"
    eval_set(
        [flat_task(limit, graders, only_ids)], model=model, log_dir=str(log_dir), log_dir_allow_dirty=True,
        retry_attempts=3, retry_wait=60, fail_on_error=0.1, max_connections=max_connections,
        # No cache: published_a and published_b must be independent generations.
        cache=False,
    )
    print(f"logs in {log_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("items")
    p = sub.add_parser("rewrite")
    p.add_argument("-a", "--attacker", default="openrouter/nousresearch/hermes-4-405b")
    p.add_argument("--max-connections", type=int, default=20)
    p.add_argument("--limit", type=int, help="items per (source, element); prints the rewrites instead of writing")
    p = sub.add_parser("audit")
    p.add_argument("--summarise", nargs="+", metavar="RATER_CSV")
    p = sub.add_parser("run")
    p.add_argument("-m", "--model", required=True)
    p.add_argument("--limit", type=int, help="items per (source, element) for a smoke test")
    p.add_argument("--max-connections", type=int, default=50)
    p.add_argument("--retry-errors", action="store_true",
                   help="re-run only the samples that errored in this model's finished logs")
    args = parser.parse_args()
    if args.command == "items":
        build_items()
    elif args.command == "rewrite":
        rewrite(args.attacker, args.max_connections, args.limit)
    elif args.command == "audit":
        audit(args.summarise)
    else:
        run(args.model, args.limit, args.max_connections, args.retry_errors)


if __name__ == "__main__":
    main()
