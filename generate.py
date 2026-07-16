'''
Generates the fixed perturbation/simulation artifacts that certify.py replays.

This is the attacker-model half of stages 2 and 3, run ONCE per artifact
refresh instead of once per evaluated model: every rewrite-family variant
(paraphrase, register, identity_strip), every deterministic framing wrapper,
and every stage-3 scenario reframing is generated here and persisted to
datasets/generated/<task_name>/<family>.jsonl (see pipeline/artifacts.py and
datasets/generated/README.md). certify.py then evaluates every target model
against these exact same frozen variants — cheaper (no attacker calls per
model) and fair (no model sees a luckier rewrite than another).

`reconsideration` is the one family with no artifacts: it challenges the
target's own control completion, so it can only run live inside certify.py.

Unlike certify.py, --perturb and --simulate compose here — each family is an
independent artifact file.

Usage:
    uv run python generate.py                          # all families, all benchmarks
    uv run python generate.py --only harm --perturb paraphrase framing --perturb-k 1
    uv run python generate.py --only harm --simulate --sim-k 1
    uv run python generate.py --missing-only           # fill gaps (e.g. failed reframings)
    uv run python generate.py --force                  # regenerate everything from scratch
'''

import asyncio
import subprocess
from argparse import ArgumentParser
from datetime import datetime, timezone

# Load .env (OPENROUTER_API_KEY, etc.) the way inspect's eval() does for
# certify.py — generate.py calls get_model() directly, outside an eval, so
# inspect's own .env loading never runs.
from dotenv import load_dotenv

load_dotenv()

from pipeline.artifacts import (
    PROMPT_VERSIONS,
    REWRITE_FAMILIES,
    artifact_path,
    framing_applies,
    load_family,
    task_name,
    write_family,
)
from pipeline.generation import generate_framing, generate_rewrites, generate_scenarios
from pipeline.registry import PREGENERATED_FAMILIES, init_benchmarks
from pipeline.stage2_perturbation.adapters import adapter_for
from pipeline.stage3_simulation.prompts import SCENARIO_FAMILY
from pipeline.utils.graders import load_graders


def parse():
    args = ArgumentParser(description="Generate the frozen perturbation/simulation artifacts certify.py replays.")
    args.add_argument(
        "--attacker", "-a", required=False, default="openrouter/deepseek/deepseek-v4-flash",
        help="Rewrite/reframing model for the generative families (default: openrouter/deepseek/deepseek-v4-flash)."
    )
    args.add_argument(
        "--perturb", required=False, nargs="+", default=sorted(PREGENERATED_FAMILIES),
        choices=sorted(PREGENERATED_FAMILIES), metavar="FAMILY",
        help="Stage-2 families to generate (default: all pregenerated families). "
             "reconsideration has no artifacts — it runs live in certify.py."
    )
    args.add_argument(
        "--no-perturb", required=False, action="store_true",
        help="Skip stage-2 families entirely (e.g. to generate only --simulate artifacts)."
    )
    args.add_argument(
        "--perturb-k", required=False, type=int, default=1,
        help="Variants per item for the rewrite families (paraphrase, register, identity_strip); default=1."
    )
    args.add_argument(
        "--simulate", required=False, action="store_true",
        help="Also generate stage-3 scenario reframings (scenario.jsonl)."
    )
    args.add_argument(
        "--sim-k", required=False, type=int, default=1,
        help="Reframed scenarios per item under --simulate; default=1."
    )
    args.add_argument(
        "--only", "-o", required=False, nargs="+", metavar="BENCHMARK",
        help="Generate only for these benchmark keys (e.g. --only harm hr)."
    )
    args.add_argument(
        "--missing-only", required=False, action="store_true",
        help="Fill gaps in existing artifact files (missing samples/variants, e.g. failed "
             "reframings) and merge, instead of skipping files that already exist."
    )
    args.add_argument(
        "--force", required=False, action="store_true",
        help="Regenerate every requested family from scratch, overwriting existing artifacts."
    )
    args.add_argument(
        "--limit", "-l", required=False, type=int, default=None,
        help="Generate for only the first N samples per task. WARNING: produces partial "
             "artifacts (marked partial in the meta sidecar) that fail certify.py's full-run "
             "validation — smoke-testing only."
    )
    return args.parse_args()


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return None


def existing_keys(name: str, family: str) -> set[tuple[str, int]]:
    """(id, variant) pairs already on disk, or empty if no file yet."""
    if not artifact_path(name, family).exists():
        return set()
    return {
        (sample_id, row.get("variant", 0))
        for sample_id, rows in load_family(name, family).items()
        for row in rows
    }


def existing_rows(name: str, family: str) -> list[dict]:
    if not artifact_path(name, family).exists():
        return []
    return [row for rows in load_family(name, family).values() for row in rows]


if __name__ == "__main__":
    args = parse()
    if args.limit:
        print(f"[WARNING] --limit {args.limit}: artifacts will be PARTIAL (marked in meta) — "
              "they will fail certify.py's full-run validation. Smoke-testing only.")

    BENCHMARKS = init_benchmarks(load_graders())  # scorers are never invoked here
    only = set(args.only) if args.only else None
    if only:
        unknown = only - set(BENCHMARKS.keys())
        if unknown:
            print(f"[WARNING] Unknown benchmark keys (ignored): {', '.join(sorted(unknown))}")
        BENCHMARKS = {key: entry for key, entry in BENCHMARKS.items() if key in only}

    perturb_families = [] if args.no_perturb else args.perturb
    summary = []  # (task, family, outcome)

    for key, entry in BENCHMARKS.items():
        print(f"Perturbing {key}...")
        for task in entry["tasks"]:
            name = task_name(task)
            adapter = adapter_for(task)
            samples = list(task.dataset)
            if args.limit:
                samples = samples[: args.limit]

            # (family, k) pairs to produce for this task
            wanted = [(f, args.perturb_k) for f in perturb_families if f in REWRITE_FAMILIES]
            if "framing" in perturb_families and framing_applies(name):
                wanted.append(("framing", 1))
            if args.simulate:
                wanted.append((SCENARIO_FAMILY, args.sim_k))

            for family, k in wanted:
                exists = artifact_path(name, family).exists()
                if exists and not (args.force or args.missing_only):
                    summary.append((name, family, "skipped (exists — use --missing-only or --force)"))
                    continue

                existing = set() if args.force else existing_keys(name, family)
                kept = [] if args.force else existing_rows(name, family)
                incomplete: list[str] = []

                if family == "framing":
                    # deterministic — cheap to rebuild wholesale every time
                    rows = generate_framing(samples, adapter)
                    kept = []
                elif family == SCENARIO_FAMILY:
                    rows, incomplete = asyncio.run(generate_scenarios(
                        samples, adapter, args.attacker, k, existing=existing,
                    ))
                else:
                    rows = asyncio.run(generate_rewrites(
                        samples, adapter, family, args.attacker, k, existing=existing,
                    ))

                if not rows and kept:
                    summary.append((name, family, "skipped (complete)"))
                    continue

                all_rows = kept + rows
                fallbacks = sum(1 for r in all_rows if r.get("fallback"))
                write_family(name, family, all_rows, meta={
                    "task": name,
                    "family": family,
                    "generator_model": None if family == "framing" else args.attacker,
                    "prompt_version": PROMPT_VERSIONS[family],
                    "k": k,
                    "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "git_commit": git_commit(),
                    "num_samples": len({r["id"] for r in all_rows}),
                    "num_variants": len(all_rows),
                    "incomplete_ids": incomplete,
                    "partial": bool(args.limit),
                })
                outcome = f"wrote {len(rows)} new / {len(all_rows)} total rows"
                if fallbacks:
                    outcome += f", {fallbacks} fallback(s)"
                if incomplete:
                    outcome += f", {len(incomplete)} incomplete id(s)"
                summary.append((name, family, outcome))

    print("\n----- generation summary -----")
    if not summary:
        print("Nothing to generate (check --only/--perturb/--simulate).")
    for name, family, outcome in summary:
        print(f"{name:28s} {family:16s} {outcome}")
