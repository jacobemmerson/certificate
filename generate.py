'''
Generates the fixed perturbation/simulation artifacts that certify.py replays.

This is the attacker-model half of stages 2 and 3, run ONCE per artifact
refresh instead of once per evaluated model: every rewrite-family variant
(paraphrase, register, past_tense, multilingual), every deterministic framing/persona wrapper,
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
    uv run python generate.py                          # all families, all clusters
    uv run python generate.py --only cyber --perturb paraphrase framing --perturb-k 1
    uv run python generate.py --only cyber --simulate --sim-k 2
    uv run python generate.py --missing-only           # fill gaps (e.g. failed reframings)
    uv run python generate.py --force                  # regenerate everything from scratch

The attacker is any inspect API provider, or a vLLM server already running
(vLLM is not in the project env; scripts/generate_hermes_slurm.sh serves it via
uvx). -M forwards model args to get_model():

    uv run python generate.py --simulate \
        --attacker vllm/NousResearch/Hermes-4-405B-FP8 \
        --model-base-url http://$VLLM_NODE:8000/v1
'''

import asyncio
import subprocess
from argparse import ArgumentParser
from datetime import datetime, timezone

import yaml

# Load .env (OPENROUTER_API_KEY, etc.) the way inspect's eval() does for
# certify.py — generate.py calls get_model() directly, outside an eval, so
# inspect's own .env loading never runs.
from dotenv import load_dotenv

load_dotenv()

from pipeline.artifacts import (
    DETERMINISTIC_FAMILIES,
    PROMPT_VERSIONS,
    REPEAT_FAMILIES,
    artifact_path,
    family_ids,
    family_meta,
    load_family,
    task_name,
    write_family,
)
from inspect_ai.model import GenerateConfig, get_model

from pipeline.generation import generate_framing, generate_persona, generate_rewrites, generate_scenarios
from pipeline.registry import PREGENERATED_FAMILIES, init_benchmarks, require_families_column
from pipeline.stage3_simulation.prompts import SCENARIO_FAMILY
from pipeline.utils.graders import load_graders


def parse():
    args = ArgumentParser(description="Generate the frozen perturbation/simulation artifacts certify.py replays.")
    args.add_argument(
        "--attacker", "-a", required=False, default="openrouter/nousresearch/hermes-4-70b",
        help="Rewrite/reframing model for the generative families (default: openrouter/nousresearch/hermes-4-70b). "
             "Any inspect API provider, or vllm/<repo> pointed at a running vLLM server via --model-base-url."
    )
    args.add_argument(
        "-M", dest="model_args", required=False, action="append", default=[], metavar="KEY=VALUE",
        help="Model argument forwarded to inspect's get_model() (repeatable), e.g. "
             "-M tensor_parallel_size=8 -M device=cuda. Values are YAML-parsed."
    )
    args.add_argument(
        "--model-base-url", required=False, default=None,
        help="Base URL of an already-running inference server (e.g. a vLLM server "
             "launched in a separate slurm job)."
    )
    args.add_argument(
        "--max-connections", required=False, type=int, default=20,
        help="Concurrent attacker generations (default: 20). Tune down for a "
             "self-hosted server, up for a large API quota."
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
        help="Variants per item for the repeat rewrite families (paraphrase, register, past_tense); default=1."
    )
    args.add_argument(
        "--simulate", required=False, action="store_true",
        help="Also generate stage-3 scenario trees (scenario.jsonl, prompt v4)."
    )
    args.add_argument(
        "--sim-k", required=False, type=int, default=2,
        help="Scenario trees per item under --simulate; default=2. Variant 2 is prompted to use a "
             "different deployment (role, sector, asker) from variant 1."
    )
    args.add_argument(
        "--reasoning", required=False, action="store_true",
        help="Request reasoning mode from the attacker for scenario reframings "
             "(thinking=True via vLLM's chat_template_kwargs, e.g. Hermes-4); the "
             "<think> block is stripped before parsing. Models/servers without the "
             "flag ignore it, but plain API providers (e.g. OpenRouter) may reject "
             "the extra body — leave it off for those."
    )
    args.add_argument(
        "--only", "-o", required=False, nargs="+", metavar="RISK",
        help="Generate only for these systemic-risk clusters (e.g. --only cyber manipulation)."
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


def parse_model_args(pairs: list[str]) -> dict:
    """-M KEY=VALUE pairs -> kwargs for get_model(), YAML-parsing each value
    (so tensor_parallel_size=8 arrives as an int, trust_remote_code=true as a
    bool) — the same convention as inspect eval's -M flag."""
    model_args = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"-M expects KEY=VALUE, got: {pair!r}")
        model_args[key] = yaml.safe_load(value)
    return model_args


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return None


def existing_rows(name: str, family: str) -> list[dict]:
    """Rows to keep on --missing-only. Fallback rows are left out so the run
    retries exactly those variants instead of treating them as done. Rows from
    another prompt version are all dropped: keeping them would restamp stale
    rows with the current version, so after a bump this equals --force."""
    if not artifact_path(name, family).exists():
        return []
    stored = (family_meta(name, family) or {}).get("prompt_version")
    if stored != PROMPT_VERSIONS[family]:
        print(f"[WARNING] {name}/{family}: stored prompt version {stored}, current "
              f"{PROMPT_VERSIONS[family]}; regenerating every row.")
        return []
    return [
        row for rows in load_family(name, family).values() for row in rows
        if not row.get("fallback")
    ]


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
    require_families_column(BENCHMARKS)

    perturb_families = [] if args.no_perturb else args.perturb

    # Instantiate once so hf/vllm attackers load weights (or spawn the vLLM
    # server) a single time, not per task/family.
    # inspect's own limiter defaults to an adaptive 20; match the semaphore.
    attacker = get_model(
        args.attacker, base_url=args.model_base_url,
        config=GenerateConfig(max_connections=args.max_connections),
        **parse_model_args(args.model_args),
    )
    summary = []  # (task, family, outcome)

    for key, entry in BENCHMARKS.items():
        print(f"Perturbing {key}...")
        for task in entry["tasks"]:
            name = task_name(task)
            samples = list(task.dataset)
            if args.limit:
                samples = samples[: args.limit]

            # (family, k) pairs to produce for this task
            wanted = [
                (f, args.perturb_k if f in REPEAT_FAMILIES else 1)
                for f in perturb_families if family_ids(task, f)
            ]
            if args.simulate:
                wanted.append((SCENARIO_FAMILY, args.sim_k))

            for family, k in wanted:
                exists = artifact_path(name, family).exists()
                if exists and not (args.force or args.missing_only):
                    summary.append((name, family, "skipped (exists — use --missing-only or --force)"))
                    continue

                kept = [] if args.force else existing_rows(name, family)
                existing = {(str(row["id"]), row.get("variant", 0)) for row in kept}
                incomplete: list[str] = []
                reasons: dict[str, str] = {}

                if family in DETERMINISTIC_FAMILIES:
                    # template-built — cheap to rebuild wholesale every time
                    rows = generate_framing(samples) if family == "framing" else generate_persona(samples)
                    kept = []
                elif family == SCENARIO_FAMILY:
                    rows, incomplete, reasons = asyncio.run(generate_scenarios(
                        samples, attacker, k, existing=existing,
                        previous_systems={
                            r["id"]: r["system"] for r in kept
                            if r.get("variant") == 1 and not r.get("fallback")
                        },
                        max_connections=args.max_connections,
                        reasoning=args.reasoning,
                    ))
                else:
                    rows = asyncio.run(generate_rewrites(
                        samples, family, attacker, k, existing=existing,
                        max_connections=args.max_connections,
                    ))

                # A batch that is 100% fallbacks (rewrites) or 100% failures
                # (scenarios) means the attacker never produced usable output
                # — e.g. a misconfigured/unreachable server erroring on every
                # call. Refuse to persist: fallback rows masquerade as a
                # complete artifact that --missing-only would then skip.
                attempted = len(rows) or len(incomplete)
                usable = sum(1 for r in rows if not r.get("fallback"))
                if family not in DETERMINISTIC_FAMILIES and attempted and not usable:
                    summary.append((name, family,
                                    f"FAILED — 0/{attempted} usable attacker outputs; nothing written"))
                    continue

                if not rows and kept:
                    summary.append((name, family, "skipped (complete)"))
                    continue

                all_rows = kept + rows
                fallbacks = sum(1 for r in all_rows if r.get("fallback"))
                write_family(name, family, all_rows, meta={
                    "task": name,
                    "family": family,
                    "generator_model": None if family in DETERMINISTIC_FAMILIES else args.attacker,
                    "reasoning": bool(args.reasoning and family == SCENARIO_FAMILY),
                    "prompt_version": PROMPT_VERSIONS[family],
                    "k": k,
                    "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "git_commit": git_commit(),
                    "num_samples": len({r["id"] for r in all_rows}),
                    "num_variants": len(all_rows),
                    "incomplete_ids": incomplete,
                    # why each dropped id failed — a systematic attacker/format
                    # mismatch shows here as one reason repeated across ids
                    "incomplete_reasons": reasons,
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
    if any(outcome.startswith("FAILED") for _, _, outcome in summary):
        raise SystemExit(1)
