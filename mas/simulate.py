"""Run stored scenarios through Concordia simulations.

Usage:
    uv run python3 -m mas.simulate \
        --target-model openai/gpt-4o-mini --sim-model openai/gpt-4o
    uv run python3 -m mas.simulate --target-model openai/gpt-4o-mini \
        --sim-model openai/gpt-4o --limit 1 --max-steps 6
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from mas import OpenRouterLanguageModel, load_scenarios
from mas.embedder import get_embedder
from mas.generation import slugify
from mas.runner import run_scenario

MAS_DIR = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run stored scenarios through Concordia simulations."
    )
    parser.add_argument("--target-model", required=True, help="OpenRouter id of the AI under test")
    parser.add_argument("--sim-model", required=True, help="OpenRouter id driving the human personas + game master")
    parser.add_argument("--scenarios", default=str(MAS_DIR / "scenarios"), help="scenario directory")
    parser.add_argument("--risk", help="only run scenarios whose risk slug contains this substring")
    parser.add_argument("--id", dest="scenario_id", help="only run the scenario with this exact id")
    parser.add_argument("--out", help="output dir (default: mas/runs/{target-slug})")
    parser.add_argument("--max-steps", type=int, help="override each scenario's max_steps")
    parser.add_argument("--limit", type=int, help="run at most this many scenarios")
    parser.add_argument("--overwrite", action="store_true", help="re-run even if output json exists")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ["OPENROUTER_API_KEY"]
    target_model = OpenRouterLanguageModel(args.target_model, api_key=api_key)
    sim_model = OpenRouterLanguageModel(args.sim_model, api_key=api_key)

    out_dir = Path(args.out) if args.out else MAS_DIR / "runs" / slugify(args.target_model)

    scenarios = load_scenarios(args.scenarios)
    if args.risk:
        scenarios = [s for s in scenarios if args.risk in slugify(s["risk"])]
    if args.scenario_id:
        scenarios = [s for s in scenarios if s["id"] == args.scenario_id]

    embedder = get_embedder()

    ran = skipped = failed = 0
    for scenario in scenarios:
        if args.limit is not None and ran >= args.limit:
            break
        if "concordia" not in scenario:
            print(f"{scenario['id']}: no concordia block, skipping")
            skipped += 1
            continue
        if (out_dir / f"{scenario['id']}.json").exists() and not args.overwrite:
            print(f"{scenario['id']}: output exists, skipping")
            skipped += 1
            continue
        print(f"{scenario['id']}: running ({scenario['framing']}, {scenario['risk']})")
        try:
            transcript = run_scenario(
                scenario, target_model, sim_model, embedder, out_dir, max_steps=args.max_steps
            )
            print(f"{scenario['id']}: done, {len(transcript['ai_responses'])} AI response(s)")
            ran += 1
        except Exception as e:  # noqa: BLE001 - one bad scenario shouldn't kill the batch
            print(f"{scenario['id']}: FAILED, {type(e).__name__}: {e}")
            failed += 1

    print(f"\nSummary: ran={ran} skipped={skipped} failed={failed} -> {out_dir}")


if __name__ == "__main__":
    main()
