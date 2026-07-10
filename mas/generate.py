"""Generate and enrich simulation scenarios from mas/risks.csv.

Usage:
    uv run python3 -m mas.generate --risks all --n-per-framing 10
    uv run python3 -m mas.generate --risks 1 --n-per-framing 2 --skip-enrich
    uv run python3 -m mas.generate --enrich-only mas/scenarios
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

import mas.generation as generation
import mas.prompts as prompts
from mas.llm import OpenRouterChatModel

MAS_DIR = Path(__file__).parent
FRAMINGS = ("emergent", "explicit")


def write_file(path: Path, metadata: dict, scenarios: list[dict]) -> None:
    path.write_text(json.dumps({"metadata": metadata, "scenarios": scenarios}, indent=2))


def enrich_file(model: OpenRouterChatModel, path: Path, overwrite: bool) -> None:
    data = json.loads(path.read_text())
    pending = [s for s in data["scenarios"] if overwrite or "concordia" not in s]
    if not pending:
        print(f"{path.name}: all scenarios already enriched, skipping")
        return
    print(f"{path.name}: enriching {len(pending)} scenario(s)")
    for s in pending:
        block = generation.enrich_scenario(model, s)
        if block is not None:
            s["concordia"] = block
        # Persist after every scenario so an interrupted run keeps its progress.
        data["metadata"]["enrichment_model"] = model.name
        write_file(path, data["metadata"], data["scenarios"])
    missing = [s["id"] for s in data["scenarios"] if "concordia" not in s]
    if missing:
        print(f"{path.name}: WARNING, enrichment failed for: {', '.join(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and enrich simulation scenarios from mas/risks.csv."
    )
    parser.add_argument("--risks", default="all", help="'all' or comma-separated row indices into risks.csv")
    parser.add_argument("--n-per-framing", type=int, default=10)
    parser.add_argument("--model", default="anthropic/claude-opus-4.7", help="OpenRouter model id")
    parser.add_argument("--out", default=str(MAS_DIR / "scenarios"), help="output directory")
    parser.add_argument("--overwrite", action="store_true", help="regenerate/re-enrich even if output exists")
    parser.add_argument("--skip-enrich", action="store_true", help="generate base scenarios only")
    parser.add_argument("--enrich-only", metavar="DIR", help="skip generation; enrich existing files in DIR")
    args = parser.parse_args()

    load_dotenv()
    model = OpenRouterChatModel(args.model)

    if args.enrich_only:
        for path in sorted(Path(args.enrich_only).glob("*.json")):
            enrich_file(model, path, args.overwrite)
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    risks = pd.read_csv(MAS_DIR / "risks.csv")
    if args.risks != "all":
        risks = risks.iloc[[int(i) for i in args.risks.split(",")]]

    for _, r in risks.iterrows():
        for framing in FRAMINGS:
            path = out_dir / f"{generation.slugify(r.risk)}__{framing}.json"
            if path.exists() and not args.overwrite:
                print(f"{path.name}: exists, skipping generation")
            else:
                print(f"{path.name}: generating {args.n_per_framing} scenario(s)")
                scenarios = generation.generate_scenarios(
                    model, args.n_per_framing, r.risk, r.description, r.examples, framing
                )
                if not scenarios:
                    print(f"{path.name}: WARNING, generation produced nothing, skipping")
                    continue
                metadata = {
                    "risk": r.risk,
                    "framing": framing,
                    "n_requested": args.n_per_framing,
                    "generation_model": model.name,
                    "prompt_version": prompts.PROMPT_VERSION,
                    "created": datetime.datetime.now().isoformat(timespec="seconds"),
                }
                write_file(path, metadata, scenarios)
            if not args.skip_enrich:
                enrich_file(model, path, args.overwrite)


if __name__ == "__main__":
    main()
