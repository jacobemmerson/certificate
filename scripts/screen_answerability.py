'''
Hermes answerability screen (spec §1.1; prepare.py tier 3b).

    uv run python3 scripts/screen_answerability.py --risk cbrn \
        --model vllm/NousResearch/Hermes-4-70B --model-base-url http://localhost:8000/v1

Reads datasets/cache/<risk>.screen_input.jsonl (written by prepare.py on a
screen-cache miss), sends each candidate's system prompt and query to the model,
and appends {key, sample_id, verdict, completion[:400], model, created_at} to
datasets/cache/screen/<risk>.jsonl. Append-only and keyed, so a requeued or
repeated run sends only what is not there yet, and the file stays reviewable.

verdict is "refused" when detectors.liberal_refusal flags the completion; mcq
candidates may answer with a bare letter, so the short-output rule is off for
them. A call that still fails after retries is not recorded: the script exits
1 and prepare.py asks for that candidate again.
'''
import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv

load_dotenv()

from inspect_ai.model import get_model

from datasets.prepare.cluster.schema import MCQ
from pipeline.generation import _attacker_call
from pipeline.stage1_evaluation.scorers.detectors import MIN_ANSWER_CHARS, liberal_refusal

CACHE_DIR = Path(__file__).resolve().parent.parent / "datasets" / "cache"


async def screen(records: list[dict], model, model_name: str, out_path: Path,
                 max_connections: int = 20) -> int:
    '''Screen every record whose key `out_path` lacks; returns the failed-call count.'''
    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(line)["key"] for line in f}
    semaphore = asyncio.Semaphore(max_connections)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    failed = 0

    with open(out_path, "a", encoding="utf-8") as out:
        async def one(record: dict) -> None:
            nonlocal failed
            async with semaphore:
                completion = await _attacker_call(
                    model, record["query"], f"screen {record['sample_id']}",
                    system=record["system_prompt"] or None,
                )
            if completion is None:
                failed += 1
                return
            min_chars = 0 if record["question_type"] == MCQ else MIN_ANSWER_CHARS
            refused = liberal_refusal(completion, min_chars=min_chars)
            out.write(json.dumps({
                "key": record["key"],
                "sample_id": record["sample_id"],
                "verdict": "refused" if refused else "answered",
                "completion": completion[:400],
                "model": model_name,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }, ensure_ascii=False) + "\n")
            out.flush()  # a preempted job keeps every verdict already written

        await asyncio.gather(*(one(r) for r in records if r["key"] not in done))
    return failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--risk", action="append", required=True)
    parser.add_argument("--model", default="openrouter/nousresearch/hermes-4-70b")
    parser.add_argument("--model-base-url", default=None,
                        help="Base URL of a running server, e.g. vLLM on slurm.")
    parser.add_argument("--max-connections", type=int, default=20)
    args = parser.parse_args()

    model = get_model(args.model, base_url=args.model_base_url)
    failed = 0
    for risk in args.risk:
        with open(CACHE_DIR / f"{risk}.screen_input.jsonl", encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
        failed += asyncio.run(screen(
            records, model, args.model, CACHE_DIR / "screen" / f"{risk}.jsonl",
            args.max_connections,
        ))
        print(f"{risk}: {len(records)} candidates screened ({failed} failed calls so far)")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
