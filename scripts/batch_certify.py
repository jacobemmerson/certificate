#!/usr/bin/env python3
"""
Batch-certify a fleet of OpenRouter models — resumable, fault-tolerant, cost-tracked.

Runs certify.py once per model (all four clusters, perturbations + --simulate) at
high concurrency, skips any model already scored in models/models.json, lets a
single model fail without sinking the batch, and after each run estimates that
model's spend from the log's recorded token usage times the per-million prices in
the table below.

    uv run python3 scripts/batch_certify.py --dry-run     # print the plan, spend nothing
    uv run python3 scripts/batch_certify.py --preflight   # ping every slug (~1 cheap call each), then stop
    uv run python3 scripts/batch_certify.py               # preflight, then run the whole batch
    uv run python3 scripts/batch_certify.py --only "Grok 4.5" "GLM 5"   # run a subset by name
    uv run python3 scripts/batch_certify.py --rerun       # re-run models already in models.json

!!! THE OPENROUTER SLUGS BELOW ARE BEST-GUESSES for the names given. Verify each
against https://openrouter.ai/models before a real run. --preflight pings every
slug and refuses to proceed if any is unreachable, so a *wrong* slug costs
nothing — but a wrong-yet-valid slug (right family, wrong model) will not be
caught. Eyeball the --dry-run table.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

from inspect_ai.log import list_eval_logs, read_eval_log

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.utils.graders import load_graders, load_models_with_check, validate_graders  # noqa: E402

# name, provider, region, openrouter slug (VERIFY), input $/1M, output $/1M
MODELS = [
    ("Claude Sonnet 5",        "Anthropic", "US",    "anthropic/claude-sonnet-5",        2.00, 10.00),
    ("Claude Haiku 5",         "Anthropic", "US",    "anthropic/claude-haiku-5",         1.00,  5.00),
    ("GPT 5.6 Terra",          "OpenAI",    "US",    "openai/gpt-5.6-terra",             1.00,  6.00),
    ("GPT-5.6 Luna Pro",       "OpenAI",    "US",    "openai/gpt-5.6-luna-pro",          0.10,  0.60),
    ("GPT-oss-120b",           "OpenAI",    "US",    "openai/gpt-oss-120b",              0.03,  0.17),
    ("Gemma 4 31B",            "Google",    "US",    "google/gemma-4-31b",               0.09,  0.34),
    ("Gemini 3.6 Flash",       "Google",    "US",    "google/gemini-3.6-flash",          1.50,  7.50),
    ("Muse Spark 1.1",         "Meta",      "US",    "meta-llama/muse-spark-1.1",        1.25,  4.25),
    ("Llama 3.1 8B Instruct",  "Meta",      "US",    "meta-llama/llama-3.1-8b-instruct", 0.02,  0.04),
    ("GLM 5.2",                "Z.ai",      "China", "z-ai/glm-5.2",                     0.28,  0.88),
    ("GLM 5",                  "Z.ai",      "China", "z-ai/glm-5",                       0.60,  1.92),
    ("Qwen 3.8 Max",           "Alibaba",   "China", "qwen/qwen3.8-max",                 2.00,  6.00),
    ("Qwen 3.7 Flash",         "Alibaba",   "China", "qwen/qwen3.7-flash",               0.03,  0.13),
    ("Grok 4.5",               "xAI",       "US",    "x-ai/grok-4.5",                    2.00,  6.00),
    ("Grok 4.3",               "xAI",       "US",    "x-ai/grok-4.3",                    1.25,  2.50),
    ("DeepSeek V4 Flash",      "DeepSeek",  "China", "deepseek/deepseek-v4-flash",       0.084, 0.168),
    ("DeepSeek V4 Pro",        "DeepSeek",  "China", "deepseek/deepseek-v4-pro",         0.0435, 0.87),
    ("Mistral Medium 3.5",     "Mistral",   "EU",    "mistralai/mistral-medium-3.5",     1.50,  7.50),
    ("Mistral Small 4",        "Mistral",   "EU",    "mistralai/mistral-small-4",        0.15,  0.60),
]

PROVIDER_PREFIX = "openrouter/"


def as_model(row):
    name, provider, region, slug, price_in, price_out = row
    return {
        "name": name, "provider": provider, "region": region,
        "slug": slug, "model": PROVIDER_PREFIX + slug,
        "price_in": price_in, "price_out": price_out,
    }


def already_scored(model_id: str) -> bool:
    """certify.py records each finished model in models/models.json under its id
    (the slug's last path segment). Present there means the batch can skip it."""
    models, _ = load_models_with_check()
    scored = {m.get("id") for m in models}
    return model_id.split("/")[-1] in scored


def usage_since(model_id: str, since: float) -> dict:
    """Token usage per model across this run's logs (those written after `since`),
    read from the model's own log dir (certify: log_dir = logs/<slug-tail>)."""
    log_dir = REPO_ROOT / "logs" / model_id.split("/")[-1]
    totals: dict[str, list[int]] = {}
    if not log_dir.exists():
        return totals
    for info in list_eval_logs(str(log_dir)):
        mtime = Path(info.name.replace("file://", "")).stat().st_mtime if info.name else 0
        if mtime < since:
            continue
        log = read_eval_log(info.name, header_only=True)
        for m, u in (getattr(log.stats, "model_usage", None) or {}).items():
            acc = totals.setdefault(m, [0, 0])
            acc[0] += getattr(u, "input_tokens", 0) or 0
            acc[1] += getattr(u, "output_tokens", 0) or 0
    return totals


def target_cost(usage: dict, model: dict) -> float:
    tin, tout = usage.get(model["model"], [0, 0])
    return tin / 1e6 * model["price_in"] + tout / 1e6 * model["price_out"]


def run_certify(model: dict, max_connections: int) -> int:
    cmd = [
        "uv", "run", "python3", "certify.py",
        "--model", model["model"],
        "--name", model["name"],
        "--provider", model["provider"],
        "--region", model["region"],
        "--simulate",
        "--max-connections", str(max_connections),
    ]
    print(f"\n$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="Print the plan and exit without spending.")
    p.add_argument("--preflight", action="store_true", help="Ping every target slug and the graders, then stop.")
    p.add_argument("--rerun", action="store_true", help="Include models already present in models.json.")
    p.add_argument("--only", nargs="+", metavar="NAME", help="Run only these models (by display name).")
    p.add_argument("--max-connections", type=int, default=128, help="Concurrent connections per run (default: 128).")
    return p.parse_args()


def main():
    args = parse()
    models = [as_model(r) for r in MODELS]
    if args.only:
        wanted = set(args.only)
        models = [m for m in models if m["name"] in wanted]
        missing = wanted - {m["name"] for m in models}
        if missing:
            sys.exit(f"Unknown model name(s): {', '.join(sorted(missing))}")

    runnable, skipped = [], []
    for m in models:
        (skipped if (not args.rerun and already_scored(m["model"])) else runnable).append(m)

    print("=" * 78)
    print(f"{'MODEL':22s} {'SLUG':34s} {'IN$/1M':>7s} {'OUT$/1M':>8s}  STATUS")
    print("-" * 78)
    for m in models:
        status = "SKIP (in models.json)" if m in skipped else "run"
        print(f"{m['name']:22s} {m['slug']:34s} {m['price_in']:7.3f} {m['price_out']:8.3f}  {status}")
    print("=" * 78)
    print(f"{len(runnable)} to run, {len(skipped)} skipped, graders: {load_graders()}")

    if args.dry_run:
        return

    # Preflight: a bad slug should cost nothing, so ping every target and grader
    # before any cluster runs. validate_graders raises SystemExit naming failures.
    print("\nPreflight: pinging targets + graders...")
    validate_graders([m["model"] for m in runnable] + load_graders())
    print("All targets and graders reachable.")
    if args.preflight:
        return

    report = REPO_ROOT / "logs" / "batch_cost_report.csv"
    if not report.exists():
        report.write_text("model,slug,target_cost_usd,target_in_tokens,target_out_tokens,returncode\n")

    grand = 0.0
    for i, m in enumerate(runnable, 1):
        print(f"\n{'#' * 78}\n# [{i}/{len(runnable)}] {m['name']}  ({m['model']})\n{'#' * 78}")
        started = time.time()
        rc = run_certify(m, args.max_connections)
        usage = usage_since(m["model"], started)
        cost = target_cost(usage, m)
        grand += cost
        tin, tout = usage.get(m["model"], [0, 0])
        judges = {k: v for k, v in usage.items() if k != m["model"]}
        judge_tok = sum(a + b for a, b in judges.values())
        with report.open("a") as f:
            f.write(f"{m['name']},{m['slug']},{cost:.4f},{tin},{tout},{rc}\n")
        status = "OK" if rc == 0 else f"NONZERO EXIT {rc}"
        print(f"\n[{m['name']}] {status} | target ${cost:.2f} "
              f"({tin:,} in / {tout:,} out) | judge tokens {judge_tok:,} (priced separately)")

    print(f"\n{'=' * 78}\nBatch done. Estimated target spend: ${grand:.2f}. "
          f"Report: {report}\n{'=' * 78}")


if __name__ == "__main__":
    main()
