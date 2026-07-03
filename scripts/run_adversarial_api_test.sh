#!/bin/bash
# Quick adversarial-retry smoke test on SocialHarmBench using only API-based models —
# judge, attacker, and LlamaGuard are all hosted models reached over an API (OpenRouter),
# so unlike run_adversarial_test.sh / run_adversarial_slurm.sh, no local GPU/vLLM/litellm
# hosting is needed. Run directly, e.g.: bash scripts/run_adversarial_api_test.sh

set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

# export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY}"

TARGET_MODEL="openrouter/deepseek/deepseek-v4-flash"

# API-based judge, attacker, and LlamaGuard — all via OpenRouter
JUDGE_MODEL="openrouter/anthropic/claude-haiku-4.5"
ATTACKER_MODEL="openrouter/meta-llama/llama-3.2-3b-instruct"
LLAMAGUARD_MODEL="openrouter/meta-llama/llama-guard-4-12b"

uv run inspect eval tasks/evals/socialharmbench.py@social_harm_bench_adversarial \
    --model "$TARGET_MODEL" \
    --max-connections 50 \
    --limit 10 \
    --log-dir logs/ \
    --sample-shuffle \
    -T attacks="["cipher", "deep_inception", "pair", "crescendo"]"\
    -T "grader=${JUDGE_MODEL}" \
    -T "attacker=${ATTACKER_MODEL}" \
    -T "llamaguard_model=${LLAMAGUARD_MODEL}" \
    -T "attacker_budget=5" \
    "$@"
