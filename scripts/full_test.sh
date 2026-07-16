#!/usr/bin/env bash

set -euo pipefail

MODEL="openrouter/openai/gpt-4o-mini"
ATTACKER="openrouter/deepseek/deepseek-v4-flash"
GRADER="openrouter/openai/gpt-4o"
ONLY="harm"

uv run generate.py \
    --attacker "$ATTACKER" \
    --only "$ONLY" \
    --simulate \

uv run certify.py \
    --model "$MODEL" \
    --grader "$GRADER" \
    --only "$ONLY" \
    --llamaguard "" \
    --simulate
