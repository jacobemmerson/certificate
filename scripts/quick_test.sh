#!/usr/bin/env bash
# Quick smoke test: generate the perturbation + scenario artifacts once, then
# certify GPT-4o-mini by replaying them. Stage 2 (--perturb, on by default)
# and stage 3 (--simulate) compose in one certify run — one control
# generation, one log, separate metric pools (lvr/consistency vs.
# lvr_scenario/consistency_scenario).
#
# Generation is NOT limited: certify --limit randomly shuffles its subset, so
# the artifacts must cover the whole benchmark for the replayed variants to
# line up. The certify runs stay limited (fast, and results aren't saved to
# models.json).
set -euo pipefail

MODEL="openrouter/openai/gpt-4o-mini"
ATTACKER="openrouter/openai/gpt-4o-mini"
GRADER="openrouter/openai/gpt-4o-mini"
ONLY="harm"
LIMIT=10

# 1. Generate the frozen artifacts (attacker model runs once) —
#    both stage-2 perturbations and stage-3 scenarios, full coverage.
uv run generate.py \
    --attacker "$ATTACKER" \
    --only "$ONLY" \
    --simulate

# 2. Certify GPT-4o-mini against the pregenerated perturbations (stage 2)
#    AND scenarios (stage 3) in one run/one log.
uv run certify.py \
    --model "$MODEL" \
    --grader "$GRADER" \
    --only "$ONLY" \
    --limit "$LIMIT" \
    --simulate
