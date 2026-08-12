#!/usr/bin/env bash
# Test run for the risk-cluster pipeline.
#
# Defaults to a plain stage-1 run: no perturbations, no scenarios. Those replay
# frozen artifacts from datasets/generated/, and none exist for the cluster
# tasks yet — certify.py fails fast if you ask for a family that has not been
# generated.
#
# To add them, generate first (GENERATE=1) and then name the families:
#   PERTURB="paraphrase framing"  GENERATE=1  bash scripts/cluster_test.sh
#   SIMULATE=1                    GENERATE=1  bash scripts/cluster_test.sh
#
# Cost: framing is deterministic templating and reconsideration runs live, so
# both are free. paraphrase, register, identity_strip and the stage-3 scenarios
# each need one attacker-model pass over every sample of every cluster.
#
# Generation is deliberately never --limit'ed: certify --limit shuffles its
# subset, so artifacts must cover the whole cluster for the replayed variants
# to line up with whichever samples get drawn. With --limit set, results are
# not written to models.json.
set -euo pipefail
cd "$(dirname "$0")/.."

# ----- what to run -----------------------------------------------------------
# Edit these, or override any of them from the environment:
#   MODEL=openai/gpt-5-mini CLUSTERS=cyber bash scripts/cluster_test.sh
MODEL="${MODEL:-openrouter/openai/gpt-4o-mini}"   # the model under certification
GRADER="${GRADER:-openrouter/anthropic/claude-sonnet-4.5}"                              # empty = the GRADERS.md ensemble
ATTACKER="${ATTACKER:-openrouter/deepseek/deepseek-v4-flash}"  # generation only

CLUSTERS="${CLUSTERS:-}"          # e.g. "cyber" or "cyber manipulation"; empty = all
PERTURB="${PERTURB:-}"            # stage-2 families; empty = no perturbation
SIMULATE="${SIMULATE:-0}"         # 1 = also run stage-3 scenarios
LIMIT="${LIMIT-20}"               # samples per cluster; LIMIT="" = full run
                                  # (no colon above, so an explicit empty
                                  #  value means "no limit" rather than 20)

BUILD_DATASETS="${BUILD_DATASETS:-1}"   # datasets/public/<risk>.csv from raw/
GENERATE="${GENERATE:-0}"               # datasets/generated/<cluster>/<family>.jsonl
CERTIFY="${CERTIFY:-1}"

# -----------------------------------------------------------------------------

# Both tools default --perturb ON, so "no perturbation" has to be asked for
# explicitly. Word-split PERTURB into the flag deliberately: these are family
# names, never paths.
perturb_args=(--no-perturb)
[[ -n "$PERTURB" ]] && perturb_args=(--perturb $PERTURB)

simulate_args=()
[[ "$SIMULATE" == 1 ]] && simulate_args=(--simulate)

if [[ "$BUILD_DATASETS" == 1 ]]; then
    echo "==> building cluster datasets"
    uv run python3 -m datasets.prepare.cluster.prepare
fi

if [[ "$GENERATE" == 1 ]]; then
    # reconsideration challenges the target's own answer, so it is produced
    # live at eval time and generate.py rejects it as a family.
    gen_perturb="$(echo "${PERTURB//reconsideration/}" | xargs)"
    gen_args=(--no-perturb)
    [[ -n "$gen_perturb" ]] && gen_args=(--perturb $gen_perturb)

    echo "==> generating artifacts"
    uv run generate.py \
        --attacker "$ATTACKER" \
        ${CLUSTERS:+--only $CLUSTERS} \
        "${gen_args[@]}" \
        "${simulate_args[@]}" \
        --missing-only
fi

if [[ "$CERTIFY" == 1 ]]; then
    echo "==> certifying $MODEL"
    uv run certify.py \
        --model "$MODEL" \
        ${GRADER:+--grader "$GRADER"} \
        ${CLUSTERS:+--only $CLUSTERS} \
        "${perturb_args[@]}" \
        "${simulate_args[@]}" \
        ${LIMIT:+--limit "$LIMIT"}
fi

echo
echo "Reported score, per-cluster and per-source breakdowns are in the run"
echo "summary above; per-sample detail is in the .eval logs under logs/."
echo "If many samples scored 'unparseable', the judge config or a source's"
echo "criterion text is wrong — not the model being safe."
