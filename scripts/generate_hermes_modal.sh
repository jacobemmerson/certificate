#!/bin/bash
# Same job as generate_hermes_slurm.sh, with Hermes-4-70B served on Modal
# (scripts/hermes_modal.py) instead of a local GPU node. Run from the repo root:
#
#     VLLM_API_KEY=<value in the hermes-vllm Modal secret> scripts/generate_hermes_modal.sh
#
# SCREEN_ONLY=1 stops after the screen. KEEP_WARM=1 skips stopping the app on
# exit; otherwise it scales to zero 15 min after the last request anyway.
#
# Smoke-test first, with the app deployed (KEEP_WARM=1 SCREEN_ONLY=1 works):
#     uv run python generate.py --attacker vllm/$MODEL --model-base-url $URL/v1 --limit 5 --only cyber
#
# modal, like vllm, runs through uvx and stays out of uv.lock (see the
# venv-drift note in pyproject.toml). inspect's vllm/ provider sends
# $VLLM_API_KEY as the bearer token.

set -euo pipefail
cd "$(dirname "$0")/.."

: "${VLLM_API_KEY:?set VLLM_API_KEY to the value stored in the hermes-vllm Modal secret}"
export VLLM_API_KEY
command -v uvx > /dev/null || { echo "uvx not on PATH"; exit 1; }

MODEL="NousResearch/Hermes-4-70B"

# ---- vLLM server on Modal ---------------------------------------------------
URL=$(uvx modal deploy scripts/hermes_modal.py | tee /dev/stderr \
    | grep -o 'https://[^ ]*\.modal\.run' | head -1 || true)
[ -n "$URL" ] || { echo "no *.modal.run URL in deploy output"; exit 1; }
[ "${KEEP_WARM:-0}" = 1 ] || trap 'uvx modal app stop hermes-vllm' EXIT

# The first request starts the container; a cold start downloads ~140 GB of
# weights into the hf-cache volume, so give it up to ~40 min.
for i in $(seq 1 80); do
    curl -sf -m 60 -H "Authorization: Bearer $VLLM_API_KEY" "$URL/health" > /dev/null && break
    sleep 30
done
curl -sf -m 60 -H "Authorization: Bearer $VLLM_API_KEY" "$URL/health" > /dev/null \
    || { echo "vLLM never became healthy; see: uvx modal app logs hermes-vllm"; exit 1; }

# ---- answerability screen ---------------------------------------------------
# prepare.py leaves datasets/cache/<risk>.screen_input.jsonl when candidates
# have no screen verdict yet. Screen them, then rebuild that risk's CSV so the
# artifacts below are generated for the screened selection. Keys already
# screened are skipped, so rerunning costs nothing.
for input in datasets/cache/*.screen_input.jsonl; do
    [ -e "$input" ] || continue
    risk=$(basename "$input" .screen_input.jsonl)
    uv run python scripts/screen_answerability.py --risk "$risk" \
        --model "vllm/$MODEL" \
        --model-base-url "$URL/v1" \
        --max-connections 32
    uv run python -m datasets.prepare.cluster.prepare --risk "$risk"
done
# SCREEN_ONLY=1 stops here, so refused_dropped in
# datasets/public/<risk>.meta.json can be reviewed before any generation.
if [ "${SCREEN_ONLY:-0}" = 1 ]; then
    exit 0
fi

# ---- artifact generation ----------------------------------------------------
# --missing-only makes this safe to rerun after an interruption:
# finished families are skipped, interrupted ones are filled in and merged.
# A family whose stored prompt_version differs from the code is regenerated
# in full, so after a prompt bump this behaves like --force for that family.
# generate.py exits nonzero if the attacker produces no usable output.
uv run python generate.py \
    --attacker "vllm/$MODEL" \
    --model-base-url "$URL/v1" \
    --max-connections 32 \
    --missing-only \
    --perturb-k 1 \
    --simulate --sim-k 2 \
    --reasoning
