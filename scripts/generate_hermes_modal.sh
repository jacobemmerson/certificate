#!/bin/bash
# Same job as generate_hermes_slurm.sh, with Hermes-4-70B served on Modal
# (scripts/hermes_modal.py) instead of a local GPU node. Run from the repo root:
#
#     VLLM_API_KEY=<value in the hermes-vllm Modal secret> scripts/generate_hermes_modal.sh
#
# MAX_CONNECTIONS (default 128) caps concurrent requests for both the screen
# and generation. SCREEN_ONLY=1 stops after the screen. FORCE=1 regenerates
# every family (--force instead of --missing-only); it is required on the
# first run after a dataset rebuild, because --missing-only keeps rows for
# sample ids that no longer exist. The app is stopped on exit unless
# KEEP_WARM=1; a kept-warm app bills 4xH100 until
# `uvx modal app stop -y <HERMES_APP>` or 24 h. The modal log (with HERMES_URL
# and HERMES_APP) is logs/hermes-modal-<timestamp>.log.
#
# Smoke tests, in order, before the full run (a throughput number is the deliverable):
# 1. `KEEP_WARM=1 SCREEN_ONLY=1` with `MAX_CONNECTIONS=128`: time the cyber screen (940 items) → requests/s. Expect the 4×H100 vLLM to sustain well above 2 req/s; if it is near 20/s-bound by the client, raise `MAX_CONNECTIONS` to 256.
# 2. `uv run python generate.py --attacker vllm/NousResearch/Hermes-4-70B --model-base-url $URL/v1 --only cyber --limit 5 --force --perturb-k 1 --simulate --sim-k 2 --reasoning --max-connections 128`; then `uv run python scripts/audit_variant_fidelity.py` on the cyber artifacts (existing tool) and eyeball `incomplete_reasons` in each `.meta.json`; zero fallback rows required (the Jul 16 failure mode).
# 3. Only then: full `FORCE=1 scripts/generate_hermes_modal.sh`.
# Record the requests/s from test 1. URL is the HERMES_URL line in the log.
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
# `modal run` blocks while streaming the container's logs, so it runs in the
# background; --detach keeps the app alive if this client dies.
mkdir -p logs
LOG="logs/hermes-modal-$(date +%Y%m%d-%H%M%S).log"
uvx modal run --detach scripts/hermes_modal.py::serve > "$LOG" 2>&1 &
CLIENT_PID=$!
APP=""
stop() {
    kill $CLIENT_PID 2>/dev/null || true
    if [ "${KEEP_WARM:-0}" = 1 ]; then
        echo "KEEP_WARM=1: app $APP still running (4xH100 billing); stop it with: uvx modal app stop -y $APP"
    elif [ -n "$APP" ]; then
        uvx modal app stop -y "$APP"
    else
        echo "no HERMES_APP in $LOG; check \`uvx modal app list\` for a running hermes-vllm app"
    fi
}
trap stop EXIT

# Image build plus scheduling 4xH100 can take a while before the tunnel opens.
for i in $(seq 1 120); do
    grep -q 'HERMES_APP=ap-' "$LOG" && break
    kill -0 $CLIENT_PID 2>/dev/null || break
    sleep 10
done
URL=$(grep -o -m1 'HERMES_URL=https://[^ ]*' "$LOG" | cut -d= -f2 || true)
APP=$(grep -o -m1 'HERMES_APP=ap-[A-Za-z0-9]*' "$LOG" | cut -d= -f2 || true)
[ -n "$URL" ] && [ -n "$APP" ] || { echo "no HERMES_URL/HERMES_APP from modal run; see $LOG"; exit 1; }
echo "HERMES_URL=$URL HERMES_APP=$APP (log: $LOG)"

# A cold start downloads ~140 GB of weights into the hf-cache volume, so give
# vLLM up to ~40 min to come up behind the tunnel.
for i in $(seq 1 80); do
    curl -sf -m 60 -H "Authorization: Bearer $VLLM_API_KEY" "$URL/health" > /dev/null && break
    sleep 30
done
curl -sf -m 60 -H "Authorization: Bearer $VLLM_API_KEY" "$URL/health" > /dev/null \
    || { echo "vLLM never became healthy; see $LOG"; exit 1; }

MAX_CONNECTIONS=${MAX_CONNECTIONS:-128}

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
        --max-connections "$MAX_CONNECTIONS"
    uv run python -m datasets.prepare.cluster.prepare --risk "$risk"
done
# SCREEN_ONLY=1 stops here, so refused_dropped in
# datasets/public/<risk>.meta.json can be reviewed before any generation.
if [ "${SCREEN_ONLY:-0}" = 1 ]; then
    exit 0
fi

# ---- artifact generation ----------------------------------------------------
# FORCE=1 regenerates everything (see the header). Otherwise
# --missing-only makes this safe to rerun after an interruption:
# finished families are skipped, interrupted ones are filled in and merged.
# A family whose stored prompt_version differs from the code is regenerated
# in full, so after a prompt bump this behaves like --force for that family.
# generate.py exits nonzero if the attacker produces no usable output.
MODE=--missing-only
[ "${FORCE:-0}" = 1 ] && MODE=--force
uv run python generate.py \
    --attacker "vllm/$MODEL" \
    --model-base-url "$URL/v1" \
    --max-connections "$MAX_CONNECTIONS" \
    "$MODE" \
    --perturb-k 1 \
    --simulate --sim-k 2 \
    --reasoning
