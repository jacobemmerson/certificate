#!/bin/bash
# Same job as generate_hermes_slurm.sh, with Hermes-4-70B served on Modal
# (scripts/hermes_modal.py) instead of a local GPU node. Run from the repo root:
#
#     VLLM_API_KEY=<value in the hermes-vllm Modal secret> scripts/generate_hermes_modal.sh
# Set HERMES_URL and HERMES_APP to reuse an app that is already up (printed by a
# previous run) instead of launching a new one.
#
# MAX_CONNECTIONS (default 128) caps concurrent requests. FORCE=1 regenerates
# every family (--force instead of --missing-only); it is required on the
# first run after a dataset rebuild, because --missing-only keeps rows for
# sample ids that no longer exist. The app is stopped on exit unless
# KEEP_WARM=1; a kept-warm app bills 4xH100 until
# `uv run modal app stop -y <HERMES_APP>` or 24 h. The modal log (with HERMES_URL
# and HERMES_APP) is logs/hermes-modal-<timestamp>.log.
#
# Smoke test before the full run (Oct 2026: ~5,000 gen tok/s at 128 connections):
# 1. `uv run python generate.py --attacker vllm/NousResearch/Hermes-4-70B --model-base-url $URL/v1 --only cyber --limit 5 --force --perturb-k 1 --simulate --sim-k 2 --reasoning --max-connections 128`; then `uv run python scripts/audit_variant_fidelity.py` on the cyber artifacts (existing tool) and eyeball `incomplete_reasons` in each `.meta.json`; zero fallback rows required (the Jul 16 failure mode).
# 2. Only then: full `FORCE=1 scripts/generate_hermes_modal.sh`. URL is the HERMES_URL line in the log.
#
# modal is a project dependency (uv run modal); vllm stays out of uv.lock (see the
# venv-drift note in pyproject.toml). inspect's vllm/ provider sends
# $VLLM_API_KEY as the bearer token.

set -euo pipefail
cd "$(dirname "$0")/.."

: "${VLLM_API_KEY:?set VLLM_API_KEY to the value stored in the hermes-vllm Modal secret}"
export VLLM_API_KEY
command -v uv > /dev/null || { echo "uv not on PATH"; exit 1; }

MODEL="NousResearch/Hermes-4-70B"

# ---- vLLM server on Modal ---------------------------------------------------
# HERMES_URL=<tunnel url> HERMES_APP=<ap-id> reuses an app that is already up
# (e.g. one left warm by KEEP_WARM=1) instead of starting a second 4xH100.
if [ -n "${HERMES_URL:-}" ]; then
    URL=$HERMES_URL
    APP=${HERMES_APP:-}
    LOG=/dev/null
    CLIENT_PID=""
    stop() {
        if [ "${KEEP_WARM:-0}" = 1 ] || [ -z "$APP" ]; then
            echo "app ${APP:-?} left running; stop it with: uv run modal app stop -y ${APP:-<id>}"
        else
            uv run modal app stop -y "$APP"
        fi
    }
    trap stop EXIT
    echo "reusing HERMES_URL=$URL HERMES_APP=$APP"
else
    # `modal run` blocks while streaming the container's logs, so it runs in the
    # background; --detach keeps the app alive if this client dies. COLUMNS keeps
    # Rich from wrapping the "View run at .../ap-<id>" line the app id is read from.
    mkdir -p logs
    LOG="logs/hermes-modal-$(date +%Y%m%d-%H%M%S).log"
    COLUMNS=200 uv run modal run --detach scripts/hermes_modal.py::serve > "$LOG" 2>&1 &
    CLIENT_PID=$!
    APP=""
    stop() {
        kill $CLIENT_PID 2>/dev/null || true
        if [ "${KEEP_WARM:-0}" = 1 ]; then
            echo "KEEP_WARM=1: app $APP still running (4xH100 billing); stop it with: uv run modal app stop -y $APP"
        elif [ -n "$APP" ]; then
            uv run modal app stop -y "$APP"
        else
            echo "no app id in $LOG; check \`uv run modal app list\` for a running hermes-vllm app"
        fi
    }
    trap stop EXIT

    # The client prints the app id as soon as the app exists, before the image
    # build and GPU scheduling, so the trap can stop it from here on.
    for i in $(seq 1 60); do
        APP=$(grep -o -m1 'ap-[A-Za-z0-9]\{22\}' "$LOG" || true)
        [ -n "$APP" ] && break
        kill -0 $CLIENT_PID 2>/dev/null || break
        sleep 5
    done
    [ -n "$APP" ] || { echo "no app id from modal run; see $LOG"; exit 1; }

    # Then the container opens the tunnel and prints HERMES_URL.
    for i in $(seq 1 120); do
        grep -q 'HERMES_URL=https' "$LOG" && break
        kill -0 $CLIENT_PID 2>/dev/null || break
        sleep 10
    done
    URL=$(grep -o -m1 'HERMES_URL=https://[^ ]*' "$LOG" | cut -d= -f2 || true)
    [ -n "$URL" ] || { echo "no HERMES_URL from modal run; see $LOG"; exit 1; }
    grep -q "HERMES_APP=$APP" "$LOG" || echo "warning: container's HERMES_APP differs from $APP; see $LOG"
    echo "HERMES_URL=$URL HERMES_APP=$APP (log: $LOG)"
fi

# A cold start downloads ~140 GB of weights into the hf-cache volume, so give
# vLLM up to ~40 min to come up behind the tunnel.
for i in $(seq 1 80); do
    curl -sf -m 60 -H "Authorization: Bearer $VLLM_API_KEY" "$URL/health" > /dev/null && break
    sleep 30
done
curl -sf -m 60 -H "Authorization: Bearer $VLLM_API_KEY" "$URL/health" > /dev/null \
    || { echo "vLLM never became healthy; see $LOG"; exit 1; }

MAX_CONNECTIONS=${MAX_CONNECTIONS:-128}

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
