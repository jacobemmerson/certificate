#!/usr/bin/env bash
# Regrade Claude Sonnet 5 with the missing GPT judge.
#
# WHY. In models/models.json, claude-sonnet-5 was scored by only one of the two
# ensemble judges named in GRADERS.md — openrouter/anthropic/claude-sonnet-4.5 —
# with openrouter/openai/gpt-5.6-luna absent from every condition. A one-judge
# score is not the ensemble mean the rest of the roster is graded on, so it is
# not comparable to it (GRADERS.md: the ensemble is averaged, not voted).
#
# The raw Inspect .eval logs for this model are gone (there is no
# logs/claude-sonnet-5/), so the gpt judge cannot be replayed over the stored
# generations. The only way back to a consistent score is to regenerate the
# model's answers and grade the fresh generations with the WHOLE ensemble at
# once: grading just the gpt judge onto answers the claude judge never saw would
# average two judges over two different sets of responses.
#
# HOW. certify.py --rerun regenerates a cluster and grades it with every judge in
# GRADERS.md (no --grader override), then merges the result into models.json,
# replacing the one-judge entry. certify.py rewrites the whole model record from
# its args and does not carry aa_intelligence_index / aa_model_match, so
# scripts/match_aa_index.py is re-run afterwards to restore them — exactly as a
# normal certification batch is followed by the AA mapping step.
#
# --simulate plus the default (all-family) perturbation reproduce the conditions
# the existing entry already carries (control, framing, identity_strip,
# paraphrase, reconsideration, register, scenario), so the regrade differs from
# the original run only in the judge set, not the elicitation.
#
# RESUMABLE. Each risk cluster is a separate certify call, and a cluster that
# already carries the gpt judge is skipped, so re-running after a mid-batch
# failure only redoes the clusters still missing it. certify.py's own skip logic
# cannot help here — the old results are marked "complete", just with one judge —
# which is why every cluster is forced with --rerun.
#
# Usage:
#   bash scripts/regrade_sonnet5.sh
#   MAX_CONN=200 bash scripts/regrade_sonnet5.sh
#   MODEL=openrouter/anthropic/claude-sonnet-5 bash scripts/regrade_sonnet5.sh
set -uo pipefail

cd "$(dirname "$0")/.."

# !!! BEST-GUESS slug — VERIFY against https://openrouter.ai/models before running.
MODEL="${MODEL:-openrouter/anthropic/claude-sonnet-5}"
MODEL_ID="${MODEL##*/}"                 # claude-sonnet-5, the models.json id
NAME="Claude Sonnet 5"
PROVIDER="Anthropic"
REGION="US"
GPT_JUDGE="openrouter/openai/gpt-5.6-luna"
CLAUDE_JUDGE="openrouter/anthropic/claude-sonnet-4.5"
RISKS=(cbrn cyber loss_of_control manipulation)
MAX_CONN="${MAX_CONN:-128}"

# Force Inspect's non-interactive display, so an idle SSH pty or a terminal
# resize cannot cancel an unattended run (see scripts/small_batch.sh).
export INSPECT_DISPLAY="${INSPECT_DISPLAY:-log}"

# The whole point is to add the gpt judge, so refuse to run unless GRADERS.md
# actually carries both judges — otherwise a --rerun would just re-bake the same
# one-judge score under a fresh generation.
for judge in "$GPT_JUDGE" "$CLAUDE_JUDGE"; do
  if ! grep -qxF "$judge" GRADERS.md; then
    echo "[ABORT] $judge is not listed in GRADERS.md; regrading would not add it."
    exit 1
  fi
done

# True when this model's stored results for a risk already mention the gpt judge —
# used to skip a cluster that a previous run already regraded. Stdlib only, so it
# runs under the system python without the uv environment.
has_gpt_judge() {  # $1 = risk
  python3 - "$MODEL_ID" "$1" "$GPT_JUDGE" <<'PY'
import json, sys
model_id, risk, judge = sys.argv[1:4]
models = json.load(open("models/models.json"))
entry = next((m for m in models if m.get("id") == model_id), None)
risk_results = (entry or {}).get("results", {}).get(risk, {})
sys.exit(0 if judge in json.dumps(risk_results) else 1)
PY
}

failed=()
for risk in "${RISKS[@]}"; do
  echo ""
  echo "################################################################"
  echo "# $risk"
  echo "################################################################"
  if has_gpt_judge "$risk"; then
    echo "[SKIP] $risk already carries $GPT_JUDGE"
    continue
  fi
  if ! uv run python3 certify.py \
        --model "$MODEL" \
        --name "$NAME" \
        --provider "$PROVIDER" \
        --region "$REGION" \
        --only "$risk" \
        --rerun \
        --simulate \
        --cheapest \
        --max-connections "$MAX_CONN"; then
    echo "[FAILED] $risk — continuing"
    failed+=("$risk")
  fi
done

if [ ${#failed[@]} -ne 0 ]; then
  echo ""
  echo "[INCOMPLETE] clusters still missing the gpt judge: ${failed[*]}"
  echo "Re-run this script to retry them; already-regraded clusters are skipped."
  exit 1
fi

# certify.py rewrites the whole model record and drops the AA fields; restore
# them the same way a normal certification batch does.
echo ""
echo "Re-applying Artificial Analysis index mapping..."
uv run python3 scripts/match_aa_index.py

# Confirm the fix landed: every risk cluster must now carry the gpt judge.
echo ""
all_ok=1
for risk in "${RISKS[@]}"; do
  if has_gpt_judge "$risk"; then
    echo "[OK]   $risk carries $GPT_JUDGE"
  else
    echo "[WARN] $risk still missing $GPT_JUDGE"
    all_ok=0
  fi
done

[ "$all_ok" -eq 1 ] && echo "Done — Claude Sonnet 5 regraded with both judges." || exit 1
