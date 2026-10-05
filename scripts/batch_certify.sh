#!/usr/bin/env bash
# Non-slurm batch: certify.py once per model in scripts/models.txt, serially.
# Same roster and flags as scripts/certify_slurm.sh. Resumable — certify.py
# skips risks already in models/results/<model>.json and resumes unfinished
# logs via eval_set. set -e is deliberately NOT set: a model that errors is
# logged and the batch continues.
#
#   bash scripts/batch_certify.sh
#   MAX_CONN=200 bash scripts/batch_certify.sh
set -uo pipefail
cd "$(dirname "$0")/.."

export INSPECT_DISPLAY="${INSPECT_DISPLAY:-log}"

failed=()
while IFS='|' read -r slug name provider region; do
  echo ""
  echo "################################################################"
  echo "# $name  (openrouter/$slug)"
  echo "################################################################"
  if ! uv run python3 certify.py \
        --model "openrouter/$slug" \
        --name "$name" \
        --provider "$provider" \
        --region "$region" \
        --simulate \
        --cheapest \
        --max-connections "${MAX_CONN:-128}"; then
    echo "[FAILED] $name (openrouter/$slug) — continuing"
    failed+=("$name")
  fi
done < <(grep -Ev '^\s*(#|$)' scripts/models.txt)

echo ""
echo "================================================================"
if [ ${#failed[@]} -eq 0 ]; then
  echo "Batch complete — all models ran."
else
  echo "Batch complete — ${#failed[@]} failed: ${failed[*]}"
fi
echo "================================================================"
