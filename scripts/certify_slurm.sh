#!/bin/bash
#SBATCH --job-name=certify
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=24:00:00
#SBATCH --array=0-29%3
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --requeue

# One certify.py run per array index, reading the Nth model of scripts/models.txt.
# CPU only: the target and judges are API calls. %3 bounds concurrent runs so
# three models share the OpenRouter rate limit, not thirty.
#
# Resumable: certify.py resumes logs/<model>/current via eval_set, so a
# preempted/requeued task re-runs only its unfinished samples. Each run writes
# its own models/results/<model>.json and rebuilds models.json atomically, so
# parallel tasks cannot corrupt each other's results.
#
# --cheapest resolves the cheapest capable OpenRouter endpoints at run time, and
# endpoints are part of task identity. If prices change between a preemption and
# the requeue, the affected tasks re-run in full (re-billed) in the same run dir,
# with the old logs left alongside. Pass --rerun or a new --run-id to start clean.
#
# Indices beyond the file's last model exit 0, so --array can stay 0-29 as the
# roster grows. Append models at the END of models.txt while an array is queued.
#
# logs/ is gitignored and slurm opens --output before this script runs, so a
# missing logs/ fails the job silently: create it at submit time.
#
#   mkdir -p logs && sbatch scripts/certify_slurm.sh            # from the repo root
#   mkdir -p logs && sbatch --array=5 scripts/certify_slurm.sh  # one model
#   MAX_CONN=64 sbatch scripts/certify_slurm.sh
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"

export UV_CACHE_DIR="$HOME/scratch/uv_cache"
export INSPECT_DISPLAY=log

mapfile -t MODELS < <(grep -Ev '^\s*(#|$)' scripts/models.txt)
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#MODELS[@]}" ]; then
  echo "no model at index $SLURM_ARRAY_TASK_ID (models.txt has ${#MODELS[@]})"
  exit 0
fi

IFS='|' read -r slug name provider region <<< "${MODELS[$SLURM_ARRAY_TASK_ID]}"
echo "# $name (openrouter/$slug)"
exec uv run python3 certify.py \
  --model "openrouter/$slug" \
  --name "$name" \
  --provider "$provider" \
  --region "$region" \
  --simulate \
  --cheapest \
  --max-connections "${MAX_CONN:-128}"
