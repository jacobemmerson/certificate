#!/bin/bash
#SBATCH --job-name=certify
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err

# ── Certificate generation for several models (parallel job array) ────────────
# One array task per model — the scheduler runs them concurrently. Each task
# runs the FULL certificate (all benchmarks, including the adversarial harm_adv
# best-of-N perturbation attack).
#
# To avoid races on the shared models/models.json, each task writes its own
# partial file (models/partials/<model_id>.json) via CERTIFY_MODELS_PATH. Fold
# them back in afterwards with scripts/merge_partials.py.
#
# certify.py's defaults route the attacker + LlamaGuard through OpenRouter, so
# this is an API-only job — no GPUs required. Just needs OPENROUTER_API_KEY.
#
# IMPORTANT: keep #SBATCH --array=0-N matched to the number of MODELS entries
# (0-indexed, so 4 models => --array=0-3).
#
# Usage:
#   # submit the array, then a merge job that runs once all tasks succeed:
#   jid=$(sbatch --parsable scripts/certify_slurm.sh)
#   sbatch --dependency=afterok:$jid --job-name=certify-merge \
#          --output=logs/merge_%j.out --wrap 'uv run scripts/merge_partials.py'
#
#   # quick smoke test (note: --limit disables saving, so no partials/merge):
#   sbatch scripts/certify_slurm.sh --limit 10
# Any extra args are forwarded to certify.py.

set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

source .venv/bin/activate

export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY}"

mkdir -p logs models/partials

# ── Models to certify ─────────────────────────────────────────────────────────
# Each entry: model_id | display name | provider | region | specialty
# Keep #SBATCH --array above in sync with the length of this list.
MODELS=(
    #"openrouter/anthropic/claude-sonnet-4.6|Claude Sonnet 4.6|Anthropic|USA Frontier Models|Everyday use and reasoning"
    #"openrouter/openai/gpt-5.4-mini|GPT-5.4 Mini|OpenAI|USA Frontier Models|High-throughput and cost efficiency"
    #"openrouter/google/gemini-3.1-pro-preview|Gemini 3.1 Pro|Google|USA Frontier Models|Reasoning"
    "openrouter/deepseek/deepseek-v4-flash|DeepSeek V4 Flash|DeepSeek|China Frontier Models|General-purpose assistant"
)

# ── Select this task's model ──────────────────────────────────────────────────
# Falls back to task 0 when run outside a job array (e.g. a bare bash run).
idx="${SLURM_ARRAY_TASK_ID:-0}"
if (( idx < 0 || idx >= ${#MODELS[@]} )); then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$idx out of range (0-$(( ${#MODELS[@]} - 1 )))." >&2
    echo "       Update #SBATCH --array to match the ${#MODELS[@]} MODELS entries." >&2
    exit 1
fi

IFS='|' read -r model name provider region specialty <<< "${MODELS[$idx]}"
model_id="${model##*/}"

# Per-task output file — merged into models/models.json later.
export CERTIFY_MODELS_PATH="models/partials/${model_id}.json"

echo "═══════════════════════════════════════════════════════════════════"
echo "[task $idx] Certifying: $name  ($model)"
echo "           Partial output: $CERTIFY_MODELS_PATH"
echo "═══════════════════════════════════════════════════════════════════"

uv run certify.py \
    --model "$model" \
    --name "$name" \
    --provider "$provider" \
    --region "$region" \
    --specialty "$specialty" \
    "$@"   # forwards e.g. --rerun, --epochs N

echo "[task $idx] Done. Partial written to $CERTIFY_MODELS_PATH"
echo "Run scripts/merge_partials.py once all array tasks finish."
