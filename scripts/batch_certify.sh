#!/usr/bin/env bash
# Batch certification over a fleet of OpenRouter models.
#
# Loops certify.py once per model (all four clusters, perturbations + --simulate).
# certify.py skips clusters already scored in models/models.json, so this is
# resumable — re-run it and finished models fall through. set -e is deliberately
# NOT set: a model that errors (bad slug, provider outage) is logged and the
# batch continues.
#
# !!! The slugs below are BEST-GUESSES for the given names. Several are
# unreleased — VERIFY each against https://openrouter.ai/models before running.
#
# Usage:
#   bash scripts/batch_certify.sh                 # all models, 128 connections
#   MAX_CONN=200 bash scripts/batch_certify.sh    # override concurrency
set -uo pipefail

cd "$(dirname "$0")/.."

MAX_CONN="${MAX_CONN:-128}"

# Force Inspect's non-interactive display. The Textual TUI's worker cancels the
# whole eval if the terminal resizes or the pty drops (SIGWINCH / idle SSH),
# which silently freezes unattended batches — seen overnight as a CancelledError
# from textual/worker.py. "log" avoids Textual entirely.
export INSPECT_DISPLAY="${INSPECT_DISPLAY:-log}"

# slug | display name | provider | region        # cost $/1M (input, output)
MODELS=(
  "anthropic/claude-sonnet-5|Claude Sonnet 5|Anthropic|US"            # (2, 10)
  "anthropic/claude-haiku-4.5|Claude Haiku 4.5|Anthropic|US"              # (1, 5)
  "openai/gpt-5.6-terra|GPT 5.6 Terra|OpenAI|US"                      # (1, 6)
  "openai/gpt-5.6-luna-pro|GPT-5.6 Luna Pro|OpenAI|US"               # (0.10, 0.60)
  "openai/gpt-oss-120b|GPT-oss-120b|OpenAI|US"                        # (0.03, 0.17)
  "google/gemma-4-31b-it|Gemma 4 31B|Google|US"                         # (0.09, 0.34)
  "google/gemini-3.6-flash|Gemini 3.6 Flash|Google|US"               # (1.50, 7.50)
  "meta-llama/muse-spark-1.2|Muse Spark 1.1|Meta|US"                 # (1.25, 4.25)
  "meta-llama/llama-3.1-8b-instruct|Llama 3.1 8B Instruct|Meta|US"    # (0.02, 0.04)
  "z-ai/glm-5.2|GLM 5.2|Z.ai|China"                                  # (0.28, 0.88)
  "z-ai/glm-5|GLM 5|Z.ai|China"                                      # (0.60, 1.92)
  "qwen/qwen3.8-max|Qwen 3.8 Max|Alibaba|China"                      # (2, 6)
  "qwen/qwen3.7-flash|Qwen 3.7 Flash|Alibaba|China"                  # (0.03, 0.13)
  "x-ai/grok-4.5|Grok 4.5|xAI|US"                                    # (2, 6)
  "x-ai/grok-4.3|Grok 4.3|xAI|US"                                    # (1.25, 2.50)
  "deepseek/deepseek-v4-flash|DeepSeek V4 Flash|DeepSeek|China"       # (0.084, 0.168)
  "deepseek/deepseek-v4-pro|DeepSeek V4 Pro|DeepSeek|China"           # (0.0435, 0.87)
  "mistralai/mistral-medium-3-5|Mistral Medium 3.5|Mistral|EU"       # (1.50, 7.50)
  "mistralai/mistral-small-2603|Mistral Small 4|Mistral|EU"             # (0.15, 0.60)
)

failed=()
for entry in "${MODELS[@]}"; do
  IFS='|' read -r slug name provider region <<< "$entry"
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
        --max-connections "$MAX_CONN"; then
    echo "[FAILED] $name (openrouter/$slug) — continuing"
    failed+=("$name")
  fi
done

echo ""
echo "================================================================"
if [ ${#failed[@]} -eq 0 ]; then
  echo "Batch complete — all models ran."
else
  echo "Batch complete — ${#failed[@]} failed: ${failed[*]}"
fi
echo "================================================================"
