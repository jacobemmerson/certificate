#!/bin/bash
#SBATCH --job-name=shb-adversarial
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# ── Configuration ────────────────────────────────────────────────────────────
# EDIT: scratch path where HF model weights will be stored (keep out of home)
SCRATCH="${SCRATCH:-/scratch/$USER}"
export HF_HOME="$SCRATCH/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"

# EDIT: target model to evaluate (through API — no local GPU needed)
TARGET_MODEL="openrouter/anthropic/claude-sonnet-4.6"

# Judge model (API — drives LLM judge and StrongREJECT)
JUDGE_MODEL="openrouter/google/gemini-3-flash-preview"

# Local model names — must match the --served-model-name below
ATTACKER_NAME="hermes-attacker"
HARMBENCH_NAME="harmbench-cls"
LLAMAGUARD_NAME="llamaguard"

# Ports
PROXY_PORT=8000
ATTACKER_PORT=8001
HARMBENCH_PORT=8002
LLAMAGUARD_PORT=8003

# ── Environment ───────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

# Load CUDA if your cluster uses modules
# module load cuda/12.4

source .venv/bin/activate   # or: conda activate <env>

export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY}"

mkdir -p logs

cleanup() {
    echo "Shutting down servers..."
    kill "$PROXY_PID" "$ATTACKER_PID" "$HARMBENCH_PID" "$LLAMAGUARD_PID" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait_for_server() {
    local port=$1 name=$2
    echo "Waiting for $name on port $port..."
    local deadline=$(( $(date +%s) + 300 ))
    until curl -sf "http://localhost:$port/health" > /dev/null 2>&1; do
        if [[ $(date +%s) -gt $deadline ]]; then
            echo "ERROR: $name did not start within 5 minutes" >&2
            exit 1
        fi
        sleep 5
    done
    echo "  $name ready"
}

CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --model "NousResearch/Hermes-3-Llama-3.1-70B-AWQ" \
    --quantization awq \
    --tensor-parallel-size 2 \
    --served-model-name "$ATTACKER_NAME" \
    --port "$ATTACKER_PORT" \
    --max-model-len 8192 \
    &
ATTACKER_PID=$!

CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
    --model "cais/HarmBench-Llama-2-13b-cls" \
    --served-model-name "$HARMBENCH_NAME" \
    --port "$HARMBENCH_PORT" \
    --max-model-len 2048 \
    &
HARMBENCH_PID=$!

CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server \
    --model "meta-llama/Llama-Guard-3-8B" \
    --served-model-name "$LLAMAGUARD_NAME" \
    --port "$LLAMAGUARD_PORT" \
    --max-model-len 4096 \
    &
LLAMAGUARD_PID=$!

wait_for_server "$ATTACKER_PORT"  "attacker"
wait_for_server "$HARMBENCH_PORT" "harmbench"
wait_for_server "$LLAMAGUARD_PORT" "llamaguard"

cat > /tmp/litellm_config_${SLURM_JOB_ID}.yaml <<EOF
model_list:
  - model_name: ${ATTACKER_NAME}
    litellm_params:
      model: openai/${ATTACKER_NAME}
      api_base: http://localhost:${ATTACKER_PORT}/v1
      api_key: dummy

  - model_name: ${HARMBENCH_NAME}
    litellm_params:
      model: openai/${HARMBENCH_NAME}
      api_base: http://localhost:${HARMBENCH_PORT}/v1
      api_key: dummy

  - model_name: ${LLAMAGUARD_NAME}
    litellm_params:
      model: openai/${LLAMAGUARD_NAME}
      api_base: http://localhost:${LLAMAGUARD_PORT}/v1
      api_key: dummy
EOF

litellm --config "/tmp/litellm_config_${SLURM_JOB_ID}.yaml" --port "$PROXY_PORT" &
PROXY_PID=$!
wait_for_server "$PROXY_PORT" "litellm-proxy"

# All openai/* calls now route through the proxy
export OPENAI_BASE_URL="http://localhost:${PROXY_PORT}/v1"
export OPENAI_API_KEY="dummy"

uv run inspect eval tasks/evals/socialharmbench.py@social_harm_bench_adversarial \
    --model "$TARGET_MODEL" \
    --max-connections 20 \
    --log-dir logs/ \
    -T "grader=${JUDGE_MODEL}" \
    -T "attacker=openai/${ATTACKER_NAME}" \
    -T "llamaguard_model=openai/${LLAMAGUARD_NAME}" \
    -T "harmbench_model=openai/${HARMBENCH_NAME}" \
    -T "attacker_budget=5" \
    "$@"   # pass through any extra args, e.g. --limit 10 for a test run

echo "Eval complete. Logs in logs/"
