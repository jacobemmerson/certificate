#!/bin/bash
#SBATCH --job-name=shb-test
#SBATCH --nodes=1
#SBATCH --gpus=h100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_test_%j.out
#SBATCH --error=logs/slurm_test_%j.err

set -euo pipefail

module load cuda 2>/dev/null || true   # needed for flashinfer JIT (nvcc)

export HF_HOME="$HOME/scratch/hf_cache"
export HF_HUB_OFFLINE=1               # never attempt network access on compute nodes
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:?}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# ── Activate venv (install on login node first with: uv venv .venv && uv pip install -r requirements.txt) ──
if [[ ! -f .venv/bin/activate ]]; then
    echo "ERROR: .venv not found. Run 'uv venv .venv && uv pip install -r requirements.txt' on the login node first." >&2
    exit 1
fi
source .venv/bin/activate

# ── Cleanup trap ──────────────────────────────────────────────────────────────
PROXY_PID="" ATTACKER_PID="" HARMBENCH_PID=""
cleanup() {
    kill $PROXY_PID $ATTACKER_PID $HARMBENCH_PID 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ── Helper ────────────────────────────────────────────────────────────────────
wait_for_server() {
    local port=$1 name=$2
    echo "Waiting for $name on port $port..."
    local deadline=$(( $(date +%s) + 900 ))
    until curl -sf "http://localhost:$port/health" > /dev/null 2>&1; do
        if [[ $(date +%s) -gt $deadline ]]; then
            echo "ERROR: $name did not start within 15 minutes" >&2; exit 1
        fi
        sleep 5
    done
    echo "  $name ready"
}

# ── vLLM: Hermes-3-70B-AWQ (attacker) on GPU 0 ───────────────────────────────
# AWQ fits in ~35 GB; leaves GPU 1 free for HarmBench
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model "NousResearch/Hermes-3-Llama-3.1-70B" \
    --quantization bitsandbytes \
    --load-format bitsandbytes \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --served-model-name hermes-attacker \
    --port 8001 &
ATTACKER_PID=$!

# ── vLLM: HarmBench classifier on GPU 1 ──────────────────────────────────────
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model "cais/HarmBench-Llama-2-13b-cls" \
    --served-model-name harmbench-cls \
    --port 8002 &
HARMBENCH_PID=$!

wait_for_server 8001 "attacker"
wait_for_server 8002 "harmbench"

# ── LiteLLM proxy: single OPENAI_BASE_URL routes by model name ───────────────
cat > /tmp/litellm_${SLURM_JOB_ID}.yaml <<EOF
model_list:
  - model_name: hermes-attacker
    litellm_params:
      model: openai/hermes-attacker
      api_base: http://localhost:8001/v1
      api_key: dummy
  - model_name: harmbench-cls
    litellm_params:
      model: openai/harmbench-cls
      api_base: http://localhost:8002/v1
      api_key: dummy
EOF

litellm --config "/tmp/litellm_${SLURM_JOB_ID}.yaml" --port 8000 &
PROXY_PID=$!
wait_for_server 8000 "litellm-proxy"

export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="dummy"

# ── Run eval ──────────────────────────────────────────────────────────────────
inspect eval tasks/evals/socialharmbench.py@social_harm_bench_adversarial \
    --model "openrouter/anthropic/claude-haiku-4-5" \
    --max-connections 10 \
    --limit 20 \
    --log-dir out/ \
    -T attacks="["cipher", "deep_inception", "pair", "crescendo"]"\
    -T "grader=openrouter/google/gemini-flash-1.5" \
    -T "attacker=openai/hermes-attacker" \
    -T "llamaguard_model=openrouter/meta-llama/llama-guard-4-12b" \
    -T "harmbench_model=openai/harmbench-cls" \
    "$@"
