#!/bin/bash
#SBATCH --job-name=shb-test
#SBATCH --nodes=1
#SBATCH --gpus=h100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
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
# Servers are launched via `setsid` so each owns its own process group; a
# timed-out or stuck vLLM process (e.g. blocked in a long model-load/compile
# call, not yet checking for signals) won't survive as an orphan burning GPU
# and wall-clock for the rest of the job — TERM is followed by a KILL to the
# whole group after a grace period.
PROXY_PID="" ATTACKER_PID="" HARMBENCH_PID=""
cleanup() {
    local pid
    for pid in "$PROXY_PID" "$ATTACKER_PID" "$HARMBENCH_PID"; do
        [[ -n "$pid" ]] && kill -TERM -- "-$pid" 2>/dev/null || true
    done
    sleep 5
    for pid in "$PROXY_PID" "$ATTACKER_PID" "$HARMBENCH_PID"; do
        [[ -n "$pid" ]] && kill -KILL -- "-$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ── Helper ────────────────────────────────────────────────────────────────────
wait_for_server() {
    local port=$1 name=$2 timeout=${3:-600}
    echo "Waiting for $name on port $port (timeout ${timeout}s)..."
    local deadline=$(( $(date +%s) + timeout ))
    until curl -sf "http://localhost:$port/health" > /dev/null 2>&1; do
        if [[ $(date +%s) -gt $deadline ]]; then
            echo "ERROR: $name did not start within $((timeout / 60)) minutes" >&2; exit 1
        fi
        sleep 5
    done
    echo "  $name ready"
}

# ── vLLM: Hermes-3-70B (attacker) on GPU 0, bitsandbytes on-the-fly quant ────
# NB: "Hermes-3-Llama-3.1-70B-AWQ" does not exist under NousResearch on HF —
# don't switch to it without first confirming a real repo. On-the-fly
# bitsandbytes quantization is slow (~11 min just for weight loading, observed
# in slurm_test_17299918), so the health-check/job timeouts below are sized
# with that in mind.
CUDA_VISIBLE_DEVICES=0 setsid python -m vllm.entrypoints.openai.api_server \
    --model "NousResearch/Hermes-3-Llama-3.1-70B" \
    --quantization bitsandbytes \
    --load-format bitsandbytes \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --served-model-name hermes-attacker \
    --port 8001 &
ATTACKER_PID=$!

# ── vLLM: HarmBench classifier on GPU 1 ──────────────────────────────────────
# cais/HarmBench-Llama-2-13b-cls ships no chat_template (it's a completion-style
# classifier, not a chat model), so vLLM's chat/responses endpoints 400 on every
# request without one. _HARMBENCH_PROMPT in tasks/scorers/harm.py already sends
# the fully pre-formatted official HarmBench prompt as a single message, so the
# template must pass it through unchanged rather than re-wrapping it.
CUDA_VISIBLE_DEVICES=1 setsid python -m vllm.entrypoints.openai.api_server \
    --model "cais/HarmBench-Llama-2-13b-cls" \
    --served-model-name harmbench-cls \
    --chat-template scripts/harmbench_chat_template.jinja \
    --port 8002 &
HARMBENCH_PID=$!

wait_for_server 8001 "attacker" 1200
wait_for_server 8002 "harmbench" 600

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

setsid litellm --config "/tmp/litellm_${SLURM_JOB_ID}.yaml" --port 8000 &
PROXY_PID=$!
wait_for_server 8000 "litellm-proxy" 300

export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="dummy"
# Force inspect-ai to use Chat Completions instead of the newer Responses API,
# which vLLM does not support (/v1/responses returns 400).
export INSPECT_OPENAI_FORCE_LEGACY_COMPLETIONS=1

# ── Run eval ──────────────────────────────────────────────────────────────────
inspect eval tasks/evals/socialharmbench.py@social_harm_bench_adversarial \
    --model "openrouter/anthropic/claude-haiku-4-5" \
    --max-connections 3 \
    --limit 3 \
    --log-dir out/ \
    -T attacks="["cipher", "deep_inception", "pair", "crescendo"]"\
    -T "grader=openrouter/google/gemini-2.5-flash" \
    -T "attacker=openai/hermes-attacker" \
    -T "llamaguard_model=openrouter/meta-llama/llama-guard-4-12b" \
    -T "harmbench_model=openai/harmbench-cls" \
    "$@"
