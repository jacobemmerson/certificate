#!/bin/bash
#SBATCH --job-name=perturbation-gen-hermes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128000
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --requeue

# Generates ALL stage-2 perturbation artifacts (paraphrase, register,
# identity_strip, framing) plus stage-3 scenario reframings with a locally
# served Hermes-4-70B attacker.
#
# vLLM is deliberately NOT a project dependency (it pins its own torch and
# would force a torch upgrade on the whole locked env — see the venv-drift
# note in pyproject.toml). Instead, `uvx` runs `vllm serve` in its own
# isolated environment, and generate.py talks to it over HTTP via
# --model-base-url; inspect's vllm/ provider only needs the vllm package
# when it launches the server itself.
#
# Sizing: 70B FP16 weights are ~140 GB, so tensor_parallel_size=4 on a
# 4xH100-80GB node (320 GB) leaves ~150 GB for KV cache.

set -euo pipefail
# Under sbatch, $0 points at slurm's spooled copy of this script, not the
# repo — so navigate from the submit directory (submit from the repo root).
cd "$SLURM_SUBMIT_DIR"

# Adjust to your cluster's convention (module load cuda/12.4, conda, etc.).
# Caches belong on scratch, not $HOME: HF weights are ~140 GB and uvx
# materializes vllm's own env (torch etc., several GB) under UV_CACHE_DIR.
export HF_HOME="$HOME/scratch/hf_cache"
export UV_CACHE_DIR="$HOME/scratch/uv_cache"
# export HF_TOKEN=...            # only needed for gated repos; Hermes-4 is open

MODEL="NousResearch/Hermes-4-70B"
PORT=8000

# ---- vLLM server (isolated env — leaves the project's uv.lock untouched) ---
uvx vllm serve "$MODEL" \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.90 \
    --port "$PORT" \
    > "logs/vllm-$SLURM_JOB_ID.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

# First run downloads weights + builds the vllm env; give it up to ~40 min.
for i in $(seq 1 80); do
    curl -sf "http://localhost:$PORT/health" > /dev/null && break
    kill -0 $SERVER_PID 2>/dev/null || { echo "vLLM died — see logs/vllm-$SLURM_JOB_ID.log"; exit 1; }
    sleep 30
done
curl -sf "http://localhost:$PORT/health" > /dev/null || { echo "vLLM never became healthy"; exit 1; }

# ---- artifact generation ----------------------------------------------------
# --missing-only makes this safe to requeue after preemption or timeout:
# finished families are skipped, interrupted ones are filled in and merged.
# generate.py exits nonzero if the attacker produces no usable output.
uv run python generate.py \
    --attacker "vllm/$MODEL" \
    --model-base-url "http://localhost:$PORT/v1" \
    --max-connections 32 \
    --perturb-k 1 \
    --simulate --sim-k 1 \
    --reasoning \
    --only auth \
    --limit 5 \
    --force
