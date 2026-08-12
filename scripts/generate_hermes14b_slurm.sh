#!/bin/bash
#SBATCH --job-name=perturbation-gen-hermes14b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --requeue

# Verification/smoke-test variant of generate_hermes_slurm.sh: same pipeline,
# but with NousResearch/Hermes-4-14B (the smallest model in the Hermes-4
# line) instead of the 70B, so it needs a fraction of the resources and
# should clear the queue much faster. Use this to sanity-check the
# generation pipeline end-to-end before burning a 4xH100 allocation on the
# real 70B run.
#
# vLLM is deliberately NOT a project dependency (it pins its own torch and
# would force a torch upgrade on the whole locked env — see the venv-drift
# note in pyproject.toml). Instead, `uvx` runs `vllm serve` in its own
# isolated environment, and generate.py talks to it over HTTP via
# --model-base-url; inspect's vllm/ provider only needs the vllm package
# when it launches the server itself.
#
# Sizing: 14B BF16 weights are ~28 GB, so tensor_parallel_size=1 on a single
# H100-80GB leaves ~50 GB for KV cache — no multi-GPU coordination needed.

set -euo pipefail
# Under sbatch, $0 points at slurm's spooled copy of this script, not the
# repo — so navigate from the submit directory (submit from the repo root).
cd "$SLURM_SUBMIT_DIR"

# flashinfer JIT-compiles its sampling kernels on first use and needs
# nvcc/CUDA_HOME on the compute node — without this the workers die with
# "Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist".
module load cuda/12.6

# Adjust to your cluster's convention (module load cuda/12.4, conda, etc.).
# Caches belong on scratch, not $HOME: HF weights are ~140 GB and uvx
# materializes vllm's own env (torch etc., several GB) under UV_CACHE_DIR.
export HF_HOME="$HOME/scratch/hf_cache"
export UV_CACHE_DIR="$HOME/scratch/uv_cache"
# export HF_TOKEN=...            # only needed for gated repos; Hermes-4 is open

MODEL="NousResearch/Hermes-4-14B"
PORT=8000

# ---- vLLM server (isolated env — leaves the project's uv.lock untouched) ---
uvx vllm serve "$MODEL" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 \
    --port "$PORT" \
    > "logs/vllm-$SLURM_JOB_ID.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

# Smaller weights than the 70B, but still give it room on a cold cache
# (first run downloads weights + builds the vllm env).
for i in $(seq 1 40); do
    curl -sf "http://localhost:$PORT/health" > /dev/null && break
    kill -0 $SERVER_PID 2>/dev/null || { echo "vLLM died — see logs/vllm-$SLURM_JOB_ID.log"; exit 1; }
    sleep 30
done
curl -sf "http://localhost:$PORT/health" > /dev/null || { echo "vLLM never became healthy"; exit 1; }

# ---- artifact generation ----------------------------------------------------
# --missing-only makes this safe to requeue after preemption or timeout:
# finished families are skipped, interrupted ones are filled in and merged.
# generate.py exits nonzero if the attacker produces no usable output.
#
# --only manipulation --limit 5 keeps this a smoke test, not a full artifact refresh
# (limited artifacts are marked partial and will fail certify.py's full-run
# validation — that's expected here, this run is for pipeline verification
# only, not for producing artifacts certify.py will replay for real).
uv run python generate.py \
    --attacker "vllm/$MODEL" \
    --model-base-url "http://localhost:$PORT/v1" \
    --max-connections 32 \
    --perturb-k 1 \
    --simulate --sim-k 1 \
    --reasoning \
    --only manipulation \
    --limit 5 \
    --force
