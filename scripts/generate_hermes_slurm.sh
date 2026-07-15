#!/bin/bash
#SBATCH --job-name=perturbation-gen-hermes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --requeue

# Generates ALL stage-2 perturbation artifacts (paraphrase, register,
# identity_strip, framing) plus stage-3 scenario reframings with a locally
# served Hermes-4-70B attacker.
#
# Sizing: 70B FP16 weights are ~140 GB, so tensor_parallel_size=4 on a
# 4xH100-80GB node (320 GB) leaves ~150 GB for KV cache. inspect's vllm/
# provider launches the vLLM server on the allocated GPUs and tears it down
# at exit — no separate serving job or Ray cluster needed at this size.

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

# Adjust to your cluster's convention (module load cuda/12.4, conda, etc.).
# HF caches belong on scratch, not $HOME — the weights are ~140 GB.
export HF_HOME="$HOME/scratch/hf_cache"
# export HF_TOKEN=...            # only needed for gated repos; Hermes-4 is open

MODEL="NousResearch/Hermes-4-70B"

# --missing-only makes this safe to requeue after preemption or timeout:
# finished families are skipped, interrupted ones are filled in and merged.
uv run generate.py \
    --attacker "vllm/$MODEL" \
    -M tensor_parallel_size=4 \
    -M gpu_memory_utilization=0.90 \
    --max-connections 32 \
    --perturb-k 1 \
    --simulate --sim-k 1 \
    --missing-only
