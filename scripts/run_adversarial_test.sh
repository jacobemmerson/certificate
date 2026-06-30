#!/bin/bash
#SBATCH --job-name=shb-test
#SBATCH --nodes=1
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm_test_%j.out
#SBATCH --error=logs/slurm_test_%j.err

export HF_HOME="${SCRATCH:-/scratch/$USER}/hf_cache"
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:?}"

cd "$SLURM_SUBMIT_DIR"
uv venv .venv --quiet 2>/dev/null || true
uv pip install -r requirements-eval.txt --quiet
source .venv/bin/activate
mkdir -p logs

inspect eval tasks/evals/socialharmbench.py@social_harm_bench_adversarial \
    --model "openrouter/anthropic/claude-haiku-4-5" \
    --max-connections 5 \
    --limit 5 \
    --log-dir logs/ \
    -T "attacker=hf/Qwen/Qwen2.5-1.5B-Instruct" \
    -T "llamaguard_model=openrouter/meta-llama/llama-guard-4-12b" \
    "$@"
