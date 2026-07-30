uv run python generate.py \
    --attacker "openrouter/meta-llama/llama-4-maverick" \
    --max-connections 32 \
    --perturb-k 1 \
    --simulate --sim-k 1 \
    --limit 5 \
    --force
