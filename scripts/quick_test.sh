uv run certify.py \
    --model "openrouter/openai/gpt-4o-mini" \
    --grader "openrouter/openai/gpt-4o-mini" \
    --attacker "openrouter/openai/gpt-4o-mini" \
    --only "harm" \
    --limit 10 \
    --simulate
