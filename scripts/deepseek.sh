
uv run certify.py \
	--model "openrouter/deepseek/deepseek-v4-pro" \
	--name "DeepSeek V4 Pro" \
	--provider "DeepSeek" \
	--region "USA Frontier Models" \
	--specialty "Advanced code generation and reasoning"

uv run certify.py \
	--model "openrouter/deepseek/deepseek-v4-flash" \
	--name "DeepSeek V4 Flash" \
	--provider "DeepSeek" \
	--region "USA Frontier Models" \
	--specialty "Fast code generation and reasoning"

uv run certify.py \
	--model "openrouter/deepseek/deepseek-v3.2" \
	--name "DeepSeek V3.2" \
	--provider "DeepSeek" \
	--region "USA Frontier Models" \
	--specialty "Code generation and reasoning"

uv run certify.py \
	--model "openrouter/deepseek/deepseek-v3.2-speciale" \
	--name "DeepSeek V3.2 Speciale" \
	--provider "DeepSeek" \
	--region "USA Frontier Models" \
	--specialty "Specialized code generation and reasoning"
