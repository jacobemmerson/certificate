
uv run certify.py \
	--model "openrouter/deepseek/deepseek-v4-pro" \
	--name "DeepSeek V4 Pro" \
	--provider "DeepSeek" \
	--region "USA Frontier Models" \
	--speciality "Advanced code generation and reasoning"

uv run certify.py \
	--model "openrouter/deepseek/deepseek-v4-flash" \
	--name "DeepSeek V4 Flash" \
	--provider "DeepSeek" \
	--region "USA Frontier Models" \
	--speciality "Fast code generation and reasoning"

uv run certify.py \
	--model "openrouter/deepseek/deepseek-v3.2" \
	--name "DeepSeek V3.2" \
	--provider "DeepSeek" \
	--region "USA Frontier Models" \
	--speciality "Code generation and reasoning"

uv run certify.py \
	--model "openrouter/deepseek/deepseek-v3.2-speciale" \
	--name "DeepSeek V3.2 Speciale" \
	--provider "DeepSeek" \
	--region "USA Frontier Models" \
	--speciality "Specialized code generation and reasoning"
