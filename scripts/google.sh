
uv run certify.py \
	--model "openrouter/google/gemini-3-flash-preview" \
	--name "Gemini 3 Flash" \
	--provider "Google" \
	--region "USA Frontier Models" \
	--speciality "Fast multimodal reasoning"

uv run certify.py \
	--model "openrouter/google/gemini-3.1-pro-preview" \
	--name "Gemini 3.1 Pro" \
	--provider "Google" \
	--region "USA Frontier Models" \
	--speciality "Multimodal understanding and reasoning"
