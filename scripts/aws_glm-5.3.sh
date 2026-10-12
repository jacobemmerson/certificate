#!/usr/bin/env bash
# GLM 5.3 on AWS Bedrock (model id zai.glm-5.3, on-demand). Needs
# AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY in .env and model access granted
# to the account in the Bedrock console (Model access -> Z.ai -> GLM 5.3);
# without it every call fails with "zai.glm-5.3 is not available for this
# account". Region defaults to us-east-1: AWS_REGION=us-west-2 bash scripts/aws_glm-5.3.sh
#
#   bash scripts/aws_glm-5.3.sh                 # full run
#   bash scripts/aws_glm-5.3.sh --limit 2 --only loc_deception   # smoke test
set -uo pipefail
cd "$(dirname "$0")/.."
export AWS_DEFAULT_REGION="${AWS_REGION:-us-east-1}"
uv run certify.py \
	--model "bedrock/zai.glm-5.3" \
	--name "GLM 5.3" \
	--provider "Z.ai" \
	--region "China" \
	--specialty "Agentic coding and reasoning" \
	--simulate --max-tasks 21 "$@"
