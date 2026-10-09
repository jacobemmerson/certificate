"""Serve the Hermes-4-70B attacker on Modal as an OpenAI-compatible vLLM endpoint.

Modal only hosts the model. screen_answerability.py and generate.py run from
this checkout and reach it via --model-base-url; scripts/generate_hermes_modal.sh
does the whole job. Like vllm on slurm, modal runs through uvx and stays out of
uv.lock (see the venv-drift note in pyproject.toml).

    uvx modal secret create hermes-vllm VLLM_API_KEY=$(openssl rand -hex 32)  # once
    uvx modal deploy scripts/hermes_modal.py
    uvx modal app stop hermes-vllm

The endpoint is https://<workspace>--hermes-vllm-serve.modal.run; deploy prints it.
Cost: 4xH100 bill for as long as a container is warm, which lasts until 15 min
after the last request. The first start downloads ~140 GB of weights into the
hf-cache volume, so later starts are much faster.
"""
import os
import subprocess

import modal

MODEL = "NousResearch/Hermes-4-70B"
MINUTES = 60

# The CUDA devel base (as in Modal's vLLM example) rather than debian_slim:
# flashinfer JIT-compiles sampling kernels and needs nvcc, the same reason the
# slurm script loads cuda/12.6. vllm is pinned to what that example pins (Oct 2026).
image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

app = modal.App("hermes-vllm")


# timeout bounds each HTTP request, not the container: the container lives
# while requests keep arriving and stops scaledown_window after the last one.
@app.function(
    image=image,
    gpu="H100:4",
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("hermes-vllm")],
    timeout=60 * MINUTES,
    scaledown_window=15 * MINUTES,
)
@modal.concurrent(max_inputs=64)
@modal.web_server(port=8000, startup_timeout=40 * MINUTES)
def serve():
    subprocess.Popen([
        "vllm", "serve", MODEL,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--tensor-parallel-size", "4",
        "--gpu-memory-utilization", "0.90",
        "--api-key", os.environ["VLLM_API_KEY"],
    ])
