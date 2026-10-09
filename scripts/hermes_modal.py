"""Serve the Hermes-4-70B attacker on Modal as an OpenAI-compatible vLLM endpoint.

Modal only hosts the model. screen_answerability.py and generate.py run from
this checkout and reach it via --model-base-url; scripts/generate_hermes_modal.sh
does the whole job. modal is a project dependency; vllm, as on slurm, stays out of
uv.lock (see the venv-drift note in pyproject.toml).

    export VLLM_API_KEY=$(openssl rand -hex 32)   # once; save it (e.g. in .env), clients need it
    uv run modal secret create hermes-vllm VLLM_API_KEY="$VLLM_API_KEY"
    uv run modal run --detach scripts/hermes_modal.py::serve
    uv run modal app stop -y <HERMES_APP>

The server sits behind a modal.forward tunnel, not a web endpoint: Modal cuts
web-endpoint requests at 150 s, and reasoning-mode scenario calls run longer.
`modal run` streams the container's stdout, which prints HERMES_URL=<tunnel
URL> and HERMES_APP=<app id>. --detach keeps the container up after the local
client exits, until `modal app stop` or the 24 h function timeout. Stop it by
the app id: `modal app stop <name>` only finds deployed apps, not this
ephemeral one (`uv run modal app list` also shows it).

Cost: 4xH100 bill for the whole time the container is up, busy or idle
(roughly $16/h at ~$4 per H100-hour; see modal.com/pricing). The first start
downloads ~140 GB of weights into the hf-cache volume, so later starts are much
faster.
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
# torch.compile and CUDA-graph artifacts, so cold starts after the first skip them.
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("hermes-vllm")


@app.function(
    image=image,
    gpu="H100:4",
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    secrets=[modal.Secret.from_name("hermes-vllm")],
    timeout=24 * 60 * MINUTES,
    max_containers=1,
)
def serve():
    # vLLM 0.21 on H100 already enables prefix caching and allows 1024 concurrent
    # seqs; it caps per-step tokens at 8192, which 32768 raises so long prompts
    # prefill in fewer steps. 16384 covers the reasoning path's max_tokens=8192
    # plus the prompts.
    proc = subprocess.Popen([
        "vllm", "serve", MODEL,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--tensor-parallel-size", "4",
        "--gpu-memory-utilization", "0.92",
        "--max-model-len", "16384",
        "--max-num-batched-tokens", "32768",
        "--api-key", os.environ["VLLM_API_KEY"],
    ])
    with modal.forward(8000) as tunnel:
        print(f"HERMES_URL={tunnel.url}", flush=True)
        print(f"HERMES_APP={app.app_id}", flush=True)
        proc.wait()
