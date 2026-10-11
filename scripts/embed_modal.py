'''
Fill the embedding cache on a Modal GPU instead of the local CPU, for embedders
too slow to run here (the smoke test of alternatives to MiniLM):

    uv run modal run scripts/embed_modal.py --model Qwen/Qwen3-Embedding-0.6B

Same inputs and outputs as scripts/embed_items.py (which it reuses): reads
datasets/cache/{items,leaves}.embed_input.jsonl, writes the cache under
datasets/cache/embeddings/<model slug>/. Texts are encoded in chunks across
containers; the local side only concatenates. HF weights land in the shared
hf-cache volume so a rerun skips the download.
'''
import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent))

CHUNK = 8192
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("sentence-transformers>=5.0", "torch", "numpy")
)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
app = modal.App("embed-smoke")


@app.function(image=image, gpu="L4", volumes={"/root/.cache/huggingface": hf_cache},
              timeout=60 * 60)
def encode(model: str, texts: list[str]) -> list[list[float]]:
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(model, trust_remote_code=True)
    # Scoring texts include system prompts of several thousand tokens; 512 is
    # what MiniLM/e5 see anyway, and keeps a 0.6B model inside an L4.
    encoder.max_seq_length = 512
    return encoder.encode(texts, batch_size=32, normalize_embeddings=True).tolist()


@app.local_entrypoint()
def main(model: str):
    import numpy as np
    from embed_items import NAMES, embed

    def encode_all(texts: list[str]):
        chunks = [texts[i:i + CHUNK] for i in range(0, len(texts), CHUNK)]
        return np.concatenate([np.asarray(part, dtype=np.float32)
                               for part in encode.map([model] * len(chunks), chunks)])

    for name in NAMES:
        print(f"{name}: embedded {embed(name, encode_all, model=model)} new texts with {model}")
