'''
Embed the texts prepare.py could not find in the embedding cache.

    uv run --no-project --with sentence-transformers --with numpy \
        python scripts/embed_items.py --risk cbrn [--risk cyber ...]

Runs in a throwaway env on purpose: sentence-transformers pulls torch, which
must never enter the project's uv.lock (spec §1.1). Reads
datasets/cache/<risk>.embed_input.jsonl ({key, text}, written by prepare.py on
a cache miss), encodes only the keys the cache lacks, and rewrites
datasets/cache/embeddings/<risk>.npz (keys, float16 unit vectors, model). CPU is
fine: all-MiniLM-L6-v2 embeds ~100k short texts in minutes.
'''
import argparse
import json
import os
from pathlib import Path

import numpy as np

# Kept equal to datasets/prepare/cluster/prepare.py::EMBEDDING_MODEL (tested).
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR = Path(__file__).resolve().parent.parent / "datasets" / "cache"


def embed(risk: str, encode, cache_dir: Path = CACHE_DIR) -> int:
    '''Add vectors for every input key the cache lacks; returns how many were added.'''
    path = cache_dir / "embeddings" / f"{risk}.npz"
    keys, vectors = [], np.zeros((0, 0), dtype=np.float16)
    if path.exists():
        with np.load(path) as data:
            # Vectors from another model are not comparable: start over.
            if str(data["model"]) == MODEL:
                keys, vectors = data["keys"].tolist(), data["vectors"]
    known = set(keys)
    todo = {}
    with open(cache_dir / f"{risk}.embed_input.jsonl", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record["key"] not in known:
                todo[record["key"]] = record["text"]
    if not todo:
        return 0

    new = np.asarray(encode(list(todo.values())), dtype=np.float32)
    new /= np.linalg.norm(new, axis=1, keepdims=True)
    vectors = np.concatenate([vectors.reshape(-1, new.shape[1]), new.astype(np.float16)])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{risk}.tmp.npz")
    np.savez(tmp, keys=np.array(keys + list(todo), dtype=str), vectors=vectors,
             model=np.array(MODEL))
    os.replace(tmp, path)
    return len(todo)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--risk", action="append", required=True)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer  # throwaway env only

    model = SentenceTransformer(MODEL)
    for risk in args.risk:
        added = embed(risk, lambda texts: model.encode(
            texts, batch_size=256, show_progress_bar=True))
        print(f"{risk}: embedded {added} new texts")


if __name__ == "__main__":
    main()
