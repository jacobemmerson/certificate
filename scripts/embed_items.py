'''
Embed the texts prepare.py could not find in the embedding cache.

    uv run --no-project --with sentence-transformers --with numpy \
        python scripts/embed_items.py [--model <hf id>]

Runs in a throwaway env on purpose: sentence-transformers pulls torch, which
must never enter the project's uv.lock (spec §1.1). Reads
datasets/cache/<name>.embed_input.jsonl ({key, text}, written by prepare.py on
a cache miss) for name in items (every source's queries) and leaves (every
leaf's anchors), encodes only the keys the cache lacks, and rewrites
datasets/cache/embeddings/<name>.npz (keys, float16 unit vectors, model). CPU is
fine: all-MiniLM-L6-v2 embeds ~100k short texts in minutes; for a larger model use
scripts/embed_modal.py (a Modal L4). --model keeps a
second cache under embeddings/<model slug>/ for a comparison (prepare.py
--embedding-model reads it); the default model's cache stays at the top level.
'''
import argparse
import json
import os
from pathlib import Path

import numpy as np

# Kept equal to datasets/prepare/cluster/prepare.py::EMBEDDING_MODEL (tested).
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR = Path(__file__).resolve().parent.parent / "datasets" / "cache"
NAMES = ("items", "leaves")


def cache_path(cache_dir: Path, name: str, model: str) -> Path:
    '''embeddings/<name>.npz for the default model, embeddings/<slug>/<name>.npz otherwise.'''
    base = cache_dir / "embeddings"
    if model != MODEL:
        base = base / model.replace("/", "__")
    return base / f"{name}.npz"


def embed(name: str, encode, cache_dir: Path = CACHE_DIR, model: str = MODEL) -> int:
    '''Add vectors for every input key the cache lacks; returns how many were added.'''
    path = cache_path(cache_dir, name, model)
    keys, vectors = [], np.zeros((0, 0), dtype=np.float16)
    if path.exists():
        with np.load(path) as data:
            # Vectors from another model are not comparable: start over.
            if str(data["model"]) == model:
                keys, vectors = data["keys"].tolist(), data["vectors"]
    known = set(keys)
    todo = {}
    input_path = cache_dir / f"{name}.embed_input.jsonl"
    if not input_path.exists():
        return 0
    with open(input_path, encoding="utf-8") as f:
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
    tmp = path.with_name(f"{name}.tmp.npz")
    np.savez(tmp, keys=np.array(keys + list(todo), dtype=str), vectors=vectors,
             model=np.array(model))
    os.replace(tmp, path)
    return len(todo)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL, help="sentence-transformers model id")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer  # throwaway env only

    model = SentenceTransformer(args.model)
    for name in NAMES:
        added = embed(name, lambda texts: model.encode(
            texts, batch_size=256, show_progress_bar=True), model=args.model)
        print(f"{name}: embedded {added} new texts")


if __name__ == "__main__":
    main()
