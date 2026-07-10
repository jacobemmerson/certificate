"""Sentence-transformer embedder for Concordia's associative memory.

Concordia needs a `Callable[[str], np.ndarray]` to embed memories. We load the
same model as `con_demo.ipynb` (all-mpnet-base-v2) once at module level so batch
simulation runs pay the load cost a single time.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
_model = None


def get_embedder() -> Callable[[str], np.ndarray]:
    """Return a cached embedding function, loading the model on first call."""
    global _model
    if _model is None:
        import sentence_transformers

        _model = sentence_transformers.SentenceTransformer(_MODEL_NAME)
    return lambda text: _model.encode(text, show_progress_bar=False)
