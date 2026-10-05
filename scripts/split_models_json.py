"""
One-off: split models/models.json into models/results/<slug>.json and rebuild
models.json from them. Safe to run twice — the second run rewrites identical
files. Every field of each record is kept (aa_* included).

    uv run python3 scripts/split_models_json.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.utils.graders import MODELS_DIR, model_result_path, rebuild_models_json, write_json_atomic

if __name__ == "__main__":
    for model in json.loads((MODELS_DIR / "models.json").read_text()):
        write_json_atomic(model_result_path(model["id"]), model)
    print(f"rebuilt {rebuild_models_json()}")
