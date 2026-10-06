"""
One-off: split models/models.json into models/results/<slug>.json and rebuild
models.json from them. Every field of each record is kept (aa_* included).
Refuses to run once models/results/ has any record: those files are the source
of truth, and an older models.json (a git checkout, a stash) would overwrite them.

    uv run python3 scripts/split_models_json.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.utils.graders import MODELS_DIR, model_result_path, rebuild_models_json, write_json_atomic

if __name__ == "__main__":
    if any((MODELS_DIR / "results").glob("*.json")):
        sys.exit(f"{MODELS_DIR / 'results'} already has per-model files, which are the source of "
                 "truth. Delete them first if you really mean to re-split models.json.")
    for model in json.loads((MODELS_DIR / "models.json").read_text()):
        write_json_atomic(model_result_path(model["id"]), model)
    print(f"rebuilt {rebuild_models_json()}")
