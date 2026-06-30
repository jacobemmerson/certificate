"""Download the HarmBench classifier (cais/HarmBench-Llama-2-13b-cls) from HuggingFace.

This model requires ~26 GB of GPU VRAM and is used by harmbench_classifier_scorer.
Run once before evaluating on the cluster:

    python scripts/download_harmbench_cls.py

The model will be saved to models/harmbench_cls/ and picked up automatically
by harmbench_classifier_scorer(model_path="models/harmbench_cls").
"""
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_REPO = "cais/HarmBench-Llama-2-13b-cls"
LOCAL_DIR = Path(__file__).resolve().parent.parent / "models" / "harmbench_cls"


def main() -> None:
    print(f"Downloading {MODEL_REPO} → {LOCAL_DIR}")
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=MODEL_REPO, local_dir=str(LOCAL_DIR))
    print(f"Done. Model saved to {LOCAL_DIR}")


if __name__ == "__main__":
    main()
