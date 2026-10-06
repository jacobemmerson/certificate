'''Shared vocabulary for datasets/raw/manifest.toml (see scripts/fetch_raw.py).'''
import tomllib
from pathlib import Path

STATUSES = ("registered", "vendored", "prompt", "partial", "agentic", "human_study", "unreleased")
HOSTS = ("github", "hf", "none")
REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = REPO_ROOT / "datasets" / "raw" / "manifest.toml"


def load_manifest(path: Path) -> list[dict]:
    with open(path, "rb") as handle:
        return tomllib.load(handle).get("benchmark", [])
