from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def load_graders(path: str | Path | None = None) -> list[str]:
    """Load grader model names from a text file (one per line, # comments ignored)."""
    if path is None:
        path = REPO_ROOT / "graders.txt"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Graders file not found: {path}")
    models = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not models:
        raise ValueError(f"No grader models found in {path}")
    return models
