'''
author: @tae

Utilities for grader and model loading
TODO: rename file to something more fitting since this is general utilties
'''

from pathlib import Path
import fcntl
import json
import os
import re
import tempfile

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = REPO_ROOT / "models"


def model_result_path(model_id: str) -> Path:
    """`models/results/<slug>.json`: the one file a certification run writes."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model_id)
    return MODELS_DIR / "results" / f"{slug}.json"


def write_json_atomic(path: Path, obj) -> None:
    """Write via a sibling tmp file + os.replace so a killed process (slurm
    preemption, Ctrl-C) leaves the previous file intact, never a half-written one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=4)
            f.write("\n")
        os.chmod(tmp, 0o644)  # mkstemp makes 0600, which os.replace would carry over
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


def load_model_results() -> list[dict]:
    """Every per-model record, sorted by id (case-insensitive) — the order models.json is rebuilt in."""
    results_dir = MODELS_DIR / "results"
    if not results_dir.is_dir():
        return []
    records = [json.loads(p.read_text()) for p in results_dir.glob("*.json")]
    return sorted(records, key=lambda m: m["id"].lower())


def rebuild_models_json() -> Path:
    """models.json is derived: rebuild it from the per-model files, atomically."""
    path = MODELS_DIR / "models.json"
    lock = MODELS_DIR / "results" / ".rebuild.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    # Without it, a rebuild that read results/ before another job's file existed
    # can os.replace last and drop that job's model from models.json.
    with open(lock, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        write_json_atomic(path, load_model_results())
    return path

def load_graders(path: str | Path | None = None) -> list[str]:
    """Load grader model names from a text file (one per line, # comments ignored)."""
    if path is None:
        path = REPO_ROOT / "GRADERS.md"
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

def validate_graders(graders: str | list[str]) -> None:
    '''
    Fail fast unless every grader model answers, before any eval spends money.

    A misconfigured judge is the worst failure this suite has. It does not stop
    the run — the model under test answers fine — so every sample errors on the
    *scoring* call and the run dies after paying for all of it, with a 400
    buried inside an escaped request dump in the .eval file. And if a bad judge
    returned garbage rather than erroring, every sample would score as an
    abstention, which is the safe end: a broken grader would report a perfect
    certification. One trivial call each is cheap insurance against both.
    '''
    import asyncio

    from inspect_ai._util.dotenv import init_dotenv
    from inspect_ai.model import get_model

    # Inspect loads .env inside eval()/the CLI, and this check runs before
    # either. Without this the preflight reports a missing API key for every
    # provider whose credentials live in .env — turning a guard against
    # misconfiguration into a source of it.
    init_dotenv()

    names = graders if isinstance(graders, list) else [graders]

    async def check(name: str) -> str | None:
        try:
            await get_model(name).generate("Reply with the single word: ok")
            return None
        except Exception as exc:
            return f"  {name}: {type(exc).__name__}: {str(exc).strip().splitlines()[-1][:200]}"

    failures = [f for f in asyncio.run(_gather(check, names)) if f]
    if failures:
        raise SystemExit(
            "Grader model(s) unusable — no evals were started:\n"
            + "\n".join(failures)
            + f"\n\nGraders come from --grader, or {REPO_ROOT / 'GRADERS.md'} when it is omitted."
        )


def validate_target(model: str, model_args: dict | None = None) -> None:
    '''
    Fail fast unless the model under test answers with something.

    The mirror of validate_graders, and for the same reason. A model that
    returns an empty completion does not error: every sample completes, every
    judge abstains, the run is recorded as a success and the score comes out
    -1. meta/muse-spark-1.2 did exactly that — four clusters, 1188 samples and
    the whole grader bill, to discover the endpoint answers `content: null` to
    everything including "hi".

    `model_args` carries the run's provider routing, so the canary is answered
    by the same endpoint the eval will be pinned to: a model that works on one
    endpoint and is mute on another must fail here rather than later.
    '''
    import asyncio

    from inspect_ai._util.dotenv import init_dotenv
    from inspect_ai.model import get_model

    init_dotenv()

    async def ask() -> str:
        return (await get_model(model, **(model_args or {})).generate(
            "Reply with the single word: ok"
        )).completion

    try:
        completion = asyncio.run(ask())
    except Exception as exc:
        raise SystemExit(
            f"Model under test unusable — no evals were started:\n"
            f"  {model}: {type(exc).__name__}: {str(exc).strip().splitlines()[-1][:200]}"
        )

    if not completion.strip():
        raise SystemExit(
            f"Model under test returned an empty completion — no evals were started:\n"
            f"  {model} answered a trivial prompt with nothing.\n\n"
            "Every sample would complete, every judge would abstain, and the run would be "
            "recorded as a success scoring -1. Check the model serves this endpoint at all."
        )


async def _gather(fn, items):
    import asyncio

    return await asyncio.gather(*(fn(item) for item in items))


def load_models_with_check(model_id: str | None = None) -> tuple[list[dict], int]:
    '''
    Return the models list and the index of `model_id` within it (-1 if not
    found, or if no model_id is given).
    '''
    models = load_model_results()
    if model_id:
        for i, m in enumerate(models):
            if m['id'] == model_id:
                return models, i
    return models, -1
