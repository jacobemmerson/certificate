"""The on-disk store for pregenerated perturbation/simulation artifacts.

Stage 2's rewrite families (paraphrase, register, identity_strip), the
deterministic framing family, and stage 3's scenario reframings are generated
*once* by `generate.py` (running the attacker model) and persisted under
`datasets/generated/<task_name>/<family>.jsonl` — see
datasets/generated/README.md for the schema. At eval time, certify.py replays
those fixed artifacts against the target model
(pipeline/stage2_perturbation/solvers.py), so every evaluated model sees the
exact same variants. Only `reconsideration` has no artifact file: it
challenges the target's own control completion, so it can only exist live.

This module is the single owner of the artifact paths, JSONL format, and the
pre-run validation certify.py uses to fail fast (with the generate.py command
to run) before any eval spends money.
"""
from __future__ import annotations

import json
from pathlib import Path

from inspect_ai import Task
from inspect_ai._util.registry import registry_info

from pipeline.stage2_perturbation.adapters import elicitation_family
from pipeline.stage2_perturbation.framing import FRAMING_TEMPLATES, FRAMING_VERSION
from pipeline.stage2_perturbation.rewrite import FAMILY_SYSTEM_PROMPTS, REWRITE_PROMPT_VERSION
from pipeline.stage3_simulation.prompts import PROMPT_VERSION as SCENARIO_PROMPT_VERSION
from pipeline.stage3_simulation.prompts import SCENARIO_FAMILY

REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATED_DIR = REPO_ROOT / "datasets" / "generated"

REWRITE_FAMILIES = tuple(sorted(FAMILY_SYSTEM_PROMPTS))  # identity_strip, paraphrase, register

# Current prompt/template version per family — compared against each artifact's
# meta sidecar so a stale artifact set produces a loud warning (never a silent
# mismatch between the prompts in the repo and the frozen variants on disk).
PROMPT_VERSIONS = {
    **{family: REWRITE_PROMPT_VERSION for family in REWRITE_FAMILIES},
    "framing": FRAMING_VERSION,
    SCENARIO_FAMILY: SCENARIO_PROMPT_VERSION,
}


def task_name(base_task: Task) -> str:
    """Recover the original @task function's registry name (e.g. "fscale")."""
    return registry_info(base_task).name


def artifact_path(task: str, family: str) -> Path:
    return GENERATED_DIR / task / f"{family}.jsonl"


def meta_path(task: str, family: str) -> Path:
    return GENERATED_DIR / task / f"{family}.meta.json"


def write_family(task: str, family: str, rows: list[dict], meta: dict) -> None:
    """Write one family's variants (JSONL, sorted by (id, variant) for stable
    diffs — these files are committed) plus its provenance sidecar.
    """
    path = artifact_path(task, family)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (str(r["id"]), r.get("variant", 0)))
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(meta_path(task, family), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_family(task: str, family: str) -> dict[str, list[dict]]:
    """Load one family's artifact file as {sample_id: [variant rows]}, rows
    ordered by variant index. Raises FileNotFoundError with the generate.py
    hint if the file was never generated (certify.py's pre-run validation
    normally catches this first, with the exact command).
    """
    path = artifact_path(task, family)
    if not path.exists():
        raise FileNotFoundError(
            f"No generated artifacts at {path} — run generate.py for this task/family first "
            f"(see datasets/generated/README.md)."
        )
    by_id: dict[str, list[dict]] = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            by_id.setdefault(str(row["id"]), []).append(row)
    for rows in by_id.values():
        rows.sort(key=lambda r: r.get("variant", 0))
    return by_id


def family_meta(task: str, family: str) -> dict | None:
    path = meta_path(task, family)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def sample_ids(task: Task) -> list[str]:
    return [str(sample.id) for sample in task.dataset]


def framing_ids(task: Task) -> set[str]:
    """Sample ids in this task that framing templates actually apply to.

    Elicitation family is per-sample (a cluster mixes several), so framing
    covers a subset — generate_framing skips the rest, and coverage checks
    must expect the same subset rather than the whole dataset.
    """
    return {
        str(sample.id)
        for sample in task.dataset
        if FRAMING_TEMPLATES.get(elicitation_family(sample))
    }


def framing_applies(task: Task) -> bool:
    """Whether any sample in this task has framing templates at all."""
    return bool(framing_ids(task))


def validate_artifacts(
    benchmarks: dict,
    families: list[str] | None,
    simulate: bool,
    perturb_k: int = 1,
    sim_k: int = 1,
    limit: int | None = None,
) -> None:
    """Fail fast (before any eval runs) unless every task in `benchmarks` has
    a complete artifact file for every requested pregenerated family.

    Per family: rewrite families must cover every dataset sample id with at
    least `perturb_k` variants; framing must cover every id it *applies* to
    (elicitation family is per-sample — see framing_ids); scenario must have a file,
    but ids with fewer than `sim_k` variants only warn — generation drops
    unparseable reframings, mirroring the old live behavior, and replay just
    runs what exists (identically for every model). `reconsideration` is
    live-only and never validated. Prompt-version mismatches warn, not fail.

    When `limit` is set the run is a non-saved smoke test, so coverage
    shortfalls are downgraded to warnings for every family (the file must still
    exist): partial artifacts from `generate.py --limit` are fine here, and
    replay simply runs the variants that happen to exist for the sampled items.
    """
    requested = [f for f in (families or []) if f != "reconsideration"]
    errors: list[str] = []
    for key, entry in benchmarks.items():
        for task in entry["tasks"]:
            name = task_name(task)
            ids = set(sample_ids(task))
            checks: list[tuple[str, int, bool]] = []  # (family, min_k, strict)
            strict_rewrite = not limit
            checks += [(f, perturb_k, strict_rewrite) for f in requested if f in REWRITE_FAMILIES]
            if "framing" in requested and framing_applies(task):
                checks.append(("framing", 1, strict_rewrite))
            if simulate:
                checks.append((SCENARIO_FAMILY, sim_k, False))

            for family, min_k, strict in checks:
                cmd = f"uv run python generate.py --only {key} " + (
                    "--simulate" if family == SCENARIO_FAMILY else f"--perturb {family}"
                )
                path = artifact_path(name, family)
                if not path.exists():
                    errors.append(f"Missing artifacts for {name}/{family} ({path}). Run: {cmd}")
                    continue

                # Framing is the one family with per-sample applicability: a
                # cluster mixes elicitation families, and samples whose family
                # has no templates are skipped by generate_framing. Expecting
                # full coverage would fail every cluster that contains one.
                expected = framing_ids(task) if family == "framing" else ids

                by_id = load_family(name, family)
                missing = expected - set(by_id)
                short = {i for i in expected & set(by_id) if len(by_id[i]) < min_k}
                if missing or short:
                    detail = (
                        f"{name}/{family}: {len(missing)} sample(s) missing, "
                        f"{len(short)} with fewer than {min_k} variant(s). Run: {cmd} --missing-only"
                    )
                    if strict:
                        errors.append(detail)
                    else:
                        print(f"[WARNING] {detail} (replaying the variants that exist)")

                meta = family_meta(name, family)
                stored_version = (meta or {}).get("prompt_version")
                if stored_version and stored_version != PROMPT_VERSIONS[family]:
                    print(
                        f"[WARNING] {name}/{family}: artifacts were generated with prompt version "
                        f"{stored_version}, current code is {PROMPT_VERSIONS[family]} — "
                        f"regenerate with --force to refresh."
                    )
                if meta and meta.get("partial"):
                    print(f"[WARNING] {name}/{family}: artifacts are marked partial (generated with --limit).")

    if errors:
        raise FileNotFoundError(
            "Pregenerated artifacts are missing or incomplete — no evals were started:\n  "
            + "\n  ".join(errors)
        )
