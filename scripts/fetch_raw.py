'''
Fetch raw benchmark data listed in datasets/raw/manifest.toml into
datasets/raw/<name>/, sparse (only `files` patterns) and pinned (the resolved
commit SHA is written to fetch.json, which prepare.py records in its meta).

    uv run python3 scripts/fetch_raw.py [--only NAME ...] [--status S ...] [--dry-run]

Gated Hugging Face datasets need HF_TOKEN in the environment.
'''
import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from datasets.prepare.cluster.manifest import STATUSES, load_manifest  # noqa: E402

RAW_DIR = REPO_ROOT / "datasets" / "raw"


def select(entries: list[dict], only: list[str], statuses: list[str]) -> list[dict]:
    if only:
        return [entry for entry in entries if entry["name"] in only]
    return [entry for entry in entries if entry["status"] in statuses]


def plan(entry: dict, dest: Path) -> list[list[str]] | dict:
    '''The git commands (github) or snapshot_download kwargs (hf) a fetch runs.'''
    revision = entry.get("revision", "")
    if entry["host"] == "hf":
        return {
            "repo_id": entry["repo"], "repo_type": "dataset", "revision": revision or None,
            "allow_patterns": entry["files"], "local_dir": dest,
        }
    # `git clone --depth 1 <sha>` rejects SHAs; fetch-by-ref works for sha, tag or branch.
    return [
        ["git", "init", "-q"],
        ["git", "remote", "add", "origin", f"https://github.com/{entry['repo']}.git"],
        ["git", "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", revision or "HEAD"],
        ["git", "sparse-checkout", "set", "--no-cone", *entry["files"]],
        ["git", "checkout", "-q", "FETCH_HEAD"],
    ]


def fetch(entry: dict, dest: Path):
    name, requested = entry["name"], entry.get("revision", "")
    record = dest / "fetch.json"
    if record.exists():
        data = json.loads(record.read_text())
        pinned = not requested or requested in (data["revision"], data.get("requested"))
        if pinned and data.get("files") == entry["files"]:
            print(f"{name}: up to date")
            return
        shutil.rmtree(dest)
    elif dest.exists() and any(dest.iterdir()):
        raise SystemExit(f"{name}: {dest} exists without fetch.json; refusing to overwrite it")
    dest.mkdir(parents=True, exist_ok=True)

    # dest is new or emptied above, so a failed download leaves nothing worth keeping.
    try:
        sha = download(entry, dest)
    except BaseException:
        shutil.rmtree(dest, ignore_errors=True)
        raise

    record.write_text(json.dumps({
        "name": name, "id": entry["id"], "host": entry["host"], "repo": entry["repo"],
        "revision": sha, "requested": requested, "files": entry["files"],
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2) + "\n")
    (dest / ".gitignore").write_text("*\n!fetch.json\n!.gitignore\n")
    print(f"{name}: fetched {sha}")


def download(entry: dict, dest: Path) -> str:
    '''Populate dest and return the resolved commit SHA.'''
    name, steps = entry["name"], plan(entry, dest)
    if entry["host"] == "hf":
        from huggingface_hub import HfApi, snapshot_download
        from huggingface_hub.errors import GatedRepoError
        try:
            sha = HfApi().dataset_info(entry["repo"], revision=steps["revision"]).sha
            snapshot_download(**{**steps, "revision": sha})
        except GatedRepoError as error:
            raise SystemExit(f"{name}: gated dataset {entry['repo']}; accept its terms on the Hub "
                             f"and set HF_TOKEN in the environment ({error})")
        shutil.rmtree(dest / ".cache", ignore_errors=True)
    else:
        for command in steps:
            subprocess.run(command, cwd=dest, check=True)
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=dest, check=True,
                             capture_output=True, text=True).stdout.strip()
        shutil.rmtree(dest / ".git")
    return sha


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", default=[], metavar="NAME")
    parser.add_argument("--status", nargs="+", default=["prompt", "partial"], choices=STATUSES)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest", type=Path, default=RAW_DIR / "manifest.toml")
    args = parser.parse_args(argv)

    entries = load_manifest(args.manifest)
    if unknown := set(args.only) - {entry["name"] for entry in entries}:
        parser.error(f"--only names not in {args.manifest}: {sorted(unknown)}")
    if bad := [entry["name"] for entry in entries if not re.fullmatch(r"[a-z0-9_]+", entry["name"])]:
        parser.error(f"manifest names must match [a-z0-9_]+: {bad}")

    for entry in select(entries, args.only, args.status):
        if entry["host"] == "none":
            print(f"{entry['name']}: no host ({entry['status']}), skipped")
            continue
        if not entry["files"]:
            print(f"{entry['name']}: no files, skipped")
            continue
        # Unpinned registered/vendored rows are the dirs whose data is committed (wmdp, sosbench, ...).
        if entry["status"] in ("registered", "vendored") and not entry.get("revision"):
            print(f"{entry['name']}: committed data, skipped")
            continue
        dest = RAW_DIR / entry["name"]
        if args.dry_run:
            print(f"{entry['name']} -> {dest}: {plan(entry, dest)}")
        else:
            fetch(entry, dest)


if __name__ == "__main__":
    main()
