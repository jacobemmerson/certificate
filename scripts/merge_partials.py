'''
author: @tae

Merge per-model partial certificate files (written by parallel certify.py
array tasks) back into the shared models/models.json.

Each partial in models/partials/*.json holds the results for a single model,
written with CERTIFY_MODELS_PATH pointing at that partial. This folds them into
models.json using the same upsert-and-union semantics as certify.update():
existing scores are preserved and overlaid with the newly computed ones.

Usage:
    uv run scripts/merge_partials.py
    uv run scripts/merge_partials.py --partials-dir models/partials --keep
'''

import json
import shutil
from pathlib import Path
from argparse import ArgumentParser

REPO_ROOT = Path(__file__).resolve().parent.parent


def upsert(models: list[dict], entry: dict) -> None:
    '''Insert or merge a single model entry, mirroring certify.update().'''
    for i, m in enumerate(models):
        if m['id'] == entry['id']:
            # union overlapping keys, new results win (right side of |)
            entry['scores'] = m.get('scores', {}) | entry.get('scores', {})
            entry['scores_meta'] = m.get('scores_meta', {}) | entry.get('scores_meta', {})
            models[i] = entry
            return
    models.append(entry)


def main() -> None:
    args = ArgumentParser(description="Merge per-model partial certificates into models.json")
    args.add_argument("--partials-dir", default=str(REPO_ROOT / "models" / "partials"))
    args.add_argument("--out", default=str(REPO_ROOT / "models" / "models.json"))
    args.add_argument("--keep", action="store_true", help="Keep partial files instead of deleting them after a successful merge.")
    opts = args.parse_args()

    partials_dir = Path(opts.partials_dir)
    out_path = Path(opts.out)

    partials = sorted(partials_dir.glob("*.json")) if partials_dir.exists() else []
    if not partials:
        print(f"No partials found in {partials_dir}; nothing to merge.")
        return

    # load current models (tolerate missing/empty)
    try:
        with open(out_path) as f:
            models = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        models = []

    # safety net, matching certify.update()'s _previous behaviour
    if models:
        shutil.copy(out_path, str(out_path).replace('.json', '_previous.json'))

    merged = 0
    for p in partials:
        try:
            with open(p) as f:
                entries = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[WARN] skipping unreadable partial {p.name}: {e}")
            continue
        for entry in entries:
            upsert(models, entry)
            merged += 1
            print(f"  merged {entry['id']} ({len(entry.get('scores', {}))} benchmark scores)")

    with open(out_path, 'w') as f:
        json.dump(models, f, indent=4)

    print(f"Merged {merged} model ent(ies) into {out_path} ({len(models)} total).")

    if not opts.keep:
        for p in partials:
            p.unlink()
        print(f"Removed {len(partials)} partial file(s) from {partials_dir}.")


if __name__ == "__main__":
    main()
