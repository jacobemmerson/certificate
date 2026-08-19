"""
Match each model in models/models.json to its Artificial Analysis (AA)
intelligence index entry and write the result into models.json as
`aa_intelligence_index` / `aa_model_match`.

AA lists many variants per base model (reasoning/non-reasoning, effort
levels), and its `name` field only roughly matches the `name` field in
models.json (e.g. AA renamed xAI to "SpaceXAI"). Base-name matching picks
the right family; VARIANT_POLICY then picks the specific variant within
that family:
  - "flash" models: lowest available reasoning effort
  - "pro" models: max/highest available reasoning effort
  - everything else: "high" effort with reasoning on, falling back to the
    closest available effort level

Run from the repo root:
    uv run python3 scripts/match_aa_index.py
"""

import difflib
import json
import os
import re
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
MODELS_PATH = ROOT / "models" / "models.json"

AA_URL = "https://artificialanalysis.ai/api/v2/language/models/free"

# AA renames/relabels some creators relative to models.json's `company` field.
COMPANY_ALIASES = {
    "xai": "spacexai",
    "z.ai": "z ai",
    "zai": "z ai",
}

VARIANT_PAREN_RE = re.compile(r"\s*\([^)]*\)\s*")

# models.json's own name field is wrong for this one ("Muse Spark 1.1" for
# id muse-spark-1.2) - match against the corrected name instead.
NAME_OVERRIDES = {
    "muse-spark-1.2": "Muse Spark 1.2",
}

# Named tiers only - deliberately excludes "reasoning"/"non-reasoning" so
# checking `word in qualifier` can't match "reasoning" as a substring of
# "non-reasoning".
TIER_ORDER = {"minimal": 1, "low": 2, "medium": 3, "high": 5, "xhigh": 6, "max": 7}
HIGH_TARGET = TIER_ORDER["high"]


def parse_effort(aa_name: str) -> tuple[bool, int]:
    """Returns (reasoning_on, effort_order) parsed from the variant's
    parenthetical qualifier, e.g. "(Adaptive Reasoning, High Effort)". A
    qualifier naming just a tier ("(max)") implies reasoning is on - AA
    labels the reasoning-off case explicitly as "(Non-reasoning)"."""
    qualifier = aa_name[len(base_name(aa_name)):].lower()
    reasoning_on = "non-reasoning" not in qualifier
    for word, order in TIER_ORDER.items():
        if word in qualifier:
            return reasoning_on, order
    return reasoning_on, 4 if reasoning_on else 0


def classify_policy(model_name: str) -> str:
    lowered = model_name.lower()
    if "flash" in lowered:
        return "flash"
    if "pro" in lowered:
        return "pro"
    return "default"


def resolve_variant(variants: list[dict], policy: str) -> dict:
    # Prefer variants AA has actually scored - some effort tiers (e.g. some
    # Claude Sonnet 5 adaptive-reasoning levels) exist but are still null.
    scored = [v for v in variants if v["intelligence_index"] is not None] or variants
    parsed = [(v, *parse_effort(v["name"])) for v in scored]
    if policy == "flash":
        key = lambda item: (item[2], item[1])  # lowest effort, prefer reasoning-off on ties
    elif policy == "pro":
        key = lambda item: (-item[2], -item[1])  # highest effort, prefer reasoning-on on ties
    else:
        key = lambda item: (not item[1], abs(item[2] - HIGH_TARGET))  # reasoning-on, closest to "high"
    return min(parsed, key=key)[0]


def fetch_aa_models() -> list[dict]:
    api_key = os.environ["ARTIFICAL_ANALYSIS_API_KEY"]
    models, page = [], 1
    while True:
        response = requests.get(AA_URL, headers={"x-api-key": api_key}, params={"page": page})
        response.raise_for_status()
        payload = response.json()
        models.extend(payload["data"])
        if not payload["pagination"]["has_more"]:
            return models
        page += 1


def normalize(name: str) -> str:
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", " ", name)
    return name.strip()


def token_sort(name: str) -> str:
    """Word order varies between sources (e.g. "Claude Haiku 4.5" vs AA's
    "Claude 4.5 Haiku"); sorting tokens before comparing makes the ratio
    order-invariant instead of penalizing reordered-but-identical names."""
    return " ".join(sorted(normalize(name).split()))


def base_name(aa_name: str) -> str:
    """AA name with trailing variant qualifiers like "(Reasoning)" stripped."""
    return VARIANT_PAREN_RE.sub("", aa_name).strip()


def company_matches(model_company: str, aa_creator: str) -> bool:
    a = normalize(model_company)
    b = normalize(aa_creator)
    a = COMPANY_ALIASES.get(a, a)
    b = COMPANY_ALIASES.get(b, b)
    return a == b


def best_matches(model: dict, aa_models: list[dict], top_n: int = 3) -> list[dict]:
    target = token_sort(NAME_OVERRIDES.get(model["id"], model["name"]))

    # Group AA entries by base name so variants (reasoning/effort/etc.) of
    # the same underlying model are scored and reported together.
    groups: dict[str, list[dict]] = {}
    for aa in aa_models:
        groups.setdefault(base_name(aa["name"]), []).append(aa)

    scored = []
    for base, variants in groups.items():
        score = difflib.SequenceMatcher(None, target, token_sort(base)).ratio()
        if company_matches(model["company"], variants[0]["model_creator"]["name"]):
            score += 0.15
        scored.append((score, base, variants))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, base, variants in scored[:top_n]:
        variants_sorted = sorted(
            variants,
            key=lambda v: (v["evaluations"]["artificial_analysis_intelligence_index"] is None,
                            -(v["evaluations"]["artificial_analysis_intelligence_index"] or 0)),
        )
        results.append(
            {
                "score": round(score, 3),
                "base_name": base,
                "creator": variants[0]["model_creator"]["name"],
                "variants": [
                    {
                        "name": v["name"],
                        "intelligence_index": v["evaluations"]["artificial_analysis_intelligence_index"],
                    }
                    for v in variants_sorted
                ],
            }
        )
    return results


def main() -> None:
    aa_models = fetch_aa_models()
    models = json.loads(MODELS_PATH.read_text(encoding="utf-8"))

    report = {}
    print(f"{'id':<22} {'policy':<8} {'index':>7}  matched AA variant")
    print("-" * 80)
    for model in models:
        matches = best_matches(model, aa_models)
        policy = classify_policy(model["name"])
        top = matches[0]
        resolved = resolve_variant(top["variants"], policy) if top["score"] >= 0.5 else None

        report[model["id"]] = {
            "name": model["name"],
            "company": model["company"],
            "policy": policy,
            "resolved": resolved,
            "candidates": matches,
        }

        if resolved is None:
            print(f"{model['id']:<22} {policy:<8} {'?':>7}  NO CONFIDENT MATCH")
            continue

        model["aa_intelligence_index"] = resolved["intelligence_index"]
        model["aa_model_match"] = resolved["name"]
        idx = resolved["intelligence_index"]
        idx_str = f"{idx:g}" if idx is not None else "None"
        print(f"{model['id']:<22} {policy:<8} {idx_str:>7}  {resolved['name']}")

    MODELS_PATH.write_text(json.dumps(models, indent=4) + "\n", encoding="utf-8")
    print(f"\nWrote aa_intelligence_index into {MODELS_PATH}")

    out_path = ROOT / "scripts" / "aa_index_matches.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
