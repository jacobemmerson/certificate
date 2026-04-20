#!/usr/bin/env python3
"""
Prepare flat CSV datasets for the Democratic-Authoritarian Bias (DAB) benchmark.

Sources (never modified):
  benchmarks/Democratic-Authoritarian-Bias-LLMs/data/leaders.json
  benchmarks/Democratic-Authoritarian-Bias-LLMs/data/phase1_questions.json
  benchmarks/Democratic-Authoritarian-Bias-LLMs/role_model_probe/main.py

Outputs:
  benchmarks/datasets/public/dab_fscale.csv     -- 30 F-scale statements (RQ1)
  benchmarks/datasets/public/dab_favscore.csv   -- leaders x questions, pre-joined (RQ2)
  benchmarks/datasets/public/dab_rolemodel.csv  -- 222 nationalities (RQ3)

Usage:
  python scripts/prepare_dab_data.py
"""
import ast
import csv
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "benchmarks" / "Democratic-Authoritarian-Bias-LLMs" / "data"
ROLE_MODEL_PROBE = (
    REPO_ROOT
    / "benchmarks"
    / "Democratic-Authoritarian-Bias-LLMs"
    / "role_model_probe"
    / "main.py"
)
OUT_DIR = REPO_ROOT / "benchmarks" / "datasets" / "public"

# ----- Helper Funcs -----

def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: list, fieldnames: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {len(rows):>5} rows → {path.relative_to(REPO_ROOT)}")


# ----- F-Scale value agreement -----

def make_fscale_csv() -> None:
    """
    Flatten implicit_value_probes from phase1_questions.json.
    Each row is one F-scale statement (English text only).
    fscale_variables is stored as a comma-separated string.
    """
    data = load_json(DATA_DIR / "phase1_questions.json")
    probes = data["implicit_value_probes"]

    rows = []
    for p in probes:
        rows.append({
            "id": p["id"],
            "statement": p["questions"]["en"],
            "fscale_variables": ",".join(p.get("fscale_variables", [])),
            "category": p.get("category", ""),
        })

    write_csv(
        OUT_DIR / "dab_fscale.csv",
        rows,
        ["id", "statement", "fscale_variables", "category"],
    )

# ----- FavScore -----

def _resolve_question_text(q_info: dict, leader_status: str, lang: str = "en") -> str:
    """
    Return the correct question string for a given leader status.
    current leaders → 'template'; former leaders → 'historical'.
    Falls back to the other variant if the preferred one is absent.
    """
    lang_data = q_info.get("questions", {}).get(lang, {})
    if isinstance(lang_data, str):
        return lang_data
    if isinstance(lang_data, dict):
        if leader_status == "former":
            return lang_data.get("historical") or lang_data.get("template", "")
        return lang_data.get("template") or lang_data.get("historical", "")
    return ""


def make_favscore_csv() -> None:
    """
    Cross-join all 196 leaders with all 39 explicit questions.
    Placeholders {leader} and {country} are resolved here so the Inspect
    script receives a ready-to-use prompt fragment (the question sentence).

    sample_id is constructed as  <leader_slug>__<question_id>
    so it is unique and human-readable.
    """
    leaders_data = load_json(DATA_DIR / "leaders.json")
    questions_data = load_json(DATA_DIR / "phase1_questions.json")
    leaders = leaders_data["leaders"]
    explicit_qs = questions_data["explicit_leader_questions"]

    rows = []
    for leader in leaders:
        name_data = leader.get("name", {})
        name = name_data.get("en", "") if isinstance(name_data, dict) else str(name_data)

        country_data = leader.get("country", {})
        country = (
            country_data.get("en", "") if isinstance(country_data, dict) else str(country_data)
        )

        classification = leader.get("classification", "")
        status = leader.get("status", "current")
        era = leader.get("era", "")

        for q in explicit_qs:
            question_text = _resolve_question_text(q, status, lang="en")
            if not question_text:
                continue

            # Substitute leader/country placeholders into the question sentence
            question_text = question_text.replace("{leader}", name).replace(
                "{country}", country
            )

            leader_slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
            sample_id = f"{leader_slug}__{q['id']}"

            rows.append({
                "sample_id": sample_id,
                "leader_name": name,
                "country": country,
                "classification": classification,
                "status": status,
                "era": era,
                "question_id": q["id"],
                "question": question_text,
                "yes_indicates": q.get("yes_indicates", ""),
                "output_format": q.get("output_format", ""),
                "category": q.get("category", ""),
            })

    fieldnames = [
        "sample_id", "leader_name", "country", "classification", "status", "era",
        "question_id", "question", "yes_indicates", "output_format", "category",
    ]
    write_csv(OUT_DIR / "dab_favscore.csv", rows, fieldnames)


# ----- Role Model Elicitation -----

def _extract_nationalities() -> list:
    """
    Parse NATIONALITIES_KEYS directly from role_model_probe/main.py using ast
    so we stay in sync with the original source without duplicating the list.
    """
    with open(ROLE_MODEL_PROBE, encoding="utf-8") as f:
        content = f.read()
    match = re.search(r"NATIONALITIES_KEYS\s*=\s*(\[.*?\])", content, re.DOTALL)
    if not match:
        raise ValueError(
            "Could not find NATIONALITIES_KEYS in role_model_probe/main.py"
        )
    return ast.literal_eval(match.group(1))


def make_rolemodel_csv() -> None:
    nationalities = _extract_nationalities()
    rows = [{"nationality": n} for n in nationalities]
    write_csv(OUT_DIR / "dab_rolemodel.csv", rows, ["nationality"])

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating DAB datasets …")
    make_fscale_csv()
    make_favscore_csv()
    make_rolemodel_csv()
    print("Done.")
