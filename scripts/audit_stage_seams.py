"""Read-only checks that each pipeline stage's outputs are what the next reads.

    uv run python3 scripts/audit_stage_seams.py [logs/*/*.eval]

B1 perturbations vs perturbation_scores vs conditions coverage
B2 artifact rows whose first k variants are all fallback (validated, replayed as nothing)
B3 reconsidered condition present; live challenge produced nothing
B4 numeric per-condition value without judge_scores (results._scorers reads it)
B5 results.build headline vs models/models.json
B7 logs with no LLM judge verdict anywhere (fallback-only runs)
"""
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inspect_ai.log import read_eval_log

from pipeline.artifacts import REWRITE_FAMILIES, load_family
from pipeline.utils.results import build

LLM_JUDGE = "openrouter/"


def audit_log(path):
    log = read_eval_log(path)
    c = Counter()
    for s in log.samples or []:
        md = s.metadata
        stored = {r["condition"] for rows in (md.get("perturbations") or {}).values() for r in rows}
        scored = (md.get("perturbation_scores") or {}).get("cluster_scorer") or {}
        c["samples"] += 1
        c["B1 stored_not_scored"] += len(stored - set(scored))
        c["B1 scored_not_stored"] += len(set(scored) - stored - {"control"})
        for label, e in scored.items():
            v = e.get("value")
            if isinstance(v, (int, float)) and v == v:
                c["cond_numeric"] += 1
                if not (e.get("metadata") or {}).get("judge_scores"):
                    c["B4 numeric_no_judge_scores"] += 1
            else:
                c["B1 cond_unscored"] += 1
            if any(j.startswith(LLM_JUDGE) for j in ((e.get("metadata") or {}).get("judge_verdicts") or {})):
                c["llm_judged"] += 1
        if "reconsideration" in (md.get("perturbations") or {}):
            c["B3 recon_rows"] += 1
            if not md["perturbations"]["reconsideration"]:
                c["B3 recon_empty"] += 1
        score = s.scores.get("cluster_scorer")
        conds = (score.metadata or {}).get("conditions") if score else None
        if conds is not None:
            c["B1 conditions_missing_vs_scored"] += len(set(scored) - set(conds))
    return log, c


def fallback_truncation(k=1):
    out = {}
    for cluster in ("cbrn", "cyber", "loss_of_control", "manipulation"):
        for fam in REWRITE_FAMILIES:
            by_id = load_family(cluster, fam)
            dead = [i for i, rows in by_id.items() if rows[:k] and all(r.get("fallback") for r in rows[:k])]
            out[(cluster, fam)] = (len(dead), len(by_id))
    return out


def main(pattern):
    models = {m["id"]: m for m in json.load(open("models/models.json"))}
    by_model = defaultdict(list)
    print("== per log ==")
    for path in sorted(glob.glob(pattern)):
        log, c = audit_log(path)
        model = log.eval.model.split("/")[-1]
        by_model[model].append(log)
        flags = {k: v for k, v in c.items() if k[:2] in ("B1", "B3", "B4") and v}
        print(f"{model:<24}{log.eval.task:<17}n={c['samples']:<5}llm_judged={c['llm_judged']:<6}"
              f"{'B7 NO LLM VERDICTS ' if c['samples'] > 2 and not c['llm_judged'] else ''}{flags or ''}")

    print("\n== B2 artifact ids whose first k rows are all fallback (k=1) ==")
    for (cluster, fam), (dead, total) in fallback_truncation().items():
        if dead:
            print(f"{cluster}/{fam}: {dead}/{total}")

    print("\n== B5 results.build headline vs models.json ==")
    for model, logs in by_model.items():
        if model not in models:
            print(f"{model}: not in models.json"); continue
        full = [l for l in logs if len(l.samples or []) > 2]
        tree = build(full)
        for risk, node in tree.items():
            rebuilt = (node.get("aggregate") or {}).get("tail")
            stored = models[model].get("scores", {}).get(risk)
            mark = "" if rebuilt == stored else "  <-- DIFF"
            print(f"{model:<24}{risk:<17}rebuilt={rebuilt!s:<8}stored={stored!s:<8}{mark}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "logs/*/*.eval")
