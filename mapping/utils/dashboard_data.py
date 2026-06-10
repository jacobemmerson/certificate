import json
import math
import os

from dag import bench, bench_map, build, model_res


def _clean(value):
    """Convert NaN/NaT to None so the result is valid JSON."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str):
        return value.strip() or None
    return value


def build_dashboard_data():
    full = build()

    domains = {}
    for harm_id, group in full.groupby("harm_id", sort=False):
        first = group.iloc[0]
        benchmark_id = first["benchmark_id"]

        # Only benchmarks wired up in dag.py's bench_map can be scored.
        score_key = bench_map.get(benchmark_id)
        if score_key is None:
            continue

        bench_row = bench[bench["benchmark_id"] == benchmark_id]
        benchmark = {
            "id": benchmark_id,
            "title": _clean(bench_row["title"].iloc[0]) if not bench_row.empty else None,
            "ref": _clean(bench_row["quick ref"].iloc[0]) if not bench_row.empty else None,
        }

        provisions = [
            {
                "provision_id": _clean(r["provision_id"]),
                "citation": _clean(r["citation"]),
                "parent_citation": _clean(r["parent_citation"]),
                "jurisdiction": _clean(r["jurisdiction"]),
                "instrument_type": _clean(r["instrument_type"]),
                "binding_force": _clean(r["binding_force"]),
                "coverage": _clean(r["coverage"]),
                "justification": _clean(r["justification"]),
            }
            for _, r in group.iterrows()
        ]

        evidence = {
            "harm_id": harm_id,
            "ev_label": _clean(first["ev_label"]),
            "harm_label": _clean(first["harm_label"]),
            "evidence_category": _clean(first["Harm: Category "]),
            "evidence_domain": _clean(first["Harm: Domain"]),
            "evidence_subdomain": _clean(first["Harm: Subdomain"]),
            "benchmark": benchmark,
            "score_key": score_key,
            "provisions": provisions,
        }

        parent_harm_label = _clean(first["parent_harm_label"])
        domains.setdefault(parent_harm_label, []).append(evidence)

    domains_list = [
        {"parent_harm_label": label, "evidences": evidences}
        for label, evidences in domains.items()
    ]

    models = [
        {
            "id": m["id"],
            "name": m["name"],
            "company": m["company"],
            "region": m["region"],
            "specialty": m["specialty"],
            "scores": m["scores"],
            "scores_meta": m.get("scores_meta", {}),
        }
        for m in model_res
    ]

    return {"domains": domains_list, "models": models}


if __name__ == "__main__":
    data = build_dashboard_data()
    out_path = os.path.join(os.path.dirname(__file__), "..", "dashboard_data.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"wrote {out_path}")
