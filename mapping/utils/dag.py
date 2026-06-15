import pandas as pd
import numpy as np
import json
import os

bench_map = {
    "B2.01.01" : "harm"
}

_maps = os.path.join(os.path.dirname(__file__), '..', 'maps')
harms      = pd.read_csv(os.path.join(_maps, 'ESAI Harm-Bench-Legal Map - harms.csv'))
provisions = pd.read_csv(os.path.join(_maps, 'ESAI Harm-Bench-Legal Map - provisions.csv'))
bmh        = pd.read_csv(os.path.join(_maps, 'ESAI Harm-Bench-Legal Map - bench_measures_harm.csv'))
bench      = pd.read_csv(os.path.join(_maps, 'ESAI Harm-Bench-Legal Map - benchmarks.csv'))
pah        = pd.read_csv(os.path.join(_maps, 'ESAI Harm-Bench-Legal Map - provision_addresses_harm.csv'))

with open(os.path.join(os.path.dirname(__file__), '../..', 'models', 'models.json'), "r") as f:
    model_res = json.load(f)

def build(columns=None):
    # merge benchmark --measures--> harm
    out = (
        bmh
        .merge(harms[["parent_id", "harm_id", "label"]], on="harm_id")
        .rename(columns={"label": "ev_label", "parent_id": "parent_harm_id"})
        .assign(
            parent_harm_id=lambda df: df["parent_harm_id"].astype(str).str[:-2],
            harm_root_id=lambda df: df["harm_id"].astype(str).str[:-2] + "00"
        )
    )

    # merge parents + harm label + evidence label
    out = (
        out
        .merge(
            harms[["harm_id", "label"]]
            .rename(columns={"harm_id": "parent_harm_id", "label": "parent_harm_label"}),
            on="parent_harm_id",
            how="left",
            suffixes=("_bmh", "_harm")
        )
        .merge(
            harms[["harm_id", "label"]]
            .rename(columns={"harm_id": "harm_root_id", "label": "harm_label"}),
            on="harm_root_id",
            how="left",
        )
    )

    # merge provisions + provision parents + citations
    out = (
        out
        .merge(pah[["provision_id", "harm_id", "coverage", "justification"]], on="harm_id")
        .merge(provisions, on="provision_id").rename(columns={"parent_id": "parent_prov_id"})
        .merge(
            provisions[["provision_id", "citation"]].rename(columns={"provision_id" : "parent_prov_id", "citation" : "parent_citation"}),
            on="parent_prov_id",
            how="left"
        )
    )
    
    columns = columns if columns else out.columns
    return out[columns]

if __name__ == "__main__":
    out = build()

    # link to results
    #benchmarks = out['benchmark_id'].apply(lambda x: bench_map[x])

    #results = {}
    #for model in model_res:
    #    results[model['name']] = ([(b, model['scores'][b]) for b in benchmarks if b in model['scores']])

    print(out.T)
    #print(results)
