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
    # merge harms + bench_measures_harm
    out = (
        harms
        .rename(columns={
            'parent_id' : 'parent_harm_id', 
            'version' : 'harm_version', 
            'notes' : 'harm_notes'
            })
        .merge(
            bmh.drop(columns='ev_id').rename(columns={'version' : 'bmh_version', 'notes' : 'bmh_notes', 'annotator' : 'bmh_annotator', 'edge_id' : 'bmh_edge'}), 
            on="harm_id")
        .rename(columns=lambda x: x.strip().lower().replace(':', '').replace(' ', '_'))
    )

    # self-merge to get parent labels
    out = (
        out
        .merge(
            harms[['harm_id', 'label']].rename(
                columns={'harm_id' : 'parent_harm_id', 'label' : 'parent_label'}
            ),
            on='parent_harm_id',
            how='left'
        )
    )

    # merge with provision_addresses_harm
    out = (
        out
        .merge(
            pah.rename(columns={'notes' : 'pah_notes', 'annotator' : 'pah_annotator', 'edge_id' : 'pah_edge'}),
            on='harm_id',
            how='left'
        )
        .drop(columns='provision_version')
    )

    # merge with provisions + self-merge with parent citations
    out = (
        out
        .merge(
            provisions.rename(columns={'version': 'provision_version', 'notes' : 'provision_notes', 'parent_id' : 'provision_parent_id'}),
            on='provision_id',
            how='left'
        )
        .merge(
            provisions[['provision_id', 'citation']].rename(columns={'provision_id' : 'provision_parent_id', 'citation' : 'parent_citation'}),
            on='provision_parent_id',
            how='left'
        )
    )
    columns = columns if columns else out.columns
    return out[columns]

if __name__ == "__main__":
    selected = ['label', 'parent_label', 'citation', 'parent_citation', 'harm_domain', 'bench_title', 'binding_force', 'harm_id', 'coverage', 'strength', 'parent_harm_id', 'ev_id', 'provision_id', 'provision_parent_id', 'benchmark_id']
    out = build()
    print(out.shape)
    out = out[selected]
    print(out.isna().sum(axis=0))
    out = out.dropna(subset=['citation'])
    print(out.head(2))
