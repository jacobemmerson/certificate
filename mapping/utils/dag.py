import pandas as pd
import numpy as np
import json
import os

bench_map = {
    "B2.01.01": "harm",   # SocialHarmBench (Pandey2026) -> scores.harm
    "B3.01.01": "hr",     # Human rights endorsement (Samway2025) -> scores.hr
    "B4.01.01": "hist",   # Historical revisionism (Ortu2026) -> scores.hist
    "B5.01.01": "auth",   # Dem-vs-authoritarian bias (Guzman2026) -> scores.auth
}

_maps = os.path.join(os.path.dirname(__file__), '..', 'maps')
harms      = pd.read_csv(os.path.join(_maps, 'ESAI Harm-Bench-Legal Map - harms.csv'))
provisions = pd.read_csv(os.path.join(_maps, 'ESAI Harm-Bench-Legal Map - provisions.csv'))
bmh        = pd.read_csv(os.path.join(_maps, 'ESAI Harm-Bench-Legal Map - bench_measures_harm.csv'))
bench      = pd.read_csv(os.path.join(_maps, 'ESAI Harm-Bench-Legal Map - benchmarks.csv'))
pah        = pd.read_csv(os.path.join(_maps, 'ESAI Harm-Bench-Legal Map - provision_addresses_harm.csv'))

with open(os.path.join(os.path.dirname(__file__), '../..', 'models', 'models.json'), "r") as f:
    model_res = json.load(f)


def _rename_cols(df):
    return df.rename(columns=lambda x: x.strip().lower().replace(':', '').replace(' ', '_'))


# Pre-renamed copies for the tree builders below, so field names match the
# `label`/`bench_title`/`harm_domain`/`binding_force`/... convention that
# `build()` produces on its merged output.
bmh_r   = _rename_cols(bmh)
pah_r   = _rename_cols(pah)
prov_r  = _rename_cols(provisions)
bench_r = _rename_cols(bench)

_harms_by_id = harms.set_index('harm_id', drop=False)
_prov_by_id  = prov_r.set_index('provision_id', drop=False)
_bench_by_id = bench_r.set_index('benchmark_id', drop=False)
_bmh_by_harm = bmh_r.groupby('harm_id')
_pah_by_harm = pah_r.groupby('harm_id')
_pah_by_prov = pah_r.groupby('provision_id')


def build(columns=None):
    # merge harms + self-merge for parent and global labels
    out = (
        harms
        .rename(columns={
            'parent_id' : 'parent_harm_id', 
            'version' : 'harm_version', 
            'notes' : 'harm_notes'
            })
        .merge(
            harms[['harm_id', 'label']].rename(
                columns={'harm_id' : 'parent_harm_id', 'label' : 'parent_label'}
            ),
            on='parent_harm_id',
            how='left'
        )
        
    )
    #print(out.head())
    # merge harms + bench_measures_harm
    out = (
        out
        .merge(
            bmh.drop(columns='ev_id').rename(columns={'version' : 'bmh_version', 'notes' : 'bmh_notes', 'annotator' : 'bmh_annotator', 'edge_id' : 'bmh_edge'}), 
            on="harm_id",
            how="left"
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
        .pipe(_rename_cols)
    )
    columns = columns if columns else out.columns
    return out[columns]


def _harm_breadcrumb(harm_id):
    """Return (domain_label, subdomain_label) for a harm by walking parent_id.

    `harm_id` itself is the leaf label; this only resolves its ancestors.
    """
    if harm_id not in _harms_by_id.index:
        return None, None
    parent_id = _harms_by_id.loc[harm_id, 'parent_id']
    if pd.isna(parent_id) or parent_id not in _harms_by_id.index:
        return None, None
    sub = _harms_by_id.loc[parent_id]
    grandparent_id = sub['parent_id']
    domain_label = None
    if pd.notna(grandparent_id) and grandparent_id in _harms_by_id.index:
        domain_label = _harms_by_id.loc[grandparent_id, 'label']
    return domain_label, sub['label']


def _model_results_for_benchmark(benchmark_id):
    score_key = bench_map.get(benchmark_id)
    if score_key is None:
        return []
    out = []
    for m in model_res:
        score = m.get('scores', {}).get(score_key)
        if score is None:
            continue
        out.append({
            'model_id': m['id'],
            'name': m['name'],
            'company': m['company'],
            'score': score,
        })
    out.sort(key=lambda r: r['score'], reverse=True)
    return out


def _benchmarks_for_harm(harm_id):
    if harm_id not in _bmh_by_harm.groups:
        return []
    out = []
    for _, b in _bmh_by_harm.get_group(harm_id).iterrows():
        quick_ref = (
            _bench_by_id.loc[b['benchmark_id'], 'quick_ref']
            if b['benchmark_id'] in _bench_by_id.index else None
        )
        out.append({
            'benchmark_id': b['benchmark_id'],
            'bench_title': b['bench_title'],
            'quick_ref': quick_ref,
            'harm_domain': b['harm_domain'],
            'harm_subdomain': b['harm_subdomain'],
            'strength': b['strength'],
            'basis': b['basis'],
            'confidence': b['confidence'],
            'model_results': _model_results_for_benchmark(b['benchmark_id']),
        })
    return out


def _provisions_for_harm(harm_id):
    if harm_id not in _pah_by_harm.groups:
        return []
    out = []
    for _, p in _pah_by_harm.get_group(harm_id).iterrows():
        prow = _prov_by_id.loc[p['provision_id']] if p['provision_id'] in _prov_by_id.index else None
        parent_citation = None
        if prow is not None and pd.notna(prow['parent_id']) and prow['parent_id'] in _prov_by_id.index:
            parent_citation = _prov_by_id.loc[prow['parent_id'], 'citation']
        out.append({
            'provision_id': p['provision_id'],
            'citation': prow['citation'] if prow is not None else None,
            'parent_citation': parent_citation,
            'coverage': p['coverage'],
            'justification': p['justification'],
            'binding_force': prow['binding_force'] if prow is not None else None,
            'jurisdiction': prow['jurisdiction'] if prow is not None else None,
            'instrument_type': prow['instrument_type'] if prow is not None else None,
        })
    return out


def harm_tree():
    """Build the harm taxonomy as a tree, scaffolded by `parent_id`.

    Roots are harms with no `parent_id` (domains). Leaves (the ones with an
    `ev_id`) carry the benchmarks that measure them and the legal provisions
    that address them; branch nodes (domains/subdomains) just carry
    `children`.
    """
    children_of = {}
    for _, row in harms.iterrows():
        parent_id = row['parent_id']
        key = None if pd.isna(parent_id) else parent_id
        children_of.setdefault(key, []).append(row['harm_id'])

    def node(harm_id):
        row = _harms_by_id.loc[harm_id]
        out = {
            'harm_id': harm_id,
            'parent_harm_id': row['parent_id'],
            'label': row['label'],
            'ev_id': row['ev_id'],
        }
        kids = children_of.get(harm_id, [])
        if kids:
            out['children'] = [node(c) for c in kids]
        else:
            out['benchmarks'] = _benchmarks_for_harm(harm_id)
            out['provisions'] = _provisions_for_harm(harm_id)
        return out

    return [node(r) for r in children_of.get(None, [])]


def provision_tree():
    """Build the legal-provision taxonomy as a tree, scaffolded by `parent_id`.

    `P.EU.AIA.R110` has `parent_id == provision_id` (data quirk); it's
    treated as a root like the true NaN-parent rows. Any node may carry
    `harms` (provisions can address harms at any level, not just leaves),
    each with its own `benchmarks` for the three-way join.
    """
    children_of = {}
    for _, row in prov_r.iterrows():
        parent_id = row['parent_id']
        key = None if (pd.isna(parent_id) or parent_id == row['provision_id']) else parent_id
        children_of.setdefault(key, []).append(row['provision_id'])

    def node(provision_id):
        row = _prov_by_id.loc[provision_id]
        parent_id = row['parent_id']
        out = {
            'provision_id': provision_id,
            'provision_parent_id': None if (pd.isna(parent_id) or parent_id == provision_id) else parent_id,
            'citation': row['citation'],
            'jurisdiction': row['jurisdiction'],
            'instrument_type': row['instrument_type'],
            'binding_force': row['binding_force'],
        }
        kids = children_of.get(provision_id, [])
        if kids:
            out['children'] = [node(c) for c in kids]
        if provision_id in _pah_by_prov.groups:
            harms_list = []
            for _, p in _pah_by_prov.get_group(provision_id).iterrows():
                harm_id = p['harm_id']
                harm_row = _harms_by_id.loc[harm_id] if harm_id in _harms_by_id.index else None
                domain_label, subdomain_label = _harm_breadcrumb(harm_id)
                harms_list.append({
                    'harm_id': harm_id,
                    'label': harm_row['label'] if harm_row is not None else None,
                    'parent_label': subdomain_label,
                    'harm_domain_label': domain_label,
                    'coverage': p['coverage'],
                    'justification': p['justification'],
                    'benchmarks': _benchmarks_for_harm(harm_id),
                })
            out['harms'] = harms_list
        return out

    return [node(r) for r in children_of.get(None, [])]


def _benchmarks_for_model(model):
    out = []
    for benchmark_id, score_key in bench_map.items():
        score = model.get('scores', {}).get(score_key)
        if score is None:
            continue
        row = _bench_by_id.loc[benchmark_id] if benchmark_id in _bench_by_id.index else None
        out.append({
            'benchmark_id': benchmark_id,
            'bench_title': row['title'] if row is not None else None,
            'quick_ref': row['quick_ref'] if row is not None else None,
            'score': score,
            'score_breakdown': model.get('scores_meta', {}).get(score_key, {}),
        })
    return out


def model_list():
    """Build a flat list of models with their benchmark scores (wired up
    via `bench_map`), sorted by aggregate score (mean across those
    benchmarks) descending."""
    out = []
    for m in model_res:
        benchmarks = _benchmarks_for_model(m)
        scores = [b['score'] for b in benchmarks if b['score'] is not None]
        aggregate_score = sum(scores) / len(scores) if scores else None
        out.append({
            'model_id': m['id'],
            'name': m['name'],
            'company': m['company'],
            'region': m['region'],
            'specialty': m['specialty'],
            'benchmarks': benchmarks,
            'aggregate_score': aggregate_score,
        })
    out.sort(key=lambda r: (r['aggregate_score'] is None, -(r['aggregate_score'] or 0)))
    return out


if __name__ == "__main__":
    selected = ['label', 'parent_label', 'citation', 'parent_citation', 'harm_domain', 'bench_title', 'binding_force', 'harm_id', 'coverage', 'strength', 'parent_harm_id', 'ev_id', 'provision_id', 'provision_parent_id', 'benchmark_id']
    out = build()
    print(out.shape)
    out = out[selected]
    print(out.isna().sum(axis=0))
    out = out.dropna(subset=['citation'])
    print(out.head(2))
