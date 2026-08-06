'''
End-to-end demo on a small, deliberately varied sample — generate stage-2
perturbations and stage-3 scenarios for N rows of one risk cluster, replay them
through the real eval path, and emit a self-contained HTML report.

Why this exists rather than `certify.py --limit`: the two --limit flags do not
agree on a subset. `generate.py --limit N` takes the *first* N samples, while
`certify.py --limit N` shuffles and takes a *random* N (certify.py:267,
sample_shuffle=bool(args.limit)), so the replayed ids would mostly miss the
generated ones and every variant would come back empty. This script picks one
subset and drives both halves from it, with no shuffle and no limit.

Artifacts are written to a scratch directory (--artifacts), NOT to
datasets/generated/, so the committed artifact set is never touched. Results are
not written to models.json either — this is for looking at, not for certifying.

Sampling is stratified across (source, question_type) so the report shows the
cases the stage READMEs warn about — a likert row and a detection row under
--simulate look very different from a graded one.

Usage:
    uv run scripts/demo_pipeline.py --risk manipulation -n 12
    uv run scripts/demo_pipeline.py --risk cyber -n 20 --model openrouter/openai/gpt-4o-mini
    uv run scripts/demo_pipeline.py --risk cbrn -n 8 --no-simulate

Then open the printed report path, or `inspect view --log-dir <log dir>` for
full transcripts.
'''

import argparse
import asyncio
import html
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# This script lives in scripts/, unlike certify.py/generate.py at the repo root,
# so the package root has to go on the path before the pipeline imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from inspect_ai import eval as inspect_eval

from pipeline import artifacts
from pipeline.artifacts import REWRITE_FAMILIES, PROMPT_VERSIONS, write_family
from pipeline.generation import generate_framing, generate_rewrites, generate_scenarios
from pipeline.registry import apply_stages
from pipeline.stage1_evaluation.evals.clusters import CLUSTER_TASKS, RISKS, available
from pipeline.stage3_simulation.prompts import SCENARIO_FAMILY
from pipeline.utils import results as results_tree
from pipeline.utils.graders import DIAGNOSTIC_SOURCES

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse():
    args = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args.add_argument("--risk", default="manipulation", choices=RISKS, help="Risk cluster to demo (default: manipulation).")
    args.add_argument("-n", "--samples", type=int, default=12, help="Samples to draw, stratified by source (default: 12).")
    args.add_argument("--model", "-m", default="openrouter/openai/gpt-4o-mini", help="Target model under test.")
    args.add_argument("--grader", "-g", default="openrouter/openai/gpt-4o-mini", help="Judge model.")
    args.add_argument("--attacker", "-a", default="openrouter/nousresearch/hermes-4-405b", help="Rewrite/reframing model.")
    args.add_argument("--perturb-k", type=int, default=1, help="Rewrite variants per item (default: 1).")
    args.add_argument("--sim-k", type=int, default=1, help="Scenario reframings per item (default: 1).")
    args.add_argument("--no-simulate", action="store_true", help="Skip stage 3.")
    args.add_argument("--no-perturb", action="store_true", help="Skip stage 2.")
    args.add_argument("--max-connections", type=int, default=20)
    args.add_argument("--reasoning", action="store_true",
                      help="Request reasoning mode from the attacker for scenario reframings "
                           "and strip the <think> block (parity with generate.py --reasoning). "
                           "vLLM-hosted models only — plain API providers may reject the extra body.")
    args.add_argument("--out", default=None, help="Output directory (default: demo/<risk>-<timestamp>/).")
    args.add_argument("--seed", type=int, default=0, help="Stratified pick is deterministic; seed reorders within a stratum.")
    return args.parse_args()


def stratified(dataset, n: int, seed: int) -> list:
    """Round-robin across (source, question_type) strata so a small sample still
    spans the shapes a cluster mixes — that variety is the whole point here."""
    strata: dict[tuple, list] = defaultdict(list)
    for sample in dataset:
        meta = sample.metadata or {}
        strata[(meta.get("source"), meta.get("question_type"))].append(sample)

    for key in strata:
        strata[key].sort(key=lambda s: str(s.id))
        if seed:
            offset = seed % len(strata[key])
            strata[key] = strata[key][offset:] + strata[key][:offset]

    picked, exhausted = [], False
    while len(picked) < n and not exhausted:
        exhausted = True
        for key in sorted(strata, key=lambda k: (str(k[0]), str(k[1]))):
            if strata[key]:
                picked.append(strata[key].pop(0))
                exhausted = False
                if len(picked) == n:
                    break
    return picked


async def build_artifacts(samples, task_name, args) -> list[str]:
    """Generate every requested family for exactly these samples."""
    applied = []
    stamp = datetime.now(timezone.utc).isoformat()

    def meta(family, extra=None):
        return {"prompt_version": PROMPT_VERSIONS[family], "attacker": args.attacker,
                "generated_at": stamp, "demo": True, **(extra or {})}

    if not args.no_perturb:
        for family in REWRITE_FAMILIES:
            print(f"  [{family}] generating {len(samples)} x {args.perturb_k}...")
            rows = await generate_rewrites(
                samples, family, args.attacker, args.perturb_k,
                max_connections=args.max_connections,
            )
            write_family(task_name, family, rows, meta(family))
            fallbacks = sum(1 for r in rows if r.get("fallback"))
            print(f"  [{family}] {len(rows)} rows" + (f" ({fallbacks} fell back to the original)" if fallbacks else ""))
            applied.append(family)

        rows = generate_framing(samples)
        if rows:
            write_family(task_name, "framing", rows, meta("framing"))
            covered = len({r["id"] for r in rows})
            print(f"  [framing] {len(rows)} rows over {covered}/{len(samples)} samples "
                  f"(generic rows have no templates)")
            applied.append("framing")
        else:
            print("  [framing] no sample in this subset has framing templates — skipped")

        applied.append("reconsideration")  # live, no artifact

    if not args.no_simulate:
        print(f"  [{SCENARIO_FAMILY}] generating {len(samples)} x {args.sim_k} (slowest step)...")
        rows, incomplete, reasons = await generate_scenarios(
            samples, args.attacker, args.sim_k, max_connections=args.max_connections,
            reasoning=args.reasoning,
        )
        write_family(task_name, SCENARIO_FAMILY, rows, meta(
            SCENARIO_FAMILY, {"incomplete_ids": incomplete, "incomplete_reasons": reasons},
        ))
        covered = len({r["id"] for r in rows})
        print(f"  [{SCENARIO_FAMILY}] {len(rows)} rows over {covered}/{len(samples)} samples"
              + (f", {len(incomplete)} never parsed" if incomplete else ""))
        # A demo exists to be looked at, so a mostly-empty stage 3 must be
        # impossible to miss rather than one line in the scroll-back.
        if covered < len(samples) / 2:
            print(f"\n  {'*' * 68}\n"
                  f"  ** stage 3 is mostly EMPTY: only {covered}/{len(samples)} samples got a scenario.\n"
                  f"  ** The report's scenario numbers are not meaningful at this coverage.\n"
                  f"  ** Causes are in {SCENARIO_FAMILY}.meta.json (incomplete_reasons);\n"
                  f"  ** a single repeated cause means this attacker does not fit the format.\n"
                  f"  {'*' * 68}\n")

    return applied


# ----- report -----

def condition_rows(sample) -> list[dict]:
    """Flatten one eval sample into control + every recorded condition."""
    meta = sample.metadata or {}
    scored = meta.get("perturbation_scores") or {}
    per_base = next(iter(scored.values()), {})  # cluster tasks have one base scorer

    rows = []
    for label, result in per_base.items():
        variant = None
        for family_variants in (meta.get("perturbations") or {}).values():
            for v in family_variants:
                if v.get("condition") == label:
                    variant = v
        rows.append({
            "label": label,
            "family": result.get("family", ""),
            "query": result.get("query") or "",
            "completion": (variant or {}).get("completion") or _control_completion(sample),
            "value": result.get("value"),
            "answer": result.get("answer"),
            "explanation": result.get("explanation") or "",
        })
    rows.sort(key=lambda r: (r["family"] != "control", r["family"], r["label"]))
    return rows


def _control_completion(sample) -> str:
    output = getattr(sample, "output", None)
    return output.completion if output else ""


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def score_class(value) -> str:
    v = as_float(value)
    if v is None:
        return "unscored"
    return "safe" if v >= 0.99 else ("unsafe" if v <= 0.01 else "partial")


def source_table(risk: str, tree: dict) -> str:
    """Per-benchmark rows straight off the results tree — baseline, per-condition
    safety, and the aggregate that a certificate would actually report.

    The columns are chosen to make the two failure modes visible: a benchmark
    whose scenario column is blank or thin (`scored`/`total`) is not a safe one,
    and a diagnostic is flagged because it never enters the layer above."""
    benchmarks = (tree.get(risk) or {}).get("benchmarks") or {}
    if not benchmarks:
        return ""

    def num(value) -> str:
        return "—" if value is None else f"{value:.1f}"

    rows = []
    for name, entry in sorted(benchmarks.items()):
        conditions = entry.get("conditions") or {}
        control = conditions.get("control") or {}
        scenario = conditions.get(SCENARIO_FAMILY) or {}
        coverage = (
            f"{scenario.get('scored', 0)}/{scenario.get('total', 0)}"
            if scenario else "—"
        )
        label = html.escape(name) + (
            ' <span class="fam">diagnostic</span>' if entry.get("diagnostic") else ""
        )
        rows.append(
            f"<tr><td>{label}</td>"
            f"<td>{num(entry.get('baseline'))}</td>"
            f"<td>{num((entry.get('aggregate') or {}).get('worst'))}</td>"
            f"<td>{num((entry.get('aggregate') or {}).get('mean'))}</td>"
            f"<td>{num(scenario.get('safety'))}</td>"
            f"<td>{coverage}</td>"
            f"<td>{num(control.get('divergence') or scenario.get('divergence'))}</td>"
            f"<td>{', '.join(sorted((scenario or control).get('scorers') or {}))}</td>"
            "</tr>"
        )

    aggregate = (tree.get(risk) or {}).get("aggregate") or {}
    return f'''
<h2>By benchmark</h2>
<p class="note">All figures are safety 0-100, higher safer. <b>Baseline</b> is the
unperturbed control and is deliberately <em>not</em> part of the aggregate —
stage 1 is the reference the perturbed conditions are read against. <b>Worst</b>
pools each sample's lowest condition; <b>mean</b> pools them evenly, and a gap
between the two means one transform is carrying the result. <b>Scored</b> is how
many samples the scenario condition actually judged: a thin column is a thin
measurement, not a safe one.</p>
<p class="note">Cluster aggregate: worst {num(aggregate.get("worst"))} ·
mean {num(aggregate.get("mean"))} · baseline {num((tree.get(risk) or {}).get("baseline"))}</p>
<table class="metrics wide">
<tr><th>benchmark</th><th>baseline</th><th>worst</th><th>mean</th>
<th>scenario</th><th>scored</th><th>divergence</th><th>scorers</th></tr>
{"".join(rows)}
</table>'''


def render_report(log, risk: str, args, tree=None) -> str:
    samples = log.samples or []
    blocks = []
    for sample in samples:
        meta = sample.metadata or {}
        rows = condition_rows(sample)
        control = next((r for r in rows if r["family"] == "control"), None)
        control_value = control["value"] if control else None

        # An absent condition is invisible otherwise — a sample whose reframing
        # never parsed just renders one row shorter than its neighbours, which
        # reads as nothing at all.
        if not getattr(args, "no_simulate", False) and not any(
            r["family"] == SCENARIO_FAMILY for r in rows
        ):
            rows.append({
                "label": "scenario_variant_1", "family": SCENARIO_FAMILY,
                "query": "", "completion": "", "value": None,
                "answer": "NOT GENERATED — reframing never parsed",
                "explanation": "See scenario.meta.json (incomplete_reasons) for the cause.",
            })

        cond_html = []
        for row in rows:
            v = as_float(row["value"])
            drift = ""
            if row["family"] != "control" and v is not None and as_float(control_value) is not None:
                drift = f'<span class="drift">Δ {abs(v - as_float(control_value)):.2f}</span>'
            cond_html.append(f'''
        <details class="cond {score_class(row["value"])}">
          <summary>
            <span class="lbl">{html.escape(row["label"])}</span>
            <span class="fam">{html.escape(row["family"])}</span>
            <span class="val">{"—" if v is None else f"{v:.2f}"}</span>
            <span class="ans">{html.escape(str(row["answer"] or ""))}</span>
            {drift}
          </summary>
          <div class="pair">
            <div><h4>Prompt sent</h4><pre>{html.escape(row["query"][:4000])}</pre></div>
            <div><h4>Model response</h4><pre>{html.escape(row["completion"][:4000])}</pre></div>
          </div>
          <h4>Judge</h4><pre class="judge">{html.escape(row["explanation"][:2000])}</pre>
        </details>''')

        blocks.append(f'''
    <section class="sample">
      <h2>{html.escape(str(sample.id))}</h2>
      <p class="tags">
        <span>{html.escape(str(meta.get("source", "")))}</span>
        <span>{html.escape(str(meta.get("question_type", "")))}</span>
        <span>{html.escape(str(meta.get("elicitation_family", "")))}</span>
        {'<span class="sys">has system prompt</span>' if meta.get("system_prompt") else ""}
      </p>
      <p class="criterion">{html.escape(str(meta.get("criterion", ""))[:400])}</p>
      {"".join(cond_html)}
    </section>''')

    results = log.results
    metrics = []
    if results:
        for s in results.scores or []:
            for name, m in (s.metrics or {}).items():
                metrics.append(f"<tr><td>{html.escape(name)}</td><td>{m.value:.2f}</td></tr>")

    return f'''<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(risk)} pipeline demo</title>
<style>
  :root {{ color-scheme: light dark; --safe:#2e7d32; --unsafe:#c62828; --partial:#ef6c00; --unscored:#777; }}
  body {{ font: 15px/1.5 ui-sans-serif, system-ui, sans-serif; margin: 0 auto; padding: 2rem; max-width: 1100px; }}
  h1 {{ margin-bottom: .2rem; }}
  .meta {{ color: #666; margin-bottom: 2rem; }}
  table.metrics {{ border-collapse: collapse; margin-bottom: 2rem; }}
  table.metrics td, table.metrics th {{ border: 1px solid #8883; padding: .3rem .7rem; }}
  table.metrics th {{ text-align: left; font-size: .8rem; color: #888; }}
  table.wide td:not(:first-child), table.wide th:not(:first-child) {{ text-align: right; }}
  .note {{ color: #666; font-size: .85rem; margin-top: -.5rem; }}
  h2 {{ font-size: 1.1rem; margin-top: 2rem; }}
  .sample {{ border: 1px solid #8884; border-radius: 8px; padding: 1rem 1.2rem; margin-bottom: 1.5rem; }}
  .sample h2 {{ font-size: 1rem; font-family: ui-monospace, monospace; margin: 0 0 .5rem; }}
  .tags span {{ display: inline-block; background: #8882; border-radius: 4px; padding: .1rem .5rem; margin-right: .3rem; font-size: .8rem; }}
  .tags .sys {{ background: #1976d233; }}
  .criterion {{ color: #666; font-size: .85rem; font-style: italic; }}
  .cond {{ border-left: 4px solid var(--unscored); padding: .3rem .6rem; margin: .35rem 0; background: #8881; border-radius: 0 4px 4px 0; }}
  .cond.safe {{ border-color: var(--safe); }} .cond.unsafe {{ border-color: var(--unsafe); }}
  .cond.partial {{ border-color: var(--partial); }}
  summary {{ cursor: pointer; display: flex; gap: .6rem; align-items: baseline; flex-wrap: wrap; }}
  .lbl {{ font-family: ui-monospace, monospace; font-size: .85rem; }}
  .fam {{ font-size: .75rem; color: #888; }}
  .val {{ font-weight: 700; margin-left: auto; }}
  .ans {{ font-size: .8rem; color: #666; max-width: 22ch; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .drift {{ font-size: .75rem; color: var(--partial); }}
  .pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
  @media (max-width: 800px) {{ .pair {{ grid-template-columns: 1fr; }} }}
  h4 {{ margin: .6rem 0 .2rem; font-size: .75rem; text-transform: uppercase; color: #888; }}
  pre {{ white-space: pre-wrap; word-break: break-word; background: #8881; padding: .6rem; border-radius: 4px;
         font-size: .8rem; max-height: 24rem; overflow: auto; margin: 0; }}
  pre.judge {{ opacity: .8; }}
</style></head><body>
<h1>{html.escape(risk)} — pipeline demo</h1>
<p class="meta">{len(samples)} samples · target <code>{html.escape(args.model)}</code> ·
judge <code>{html.escape(args.grader)}</code> · attacker <code>{html.escape(args.attacker)}</code><br>
Score is safety in [0,1], higher = safer. Δ is drift from the control.<br>
Harm propensity is 0-100, higher = <em>more harmful</em>, and pools each sample's
worst condition. <code>perturbation_uplift</code>/<code>scenario_uplift</code> are the
share of samples a transform pushed below their own control.</p>
<table class="metrics">{"".join(metrics)}</table>
{source_table(risk, tree or {})}
<h2>By sample</h2>
{"".join(blocks)}
</body></html>'''


def main():
    args = parse()
    if not available(args.risk):
        raise SystemExit(f"{args.risk}.csv has not been built — run: uv run python3 -m datasets.prepare.cluster.prepare --risk {args.risk}")
    if args.no_perturb and args.no_simulate:
        raise SystemExit("Nothing to demo: both --no-perturb and --no-simulate were passed.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = Path(args.out) if args.out else REPO_ROOT / "demo" / f"{args.risk}-{stamp}"
    artifact_dir, log_dir = out / "artifacts", out / "logs"
    out.mkdir(parents=True, exist_ok=True)

    # Keep the committed artifact set out of harm's way: artifact_path() reads
    # this module global at call time, so both the writes below and the reads
    # inside registry.apply_stages land in the scratch dir.
    artifacts.GENERATED_DIR = artifact_dir

    task = CLUSTER_TASKS[args.risk](grader=args.grader)
    picked = stratified(task.dataset, args.samples, args.seed)
    ids = {str(s.id) for s in picked}
    task.dataset = task.dataset.filter(lambda s: str(s.id) in ids)

    print(f"\n=== {args.risk}: {len(picked)} samples ===")
    for sample in picked:
        meta = sample.metadata or {}
        flag = " +system" if meta.get("system_prompt") else ""
        print(f"  {sample.id}  [{meta.get('source')}/{meta.get('question_type')}{flag}]")

    print("\n=== generating artifacts ===")
    applied = asyncio.run(build_artifacts(picked, args.risk, args))

    print("\n=== running eval ===")
    entry = {args.risk: {"tasks": [task], "name": args.risk}}
    built = apply_stages(
        entry,
        families=[f for f in applied if f != SCENARIO_FAMILY],
        k=args.perturb_k,
        sim_k=None if args.no_simulate else args.sim_k,
    )
    logs = inspect_eval(
        built[args.risk]["tasks"],
        model=args.model,
        log_dir=str(log_dir),
        continue_on_fail=True,
        retry_on_error=2,
        fail_on_error=0.2,
        max_connections=args.max_connections,
        cache=True,
    ) or []

    if not logs:
        raise SystemExit("Eval produced no logs.")

    # The same breakdowns certify.py stores in models.json, so a demo run shows
    # the per-source detail the results panel deliberately leaves out.
    tree = results_tree.build(logs, DIAGNOSTIC_SOURCES)
    aggregate = results_tree.model_aggregate(tree)

    report = out / "report.html"
    report.write_text(
        render_report(logs[0], args.risk, args, tree),
        encoding="utf-8",
    )

    summary = {
        "risk": args.risk,
        "samples": sorted(ids),
        "families": applied + ([] if args.no_simulate else [SCENARIO_FAMILY]),
        "model": args.model, "grader": args.grader, "attacker": args.attacker,
        "aggregate": aggregate,
        "results": tree,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"\n=== done ===\nReport:     {report}")
    print(f"Transcripts: uv run inspect view --log-dir {log_dir}")


if __name__ == "__main__":
    main()
