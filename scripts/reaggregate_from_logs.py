"""
Rebuild a model's models.json entry from the .eval logs already on disk, without
re-running generation or grading.

certify.py builds a model's results from the logs it produced in the same run. A
cluster that failed on fail_on_error — a provider content-filter refusal tripping
the >10% error threshold — still leaves a complete, gradeable log behind: the
surviving samples carry real judgments and the refusals are recorded per source.
This reads those logs back and rebuilds the per-benchmark results tree over the
samples that scored, with coverage (pipeline/utils/results.py::_coverage) folding
the refusals into the denominator so the partial figure is self-describing rather
than hidden behind a 100% computed only over the prompts that got through.

It only re-derives clusters that never certified cleanly (status != success, or
absent from `scores`); a cluster that ran complete is already correct and left
untouched. Every other field is preserved — including aa_intelligence_index /
aa_model_match — and update() recomputes the headline across all four risks and
clears the now-superseded partial_scores.

Runs offline; touches no model or judge. update() writes a models_previous.json
backup before it saves.

Usage:
    uv run python3 scripts/reaggregate_from_logs.py gpt-5.6-sol
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Run from scripts/, so the repo root is not on the path; certify and pipeline
# both live there.
sys.path.insert(0, str(REPO_ROOT))

from inspect_ai.log import read_eval_log

from certify import check_status, update
from pipeline.utils import results as results_tree
from pipeline.utils.graders import DIAGNOSTIC_SOURCES, load_models_with_check


def latest_log_per_cluster(model_id: str) -> dict[str, object]:
    """The most recent .eval log for each risk cluster, keyed by task name.

    A rerun leaves several logs per cluster; filenames are timestamp-prefixed, so
    the lexically greatest name is the newest run — the one models.json reflects.
    """
    newest: dict[str, tuple[str, object]] = {}
    for path in sorted((REPO_ROOT / "logs" / model_id).glob("*.eval")):
        log = read_eval_log(str(path))
        task = str(log.eval.task)
        if task not in newest or path.name > newest[task][0]:
            newest[task] = (path.name, log)
    return {task: log for task, (_, log) in newest.items()}


def _benchmark_coverage(subtree: dict) -> str:
    """A one-line scored/total summary across a risk's benchmarks."""
    parts = []
    for bench, data in subtree.get("benchmarks", {}).items():
        scored = total = 0
        for name, cond in data.get("conditions", {}).items():
            if name == "control":
                continue
            scored += cond.get("scored", 0)
            total += cond.get("total", 0)
        pct = f"{100 * scored / total:.0f}%" if total else "n/a"
        parts.append(f"{bench} {scored}/{total} ({pct})")
    return ", ".join(parts)


def main(model_id: str) -> None:
    models, idx = load_models_with_check(model_id)
    if idx == -1:
        raise SystemExit(f"{model_id} is not in models.json")
    prev = models[idx]
    prev_status = prev.get("status", {})

    logs = latest_log_per_cluster(model_id)
    if not logs:
        raise SystemExit(f"no logs under logs/{model_id}")

    incomplete = [
        risk for risk in logs
        if prev_status.get(risk, {}).get("status") != "success"
        or risk not in prev.get("scores", {})
    ]
    if not incomplete:
        print(f"{model_id}: every cluster already complete — nothing to do.")
        return

    # {**prev} preserves identity and the AA fields; only the incomplete clusters
    # are rebuilt, and update() merges them back over the complete ones.
    new = {**prev, "scores": {}, "results": {}, "status": {}}
    for risk in incomplete:
        tree = results_tree.build([logs[risk]], DIAGNOSTIC_SOURCES)
        subtree = tree.get(risk)
        if not subtree:
            print(f"[skip] {risk}: log produced no scorable samples")
            continue
        status = check_status([logs[risk]])
        # Routing endpoints came from a live API call at run time and are not in
        # the log, so carry them over from the previous status.
        if "endpoints" in prev_status.get(risk, {}):
            status["endpoints"] = prev_status[risk]["endpoints"]

        agg = subtree.get("aggregate") or {}
        new["results"][risk] = subtree
        new["scores"][risk] = agg.get("worst") if agg.get("worst") is not None else -1
        new["status"][risk] = status
        print(
            f"[ok] {risk}: worst={agg.get('worst')} status={status['status']} "
            f"({status['completed_samples']}/{status['total_samples']} answered)\n"
            f"      coverage: {_benchmark_coverage(subtree)}"
        )

    if not new["results"]:
        print("Nothing rebuilt.")
        return

    update(new, models, idx)
    print(f"\nWrote {model_id} to models/models.json (backup: models/models_previous.json).")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: reaggregate_from_logs.py <model_id>")
    main(sys.argv[1])
