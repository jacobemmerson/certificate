'''
author: @tae

Runs the certification pipeline:
  stage 1 — plain benchmark evals (pipeline/stage1_evaluation)
  stage 2 — surface-perturbation reliability auditing (pipeline/stage2_perturbation)
  stage 3 — scenario simulation (pipeline/stage3_simulation)
Benchmarks are registered in pipeline/registry.py.

Stages 2 and 3 replay *pregenerated* artifacts from datasets/generated/
(produced once by generate.py) — the only models called here are the target
and the judges, and every evaluated model sees the exact same perturbed
variants and scenarios. Run generate.py before the first --perturb/--simulate
certification (this script validates the artifacts and fails fast with the
exact command otherwise).
'''

import json
import os
import sys
from argparse import ArgumentParser

from inspect_ai import eval


def display_mode() -> str:
    '''
    Inspect's default is the Textual TUI, whose worker cancels the whole eval if
    the terminal goes away or resizes under it (a disconnected/idle SSH pty, a
    SIGWINCH) — which silently kills unattended overnight batches with a
    CancelledError, not a model error. Force a non-interactive display whenever
    stdout is not a real terminal, honouring an explicit INSPECT_DISPLAY override.
    '''
    return os.environ.get("INSPECT_DISPLAY") or ("full" if sys.stdout.isatty() else "log")
from pipeline.artifacts import validate_artifacts
from pipeline.registry import init_benchmarks, apply_stages, ALL_PERTURB_FAMILIES
from pipeline.utils import results as results_tree
from pipeline.utils.graders import (
    DIAGNOSTIC_SOURCES, load_graders, load_models_with_check, validate_graders,
)

def parse():
    
    args = ArgumentParser()
    args.add_argument(
        "--model", "-m", required=True, help="The model to be evaluated using AISI inspect."
    )
    args.add_argument(
        "--grader", "-g", required=False, default=None, help="Grader model override (single model). If omitted, loads from GRADERS.md"
    )
    args.add_argument(
        "--name", "-n", required=False, default=None, help="The name of the model for formatting the certificate table."
    )
    args.add_argument(
        "--provider", "-p", required=False, default=None, help="The provider of the model for formatting the certificate table."
    )
    args.add_argument(
        "--region", "-r", required=False, default=None, help="The region of the world where the model is developed and data is sourced."
    )
    args.add_argument(
        "--specialty", "-s", required=False, default=None, help="What the model has been tuned or designated to do (i.e. coding, math, etc)."
    )
    args.add_argument(
        "--epochs", "-e", required=False, default=1, help="The number of turns to generate a response per sample and average over."
    )
    args.add_argument(
        "--rerun", required=False, action='store_true', help="Reruns all results regardless of whether they are present in an existing file."
    )
    args.add_argument(
        "--llamaguard", required=False, default="openrouter/meta-llama/llama-guard-4-12b",
        help="LlamaGuard model for adversarial harm scoring (default: openrouter/meta-llama/llama-guard-4-12b)."
    )
    args.add_argument(
        "--limit", "-l", required=False, type=int, default=None,
        help="Randomly sample this many examples per task (default: run the full dataset). WARNING: if limit is present, results will not be saved. They can still be accessed in logs/MODEL_NAME/"
    )
    args.add_argument(
        "--only", "-o", required=False, nargs="+", metavar="RISK",
        help="Run only these systemic-risk clusters (e.g. --only cyber manipulation). "
             "Other existing results are preserved."
    )
    args.add_argument(
        "--perturb", required=False, nargs="+", default=ALL_PERTURB_FAMILIES, choices=sorted(ALL_PERTURB_FAMILIES),
        metavar="FAMILY",
        help="Run surface-perturbation reliability auditing (see pipeline/stage2_perturbation/) for these families "
             "(e.g. --perturb paraphrase reconsideration) on top of every benchmark in --only "
             "(or all benchmarks if --only is omitted). All by default. Replays the pregenerated "
             "variants from datasets/generated/ (run generate.py first); reconsideration runs live."
    )
    args.add_argument(
        "--no-perturb", required=False, action='store_true',
        help="Skip stage-2 perturbation auditing entirely (e.g. for a --simulate-only run — "
             "--perturb is on by default)."
    )
    args.add_argument(
        "--perturb-k", required=False, type=int, default=1,
        help="Use up to this many stored variants per item for the rewrite families "
             "(paraphrase, register, identity_strip); default=1. Must not exceed the k the "
             "artifacts were generated with."
    )
    args.add_argument(
        "--simulate", required=False, action='store_true',
        help="Run stage-3 scenario simulation (see pipeline/stage3_simulation/) on top of every "
             "benchmark in --only (or all benchmarks if --only is omitted): the target is re-run "
             "on the pregenerated deployment-scenario reframings from datasets/generated/ "
             "(run generate.py --simulate first). Composes with --perturb in one run/one log: "
             "the panel reports safety_scenario/stability_scenario next to stage 2's safety_perturbed/stability."
    )
    args.add_argument(
        "--sim-k", required=False, type=int, default=1,
        help="Use up to this many stored scenarios per item under --simulate; default=1."
    )
    args.add_argument(
        "--max-connections", required=False, type=int, default=100,
        help="Max concurrent model connections Inspect opens (default: 100)."
    )
    args.add_argument(
        "--working-limit", required=False, type=int, default=900,
        help="Max working seconds per sample before it fails and retries; bounds a "
             "hung provider connection so one stuck request can't wedge the whole "
             "run (default: 900). Excludes time spent waiting on rate limits/retries."
    )

    return args.parse_args()

# ----- Updates models/models.json -----

def update(results, models, idx):
    '''
    Summarises results and updates models/models.json
    '''

    if models: # store previous only if previous results exist
        with open('models/models_previous.json', 'w') as f: # store as a safety net
            json.dump(models, f, indent=4)

    # ----- store ------
    # if idx != -1, model results already exist
    if idx != -1:
        prev = models[idx]
        prev_status = prev.get('status', {})

        # never overwrite a previously-complete benchmark result with a
        # partial/failed rerun — drop the demoted rerun and keep the old one
        for benchmark, status in list(results.get('status', {}).items()):
            previously_complete = (
                benchmark in prev.get('scores', {})
                and prev_status.get(benchmark, {}).get('status', 'success') == 'success'
            )
            if status.get('status') != 'success' and previously_complete:
                print(f"[WARNING] {benchmark}: rerun was {status.get('status')}; keeping previous complete result")
                results['scores'].pop(benchmark, None)
                results['results'].pop(benchmark, None)
                results['status'].pop(benchmark, None)

        # take values from overlapping keys from the new results (right side of pipe operator)
        results['scores'] = prev.get('scores', {}) | results['scores']
        results['results'] = prev.get('results', {}) | results.get('results', {})
        results['status'] = prev_status | results.get('status', {})
        # Recomputed after the merge, so a --only rerun reports across every
        # risk the model has, not just the ones this run touched.
        results['aggregate'] = results_tree.model_aggregate(results['results'])
        models[idx] = results
    else:
        # add new entry
        models.append(results)

    # write models file back
    with open('models/models.json', 'w') as f:
        json.dump(models, f, indent=4)


# ----- main ------

if __name__ == "__main__":

    args = parse()
    grader = args.grader if args.grader else load_graders()
    model_id = args.model.split("/")[-1]
    log_dir = f"logs/{model_id}"

    print(f"Model: {model_id}")
    print(f"Grader(s): {grader}")
    print(f"Log Directory: {log_dir}")

    # ----- task master list -----
    BENCHMARKS = init_benchmarks(grader, llamaguard_model=args.llamaguard)  # see pipeline/registry.py for all tasks

    # Stage 2 and stage 3 compose in one run: both layer condition families
    # onto the same Task (one control generation, one log), and the wrapped
    # scorers report them under separate metric pools — safety_perturbed/stability for
    # the perturbation families, safety_scenario/stability_scenario for the
    # scenario family. The certification score is the worst condition across
    # every enabled family (see pipeline/utils/scoring.py).
    run_perturb = bool(args.perturb) and not args.no_perturb

    # check for existing model results
    models, idx = load_models_with_check(model_id)
    if idx != -1:
        print(f"Results Found: Model index at {idx}")

    only = set(args.only) if args.only else None

    tasks_to_skip = set()
    if only:
        # run only the requested benchmarks; skip everything else
        tasks_to_skip = set(BENCHMARKS.keys()) - only
        unknown = only - set(BENCHMARKS.keys())
        if unknown:
            print(f"[WARNING] Unknown benchmark keys (ignored): {', '.join(sorted(unknown))}")
    elif idx != -1 and not args.rerun:
        # default: skip benchmarks that already have results
        tasks_to_skip = set(models[idx]['scores'].keys())

    if tasks_to_skip:
        print(f"Skipping: {', '.join(sorted(tasks_to_skip))}")
        BENCHMARKS = {key: entry for key, entry in BENCHMARKS.items() if key not in tasks_to_skip}

    # Fail fast — before any eval spends money — on the two things that make a
    # whole run worthless: an unusable judge (every sample errors on scoring, or
    # worse, silently abstains into a perfect score), and missing artifacts.
    validate_graders(grader)

    validate_artifacts(
        BENCHMARKS,
        families=args.perturb if run_perturb else None,
        simulate=args.simulate,
        perturb_k=args.perturb_k,
        sim_k=args.sim_k,
        limit=args.limit,
    )

    if run_perturb or args.simulate:
        # Attaches one replay solver per enabled condition family (stage-2
        # perturbations and/or the stage-3 scenario) directly onto each
        # benchmark's own Task (see pipeline/registry.py::apply_stages) — same
        # benchmark keys/log paths as a plain run, one log per benchmark. The
        # variants come from datasets/generated/; no rewrite/reframing model
        # is called (reconsideration runs live).
        BENCHMARKS = apply_stages(
            BENCHMARKS,
            families=args.perturb if run_perturb else [],
            k=args.perturb_k,
            sim_k=args.sim_k if args.simulate else None,
        )

    def check_status(evaluations):
        '''
        Summarise a benchmark's EvalLogs into a status record:
        success (every task log succeeded), partial (some did), or failed —
        plus completed/total sample counts. Stored per benchmark in
        models.json so an incomplete run is distinguishable from a clean one.
        '''
        ok = sum(1 for log in evaluations if log.status == "success")
        # log.results is None on an errored task, so counting only from it
        # reports a failed run as "0/0 samples" — which reads like an empty
        # dataset rather than a run that broke. Fall back to the samples.
        completed = sum(
            (getattr(log.results, "completed_samples", None) if log.results else None)
            or sum(1 for s in (log.samples or []) if not s.error)
            for log in evaluations
        )
        total = sum(
            (getattr(log.results, "total_samples", None) if log.results else None)
            or len(log.samples or [])
            for log in evaluations
        )
        errored = sum(1 for log in evaluations for s in (log.samples or []) if s.error)
        if errored:
            first = next(s for log in evaluations for s in (log.samples or []) if s.error)
            reason = str(first.error).strip().splitlines()[-1][:200]
            print(f"[ERROR] {errored} sample(s) errored; first: {reason}")
        status = "success" if ok == len(evaluations) else ("partial" if ok else "failed")
        return {"status": status, "completed_samples": completed, "total_samples": total}

    def start_eval(tasks: list):
        return eval(
            tasks,
            model=args.model,
            log_dir=log_dir,
            continue_on_fail=True,
            retry_on_error=2,
            # tolerate scattered sample-level errors (e.g. an unparseable
            # OpenRouter response that slips past retries) instead of failing
            # the whole task — only fail if >10% of samples error
            fail_on_error=0.1,
            epochs=args.epochs,
            sample_shuffle=bool(args.limit),
            limit=args.limit,
            max_connections=args.max_connections,
            working_limit=args.working_limit,
            display=display_mode(),
            # Eval-level cache benefits judge/grader calls (the bulk of API
            # traffic under --perturb) and retries. The replay/reconsideration
            # solvers opt out explicitly (cache=False in
            # pipeline/utils/replay.py): their target calls
            # replay identical prompts across epochs and must stay independent
            # generations, so inheriting this would collapse them.
            cache=True,
        )

    # ----- run -----
    # One eval() over every cluster, not one per cluster in a Python loop.
    # Each cluster is now a single task, so a serial loop would leave the
    # connection pool idle while one cluster drained — Inspect schedules
    # across tasks itself. Per-cluster reporting comes from partitioning the
    # returned logs by task name afterwards; `continue_on_fail` and
    # `fail_on_error` keep one bad cluster from sinking the rest.
    scores = {}
    results_by_risk = {}
    statuses = {}

    all_tasks = [task for entry in BENCHMARKS.values() for task in entry["tasks"]]
    try:
        logs = start_eval(all_tasks) or []
    except Exception as e:
        print(f"[ERROR] evaluation failed: {e}")
        logs = []
        statuses = {key: {"status": "failed", "error": str(e)} for key in BENCHMARKS}

    by_cluster: dict[str, list] = {}
    for log in logs:
        by_cluster.setdefault(str(log.eval.task), []).append(log)

    for benchmark, entry in BENCHMARKS.items():
        res = by_cluster.get(entry["name"], [])
        if not res:
            statuses.setdefault(benchmark, {"status": "failed", "error": "no log produced"})
            print(f"[ERROR] {benchmark}: no log produced")
            continue

        statuses[benchmark] = check_status(res)
        if statuses[benchmark]['status'] != 'success':
            print(f"[WARNING] {benchmark}: run was {statuses[benchmark]['status']} "
                  f"({statuses[benchmark]['completed_samples']}/{statuses[benchmark]['total_samples']} samples)")

        # score only the task logs that succeeded; a partial run's
        # scores are stored but flagged by its status record
        ok = [log for log in res if log.status == "success"]
        if ok:
            # One tree per risk, replacing what used to be three parallel
            # sections joined by hand. Every condition of every benchmark comes
            # out of the same log (pipeline/registry.py::apply_stages), so the
            # builder splits them by family itself rather than needing the run
            # to be sliced up here.
            tree = results_tree.build(ok, DIAGNOSTIC_SOURCES)
            results_by_risk.update(tree)

            # The flat headline stays: certify.py's own skip logic reads
            # `scores.keys()` to decide what a rerun can leave alone, and a
            # reader wants one number per risk without walking the tree.
            aggregate = (tree.get(entry["name"], {}).get("aggregate") or {})
            scores[benchmark] = (
                aggregate.get("worst")
                if aggregate.get("worst") is not None else -1
            )

    if (not args.limit):
        # ----- format and store results -----
        results = {
            "id": model_id,
            "name": args.name,
            "company": args.provider,
            "region": args.region,
            "specialty": args.specialty,
            "scores": scores,
            "aggregate": results_tree.model_aggregate(results_by_risk),
            "results": results_by_risk,
            "status": statuses,
        }

        update(results, models, idx)
