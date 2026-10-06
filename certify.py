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

import os
import shutil
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

from inspect_ai import eval_set
from inspect_ai.log import read_eval_log


def display_mode() -> str:
    '''
    Inspect's default is the Textual TUI, whose worker cancels the whole eval if
    the terminal goes away or resizes under it (a disconnected/idle SSH pty, a
    SIGWINCH) — which silently kills unattended overnight batches with a
    CancelledError, not a model error. Force a non-interactive display whenever
    stdout is not a real terminal, honouring an explicit INSPECT_DISPLAY override.
    '''
    return os.environ.get("INSPECT_DISPLAY") or ("full" if sys.stdout.isatty() else "log")
from pipeline.artifacts import REPEAT_FAMILIES, family_ids, load_family, task_name, validate_artifacts
from pipeline.registry import init_benchmarks, apply_stages, require_families_column, ALL_PERTURB_FAMILIES
from pipeline.stage3_simulation.classify import DEFAULT_CLASSIFIER
from pipeline.stage3_simulation.prompts import DEPTH
from pipeline.utils.scoring import SCENARIO
from pipeline.utils import results as results_tree
from pipeline.utils import retry_policy
from pipeline.utils import routing as provider_routing_api
from pipeline.utils.graders import (
    load_graders, load_models_with_check, validate_graders, validate_target,
    model_result_path, rebuild_models_json, write_json_atomic,
)

# OpenRouter's provider routing accepts a price ceiling per token class, in USD
# per million tokens: https://openrouter.ai/docs/guides/routing/provider-selection
MAX_PRICE_KEYS = ("prompt", "completion", "request", "image")


def parse_max_price(pairs: list[str] | None) -> dict[str, float] | None:
    '''
    Turn `--max-price prompt=1.25 completion=4.25` into OpenRouter's max_price
    object. Returns None when the flag is absent, i.e. no routing constraint.

    KEY=VALUE rather than positional values: the ceilings are per token class
    and completion is typically several times prompt, so a swapped pair would
    quietly route the run to the wrong endpoints instead of erroring. Every
    failure here is a SystemExit before any eval starts — a cap that parsed to
    nothing would run the whole suite at full price and report no problem.
    '''
    if not pairs:
        return None

    caps: dict[str, float] = {}
    for pair in pairs:
        key, delimiter, value = pair.partition("=")
        if not delimiter or key not in MAX_PRICE_KEYS:
            raise SystemExit(
                f"--max-price expects KEY=VALUE with KEY one of "
                f"{', '.join(MAX_PRICE_KEYS)} (got {pair!r})."
            )
        try:
            price = float(value)
        except ValueError:
            raise SystemExit(f"--max-price {key}: {value!r} is not a number.")
        if price < 0:
            raise SystemExit(f"--max-price {key}: price must not be negative.")
        caps[key] = price

    return caps


def provider_routing(
    max_price: dict[str, float] | None, endpoints: list[dict] | None
) -> dict | None:
    '''
    The OpenRouter `provider` routing object for this run, or None if the run
    asked for no routing preferences.

    `sort: "price"` is deliberately not used for --cheapest: it orders the whole
    endpoint pool, so its cheapest candidate is routinely a quantized or
    short-context deployment — a numerically different model, certified under
    the capable one's name. pipeline/utils/routing.py resolves the capable
    endpoints instead and pins them by tag, cheapest first.
    '''
    routing: dict = {}
    if max_price:
        routing["max_price"] = max_price
    if endpoints:
        routing.update(provider_routing_api.cheapest_capable_routing(endpoints))
    return routing or None


def estimate_calls(benchmarks, families, k: int, sim_k: int | None, graders, limit: int | None = None, epochs: int = 1) -> dict:
    '''
    Upper-bound target / judge / classifier call counts per risk, from the
    stored artifact rows — printed before any canary call so a 60k-call run is
    a decision, not a surprise. See docs/superpowers/plans/2026-10-05-ws-e-scale.md Task 6.
    '''
    n_graders = len(graders) if isinstance(graders, list) else 1
    estimate = {}
    for key, entry in benchmarks.items():
        totals = estimate.setdefault(key, dict.fromkeys(("samples", "target", "judge", "classifier"), 0))
        for base in entry["tasks"]:
            size = len(base.dataset)
            n = size if limit is None else min(limit, size)
            share = n / size if size else 0.0
            target, classifier = n, 0
            for family in families or []:
                applicable = family_ids(base, family)
                if not applicable:
                    continue
                if family == "reconsideration":
                    target += round(share * len(applicable))
                    continue
                stored = sum(
                    min(k, len([r for r in v if not r.get("fallback")])) if family in REPEAT_FAMILIES else len(v)
                    for i, v in load_family(task_name(base), family).items() if i in applicable
                )
                target += round(share * stored)
            if sim_k is not None:
                applicable = family_ids(base, SCENARIO)
                trees = sum(
                    min(sim_k, len(v)) for i, v in load_family(task_name(base), SCENARIO).items() if i in applicable
                )
                walked = round(share * trees)
                target += DEPTH * walked
                classifier = (DEPTH - 1) * walked  # the last turn is never classified
            # ponytail: judge counts every target call x graders; detection rows are
            # regex-scored, so this is an upper bound — subtract per-source shapes if it matters
            for column, value in zip(totals, (n, target * epochs, target * n_graders * epochs, classifier * epochs)):
                totals[column] += value
    return estimate


LIMIT_SHUFFLE_SEED = 1


def run_dir(model_id: str, run_id: str, limit: int | None) -> Path:
    '''
    logs/<model>/<run_id>, or logs/<model>/<run_id>-limitN under --limit: a
    limit is not part of eval_set's task identity, so a 3-sample success log in
    `current` would otherwise be reused as the full run's result.
    '''
    suffix = f"-limit{limit}" if limit else ""
    return Path("logs") / model_id / f"{run_id}{suffix}"


def move_aside(path: Path) -> Path | None:
    '''--rerun: keep the old run dir next to the new one under a timestamp.'''
    if not path.exists():
        return None
    target = path.with_name(f"{path.name}-{datetime.now():%Y%m%dT%H%M%S}")
    shutil.move(str(path), str(target))
    return target


def start_eval(tasks: list, model: str, model_args: dict, log_dir: Path, args) -> list:
    '''
    eval_set over one run dir resumes the matching log per task: a `success`
    log is skipped and an unfinished one re-runs only its missing samples, so
    a preempted or killed run resumes where it stopped. Logs of other tasks
    (clusters already scored, or outside --only) and of older configurations
    are left alone; a task whose configuration changed re-runs in full.
    Skipped tasks come back as header-only logs (no samples); the results tree
    needs samples, so those are re-read.
    '''
    _, logs = eval_set(
        tasks,
        log_dir=str(log_dir),
        log_dir_allow_dirty=True,
        retry_attempts=3,
        retry_wait=60,
        model=model,
        model_args=model_args,
        continue_on_fail=True,
        # tolerate scattered sample-level errors (e.g. an unparseable
        # OpenRouter response that slips past retries) instead of failing
        # the whole task — only fail if >10% of samples error
        fail_on_error=0.1,
        epochs=args.epochs,
        # a fixed seed, not True: a resumed --limit run must select the
        # same subset or eval_set discards the previous log
        sample_shuffle=LIMIT_SHUFFLE_SEED if args.limit else None,
        limit=args.limit,
        max_connections=args.max_connections,
        max_retries=args.max_retries,
        attempt_timeout=args.attempt_timeout,
        timeout=args.timeout,
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
    reused = []
    for log in logs:
        if not log.samples:
            print(f"[resume] reused finished log for {log.eval.task} from {log_dir} (--rerun or a new --run-id to regenerate)")
            log = read_eval_log(log.location)
        reused.append(log)
    return reused


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
        "--epochs", "-e", required=False, type=int, default=1, help="The number of turns to generate a response per sample and average over."
    )
    args.add_argument(
        "--rerun", required=False, action='store_true',
        help="Rerun every requested risk even if it already has results, and move the run's "
             "log directory (logs/MODEL/RUN_ID) aside under a timestamp first, so the resume "
             "logic cannot pick up its old logs."
    )
    args.add_argument(
        "--run-id", required=False, default="current",
        help="Name of the log directory under logs/MODEL/ this run writes to and resumes from "
             "(default: current). A finished task in it is skipped and an unfinished one "
             "re-runs only its missing samples; a --limit run uses RUN_ID-limitN."
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
             "Other existing results are preserved. A finished log in the run dir is reused, not re-run."
    )
    args.add_argument(
        "--perturb", required=False, nargs="+", default=sorted(ALL_PERTURB_FAMILIES), choices=sorted(ALL_PERTURB_FAMILIES),
        metavar="FAMILY",
        help="Run surface-perturbation auditing for these families (paraphrase register past_tense "
             "multilingual framing persona reconsideration; default: all). Replays datasets/generated/ "
             "(run generate.py first); reconsideration runs live."
    )
    args.add_argument(
        "--no-perturb", required=False, action='store_true',
        help="Skip stage-2 perturbation auditing entirely (e.g. for a --simulate-only run — "
             "--perturb is on by default)."
    )
    args.add_argument(
        "--perturb-k", required=False, type=int, default=1,
        help="Use up to this many stored variants per item for the rewrite families "
             "(paraphrase, register, past_tense); default=1. Must not exceed the k the "
             "artifacts were generated with."
    )
    args.add_argument(
        "--simulate", required=False, action='store_true',
        help="Run stage-3 scenario simulation (see pipeline/stage3_simulation/) on top of every "
             "benchmark in --only (or all benchmarks if --only is omitted): the target walks the "
             "pregenerated deployment-scenario trees (up to 3 turns; branch chosen by --sim-classifier) "
             "from datasets/generated/ "
             "(run generate.py --simulate first). Composes with --perturb in one run/one log: "
             "the panel reports safety_scenario next to stage 2's safety_<family>/stability_under_attack."
    )
    args.add_argument(
        "--sim-k", required=False, type=int, default=2,
        help="Use up to this many stored scenario trees per item under --simulate; default=2."
    )
    args.add_argument(
        "--sim-classifier", required=False, default=DEFAULT_CLASSIFIER,
        help="Model that labels each target reply refuse/hedge/comply to choose the next stored "
             "turn under --simulate (pipeline/stage3_simulation/classify.py). Regex-prefiltered, "
             "disk-cached with no expiry, canaried with the graders. "
             f"Default: {DEFAULT_CLASSIFIER}."
    )
    args.add_argument(
        "--max-connections", required=False, type=int, default=100,
        help="Max concurrent model connections Inspect opens (default: 100)."
    )
    args.add_argument(
        "--max-retries", required=False, type=int, default=10,
        help="Give up on a model request after this many retries (default: 10). Inspect's own "
             "default is to retry transient provider errors forever, and the backoff sleep is "
             "reported as waiting time so --working-limit never ends it — which turns a provider "
             "outage into an unattended batch that hangs instead of failing the model and moving "
             "on to the next one. Raise it for a run against a heavily rate-limited provider."
    )
    args.add_argument(
        "--attempt-timeout", required=False, type=int, default=600,
        help="Abandon a single model request after this many seconds and retry it "
             "(default: 600). Bounds a connection that has stopped responding; unset, one "
             "wedged request waits indefinitely because --working-limit excludes time spent "
             "waiting on retries."
    )
    args.add_argument(
        "--timeout", required=False, type=int, default=1800,
        help="Give up on a model request this many seconds after its first attempt, however "
             "many retries are left (default: 1800). --max-retries bounds the number of "
             "attempts but not their duration: backoff caps at 30 minutes a wait, so a "
             "handful of retries can hold a sample for hours. Whichever bound is hit first "
             "ends the request."
    )
    args.add_argument(
        "--working-limit", required=False, type=int, default=900,
        help="Max working seconds per sample before it fails and retries; bounds a "
             "hung provider connection so one stuck request can't wedge the whole "
             "run (default: 900). Excludes time spent waiting on rate limits/retries."
    )
    args.add_argument(
        "--max-price", required=False, nargs="+", metavar="KEY=VALUE", default=None,
        help="Cap what OpenRouter is allowed to pay for the target model, as one or more of "
             "prompt/completion/request/image in USD per million tokens "
             "(e.g. --max-price prompt=1.25 completion=4.25). Endpoints above the cap are "
             "dropped from routing, and a request with no qualifying endpoint fails outright "
             "rather than falling back to a pricier one — so a cap under the model's floor "
             "price errors every sample. openrouter/ models only, and the target only: the "
             "graders and --llamaguard route unconstrained."
    )

    args.add_argument(
        "--cheapest", required=False, action='store_true',
        help="Route every target call to the cheapest OpenRouter endpoint that still serves the "
             "model's most capable configuration — full weight precision, longest context, widest "
             "parameter support (see pipeline/utils/routing.py). Falls back only to other equally "
             "capable endpoints, never onto a degraded one, and fails the run rather than guess "
             "when the endpoint list cannot be read. Every endpoint considered is recorded per "
             "benchmark in models.json. Composes with --max-price. openrouter/ models only, and "
             "the target only: the graders and --llamaguard route unconstrained."
    )

    return args.parse_args()

# Share of a benchmark's samples that may come back with an empty completion
# before the run stops counting as a clean one. Matches eval()'s fail_on_error:
# scattered blanks are provider noise, a tenth of the dataset is not.
EMPTY_COMPLETION_LIMIT = 0.1


def refusal_record(evaluations) -> dict:
    '''
    Which samples the provider refused to run, and which sources they came from.

    A content-filter refusal is not the model declining — the request never
    reached it. It is still a fact worth certifying (the deployment will not
    answer these prompts), but it says far less about the model than a refusal
    the model itself produced, which the scorers already handle via
    refusal_regex.

    Recorded per sample id because the blocked set is not a random sample of
    the dataset: it is the hardest end of it. gpt-5.6-sol's 2026-08-30 cyber
    run lost 57 of cysecbench's 90 items, so anything computed over what
    remained would be a score on an easier benchmark wearing the same name.
    Ordinary errors are counted separately — a timeout is a run problem to fix,
    a refusal is a property of the deployment.
    '''
    refused: list[str] = []
    by_source: dict[str, int] = {}
    other = 0

    for log in evaluations:
        for sample in (log.samples or []):
            if not sample.error:
                continue
            if retry_policy.matches_content_filter(str(sample.error)):
                refused.append(str(sample.id))
                source = (sample.metadata or {}).get("source") or "unknown"
                by_source[source] = by_source.get(source, 0) + 1
            else:
                other += 1

    return {
        "provider_refused": len(refused),
        "other_errors": other,
        "by_source": dict(sorted(by_source.items())),
        "sample_ids": sorted(refused),
    }


def check_status(evaluations, run_id: str | None = None):
    '''
    Summarise a benchmark's EvalLogs into a status record:
    success (every task log succeeded), partial (some did), or failed —
    plus completed/total sample counts and the number of samples the model
    answered with nothing. Stored per benchmark in models.json so an
    incomplete run is distinguishable from a clean one.

    A sample that errors is loud: it is counted, printed, and past
    fail_on_error it sinks the task. A sample the model answers with an empty
    string is silent — it completes, it scores NaN, and it leaves the run
    looking flawless while measuring nothing. meta/muse-spark-1.2 returned an
    empty assistant message for all 562 manipulation samples and the run was
    recorded as `success 562/562`, so the empty count is tallied here and
    demotes the status exactly as errors do.

    Also sums each log's stats.model_usage into "usage" (per model name,
    tokens and total_cost — null when the provider reports no price) and
    records the run_id the logs came from.
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

    # Errored samples have no output by definition; counting them here too
    # would report one failure as two unrelated problems.
    answerable = [
        s for log in evaluations for s in (log.samples or []) if not s.error
    ]
    empty = sum(
        1 for s in answerable
        if not (getattr(s.output, "completion", "") or "").strip()
    )

    status = "success" if ok == len(evaluations) else ("partial" if ok else "failed")
    if answerable and empty / len(answerable) > EMPTY_COMPLETION_LIMIT:
        silent = "every sample" if empty == len(answerable) else f"{empty}/{len(answerable)} samples"
        print(f"[ERROR] model returned an empty completion for {silent}")
        status = "failed" if empty == len(answerable) else "partial"

    refusals = refusal_record(evaluations)
    if refusals["provider_refused"]:
        sources = ", ".join(f"{s} {n}" for s, n in refusals["by_source"].items())
        print(f"[WARNING] provider refused {refusals['provider_refused']} sample(s) "
              f"on content policy ({sources})")

    usage: dict[str, dict] = {}
    for log in evaluations:
        model_usage = getattr(getattr(log, "stats", None), "model_usage", None) or {}
        for model, used in model_usage.items():
            tally = usage.setdefault(model, {"input_tokens": 0, "output_tokens": 0, "total_cost": None})
            tally["input_tokens"] += used.input_tokens
            tally["output_tokens"] += used.output_tokens
            if used.total_cost is not None:
                tally["total_cost"] = (tally["total_cost"] or 0.0) + used.total_cost

    return {
        "status": status,
        "completed_samples": completed,
        "total_samples": total,
        "empty_completions": empty,
        "refusals": refusals,
        "usage": usage,
        "run_id": run_id,
    }


def record_routing(statuses: dict, endpoints: list[dict] | None) -> dict:
    '''
    Store the endpoints this run could have used, per benchmark, alongside the
    benchmark's own status.

    Per benchmark and not per model because models.json is merged one benchmark
    at a time (see update): a --only rerun would otherwise restate its routing
    over risks that were scored on a different day, from a different endpoint
    list, at different prices.
    '''
    if not endpoints:
        return statuses
    return {
        benchmark: {**status, "endpoints": endpoints}
        for benchmark, status in statuses.items()
    }


# ----- Updates models/models.json -----

def completed_risks(entry: dict) -> set[str]:
    '''Risks a rerun may skip: status success, a non-null headline, and a
    current-schema tree (pre-refactor records have no aggregate `tail`, so their
    headline means something else and must be recomputed).'''
    return {
        risk for risk, status in (entry.get('status') or {}).items()
        if status.get('status') == 'success'
        and entry.get('scores', {}).get(risk) is not None
        and "tail" in (((entry.get('results') or {}).get(risk) or {}).get('aggregate') or {})
    }


RUN_OWNED_FIELDS = ("scores", "results", "status", "aggregate")
IDENTITY_FIELDS = ("id", "name", "company", "region", "specialty")


def update(results, models, idx):
    '''Merge this run's results over the stored record, write models/results/<slug>.json, rebuild models/models.json.'''

    # ----- store ------
    # if idx != -1, model results already exist
    if idx != -1:
        prev = models[idx]
        prev_status = prev.get('status', {})

        # never overwrite a previously-complete benchmark result with a
        # partial/failed rerun — drop the demoted rerun and keep the old one
        for benchmark, status in list(results.get('status', {}).items()):
            if status.get('status') != 'success' and benchmark in completed_risks(prev):
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

        # Start from the stored record so keys this run does not own
        # (aa_* from scripts/match_aa_index.py, hand edits) survive; identity
        # comes from the CLI only where the record has none.
        results = (
            prev
            | {key: results[key] for key in RUN_OWNED_FIELDS}
            | {key: results.get(key) for key in IDENTITY_FIELDS if prev.get(key) is None}
        )

    write_json_atomic(model_result_path(results["id"]), results)
    rebuild_models_json()


# ----- main ------

if __name__ == "__main__":

    args = parse()

    # A provider content-filter refusal is a decision about the prompt, not a
    # blip: without this each refused sample burns the full retry ladder before
    # failing anyway (see pipeline/utils/retry_policy.py).
    retry_policy.install()

    grader = args.grader if args.grader else load_graders()
    model_id = args.model.split("/")[-1]
    log_dir = run_dir(model_id, args.run_id, args.limit)
    if args.rerun:
        moved = move_aside(log_dir)
        if moved:
            print(f"--rerun: moved previous logs to {moved}")

    print(f"Model: {model_id}")
    print(f"Grader(s): {grader}")
    max_price = parse_max_price(args.max_price)
    if (max_price or args.cheapest) and not args.model.startswith("openrouter/"):
        raise SystemExit(
            f"--max-price/--cheapest are OpenRouter provider routing; --model {args.model} "
            "is not an openrouter/ model."
        )

    # Resolved before the grader preflight and before any eval starts; an
    # unreadable endpoint list is a hard stop (see routing.fetch_endpoints).
    endpoints = provider_routing_api.fetch_endpoints(args.model) if args.cheapest else None
    routing = provider_routing(max_price, endpoints)
    routing_record = provider_routing_api.endpoint_record(endpoints) if endpoints else None

    model_args: dict = {}
    if routing:
        # Inspect's OpenRouter provider takes a `provider` model arg and
        # forwards it as the request's `provider` field
        # (inspect_ai/model/_providers/openrouter.py::OpenRouterAPI). It lands
        # in the log's eval.model_args, so a certification records how it was
        # routed.
        model_args["provider"] = routing
        print(f"Provider routing: {routing}")
    for endpoint in routing_record or []:
        selected = "*" if endpoint["selected"] else " "
        print(f"  {selected} {endpoint['tag']:26} "
              f"${endpoint['prompt_usd_per_m']:.3f}/${endpoint['completion_usd_per_m']:.3f} per M"
              f"  quant={endpoint['quantization']} ctx={endpoint['context_length']}")

    print(f"Log Directory: {log_dir}")

    # ----- task master list -----
    BENCHMARKS = init_benchmarks(grader, llamaguard_model=args.llamaguard)  # see pipeline/registry.py for all tasks

    # Stage 2 and stage 3 compose in one run: both layer condition families
    # onto the same Task (one control generation, one log), and the wrapped
    # scorers report them under separate metric pools — safety_<family>/stability_under_attack
    # for the perturbation families, safety_scenario for the scenario family, plus a
    # safety_worst roll-up over every condition, control included. The certification score is the
    # tail of the per-item worst case (see pipeline/utils/results.py).
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
        # default: skip risks that already certified cleanly
        tasks_to_skip = completed_risks(models[idx])

    if tasks_to_skip:
        print(f"Skipping: {', '.join(sorted(tasks_to_skip))}")
        BENCHMARKS = {key: entry for key, entry in BENCHMARKS.items() if key not in tasks_to_skip}
    require_families_column(BENCHMARKS)

    # Inspect treats an empty task list exactly like None: it discovers every
    # @task in the working directory and runs those instead (see
    # inspect_ai/_eval/loader.py::resolve_tasks). Those are the raw cluster
    # tasks with their default grader — the full unfiltered datasets, no
    # stage-2/3 solvers, and an openai/ judge this run never configured. So a
    # fully-certified model must exit here rather than fall through to eval().
    if not BENCHMARKS:
        print("Nothing to run — every requested benchmark already has results (use --rerun to force).")
        sys.exit(0)

    # Fail fast — before any eval spends money — on the three things that make
    # a whole run worthless: missing artifacts (checked first, offline), an
    # unusable judge (every sample errors on scoring, or worse, silently
    # abstains into a perfect score), and a target that answers with nothing
    # (the same failure one layer up). The call estimate prints before the
    # two network canaries.
    validate_artifacts(
        BENCHMARKS,
        families=args.perturb if run_perturb else None,
        simulate=args.simulate,
        perturb_k=args.perturb_k,
        sim_k=args.sim_k,
        limit=args.limit,
    )

    estimate = estimate_calls(
        BENCHMARKS,
        families=args.perturb if run_perturb else [],
        k=args.perturb_k,
        sim_k=args.sim_k if args.simulate else None,
        graders=grader,
        limit=args.limit,
        epochs=args.epochs,
    )
    print(f"\n{'risk':<18}{'samples':>8}{'target':>9}{'judge':>9}{'classifier':>11}")
    for risk, calls in estimate.items():
        print(f"{risk:<18}{calls['samples']:>8}{calls['target']:>9}{calls['judge']:>9}{calls['classifier']:>11}")
    totals = {column: sum(calls[column] for calls in estimate.values()) for column in ("samples", "target", "judge", "classifier")}
    print(f"{'total':<18}{totals['samples']:>8}{totals['target']:>9}{totals['judge']:>9}{totals['classifier']:>11}  (upper bounds)\n")

    graders_to_check = list(grader) if isinstance(grader, list) else [grader]
    if args.simulate:
        graders_to_check.append(args.sim_classifier)
    validate_graders(graders_to_check)
    validate_target(args.model, model_args)

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
            sim_classifier=args.sim_classifier,
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
        logs = start_eval(all_tasks, args.model, model_args, log_dir, args)
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
            scores[benchmark] = None
            continue

        statuses[benchmark] = check_status(res, run_id=args.run_id)
        if statuses[benchmark]['status'] != 'success':
            print(f"[WARNING] {benchmark}: run was {statuses[benchmark]['status']} "
                  f"({statuses[benchmark]['completed_samples']}/{statuses[benchmark]['total_samples']} samples)")

        # Build the tree from every log's surviving samples, even a run that
        # failed on fail_on_error. The samples a provider filter blocked are the
        # hardest end of the dataset, so a partial figure is measured over an
        # easier subset — but it is published rather than withheld: the status
        # record says how many were refused and which, and _coverage
        # (pipeline/utils/results.py) folds those refusals into `total`, so the
        # shortfall rides next to the number as a coverage bar and a partial
        # score reads as partial rather than passing as a complete one.
        #
        # One tree per risk: every condition of every benchmark comes out of the
        # same log (pipeline/registry.py::apply_stages), so the builder splits
        # them by family itself rather than needing the run sliced up here.
        tree = results_tree.build(res)
        results_by_risk.update(tree)

        # The flat headline: the tail (CVaR@10% of per-item worsts, spec §4.2).
        # A run that did not finish cleanly publishes its tree but no headline;
        # completed_risks() reads `scores` to decide what a rerun can skip.
        aggregate = (tree.get(entry["name"], {}).get("aggregate") or {})
        scores[benchmark] = (
            aggregate.get("tail") if statuses[benchmark]["status"] == "success" else None
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
            "status": record_routing(statuses, routing_record),
        }

        update(results, models, idx)
