"""Run Certificate benchmarks under controlled C0-C4 conditions."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

from inspect_ai import eval

from pipeline.agentic import (
    AUDIT_METADATA_KEY,
    PROTOCOL_VERSION,
    AgenticConfig,
    Condition,
)
from pipeline.utils.task_transforms import task_name
from pipeline.registry import (
    ALL_PERTURB_FAMILIES,
    apply_condition,
    apply_perturbations,
    apply_simulation,
    condition_benchmark_key,
    init_benchmarks,
    select_paired_samples,
    selected_ids_cover_benchmarks,
)
from pipeline.utils.errors import safe_exception
from pipeline.utils.graders import (
    aggregate_score,
    consistency_rate,
    load_graders,
    load_models_with_check,
)

_KNOWN_BENCHMARK_KEYS = {"auth", "harm", "hist", "hr"}

_BUDGET_ARGUMENTS = {
    "aggregate_output_tokens": "aggregate_output_tokens",
    "final_output_tokens": "final_output_tokens",
    "draft_output_tokens": "draft_output_tokens",
    "review_output_tokens": "review_output_tokens",
    "consultation_output_tokens": "consultation_output_tokens",
    "helper_output_tokens": "helper_output_tokens",
    "max_consultation_turns": "max_consultation_turns",
    "max_final_turns": "max_final_turns",
    "max_helper_calls": "max_helper_calls",
    "protocol_time_seconds": "time_seconds",
    "protocol_working_seconds": "working_seconds",
}


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--model", "-m", required=True, help="Model to evaluate with Inspect."
    )
    parser.add_argument(
        "--grader",
        "-g",
        default=None,
        help="Grader model override. If omitted, load GRADERS.md.",
    )
    parser.add_argument("--name", "-n", default=None, help="Display model name.")
    parser.add_argument("--provider", "-p", default=None, help="Model provider.")
    parser.add_argument("--region", "-r", default=None, help="Development region.")
    parser.add_argument("--specialty", "-s", default=None, help="Model specialty.")
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=1,
        help="Number of generations per sample to average.",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Rerun results even when they already exist.",
    )
    parser.add_argument(
        "--attacker",
        "-a",
        default="openrouter/deepseek/deepseek-v4-flash",
        help="Rewrite model for stage 2 and scenario model for stage 3.",
    )
    parser.add_argument(
        "--llamaguard",
        default="openrouter/meta-llama/llama-guard-4-12b",
        help="LlamaGuard model for harm scoring.",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help=(
            "Deterministically select this many canonical IDs per task. "
            "Runs without exact full coverage are not written to models/models.json."
        ),
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="Stable local pairing seed (default: 0; seed 0 is supported).",
    )
    parser.add_argument(
        "--sample-ids-in",
        type=Path,
        default=None,
        help="Reuse a JSON selected-sample-ID map from an earlier condition.",
    )
    parser.add_argument(
        "--sample-ids-out",
        type=Path,
        default=None,
        help="Write the selected canonical task-to-IDs map as JSON.",
    )
    parser.add_argument(
        "--only",
        "-o",
        nargs="+",
        metavar="BENCHMARK",
        help="Run only these base benchmark keys (for example: harm hr).",
    )
    parser.add_argument(
        "--condition",
        choices=[condition.value for condition in Condition],
        default="c0",
        help=(
            "Solving condition c0..c4 (default: c0 clean direct baseline). "
            "C1-C4 support only the 12 standard tasks."
        ),
    )
    parser.add_argument(
        "--perturb",
        nargs="+",
        default=None,
        choices=sorted(ALL_PERTURB_FAMILIES),
        metavar="FAMILY",
        help=(
            "Explicit opt-in to surface perturbations; clean C0 applies none. "
            "Incompatible with --simulate and C1-C4."
        ),
    )
    parser.add_argument(
        "--perturb-k",
        type=int,
        default=1,
        help="Generative perturbation variants per sample (default: 1).",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help=(
            "Run stage-3 scenario simulation on direct C0 tasks. Reuses "
            "--attacker as the scenario-reframing model. Incompatible with "
            "--perturb and C1-C4."
        ),
    )
    parser.add_argument(
        "--sim-k",
        type=int,
        default=1,
        help="Reframed scenarios per sample under --simulate (default: 1).",
    )
    parser.add_argument(
        "--analyst-model",
        default=None,
        help="C3/C4 analyst role model; defaults to --attacker.",
    )
    parser.add_argument(
        "--critic-model",
        default=None,
        help="C3/C4 critic role model; defaults to the first grader model.",
    )

    parser.add_argument("--aggregate-output-tokens", type=int, default=None)
    parser.add_argument("--final-output-tokens", type=int, default=None)
    parser.add_argument("--draft-output-tokens", type=int, default=None)
    parser.add_argument("--review-output-tokens", type=int, default=None)
    parser.add_argument("--consultation-output-tokens", type=int, default=None)
    parser.add_argument("--helper-output-tokens", type=int, default=None)
    parser.add_argument("--max-consultation-turns", type=int, default=None)
    parser.add_argument("--max-final-turns", type=int, default=None)
    parser.add_argument("--max-helper-calls", type=int, default=None)
    parser.add_argument("--protocol-time-seconds", type=float, default=None)
    parser.add_argument("--protocol-working-seconds", type=float, default=None)
    return parser


def build_protocol_config(args: Namespace) -> AgenticConfig:
    """Resolve CLI budget overrides against condition defaults."""
    config = AgenticConfig.default(args.condition)
    requested = {
        budget_field: getattr(args, argument)
        for argument, budget_field in _BUDGET_ARGUMENTS.items()
        if getattr(args, argument) is not None
    }
    return config.with_budget_overrides(**requested)


def parse(argv: list[str] | None = None) -> Namespace:
    """Parse and validate all unsupported combinations before model loading."""
    parser = build_parser()
    args = parser.parse_args(argv)

    unknown_only = set(args.only or []) - _KNOWN_BENCHMARK_KEYS
    if unknown_only:
        parser.error(
            "unknown --only benchmark keys: " + ", ".join(sorted(unknown_only))
        )
    if args.condition != "c0" and (args.perturb or args.simulate):
        parser.error("C1-C4 cannot be combined with --perturb or --simulate")
    if args.perturb and args.simulate:
        parser.error("--perturb and --simulate cannot be combined")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.perturb_k <= 0:
        parser.error("--perturb-k must be positive")
    if args.sim_k <= 0:
        parser.error("--sim-k must be positive")
    try:
        build_protocol_config(args)
    except ValueError as error:
        parser.error(str(error))
    return args


def _read_ids(path: Path | None) -> dict[str, list[str | int]] | None:
    if path is None:
        return None
    with path.open() as file:
        payload = json.load(file)
    if isinstance(payload, dict) and "selected_sample_ids" in payload:
        payload = payload["selected_sample_ids"]
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) and isinstance(value, list)
        for key, value in payload.items()
    ):
        raise ValueError("sample IDs JSON must be a task-to-list map")
    return payload


def _write_ids(
    path: Path,
    *,
    seed: int,
    selected_ids: dict[str, list[str | int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(
            {
                "version": 1,
                "sample_seed": seed,
                "selected_sample_ids": selected_ids,
            },
            file,
            indent=2,
        )


def _merge_usage(
    destination: dict[str, dict[str, int | float]],
    source: dict,
) -> None:
    for group, usage in source.items():
        if not isinstance(usage, dict):
            continue
        totals = destination.setdefault(str(group), {})
        for field in (
            "model_events",
            "events_with_usage",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "reasoning_tokens",
            "input_tokens_cache_write",
            "input_tokens_cache_read",
            "events_with_cost",
            "total_cost",
        ):
            value = usage.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[field] = totals.get(field, 0) + value


def summarize_process(evaluations: list) -> dict:
    """Aggregate only pre-scoring audit metadata; never inspect scorer order."""
    total = 0
    audited = 0
    valid = 0
    violations: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    phases: Counter[str] = Counter()
    dynamic_helper_calls = 0
    fixed_reviewers = 0
    terminations: Counter[str] = Counter()
    usage_statuses: Counter[str] = Counter()
    aggregate_usage: dict[str, dict[str, int | float]] = {}
    phase_usage: dict[str, dict[str, int | float]] = {}
    role_usage: dict[str, dict[str, int | float]] = {}

    for log in evaluations:
        for sample in getattr(log, "samples", None) or []:
            total += 1
            audit = (getattr(sample, "metadata", None) or {}).get(AUDIT_METADATA_KEY)
            if not isinstance(audit, dict):
                continue
            audited += 1
            valid += bool(audit.get("valid"))
            violations.update(audit.get("violations") or [])
            roles.update(audit.get("role_model_counts") or {})
            phases.update(audit.get("phase_model_counts") or {})
            dynamic_helper_calls += int(
                audit.get("dynamic_helper_count", audit.get("helper_count", 0)) or 0
            )
            fixed_reviewers += int(audit.get("fixed_reviewer_count") or 0)
            reason = audit.get("termination_reason")
            if reason:
                terminations[str(reason)] += 1
            usage_statuses[str(audit.get("usage_status") or "unknown")] += 1
            _merge_usage(aggregate_usage, {"all": audit.get("aggregate_usage") or {}})
            _merge_usage(phase_usage, audit.get("phase_usage") or {})
            _merge_usage(role_usage, audit.get("role_usage") or {})

    return {
        "applicable": audited > 0,
        "total_samples": total,
        "audited_samples": audited,
        "valid_samples": valid,
        "invalid_samples": audited - valid,
        "missing_audit_samples": total - audited,
        "valid_rate": (valid / audited) if audited else None,
        "violations": dict(sorted(violations.items())),
        "termination_reasons": dict(sorted(terminations.items())),
        "observed_role_usage": dict(sorted(roles.items())),
        "observed_phase_usage": dict(sorted(phases.items())),
        "observed_dynamic_helper_calls": dynamic_helper_calls,
        # Backward-compatible aggregate name.
        "observed_helper_calls": dynamic_helper_calls,
        "observed_fixed_reviewer_calls": fixed_reviewers,
        "usage_statuses": dict(sorted(usage_statuses.items())),
        "aggregate_token_cost_usage": aggregate_usage.get("all", {}),
        "phase_token_cost_usage": dict(sorted(phase_usage.items())),
        "role_token_cost_usage": dict(sorted(role_usage.items())),
    }


def _canonical_tasks(entry: dict) -> list[str]:
    if entry.get("base_tasks"):
        return list(entry["base_tasks"])
    return [
        (task.metadata or {}).get("agentic_protocol", {}).get("base_task")
        or task_name(task)
        for task in entry["tasks"]
    ]


def result_benchmark_key(benchmark: str, entry: dict, args: Namespace) -> str:
    """Return a mode-specific score key without changing models.json's schema."""
    base = entry.get("base_benchmark", benchmark)
    if args.condition != Condition.C0.value:
        return condition_benchmark_key(base, args.condition)
    if args.perturb:
        return f"{base}_perturbation"
    if args.simulate:
        return f"{base}_simulation"
    return base


def _existing_result_keys(model_or_scores: dict) -> set[str]:
    """Resolve current and legacy flat models.json records to mode identities."""
    if "scores" not in model_or_scores:
        return set(model_or_scores)

    scores = set(model_or_scores.get("scores", {}))
    perturbations = set(model_or_scores.get("perturbations", {}))
    simulations = set(model_or_scores.get("simulations", {}))
    identities: set[str] = set()
    for key in scores:
        marked = False
        if key in perturbations:
            identities.add(
                key if key.endswith("_perturbation") else f"{key}_perturbation"
            )
            marked = True
        if key in simulations:
            identities.add(key if key.endswith("_simulation") else f"{key}_simulation")
            marked = True
        if not marked:
            identities.add(key)
    return identities


def skip_benchmark_keys(
    benchmarks: dict,
    args: Namespace,
    existing_model_or_scores: dict,
) -> set[str]:
    """Select execution keys to skip using canonical selection and result mode."""
    if args.only:
        only = set(args.only)
        return {
            benchmark
            for benchmark, entry in benchmarks.items()
            if entry.get("base_benchmark", benchmark) not in only
        }
    if args.rerun:
        return set()
    existing_results = _existing_result_keys(existing_model_or_scores)
    return {
        benchmark
        for benchmark, entry in benchmarks.items()
        if result_benchmark_key(benchmark, entry, args) in existing_results
    }


def persisted_config(
    *,
    benchmark: str,
    entry: dict,
    config: AgenticConfig,
    args: Namespace,
    analyst_model: str,
    critic_model: str | list[str],
    selected_ids: dict[str, list[str | int]],
    process: dict,
) -> dict:
    """Build the stable per-benchmark configuration persisted in models.json."""
    base_benchmark = entry.get("base_benchmark", benchmark)
    canonical_tasks = _canonical_tasks(entry)
    budget_overrides = {
        budget_field: getattr(args, argument)
        for argument, budget_field in _BUDGET_ARGUMENTS.items()
        if getattr(args, argument) is not None
    }
    effective_budget = asdict(config.budget) if config.budget is not None else None
    # Defaults are also requested limits; retain the raw overrides separately
    # so a run remains reproducible without reconstructing versioned defaults.
    requested_budget = (
        ({**effective_budget, **budget_overrides})
        if effective_budget is not None
        else None
    )
    return {
        "condition": config.condition.value,
        "protocol_version": config.protocol_version,
        "canonical_base_benchmark": base_benchmark,
        "canonical_base_tasks": canonical_tasks,
        "configured_role_models": {
            config.roles.analyst: analyst_model,
            config.roles.critic: critic_model,
        },
        "helper_models": (
            {
                config.roles.analyst: analyst_model,
                config.roles.critic: critic_model,
            }
            if config.condition in (Condition.C3, Condition.C4)
            else {}
        ),
        "requested_budget": requested_budget,
        "budget_overrides": budget_overrides,
        "effective_budget": effective_budget,
        "sample_seed": args.sample_seed,
        "sample_ids_source": str(args.sample_ids_in) if args.sample_ids_in else None,
        "selected_sample_ids": {
            task: list(selected_ids[task]) for task in canonical_tasks
        },
        "perturbations": list(args.perturb or []),
        "simulation": (
            {"model": args.attacker, "variants": args.sim_k} if args.simulate else None
        ),
        "process_validity": {
            "aggregate_policy": (
                "protocol_valid_samples_only"
                if config.condition is not Condition.C0
                else "all_samples"
            ),
            **{
                key: process[key]
                for key in (
                    "applicable",
                    "total_samples",
                    "audited_samples",
                    "valid_samples",
                    "invalid_samples",
                    "missing_audit_samples",
                    "valid_rate",
                    "violations",
                    "termination_reasons",
                )
            },
        },
        "observed_usage": {
            "roles": process["observed_role_usage"],
            "phases": process["observed_phase_usage"],
            "dynamic_helper_calls": process["observed_dynamic_helper_calls"],
            # Backward-compatible name.
            "helper_calls": process["observed_helper_calls"],
            "fixed_reviewer_calls": process["observed_fixed_reviewer_calls"],
            "usage_statuses": process["usage_statuses"],
            "aggregate_tokens_and_cost": process["aggregate_token_cost_usage"],
            "phase_tokens_and_cost": process["phase_token_cost_usage"],
            "role_tokens_and_cost": process["role_token_cost_usage"],
        },
    }


def update(results, models, idx):
    """Summarize results while preserving complete prior benchmark runs."""
    if models:
        with open("models/models_previous.json", "w") as file:
            json.dump(models, file, indent=4)

    if idx != -1:
        previous = models[idx]
        previous_status = previous.get("status", {})
        for benchmark, status in list(results.get("status", {}).items()):
            previously_complete = (
                benchmark in previous.get("scores", {})
                and previous_status.get(benchmark, {}).get("status", "success")
                == "success"
            )
            if status.get("status") != "success" and previously_complete:
                print(
                    f"[WARNING] {benchmark}: rerun was {status.get('status')}; "
                    "keeping previous complete result"
                )
                for field in (
                    "scores",
                    "scores_meta",
                    "perturbations",
                    "simulations",
                    "agentic_config",
                    "status",
                ):
                    results.get(field, {}).pop(benchmark, None)

        for field in (
            "scores",
            "scores_meta",
            "perturbations",
            "simulations",
            "agentic_config",
            "status",
        ):
            results[field] = previous.get(field, {}) | results.get(field, {})
        models[idx] = results
    else:
        models.append(results)

    with open("models/models.json", "w") as file:
        json.dump(models, file, indent=4)


def _safe_exception(error: BaseException) -> str:
    """Backward-compatible local wrapper for safe persisted eval errors."""
    return safe_exception(error, "evaluation failed; provider details omitted")


def _status(evaluations: list) -> dict:
    ok = sum(1 for log in evaluations if log.status == "success")
    completed = sum(
        getattr(log.results, "completed_samples", 0) or 0
        for log in evaluations
        if log.results
    )
    total = sum(
        getattr(log.results, "total_samples", 0) or 0
        for log in evaluations
        if log.results
    )
    status = "success" if ok == len(evaluations) else ("partial" if ok else "failed")
    return {"status": status, "completed_samples": completed, "total_samples": total}


def main(argv: list[str] | None = None) -> None:
    args = parse(argv)
    grader = args.grader if args.grader else load_graders()
    analyst_model = args.analyst_model or args.attacker
    critic_model = args.critic_model or (
        grader[0] if isinstance(grader, list) else grader
    )
    config = build_protocol_config(args)
    model_id = args.model.split("/")[-1]
    log_dir = f"logs/{model_id}"

    print(f"Model: {model_id}")
    print(f"Grader(s): {grader}")
    print(f"Log Directory: {log_dir}")
    print(f"Condition: {config.condition.value}")
    if config.condition is not Condition.C0:
        print(f"Roles: analyst={analyst_model}, critic={critic_model}")

    benchmarks = init_benchmarks(grader, llamaguard_model=args.llamaguard)

    supplied_ids = _read_ids(args.sample_ids_in)
    source_benchmarks = benchmarks
    benchmarks, selected_ids = select_paired_samples(
        source_benchmarks,
        limit=args.limit,
        seed=args.sample_seed,
        selected_ids=supplied_ids,
    )
    complete_sample_coverage = selected_ids_cover_benchmarks(
        source_benchmarks, selected_ids
    )

    if config.condition is not Condition.C0:
        benchmarks = apply_condition(
            benchmarks,
            config,
            analyst_model=analyst_model,
            critic_model=critic_model,
        )
    elif args.perturb:
        benchmarks = apply_perturbations(
            benchmarks,
            families=args.perturb,
            rewrite_model=args.attacker,
            k=args.perturb_k,
        )
    elif args.simulate:
        benchmarks = apply_simulation(
            benchmarks,
            sim_model=args.attacker,
            k=args.sim_k,
        )

    ids_output = args.sample_ids_out
    if ids_output is None and (
        args.limit is not None or args.sample_ids_in is not None
    ):
        ids_output = Path(log_dir) / (
            f"selected_sample_ids.seed-{args.sample_seed}.json"
        )
    if ids_output is not None:
        _write_ids(ids_output, seed=args.sample_seed, selected_ids=selected_ids)
        print(f"Selected sample IDs: {ids_output}")

    budget_overrides = {
        budget_field: getattr(args, argument)
        for argument, budget_field in _BUDGET_ARGUMENTS.items()
        if getattr(args, argument) is not None
    }
    effective_budget = asdict(config.budget) if config.budget is not None else None

    def start_eval(benchmark: str, entry: dict):
        canonical_tasks = _canonical_tasks(entry)
        canonical_benchmark = entry.get("base_benchmark", benchmark)
        return eval(
            entry["tasks"],
            model=args.model,
            log_dir=log_dir + f"/{entry['name']}",
            continue_on_fail=True,
            retry_on_error=2,
            fail_on_error=0.1,
            epochs=args.epochs,
            # Selection already happened through immutable filtered views.
            sample_shuffle=False,
            limit=None,
            max_connections=100,
            cache=True,
            metadata={
                "certificate_run": {
                    "condition": config.condition.value,
                    "protocol_version": PROTOCOL_VERSION,
                    "canonical_base_benchmark": canonical_benchmark,
                    "canonical_base_tasks": canonical_tasks,
                    "configured_role_models": {
                        config.roles.analyst: analyst_model,
                        config.roles.critic: critic_model,
                    },
                    "helper_models": (
                        {
                            config.roles.analyst: analyst_model,
                            config.roles.critic: critic_model,
                        }
                        if config.condition in (Condition.C3, Condition.C4)
                        else {}
                    ),
                    "requested_budget": (
                        {**effective_budget, **budget_overrides}
                        if effective_budget is not None
                        else None
                    ),
                    "budget_overrides": budget_overrides,
                    "effective_budget": effective_budget,
                    "sample_seed": args.sample_seed,
                    "selected_sample_ids": {
                        task: selected_ids[task] for task in canonical_tasks
                    },
                }
            },
        )

    models, idx = load_models_with_check(model_id)
    if idx != -1:
        print(f"Results Found: Model index at {idx}")

    existing_model = models[idx] if idx != -1 else {}
    tasks_to_skip = skip_benchmark_keys(benchmarks, args, existing_model)
    if tasks_to_skip:
        skipped_results = {
            result_benchmark_key(benchmark, benchmarks[benchmark], args)
            for benchmark in tasks_to_skip
        }
        print(f"Skipping: {', '.join(sorted(skipped_results))}")

    scores: dict = {}
    scores_meta: dict = {}
    perturbations: dict = {}
    simulations: dict = {}
    agentic_config: dict = {}
    statuses: dict = {}

    for benchmark, entry in benchmarks.items():
        if benchmark in tasks_to_skip:
            continue
        result_key = result_benchmark_key(benchmark, entry, args)
        evaluations = []
        try:
            evaluations = start_eval(benchmark, entry)
            if not evaluations:
                statuses[result_key] = {
                    "status": "failed",
                    "error": "Inspect returned no evaluation logs",
                }
            else:
                statuses[result_key] = _status(evaluations)
                if statuses[result_key]["status"] != "success":
                    print(
                        f"[WARNING] {result_key}: run was "
                        f"{statuses[result_key]['status']} "
                        f"({statuses[result_key]['completed_samples']}/"
                        f"{statuses[result_key]['total_samples']} samples)"
                    )
                successful = [log for log in evaluations if log.status == "success"]
                if successful:
                    average, meta = aggregate_score(
                        successful,
                        valid_process_only=config.condition is not Condition.C0,
                    )
                    scores_meta[result_key] = meta
                    if average is not None:
                        scores[result_key] = average
                    if args.perturb:
                        perturbations[result_key] = consistency_rate(successful)
                    if args.simulate:
                        simulations[result_key] = consistency_rate(successful)
        except Exception as error:
            safe_error = _safe_exception(error)
            print(f"[ERROR] on {result_key}: {safe_error}")
            statuses[result_key] = {"status": "failed", "error": safe_error}
        finally:
            process = summarize_process(evaluations)
            if result_key in statuses:
                statuses[result_key].update(
                    {
                        "process_valid_samples": process["valid_samples"],
                        "process_invalid_samples": process["invalid_samples"],
                        "process_missing_audit_samples": process[
                            "missing_audit_samples"
                        ],
                    }
                )
            agentic_config[result_key] = persisted_config(
                benchmark=benchmark,
                entry=entry,
                config=config,
                args=args,
                analyst_model=analyst_model,
                critic_model=critic_model,
                selected_ids=selected_ids,
                process=process,
            )

    # Persistence depends on actual canonical-ID coverage. A complete explicit
    # map is a full run; a partial map remains log-only even when no --limit was
    # supplied.
    sample_limited = not complete_sample_coverage
    if not sample_limited:
        results = {
            "id": model_id,
            "name": args.name,
            "company": args.provider,
            "region": args.region,
            "specialty": args.specialty,
            "scores": scores,
            "scores_meta": scores_meta,
            "perturbations": perturbations,
            "simulations": simulations,
            "agentic_config": agentic_config,
            "status": statuses,
        }
        update(results, models, idx)


if __name__ == "__main__":
    main()
