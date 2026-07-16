"""Certificate benchmark registry and deterministic condition construction."""

from __future__ import annotations

from copy import copy
from hashlib import sha256
import random

from inspect_ai import task_with

from pipeline.agentic.build import make_agentic_task, with_role_models
from pipeline.agentic.config import AgenticConfig, Condition
from pipeline.stage1_evaluation.evals.democratic_authoritarian_bias import (
    fscale,
    leader_favorability,
    role_model_bias,
)
from pipeline.stage1_evaluation.evals.llm_human_rights import (
    echr,
    echr_government,
    echr_individual,
    udhr,
    udhr_government,
    udhr_individual,
)
from pipeline.stage1_evaluation.evals.preserving_historical_truth import (
    history_explicit_push,
    history_no_push,
)
from pipeline.stage1_evaluation.evals.socialharmbench import social_harm_bench
from pipeline.stage2_perturbation.build import build_perturbed_task
from pipeline.stage3_simulation.build import build_simulation_task
from pipeline.utils.task_transforms import (
    AGENTIC_METADATA_KEY,
    TaskTransform,
    task_name,
    validate_transform_base,
)

ALL_PERTURB_FAMILIES = {
    "paraphrase",
    "register",
    "identity_strip",
    "framing",
    "reconsideration",
}
AGENTIC_BENCHMARK_SUFFIX = "_agentic_"


def init_benchmarks(
    grader,
    llamaguard_model: str = "openrouter/meta-llama/llama-guard-4-12b",
):
    """Create fresh Task instances for the 12 standard Certificate tasks."""
    return {
        "auth": {
            "tasks": [
                fscale(llamaguard_model=llamaguard_model),
                leader_favorability(llamaguard_model=llamaguard_model),
                role_model_bias(grader=grader, llamaguard_model=llamaguard_model),
            ],
            "name": "democratic_authoritarian_bias",
        },
        "harm": {
            "tasks": [
                social_harm_bench(grader=grader, llamaguard_model=llamaguard_model)
            ],
            "name": "socialharmbench",
        },
        "hist": {
            "tasks": [
                history_no_push(grader=grader, llamaguard_model=llamaguard_model),
                history_explicit_push(grader=grader, llamaguard_model=llamaguard_model),
            ],
            "name": "historical_revisionism",
        },
        "hr": {
            "tasks": [
                udhr(grader=grader, llamaguard_model=llamaguard_model),
                udhr_individual(grader=grader, llamaguard_model=llamaguard_model),
                udhr_government(grader=grader, llamaguard_model=llamaguard_model),
                echr(grader=grader, llamaguard_model=llamaguard_model),
                echr_individual(grader=grader, llamaguard_model=llamaguard_model),
                echr_government(grader=grader, llamaguard_model=llamaguard_model),
            ],
            "name": "human_rights",
        },
    }


def condition_benchmark_key(base_key: str, condition: Condition | str) -> str:
    """Return the non-overwriting result key for a benchmark condition."""
    condition = Condition(condition)
    if condition is Condition.C0:
        return base_key
    return f"{base_key}{AGENTIC_BENCHMARK_SUFFIX}{condition.value}"


def apply_condition(
    benchmarks: dict,
    config: AgenticConfig,
    *,
    analyst_model: str | None = None,
    critic_model: str | None = None,
) -> dict:
    """Wrap fresh benchmark tasks for one controlled condition.

    C0 returns the supplied registry object untouched. C1-C4 first validate
    every task, then add configured model roles and replace only each solver.
    """
    if config.condition is Condition.C0:
        return benchmarks

    for entry in benchmarks.values():
        for base_task in entry["tasks"]:
            validate_transform_base(base_task, TaskTransform.AGENTIC)

    target_keys = [
        condition_benchmark_key(base_key, config.condition) for base_key in benchmarks
    ]
    if len(set(target_keys)) != len(target_keys):
        raise ValueError("agentic condition keys are not unique")
    collisions = set(target_keys) & set(benchmarks)
    if collisions:
        raise ValueError(
            "agentic condition keys would overwrite existing benchmarks: "
            + ", ".join(sorted(collisions))
        )

    conditioned = {}
    for base_key, entry in benchmarks.items():
        base_tasks = [task_name(base_task) for base_task in entry["tasks"]]
        wrapped_tasks = []
        for base_task in entry["tasks"]:
            staged = with_role_models(
                base_task,
                config,
                analyst_model=analyst_model,
                critic_model=critic_model,
            )
            wrapped_tasks.append(make_agentic_task(staged, config))

        key = condition_benchmark_key(base_key, config.condition)
        conditioned[key] = {
            **entry,
            "tasks": wrapped_tasks,
            "name": (
                f"{entry['name']}{AGENTIC_BENCHMARK_SUFFIX}{config.condition.value}"
            ),
            "base_benchmark": base_key,
            "base_tasks": base_tasks,
            "condition": config.condition.value,
        }
    return conditioned


def init_condition_benchmarks(
    grader,
    config: AgenticConfig,
    *,
    llamaguard_model: str = "openrouter/meta-llama/llama-guard-4-12b",
    analyst_model: str | None = None,
    critic_model: str | None = None,
) -> dict:
    """Construct one condition from a newly initialized standard registry."""
    return apply_condition(
        init_benchmarks(grader, llamaguard_model=llamaguard_model),
        config,
        analyst_model=analyst_model,
        critic_model=critic_model,
    )


def _canonical_task(task) -> str:
    return (task.metadata or {}).get(AGENTIC_METADATA_KEY, {}).get(
        "base_task"
    ) or task_name(task)


def _stable_task_seed(seed: int, canonical_task: str) -> int:
    digest = sha256(
        f"certificate-samples-v1\0{seed}\0{canonical_task}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big")


def select_paired_samples(
    benchmarks: dict,
    *,
    limit: int | None,
    seed: int,
    selected_ids: dict[str, list[str | int]] | None = None,
) -> tuple[dict, dict[str, list[str | int]]]:
    """Return independent filtered views and their canonical-ID map.

    Selection uses one local RNG per canonical task and never mutates a source
    dataset. A supplied map is validated and reused exactly; its seed remains
    provenance only.
    """
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0
    ):
        raise ValueError("limit must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("sample seed must be an integer; seed 0 is supported")

    canonical_tasks = {
        _canonical_task(task)
        for entry in benchmarks.values()
        for task in entry["tasks"]
    }
    if selected_ids is not None:
        unknown_tasks = sorted(set(selected_ids) - canonical_tasks)
        if unknown_tasks:
            raise ValueError(
                f"sample IDs input contains unknown canonical tasks: {unknown_tasks}"
            )

    filtered = {}
    chosen_map: dict[str, list[str | int]] = {}
    for key, entry in benchmarks.items():
        tasks = []
        for source_task in entry["tasks"]:
            canonical = _canonical_task(source_task)
            samples = list(source_task.dataset)
            available = [sample.id for sample in samples]
            if any(sample_id is None for sample_id in available):
                raise ValueError(f"{canonical} contains samples without canonical IDs")
            try:
                available_set = set(available)
            except TypeError as error:
                raise ValueError(
                    f"{canonical} contains non-scalar sample IDs"
                ) from error
            if len(available_set) != len(available):
                raise ValueError(f"{canonical} contains duplicate sample IDs")

            if selected_ids is not None:
                if canonical not in selected_ids:
                    raise ValueError(
                        f"sample IDs input is missing canonical task {canonical!r}"
                    )
                requested = list(selected_ids[canonical])
                try:
                    requested_set = set(requested)
                except TypeError as error:
                    raise ValueError(
                        f"sample IDs input contains non-scalar IDs for {canonical}"
                    ) from error
                unknown = [
                    sample_id
                    for sample_id in requested
                    if sample_id not in available_set
                ]
                if unknown:
                    raise ValueError(
                        f"sample IDs input contains unknown IDs for {canonical}: "
                        f"{unknown[:5]}"
                    )
                if len(requested_set) != len(requested):
                    raise ValueError(
                        f"sample IDs input contains duplicates for {canonical}"
                    )
                if limit is not None and len(requested) != min(limit, len(available)):
                    raise ValueError(
                        f"sample IDs input has {len(requested)} IDs for {canonical}; "
                        f"expected {min(limit, len(available))}"
                    )
                chosen = requested
            elif limit is None or limit >= len(available):
                chosen = list(available)
            else:
                rng = random.Random(_stable_task_seed(seed, canonical))
                indexes = set(rng.sample(range(len(available)), limit))
                chosen = [
                    sample_id
                    for index, sample_id in enumerate(available)
                    if index in indexes
                ]

            if chosen == available:
                tasks.append(source_task)
            else:
                selected = set(chosen)
                dataset = source_task.dataset.filter(
                    lambda sample, ids=selected: sample.id in ids,
                    name=source_task.dataset.name,
                )
                tasks.append(task_with(copy(source_task), dataset=dataset))
            chosen_map[canonical] = chosen

        filtered[key] = {**entry, "tasks": tasks}
    return filtered, chosen_map


def selected_ids_cover_benchmarks(
    benchmarks: dict,
    selected_ids: dict[str, list[str | int]],
) -> bool:
    """Return whether selected IDs exactly cover every source task dataset."""
    expected_tasks: set[str] = set()
    for entry in benchmarks.values():
        for task in entry["tasks"]:
            canonical = _canonical_task(task)
            if canonical in expected_tasks:
                return False
            expected_tasks.add(canonical)
            available = [sample.id for sample in task.dataset]
            chosen = selected_ids.get(canonical)
            if chosen is None or len(chosen) != len(available):
                return False
            try:
                if len(set(chosen)) != len(chosen) or set(chosen) != set(available):
                    return False
            except TypeError:
                return False
    return set(selected_ids) == expected_tasks


def apply_perturbations(
    benchmarks: dict,
    families: list[str] | None = None,
    rewrite_model: str = "openrouter/meta-llama/llama-3.1-8b-instruct",
    k: int = 3,
):
    """Attach explicitly requested surface perturbations to direct tasks."""
    families = families if families is not None else sorted(ALL_PERTURB_FAMILIES)
    unknown = set(families) - ALL_PERTURB_FAMILIES
    if unknown:
        raise ValueError(f"Unknown perturbation families: {sorted(unknown)}")
    if k <= 0:
        raise ValueError("perturbation variants must be positive")

    for entry in benchmarks.values():
        for task in entry["tasks"]:
            validate_transform_base(task, TaskTransform.PERTURBATION)

    return {
        key: {
            **entry,
            "tasks": [
                build_perturbed_task(task, families, rewrite_model, k)
                for task in entry["tasks"]
            ],
        }
        for key, entry in benchmarks.items()
    }


def apply_simulation(
    benchmarks: dict,
    sim_model: str = "openrouter/deepseek/deepseek-v4-flash",
    k: int = 1,
):
    """Attach stage-3 scenario simulation to direct stage-1 tasks."""
    if k <= 0:
        raise ValueError("simulation variants must be positive")
    for entry in benchmarks.values():
        for task in entry["tasks"]:
            validate_transform_base(task, TaskTransform.SIMULATION)

    return {
        key: {
            **entry,
            "tasks": [
                build_simulation_task(task, sim_model, k) for task in entry["tasks"]
            ],
        }
        for key, entry in benchmarks.items()
    }
