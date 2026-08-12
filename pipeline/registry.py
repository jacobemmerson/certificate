from pipeline.stage1_evaluation.evals.clusters import CLUSTER_TASKS, RISKS, available

from inspect_ai import Task

from pipeline.artifacts import framing_applies, load_family, task_name
from pipeline.stage2_perturbation.solvers import REPLAY_SOLVERS, framing, reconsideration
from pipeline.stage3_simulation.solvers import scenario
from pipeline.utils.replay import truncated
from pipeline.utils.scoring import SCENARIO, scoring_step, wrap_scorers

ALL_PERTURB_FAMILIES = {"paraphrase", "register", "identity_strip", "framing", "reconsideration"}

# The families generate.py pregenerates to datasets/generated/ and certify.py
# replays. reconsideration is the one live-only family: it challenges the
# target's own control completion, so it cannot exist before the eval runs.
PREGENERATED_FAMILIES = ALL_PERTURB_FAMILIES - {"reconsideration"}

def init_benchmarks(grader, llamaguard_model: str | None = None):
    '''
    One entry per systemic-risk cluster, keyed by risk name.

    A cluster's dataset is a filtered union of several benchmarks under one
    schema (datasets/public/<risk>.csv), so each entry holds a single task
    rather than a list of per-benchmark ones. Risks whose dataset has not been
    built are skipped — loss_of_control has no sources yet.

    `llamaguard_model` is accepted but unused: cluster tasks do not attach
    LlamaGuard. In a mixed cluster it would score likert and mcq rows where it
    means nothing, and each row's own `criterion` is strictly more specific
    than its generic taxonomy. See datasets/CLUSTERING.md.
    '''
    return {
        risk: {"tasks": [CLUSTER_TASKS[risk](grader=grader)], "name": risk}
        for risk in RISKS
        if available(risk)
    }


def _build_task(
    base_task: Task,
    families: list[str],
    k: int,
    sim_k: int | None = None,
) -> Task:
    """Return base_task with one replay solver appended per requested,
    applicable condition family — the stage-2 families from
    pipeline/stage2_perturbation/solvers.py, then stage 3's scenario solver
    (pipeline/stage3_simulation/solvers.py) when `sim_k` is set — one labeled
    scoring step per condition family (control included), and its scorer list
    wrapped one-per-base-judge (pipeline/utils/scoring.py::wrap_scorers).

    Returns base_task unchanged if nothing applies (e.g. only "framing" was
    requested against a benchmark whose elicitation_family has no registered
    framing templates, and sim_k is None).
    """
    name = task_name(base_task)

    solver_chain = [base_task.solver]
    applied: list[str] = []
    for family, replay_solver in REPLAY_SOLVERS.items():
        if family in families:
            solver_chain.append(replay_solver(truncated(load_family(name, family), k)))
            applied.append(family)
    if "framing" in families and framing_applies(base_task):
        solver_chain.append(framing(load_family(name, "framing")))
        applied.append("framing")
    if "reconsideration" in families:
        solver_chain.append(reconsideration())
        applied.append("reconsideration")
    if sim_k is not None:
        solver_chain.append(scenario(truncated(load_family(name, SCENARIO), sim_k)))
        applied.append(SCENARIO)

    if not applied:
        return base_task  # no requested family applies to this benchmark

    solver_chain.append(scoring_step("generate", base_task.scorer))
    solver_chain += [scoring_step(family, base_task.scorer) for family in applied]

    return Task(
        dataset=base_task.dataset,
        solver=solver_chain,
        scorer=wrap_scorers(base_task.scorer, applied),
        name=name,
    )


def apply_stages(
    benchmarks: dict,
    families: list[str] | None = None,
    k: int = 1,
    sim_k: int | None = None,
):
    '''
    Attach stage-2 surface-perturbation auditing (`families`) and/or stage-3
    scenario simulation (`sim_k` — None means "no simulation") directly onto
    an already-built BENCHMARKS dict (from init_benchmarks), in place of
    running separate Tasks/logs. For every Task in every benchmark,
    `_build_task` appends one solver per requested, applicable condition
    family onto that Task's own solver — its own solver stays first in the
    array as the control condition, generated and judged exactly once no
    matter how many stages are enabled.

    Keeps the same benchmark keys/'name' as `benchmarks`, so the result runs
    through the exact same log path as a plain run — a combined --perturb
    --simulate certification is one task run and ONE log, with the stages
    reported per attack family (safety_<family>) plus a single
    safety_under_attack roll-up, so the stages compare at equal depth.

    All pregenerated families replay fixed variants from datasets/generated/
    (produced once by generate.py — no rewrite/reframing model runs at eval
    time), using up to `k` stored variants per item for the rewrite families
    and up to `sim_k` stored scenarios. Building raises FileNotFoundError if
    the artifacts were never generated; certify.py validates them first
    (pipeline/artifacts.py::validate_artifacts) so the error carries the
    exact generate.py command.
    '''
    families = list(families) if families is not None else []
    unknown = set(families) - ALL_PERTURB_FAMILIES
    if unknown:
        raise ValueError(f"Unknown perturbation families: {sorted(unknown)}")

    return {
        key: {
            'tasks': [_build_task(t, families, k, sim_k=sim_k) for t in entry['tasks']],
            'name': entry['name'],
        }
        for key, entry in benchmarks.items()
    }
