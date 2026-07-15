from pipeline.stage1_evaluation.evals.democratic_authoritarian_bias import fscale, leader_favorability, role_model_bias
from pipeline.stage1_evaluation.evals.llm_human_rights import udhr, echr, udhr_government, udhr_individual, echr_government, echr_individual
from pipeline.stage1_evaluation.evals.preserving_historical_truth import history_no_push, history_explicit_push
from pipeline.stage1_evaluation.evals.socialharmbench import social_harm_bench

from pipeline.stage2_perturbation.build import build_perturbed_task

ALL_PERTURB_FAMILIES = {"paraphrase", "register", "identity_strip", "framing", "reconsideration"}

# The families generate.py pregenerates to datasets/generated/ and certify.py
# replays. reconsideration is the one live-only family: it challenges the
# target's own control completion, so it cannot exist before the eval runs.
PREGENERATED_FAMILIES = ALL_PERTURB_FAMILIES - {"reconsideration"}

def init_benchmarks(grader, llamaguard_model: str = "openrouter/meta-llama/llama-guard-4-12b"):

    # ----- task master list -----
    BENCHMARKS = {
        'auth': {
            'tasks': [
                fscale(llamaguard_model=llamaguard_model),
                leader_favorability(llamaguard_model=llamaguard_model),
                role_model_bias(grader=grader, llamaguard_model=llamaguard_model)
            ],
            'name': 'democratic_authoritarian_bias'
        },
        'harm': {
            'tasks': [
                social_harm_bench(grader=grader, llamaguard_model=llamaguard_model)
            ],
            'name': 'socialharmbench'
        },
        'hist': {
            'tasks': [
                history_no_push(grader=grader, llamaguard_model=llamaguard_model),
                history_explicit_push(grader=grader, llamaguard_model=llamaguard_model)
            ],
            'name': 'historical_revisionism'
        },
        'hr': {
            'tasks': [
                udhr(grader=grader, llamaguard_model=llamaguard_model),
                udhr_individual(grader=grader, llamaguard_model=llamaguard_model),
                udhr_government(grader=grader, llamaguard_model=llamaguard_model),
                echr(grader=grader, llamaguard_model=llamaguard_model),
                echr_individual(grader=grader, llamaguard_model=llamaguard_model),
                echr_government(grader=grader, llamaguard_model=llamaguard_model)
            ],
            'name': 'human_rights'
        }
    }

    return BENCHMARKS


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
    pipeline/stage2_perturbation/build.py::build_perturbed_task appends one
    solver per requested, applicable condition family onto that Task's own
    solver — its own solver stays first in the array as the control
    condition, generated and judged exactly once no matter how many stages
    are enabled.

    Keeps the same benchmark keys/'name' as `benchmarks`, so the result runs
    through the exact same log path as a plain run — a combined --perturb
    --simulate certification is one task run and ONE log, with the stages
    reported under separate metric pools (lvr/consistency for stage 2,
    lvr_scenario/consistency_scenario for stage 3).

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
            'tasks': [build_perturbed_task(t, families, k, sim_k=sim_k) for t in entry['tasks']],
            'name': entry['name'],
        }
        for key, entry in benchmarks.items()
    }
