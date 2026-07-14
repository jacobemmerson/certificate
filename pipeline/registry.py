from pipeline.stage1_evaluation.evals.democratic_authoritarian_bias import fscale, leader_favorability, role_model_bias
from pipeline.stage1_evaluation.evals.llm_human_rights import udhr, echr, udhr_government, udhr_individual, echr_government, echr_individual
from pipeline.stage1_evaluation.evals.preserving_historical_truth import history_no_push, history_explicit_push
from pipeline.stage1_evaluation.evals.socialharmbench import social_harm_bench

from pipeline.stage2_perturbation.build import build_perturbed_task

ALL_PERTURB_FAMILIES = {"paraphrase", "register", "identity_strip", "framing", "reconsideration"}

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


def apply_perturbations(
    benchmarks: dict,
    families: list[str] | None = None,
    rewrite_model: str = "openrouter/meta-llama/llama-3.1-8b-instruct",
    k: int = 3,
):
    '''
    Attach surface-perturbation auditing directly onto an already-built
    BENCHMARKS dict (from init_benchmarks), in place of running a separate
    perturbation Task/log. For every Task in every benchmark,
    pipeline/stage2_perturbation/build.py::build_perturbed_task appends one
    solver per requested, applicable family onto that Task's own solver —
    its own solver stays first in the array as the control condition.

    Keeps the same benchmark keys/'name' as `benchmarks`, so the result runs
    through the exact same log path as an unperturbed run — no separate
    "_perturb" Task or log directory.

    rewrite_model is the paraphrase/rewrite model for the generative
    families (paraphrase, register, identity_strip); it should
    differ from both the target and grader models (certify.py's --attacker
    default is used for this by convention).
    '''
    families = families if families is not None else sorted(ALL_PERTURB_FAMILIES)
    unknown = set(families) - ALL_PERTURB_FAMILIES
    if unknown:
        raise ValueError(f"Unknown perturbation families: {sorted(unknown)}")

    return {
        key: {
            'tasks': [build_perturbed_task(t, families, rewrite_model, k) for t in entry['tasks']],
            'name': entry['name'],
        }
        for key, entry in benchmarks.items()
    }
