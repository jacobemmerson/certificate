from tasks.evals.democratic_authoritarian_bias import fscale, favscore, rolemodel
from tasks.evals.llm_human_rights import udhr, echr, udhr_government, udhr_individual, echr_government, echr_individual
from tasks.evals.preserving_historical_truth import no_push, explicit_push
from tasks.evals.socialharmbench import social_harm_bench, social_harm_bench_adversarial

def init_benchmarks(grader, attacker: str = "openrouter/meta-llama/llama-3.1-8b-instruct", llamaguard_model: str = "openrouter/meta-llama/llama-guard-4-12b"):

    # ----- task master list -----
    BENCHMARKS = {
        'auth': {
            'tasks': [
                fscale(llamaguard_model=llamaguard_model),
                favscore(llamaguard_model=llamaguard_model),
                rolemodel(grader=grader, llamaguard_model=llamaguard_model)
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
                no_push(grader=grader, llamaguard_model=llamaguard_model),
                explicit_push(grader=grader, llamaguard_model=llamaguard_model)
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
        },
        'harm_adv': {
            'tasks': [
                social_harm_bench_adversarial(grader=grader, attacker=attacker, llamaguard_model=llamaguard_model)
            ],
            'name': 'socialharmbench_adversarial'
        }
    }

    return BENCHMARKS
