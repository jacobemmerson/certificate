from tasks.evals.democratic_authoritarian_bias import fscale, favscore, rolemodel
from tasks.evals.llm_human_rights import udhr, echr, udhr_government, udhr_individual, echr_government, echr_individual
from tasks.evals.preserving_historical_truth import no_push, explicit_push
from tasks.evals.socialharmbench import social_harm_bench

def init_benchmarks(grader):

    # ----- task master list -----
    BENCHMARKS = {
        'auth': {
            'tasks': [
                fscale(), 
                #favscore(), 
                rolemodel(grader=grader)
            ],
            'name': 'democratic_authoritarian_bias'
        },
        'harm': {
            'tasks': [
                social_harm_bench(grader=grader)
            ],
            'name': 'socialharmbench'
        },
        'hist': {
            'tasks': [
                no_push(grader=grader),
                explicit_push(grader=grader)
            ],
            'name': 'historical_revisionism'
        },
        'hr': {
            'tasks': [
                udhr(grader=grader),
                udhr_individual(grader=grader),
                udhr_government(grader=grader),
                echr(grader=grader),
                echr_individual(grader=grader),
                echr_government(grader=grader)
            ],
            'name': 'human_rights'
        }
    }

    return BENCHMARKS