'''
author: @tae

Runs all benchmarks.
Tasks are defined in scripts/evals
'''

import json
from argparse import ArgumentParser

from inspect_ai import eval
from scripts.evals.democratic_authoritarian_bias import fscale, favscore, rolemodel
from scripts.evals.llm_human_rights import udhr, echr, udhr_government, udhr_individual, echr_government, echr_individual
from scripts.evals.preserving_historical_truth import no_push, explicit_push
from scripts.evals.socialharmbench import social_harm_bench

'''
Website expects a JSON Array in this format:

[
  {
    "id": "gpt-5.2",
    "name": "GPT-5.2",
    "company": "OpenAI",
    "region": "Frontier Leaders (USA)",
    "specialty": "Peak general intelligence & complex reasoning",
    "scores": {
      "hr": 79.2,
      "harm": 100,
      "hist": 79.1,
      "auth": 79
    }
  },
...
]
'''

def update(results):
    '''
    Summarises results and updates models/models.json
    '''
    with open('models/models.json', 'r') as f:
        models = json.load(f) # JSON array : [{model: x, id: y}, {model: x, id: y}, etc]

    with open('models/models_previous.json', 'w') as f: # store as a safety net
        json.dump(models, f)

    # ----- Format -----
    scores = {}
    for k,v in results['scores'].items():

        value = v.results.scores[0] # primary metric must go first
        # TODO: Add support for multiple scores / score selection / score reducers
        if value.name == "favscore_scorer":
            value = value.metrics['democratic_bias_score']
        
        else:
            value = value.metrics['mean'] 
        scores[k] = value

    results['scores'] = scores

    # ----- Store ------
    found = False
    for i, m in enumerate(models):
        if m['id'] == results['id']:
            # Update file
            found = True
            models[i] = results
            break

    if not found:
        # add new entry
        models.append(results)
        pass

    # write models file back
    with open('models/models.json', 'w') as f:
        json.dump(models, f)
    pass

def parse():
    
    args = ArgumentParser()
    args.add_argument(
        "--model", "-m", required=True, help="The model to be evaluated using AISI inspect."
    )
    args.add_argument(
        "--grader", "-g", required=True, help="The model to grade LLM-as-a-judge grader responses."
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
        "--speciality", "-s", required=False, default=None, help="What the model has been tuned or designated to do (i.e. coding, math, etc)."
    )
    args.add_argument(
        "--epochs", "-e", required=False, default=1, help="The number of turns to generate a response per sample and average over."
    )

    return args.parse_args()

if __name__ == "__main__":

    args = parse()
    log_dir = f"logs/{args.model.split('/')[-1]}"

    def check_status(evaluations):
        if evaluations['status'] != 'success':
            pass
        # TODO: implement a catch to see when a task fails

    def start_eval(tasks: list, task_name: str):
        return eval(
            tasks,
            model=args.model,
            log_dir=log_dir + f"/{task_name}",
            continue_on_fail=True,
            retry_on_error=2,
            epochs=args.epochs,
            sample_shuffle=False
        )
    
    # ----- Democratic vs. Authoritarian Bias -----
    dab = start_eval(
        #[fscale(), favscore(), rolemodel(grader=args.grader)],
        [favscore()],
        task_name="democratic_authoritarian"
    )

    # ----- SocialHarmBench -----
    shb = start_eval(
        [social_harm_bench(grader=args.grader)],
        task_name="socialharmbench"
    )

    # ----- Historical Revisionism -----
    histres = start_eval(
        [no_push(grader=args.grader)],
        task_name="historical_revisionism"
    )

    # ----- When do LLMs endorse Human Rights Limitations -----
    humanrights = start_eval(
        [udhr(grader=args.grader), echr(grader=args.grader)],
        task_name="human_rights"
    )

    results = {
        "id": args.model.split("/")[-1],
        "name": args.name,
        "company": args.provider,
        "region": None,
        "speciality": None,
        "scores": {
            "hr": humanrights,
            "harm": shb,
            "hist": histres,
            "auth": dab
        }
    }

    #update(results)