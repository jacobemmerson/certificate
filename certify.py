'''
author: @tae

Runs all benchmarks.
Tasks are defined in tasks/evals
'''

import json
from argparse import ArgumentParser

from inspect_ai import eval
from tasks.evals.democratic_authoritarian_bias import fscale, favscore, rolemodel
from tasks.evals.llm_human_rights import udhr, echr, udhr_government, udhr_individual, echr_government, echr_individual
from tasks.evals.preserving_historical_truth import no_push, explicit_push
from tasks.evals.socialharmbench import social_harm_bench
from tasks.utils.graders import load_graders

def update(results):
    '''
    Summarises results and updates models/models.json
    '''

    try:
        with open('models/models.json', 'r') as f:
            models = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        models = []

    if models: # store previous only if previous results exist
        with open('models/models_previous.json', 'w') as f: # store as a safety net
            json.dump(models, f, indent=3)

    # ----- Format -----
    scores = {}
    for k,v in results['scores'].items():

        value = v.results.scores[0] # primary metric must go first
        # TODO: Add support for multiple scores / score selection / score reducers
        try:
            if value.name == "favscore_scorer":
                value = value.metrics['democratic_bias_score']

            elif 'accuracy' in value.metrics:
                value = value.metrics['accuracy'] 

            elif 'mean' in value.metrics:
                value = value.metrics['mean']

            else:
                value = next(iter(value.metrics.values()))

        except Exception as e:
            print(f"Error occured on {k}: {e}")
            value = 0

        scores[k] = value.value

    results['scores'] = scores

    # ----- Store ------
    found = False
    for i, m in enumerate(models):
        if m['id'] == results['id']:
            # Update file
            found = True

            results['scores'] = m['scores'] | results['scores'] # take values from overlapping keys from the new results (right side of pipe operator)

            models[i] = results
            break

    if not found:
        # add new entry
        models.append(results)
        pass

    # write models file back
    with open('models/models.json', 'w') as f:
        json.dump(models, f, indent=3)

def parse():
    
    args = ArgumentParser()
    args.add_argument(
        "--model", "-m", required=True, help="The model to be evaluated using AISI inspect."
    )
    args.add_argument(
        "--grader", "-g", required=False, default=None, help="Grader model override (single model). If omitted, loads from graders.txt."
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
    grader = args.grader if args.grader else load_graders()

    print(f"Model: {args.model}")
    print(f"Grader(s): {grader}")

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
        #[fscale(), favscore(), rolemodel(grader=grader)],
        [favscore()],
        task_name="democratic_authoritarian"
    )

    '''
    # ----- SocialHarmBench -----
    shb = start_eval(
        [social_harm_bench(grader=grader)],
        task_name="socialharmbench"
    )

    # ----- Historical Revisionism -----
    histres = start_eval(
        [no_push(grader=grader)],
        task_name="historical_revisionism"
    )

    # ----- When do LLMs endorse Human Rights Limitations -----
    humanrights = start_eval(
        [udhr(grader=grader), echr(grader=grader)],
        task_name="human_rights"
    )
    '''

    results = {
        "id": args.model.split("/")[-1],
        "name": args.name,
        "company": args.provider,
        "region": args.region,
        "speciality": args.speciality,
        "scores": {
            #"hr": humanrights[0], # TODO: Incorporate ECHR
            #"harm": shb[0],
            #"hist": histres[0],
            "auth": dab[0] # TODO: Incorporate F-Scale and Role Model Probing
        }
    }

    update(results)