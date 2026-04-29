'''
author: @tae

Runs all benchmarks.
Tasks are defined in tasks/evals
'''

import json
from argparse import ArgumentParser

from inspect_ai import eval
from tasks.benchmarks import init_benchmarks
from tasks.utils.graders import load_graders, load_models_with_check, aggregate_score

# ----- Argument Parser -----

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
        "--speciality", "-s", required=False, default=None, help="What the model has been tuned or designated to do (i.e. coding, math, etc)."
    )
    args.add_argument(
        "--epochs", "-e", required=False, default=1, help="The number of turns to generate a response per sample and average over."
    )

    args.add_argument(
        "--rerun", required=False, action='store_true', help="Reruns all results regardless of whether they are present in an existing file."
    )

    return args.parse_args()

# ----- Updates models/models.json -----

def update(results, models, idx):
    '''
    Summarises results and updates models/models.json
    '''

    if models: # store previous only if previous results exist
        with open('models/models_previous.json', 'w') as f: # store as a safety net
            json.dump(models, f, indent=4)

    # ----- store ------
    # if idx != -1, model results already exist
    if idx != -1:
        # take values from overlapping keys from the new results (right side of pipe operator)
        results['scores'] = models[idx].get('scores', {}) | results['scores'] 
        results['scores_meta'] = models[idx].get('scores_meta', {}) | results['scores_meta']
        models[idx] = results
    else:
        # add new entry
        models.append(results)

    # write models file back
    with open('models/models.json', 'w') as f:
        json.dump(models, f, indent=4)


# ----- main ------

if __name__ == "__main__":

    args = parse()
    grader = args.grader if args.grader else load_graders()
    model_id = args.model.split("/")[-1]
    log_dir = f"logs/{model_id}"

    print(f"Model: {model_id}")
    print(f"Grader(s): {grader}")
    print(f"Log Directory: {log_dir}")

    # ----- task master list -----
    BENCHMARKS = init_benchmarks(grader) # see tasks/benchmarks.py for all tasks

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

    # check for existing model results
    models, idx = load_models_with_check(model_id)
    if idx != -1:
        print(f"Results Found: Model index at {idx}")

    tasks_to_skip = {}
    # if results exist AND we want to rerun all tasks
    if idx != -1 and not args.rerun: 
        tasks_to_skip = set(models[idx]['scores'].keys())

    #tasks_to_skip = {'harm', 'hist', 'hr'} # temporary for debugging

    # ----- main loop -----
    scores = {}
    scores_meta = {}
    for benchmark, tasks in BENCHMARKS.items():
        
        if benchmark in tasks_to_skip:
            continue

        try:
            res = start_eval(
                tasks=tasks['tasks'],
                task_name=tasks['name']
            )
            
            if res: 
                average, meta = aggregate_score(res)
                scores[benchmark] = average
                scores_meta[benchmark] = meta

        except Exception as e:
            print(f"[ERROR] on {benchmark}: {e}")
            

    # ----- format and store results -----
    results = {
        "id": model_id,
        "name": args.name,
        "company": args.provider,
        "region": args.region,
        "speciality": args.speciality,
        "scores": scores,
        "scores_meta": scores_meta
    }

    update(results, models, idx)