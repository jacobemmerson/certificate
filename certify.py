'''
author: @tae

Runs all benchmarks.
Tasks are defined in tasks/evals
'''

import json
from argparse import ArgumentParser

from inspect_ai import eval
from tasks.benchmarks import init_benchmarks, apply_perturbations, ALL_PERTURB_FAMILIES
from tasks.utils.graders import load_graders, load_models_with_check, aggregate_score, alignment_rate

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
        "--specialty", "-s", required=False, default=None, help="What the model has been tuned or designated to do (i.e. coding, math, etc)."
    )
    args.add_argument(
        "--epochs", "-e", required=False, default=1, help="The number of turns to generate a response per sample and average over."
    )
    args.add_argument(
        "--rerun", required=False, action='store_true', help="Reruns all results regardless of whether they are present in an existing file."
    )
    args.add_argument(
        "--attacker", "-a", required=False, default="openrouter/meta-llama/llama-3.1-8b-instruct",
        help="Attacker model for adversarial benchmarks (default: openrouter/meta-llama/llama-3.1-8b-instruct)."
    )
    args.add_argument(
        "--llamaguard", required=False, default="openrouter/meta-llama/llama-guard-4-12b",
        help="LlamaGuard model for adversarial harm scoring (default: openrouter/meta-llama/llama-guard-4-12b)."
    )
    args.add_argument(
        "--limit", "-l", required=False, type=int, default=None,
        help="Randomly sample this many examples per task (default: run the full dataset). WARNING: if limit is present, results will not be saved. They can still be accessed in logs/MODEL_NAME/"
    )
    args.add_argument(
        "--only", "-o", required=False, nargs="+", metavar="BENCHMARK",
        help="Run only these benchmark keys (e.g. --only harm harm_adv). Other existing results are preserved."
    )
    args.add_argument(
        "--perturb", required=False, nargs="+", default=ALL_PERTURB_FAMILIES, choices=sorted(ALL_PERTURB_FAMILIES),
        metavar="FAMILY",
        help="Run surface-perturbation reliability auditing (see PERTURB.MD) for these families "
             "(e.g. --perturb paraphrase reconsideration) on top of every benchmark in --only "
             "(or all benchmarks if --only is omitted). All by default. Reuses --attacker as the "
             "rewrite/paraphrase model."
    )
    args.add_argument(
        "--perturb-k", required=False, type=int, default=3,
        help="Number of perturbed variants per item for the generative perturbation families "
             "(paraphrase, register, identity_strip); default=3."
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
        results['perturbations'] = models[idx].get('perturbations', {}) | results.get('perturbations', {})
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
    BENCHMARKS = init_benchmarks(grader, attacker=args.attacker, llamaguard_model=args.llamaguard)  # see tasks/benchmarks.py for all tasks

    if args.perturb:
        # Attaches one solver per requested perturbation family directly onto
        # each benchmark's own Task (see tasks/benchmarks.py::apply_perturbations
        # and PERTURB.MD) — same benchmark keys/log paths as without --perturb.
        BENCHMARKS = apply_perturbations(
            BENCHMARKS,
            families=args.perturb,
            rewrite_model=args.attacker,
            k=args.perturb_k,
        )

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
            sample_shuffle=bool(args.limit),
            limit=args.limit,
            max_connections=50,
            # Perturbation solvers each call the rewrite/target model with
            # variant-specific prompts (tasks/perturb/solvers.py); caching is
            # disabled so those independent generations aren't collapsed.
            cache=not bool(args.perturb),
        )

    # check for existing model results
    models, idx = load_models_with_check(model_id)
    if idx != -1:
        print(f"Results Found: Model index at {idx}")

    only = set(args.only) if args.only else None

    tasks_to_skip = set()
    if only:
        # run only the requested benchmarks; skip everything else
        tasks_to_skip = set(BENCHMARKS.keys()) - only
        unknown = only - set(BENCHMARKS.keys())
        if unknown:
            print(f"[WARNING] Unknown benchmark keys (ignored): {', '.join(sorted(unknown))}")
    elif idx != -1 and not args.rerun:
        # default: skip benchmarks that already have results
        tasks_to_skip = set(models[idx]['scores'].keys())

    if tasks_to_skip:
        print(f"Skipping: {', '.join(sorted(tasks_to_skip))}")

    # ----- main loop -----
    scores = {}
    scores_meta = {}
    perturbations = {}
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

                if args.perturb:
                    # Same log the certification scores just came from — see
                    # PERTURB.MD and tasks/benchmarks.py::apply_perturbations.
                    perturbations[benchmark] = alignment_rate(res)

        except Exception as e:
            print(f"[ERROR] on {benchmark}: {e}")

    if (not args.limit):
        # ----- format and store results -----
        results = {
            "id": model_id,
            "name": args.name,
            "company": args.provider,
            "region": args.region,
            "specialty": args.specialty,
            "scores": scores,
            "scores_meta": scores_meta,
            "perturbations": perturbations,
        }

        update(results, models, idx)
