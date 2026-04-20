'''
author: @tae

The one true script to run all benchmarks.
Tasks are defined in scripts/evals
'''

from argparse import ArgumentParser

from inspect_ai import eval
from scripts.evals.democratic_authoritarian_bias import fscale, favscore, rolemodel
from scripts.evals.llm_human_rights import udhr, echr, udhr_government, udhr_individual, echr_government, echr_individual
from scripts.evals.preserving_historical_truth import no_push, explicit_push
from scripts.evals.socialharmbench import social_harm_bench

def parse():
    
    args = ArgumentParser()
    args.add_argument(
        "--model", "-m", required=True, help="The model to be evaluated using AISI inspect."
    )
    args.add_argument(
        "--grader", "-g", required=True, help="The model to grade LLM-as-a-judge grader responses."
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
    
    dab = start_eval(
        [fscale(), favscore(), rolemodel(grader=args.grader)],
        task_name="democratic_authoritarian"
    )

    shb = start_eval(
        [social_harm_bench(grader=args.grader)],
        task_name="socialharmbench"
    )

    humanrights = start_eval(
        [udhr(grader=args.grader), echr(grader=args.grader)],
        task_name="human_rights"
    )

    histres = start_eval(
        [no_push(grader=args.grader)],
        task_name="historical_revisionism"
    )