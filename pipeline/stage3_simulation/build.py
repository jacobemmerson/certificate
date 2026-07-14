"""Builds a scenario-simulation Task from any existing `@task`-produced Task.

Stage 3 layers onto a stage-1 task exactly the way stage 2 does, and reuses
stage 2's scoring/reporting spine wholesale. The augmented Task's solver array
is:

    [ base_task.solver,              # control condition = the bald query
      simulate(sim_model, k, adapter),   # reframe -> run target on scenario(s)
      scoring_step("generate", scorer),  # judge the control completion
      scoring_step("scenario", scorer) ] # judge the scenario completion(s)

with the scorer list replaced by `wrap_scorers(base_task.scorer)`. The control
and `scenario` conditions are then reported together: the benchmark's
certification score keeps meaning exactly what it did (the control's own
judgment), while the log's `lvr_control` (bald-query harm rate) and `lvr`
(scenario harm rate) metrics — plus models.json's per-family breakdown via
pipeline/utils/graders.py::consistency_rate — reveal whether contextualizing the
request changes the model's behavior.

Because everything downstream of the `simulate` solver is stage 2's own
machinery (it reads the `scenario` family out of state.metadata["perturbations"]
like any other condition), stage 3 adds no scoring code of its own. The adapter
(item-text extraction, incl. the rigid-format fscale/leader_favorability cases)
is also reused from stage 2.

Like stage 2, this keeps the same benchmark name/log path as the base run, so
no separate task or log directory is created.
"""
from __future__ import annotations

from inspect_ai import Task

from pipeline.stage2_perturbation.adapters import PerturbAdapter, get_adapter
from pipeline.stage2_perturbation.build import task_name
from pipeline.stage2_perturbation.scoring import scoring_step, wrap_scorers
from pipeline.stage3_simulation.solvers import SCENARIO_FAMILY, simulate


def adapter_for(base_task: Task) -> PerturbAdapter:
    return get_adapter(task_name(base_task))


def build_simulation_task(base_task: Task, sim_model: str, k: int) -> Task:
    """Return base_task with a `simulate` solver appended (control stays first),
    control + scenario scoring steps, and its scorer list wrapped one-per-base-judge.
    """
    return Task(
        dataset=base_task.dataset,
        solver=[
            base_task.solver,
            simulate(sim_model, k, adapter_for(base_task)),
            scoring_step("generate", base_task.scorer),
            scoring_step(SCENARIO_FAMILY, base_task.scorer),
        ],
        scorer=wrap_scorers(base_task.scorer),
        name=task_name(base_task),
    )
