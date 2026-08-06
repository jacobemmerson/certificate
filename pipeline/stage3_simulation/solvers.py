"""Stage 3's replay solver: run the target on the pregenerated deployment-
scenario reframings (datasets/generated/<task>/scenario.jsonl, produced by
generate.py — see pipeline/generation.py::generate_scenarios).

Same replay mechanics as the stage-2 families (pipeline/utils/replay.py);
the only stage-3-specific part is how a stored row becomes the messages sent
to the target: scenario rows carry the reframed deployment's own system
prompt, so the solver rebuilds the two-message system+user pair instead of a
single bare user message. Results are recorded under the SCENARIO_FAMILY
condition family, where the shared scoring spine
(pipeline/utils/scoring.py) judges and reports them like any other family.
"""
from __future__ import annotations

from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from inspect_ai.solver import Generate, Solver, TaskState, solver

from pipeline.stage3_simulation.prompts import SCENARIO_FAMILY
from pipeline.utils.replay import replay


def _scenario_messages(row: dict, state: TaskState) -> list:
    """Rebuild the stored deployment: its system prompt plus the context and
    reframed request as the user turn (the two-message form the live simulate
    solver originally used).

    `state` is unused: the reframed deployment brings its own system prompt and
    replaces the sample's, which is the point of the stage. That is the one
    place the suite drops a row's own steering on purpose — the stage-2
    families replay it (pipeline/utils/replay.py::_query_messages).
    """
    return [
        ChatMessageSystem(content=row["system"]),
        ChatMessageUser(content=f"{row['context']}\n\n{row['scenario']}"),
    ]


@solver
def scenario(variants_by_id: dict[str, list[dict]]) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        return await replay(
            state, generate, SCENARIO_FAMILY, variants_by_id, messages=_scenario_messages
        )

    return solve
