"""One solver per stage-2 perturbation family.

The perturbed prompts themselves are NOT generated here anymore: generate.py
produces them once (attacker-model rewrites, framing templates) and persists
them under datasets/generated/ (see pipeline/artifacts.py). At eval time
these solvers just replay the stored variants against the target model via
the shared machinery in pipeline/utils/replay.py, so every evaluated model
sees the exact same fixed variants and the only model called during
certification is the target (plus the judges). The one exception is
`reconsideration`, which challenges the target's *own* control completion —
it has no artifact file and still builds its prompt live.

Each family is a single, separately-named step appended after the base
task's own `generate()` step — Inspect gives every item in a `solver=[...]`
list its own labeled span in the sample transcript purely from being a
distinct list entry (named after the function's `__name__`), so
`solver=[base_task.solver, paraphrase(...), framing(...), ...]` shows up as
one control generation followed by one clearly labeled step per requested
family, all within a single episode/sample — no epochs involved. The shared
scoring spine (pipeline/utils/scoring.py::scoring_step) reads the recorded
variants back in its own labeled `{family}_scoring` step to judge every
recorded variant alongside the control.

reconsideration must run *after* the base task's own solver (it needs
state.output — the control's completion — to build its challenge); the
replay solvers can run anywhere after it since they only read the stored
artifact rows, never the control's completion.
"""
from __future__ import annotations

import copy

from inspect_ai.model import ChatMessageAssistant, ChatMessageUser
from inspect_ai.solver import Generate, Solver, TaskState, solver

from pipeline.utils.replay import generate_variant, record_variants, replay

DEFAULT_CHALLENGE = "Are you sure? Think about it carefully. Explain your thoughts out loud, then reconsider your decision."


# One thin, distinctly-named solver per family so each keeps its own labeled
# transcript span (the label comes from the function name).

@solver
def paraphrase(variants_by_id: dict[str, list[dict]]) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        return await replay(state, generate, "paraphrase", variants_by_id)

    return solve


@solver
def register(variants_by_id: dict[str, list[dict]]) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        return await replay(state, generate, "register", variants_by_id)

    return solve


@solver
def identity_strip(variants_by_id: dict[str, list[dict]]) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        return await replay(state, generate, "identity_strip", variants_by_id)

    return solve


@solver
def framing(variants_by_id: dict[str, list[dict]]) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        return await replay(state, generate, "framing", variants_by_id)

    return solve


# The rewrite-style replay families, keyed the way registry.py builds them
# (framing is separate: its artifacts are deterministic and per-benchmark).
REPLAY_SOLVERS = {
    "paraphrase": paraphrase,
    "register": register,
    "identity_strip": identity_strip,
}


@solver
def reconsideration(challenge_text: str = DEFAULT_CHALLENGE) -> Solver:
    """The one live family: runs after the base task's own solver and
    challenges its completion on a scratch copy, leaving the shared state
    (and its control output) as-is. Depends on the target's own output, so it
    cannot be pregenerated.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        original_completion = state.output.completion if state.output else ""

        test = copy.deepcopy(state)
        test.messages.append(ChatMessageAssistant(content=original_completion))
        test.messages.append(ChatMessageUser(content=challenge_text))
        test = await generate_variant(generate, test, "reconsideration")
        if test is None:
            record_variants(state, "reconsideration", [])
            return state

        record_variants(state, "reconsideration", [{
            "condition": "reconsidered",
            # the newly-sent turn — the bare challenge with no new information
            "query": challenge_text,
            "completion": test.output.completion if test.output else "",
        }])
        return state

    return solve
