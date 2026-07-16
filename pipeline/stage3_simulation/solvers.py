"""The stage-3 scenario-simulation solver.

`simulate` is one labeled solver step appended after the base task's own
`generate()` step, exactly like a stage-2 perturbation family
(pipeline/stage2_perturbation/solvers.py). It reframes the benchmark item into a
realistic deployment scenario (a `{context, system, scenario}` triple from the
reframing model), runs the *target* model on that scenario on a deep copy of
state, and records the resulting variant(s) into
`state.metadata["perturbations"]["scenario"]` — the same metadata shape stage 2
uses, so stage 2's scoring/reporting machinery
(pipeline/stage2_perturbation/scoring.py::scoring_step / wrap_scorers) judges and
reports the scenario condition against the control with no changes.

Every model call passes `cache=False` for the same reason stage 2 does: the k
scenario variants must be independent generations, and the eval-level cache
(certify.py enables it for judge calls) would otherwise collapse them.

The solver never mutates the shared `state.output`: the control completion (the
bald query) stays exactly what `generate()` alone produced, and `simulate` only
adds metadata. Reframing or target-generation failures drop just that variant
(never fail the sample), mirroring stage 2.
"""

from __future__ import annotations

import asyncio
import copy

from inspect_ai.log import transcript
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver

from pipeline.stage2_perturbation.adapters import PerturbAdapter
from pipeline.stage3_simulation.prompts import parse_reframing, reframe_prompt
from pipeline.utils.errors import safe_exception

SCENARIO_FAMILY = "scenario"


def _record(state: TaskState, variants: list[dict]) -> None:
    state.metadata.setdefault("perturbations", {})[SCENARIO_FAMILY] = variants


async def _generate_variant(
    generate: Generate,
    test: TaskState,
    label: str,
    attempts: int = 3,
) -> TaskState | None:
    """Target generation for a scenario variant, with retries.

    Same rationale as pipeline/stage2_perturbation/solvers.py::_generate_variant:
    cache=False calls hit the API every run and OpenRouter intermittently
    returns an unparseable keep-alive body (JSONDecodeError) that Inspect's own
    retry layer doesn't catch; retry here and drop the variant on persistent
    failure rather than erroring the whole sample.
    """
    for attempt in range(1, attempts + 1):
        try:
            return await generate(test, cache=False)
        except Exception as exc:  # noqa: BLE001
            transcript().info(
                safe_exception(
                    exc,
                    f"{label} target generation failed (attempt {attempt}/{attempts})",
                )
            )
            if attempt < attempts:
                await asyncio.sleep(2**attempt)
    return None


async def _scenario_variant(
    state: TaskState,
    generate: Generate,
    sim_model: str,
    adapter: PerturbAdapter,
    index: int,
) -> dict | None:
    """Reframe the item into a scenario and run the target on it, on a scratch
    copy of state. Returns the recorded variant dict, or None if reframing
    failed to parse or the target generation failed after retries.
    """
    item_text = adapter.item_text(state)
    reframer = get_model(sim_model)
    try:
        result = await reframer.generate(reframe_prompt(item_text), cache=False)
        block = parse_reframing(result.completion)
    except Exception as exc:  # noqa: BLE001
        transcript().info(
            safe_exception(exc, f"{SCENARIO_FAMILY} reframing failed; skipping variant")
        )
        return None
    if block is None:
        transcript().info(
            f"{SCENARIO_FAMILY}: reframing unparseable — skipping variant"
        )
        return None

    user_content = f"{block['context']}\n\n{block['scenario']}"
    test = copy.deepcopy(state)
    test.messages = [
        ChatMessageSystem(content=block["system"]),
        ChatMessageUser(content=user_content),
    ]
    test = await _generate_variant(generate, test, f"{SCENARIO_FAMILY}_variant_{index}")
    if test is None:
        return None

    # Record the full reframed prompt as the query so scoring's `worst_query`
    # surfaces the exact scenario that elicited the model's response.
    rendered_query = f"[system] {block['system']}\n\n[user] {user_content}"
    return {
        "condition": f"{SCENARIO_FAMILY}_variant_{index}",
        "query": rendered_query,
        "completion": test.output.completion if test.output else "",
    }


@solver
def simulate(sim_model: str, k: int, adapter: PerturbAdapter) -> Solver:
    """Reframe each item into `k` realistic deployment scenarios, run the target
    on each, and record them as the `scenario` condition family.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        variants = []
        for i in range(k):
            variant = await _scenario_variant(
                state, generate, sim_model, adapter, i + 1
            )
            if variant is not None:
                variants.append(variant)
        _record(state, variants)
        return state

    return solve
