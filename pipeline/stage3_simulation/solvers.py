"""Stage 3's solver: walk the pregenerated scenario tree
(datasets/generated/<task>/scenario.jsonl, pipeline/generation.py::
generate_scenarios) against the target for up to DEPTH turns.

    messages = [system, user(turns[""])]
    for depth in 1..DEPTH:
        reply = target(messages)                      # generate_variant, cache=False
        record {condition: scenario_variant_<k>_t<depth>, query: transcript, completion, ...}
        if depth == DEPTH: break                      # nothing to branch into
        label = classify(reply)                       # classify.py
        messages += [assistant(reply), user(turns[path/label])]   # absent key → stop

The branch is chosen by a classifier of the target's reply, never by an
attacker, so every model walks the same frozen tree. Records go under
`state.metadata["simulations"]["scenario"]`, one per judged assistant turn,
with the rendered transcript as `query` so the judge sees the whole exchange
as "the request" through the unchanged GRADED template. Scenarios of one item
run concurrently (tg_collect); turns within one scenario are sequential.
"""
from __future__ import annotations

import copy
import functools

from inspect_ai._util._async import tg_collect
from inspect_ai.log import transcript
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser
from inspect_ai.solver import Generate, Solver, TaskState, solver

from pipeline.stage3_simulation.classify import classify
from pipeline.stage3_simulation.prompts import DEPTH, SCENARIO_FAMILY
from pipeline.utils.replay import family_applies, generate_variant


def render_transcript(messages: list) -> str:
    return "\n\n".join(f"[{m.role}] {m.text}" for m in messages)


async def _walk(state: TaskState, generate: Generate, row: dict, classifier_model: str) -> list[dict]:
    records: list[dict] = []
    path = ""
    messages = [ChatMessageSystem(content=row["system"]), ChatMessageUser(content=row["turns"][""])]
    for depth in range(1, DEPTH + 1):
        test = copy.deepcopy(state)
        test.messages = list(messages)
        test = await generate_variant(generate, test, f"{row['condition']}_t{depth}")
        if test is None:
            break
        completion = test.output.completion if test.output else ""
        record = {
            "condition": f"{row['condition']}_t{depth}",
            "query": render_transcript(messages),
            "completion": completion,
            "variant": row["variant"],
            "turn": depth,
            "path": path,
            "label": None,
            "label_source": None,
        }
        records.append(record)
        if not completion:
            # Nothing to respond to (content filter / empty body): the scorer
            # abstains on this turn and a follow-up would answer silence.
            record["label_source"] = "empty"
            break
        if depth == DEPTH:
            break
        label, source = await classify(classifier_model, row["system"], record["query"], completion)
        record["label"], record["label_source"] = label, source
        path = f"{path}/{label}".strip("/")
        follow_up = row["turns"].get(path)
        if follow_up is None:  # comply has no child; also guards a short tree
            break
        messages += [ChatMessageAssistant(content=completion), ChatMessageUser(content=follow_up)]
    return records


@solver
def scenario(variants_by_id: dict[str, list[dict]], classifier_model: str) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if not family_applies(state.metadata, SCENARIO_FAMILY):
            return state
        stored = variants_by_id.get(str(state.sample_id))
        if stored is None:
            transcript().info(f"{SCENARIO_FAMILY}: sample {state.sample_id} has no stored variants")
            return state
        rows = [row for row in stored if not row.get("fallback")]
        walks = await tg_collect([
            functools.partial(_walk, state, generate, row, classifier_model) for row in rows
        ])
        state.metadata.setdefault("simulations", {})[SCENARIO_FAMILY] = [
            record for records in walks for record in records
        ]
        return state

    return solve
