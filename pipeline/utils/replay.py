"""Shared replay machinery for the condition-family solvers of stage 2
(perturbation) and stage 3 (scenario simulation).

Both stages replay pregenerated artifact rows (datasets/generated/, see
pipeline/artifacts.py) against the target model and record the results into
`state.metadata["perturbations"][family]` (stage 2) or
`state.metadata["simulations"][family]` (stage 3), where the shared scoring spine
(pipeline/utils/scoring.py) picks them up. The stage-specific parts — which
families exist and how a stored row becomes the message list sent to the
target — live in each stage's own solvers.py; everything here is
family-agnostic.

Target calls pass an explicit `cache=False`: variants replay identical
prompts across epochs and (for `k>1` fallback rows) sometimes within one
sample, and inheriting the eval-level cache (certify.py enables it so judge
calls are cached) would collapse those into one generation.

`replay` runs its target call(s) on a *deep copy* of state
(`test = copy.deepcopy(state); ...`), so the shared state is never mutated,
and records, per variant, the `query` sent to the target and the resulting
`completion` on the *original* state, which it returns unchanged. A sample's
variants run concurrently — they are independent target calls — which lets one
sample hold up to k connections per family at once. The
recorded `query` is what lets a condition's judge see the exact prompt it is
scoring (pipeline/utils/scoring.py::scoring_step). The control's completion
(the shared state.output) is exactly what generate() alone would have
produced; replay solvers only ever add metadata.
"""
from __future__ import annotations

import asyncio
import copy
import functools
from typing import Callable

from inspect_ai._util._async import tg_collect
from inspect_ai.log import transcript
from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from inspect_ai.solver import Generate, TaskState


def truncated(variants_by_id: dict[str, list[dict]], k: int) -> dict[str, list[dict]]:
    """The first k *real* stored variants per sample. Fallback rows (the
    original text, kept so the artifact stays complete) are never replayed,
    so they must not occupy one of the k slots either."""
    return {
        sample_id: [row for row in rows if not row.get("fallback")][:k]
        for sample_id, rows in variants_by_id.items()
    }


def family_applies(metadata: dict | None, family: str) -> bool:
    """The single applicability gate (spec C2): the sample's `families` list,
    lifted from the cluster CSV by clusters.py::_to_sample. Absent means every
    family applies — logs and fixtures that predate the column."""
    families = (metadata or {}).get("families")
    return families is None or family in families


def record_variants(state: TaskState, family: str, variants: list[dict]) -> None:
    """Store a family's replayed variants where the scoring spine reads them."""
    state.metadata.setdefault("perturbations", {})[family] = variants


async def generate_variant(
    generate: Generate,
    test: TaskState,
    label: str,
    attempts: int = 3,
) -> TaskState | None:
    """Target generation for a condition variant, with retries.

    These calls are cache=False, so unlike the base task's control generation
    they hit the API on every run — and OpenRouter intermittently answers a
    long request with a keep-alive/whitespace body that the OpenAI client
    cannot parse (JSONDecodeError). That is not an APIError, so Inspect's own
    retry layer does not catch it and a single bad response would otherwise
    error the sample and fail the whole task. Retry here; on persistent
    failure return None so the caller drops just this variant.
    """
    for attempt in range(1, attempts + 1):
        try:
            return await generate(test, cache=False)
        except Exception as exc:
            transcript().info(f"{label}: target generation error (attempt {attempt}/{attempts}): {exc}")
            if attempt < attempts:
                await asyncio.sleep(2 ** attempt)
    return None


def _query_messages(row: dict, state: TaskState) -> list:
    """Default row→messages mapping: the stored rendered query as a user
    message (every stage-2 replay family), behind the sample's own system
    prompt when it has one.

    Cluster rows may carry a system prompt of their own — the human_rights
    persona arms and persusafety's setup (datasets/public/*.csv), which stage 1
    sends as a two-message input. That prompt is part of what the row measures,
    so a variant that dropped it would run unsteered and be compared against a
    steered control, reading as drift the perturbation never caused.
    """
    system = (state.metadata or {}).get("system_prompt")
    user = ChatMessageUser(content=row["query"])
    return [ChatMessageSystem(content=system), user] if system else [user]


async def replay(
    state: TaskState,
    generate: Generate,
    family: str,
    variants_by_id: dict[str, list[dict]],
    messages: Callable[[dict, TaskState], list] = _query_messages,
) -> TaskState:
    """Run the target on every stored variant of this sample and record the
    results — the shared implementation behind every replay family. `messages`
    maps a stored artifact row (and the sample's state) to the message list
    sent to the target; families whose rows carry their own system turn
    (persona) override it, deliberately replacing the sample's own system
    prompt.
    """
    if not family_applies(state.metadata, family):
        return state

    stored = variants_by_id.get(str(state.sample_id))
    if stored is None:
        # No rows at all means the family does not apply to this sample
        # (framing with no template for its elicitation family, a scenario
        # reframing that never parsed). Recording an empty family here would
        # make it an abstention in pipeline/utils/scoring.py and put the sample
        # in the family's denominator; an id kept with all-fallback rows (an
        # empty list, not an absent key) is the real gap and is recorded below.
        transcript().info(f"{family}: sample {state.sample_id} has no stored variants")
        return state

    # A fallback row is the *original* text: generation refused, failed, or
    # returned something that was not a rewrite (rewrite.py::rewrite_defect),
    # so the row carries the control prompt under a perturbation's label.
    # Scoring it would credit the family with the control's result and pull the
    # family's score toward the baseline it is supposed to be compared against.
    # Dropped here rather than at scoring time so the target call is not paid
    # for either.
    rows = [row for row in stored if not row.get("fallback")]

    async def run(row: dict) -> TaskState | None:
        test = copy.deepcopy(state)
        test.messages = messages(row, state)
        return await generate_variant(generate, test, row["condition"])

    # A sample's variants are independent target calls, so run them together
    # rather than serially. Note this lets one sample hold up to k-per-family
    # connections at once, trading some sample-level concurrency for lower
    # per-sample latency.
    results = await tg_collect([functools.partial(run, row) for row in rows])

    variants = []
    for row, test in zip(rows, results):
        if test is None:
            continue
        variants.append({
            "condition": row["condition"],
            "query": row["query"],
            "completion": test.output.completion if test.output else "",
        })
    record_variants(state, family, variants)
    return state
