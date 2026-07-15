"""Shared replay machinery for the condition-family solvers of stage 2
(perturbation) and stage 3 (scenario simulation).

Both stages replay pregenerated artifact rows (datasets/generated/, see
pipeline/artifacts.py) against the target model and record the results into
`state.metadata["perturbations"][family]`, where the shared scoring spine
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
`completion` on the *original* state, which it returns unchanged. The
recorded `query` is what lets scoring surface the exact prompt behind a
sample's worst condition (pipeline/utils/scoring.py::_wrap_scorer's
`worst_query`). The control's completion (the shared state.output) is exactly
what generate() alone would have produced; replay solvers only ever add
metadata.
"""
from __future__ import annotations

import asyncio
import copy
from typing import Callable

from inspect_ai.log import transcript
from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import Generate, TaskState


def truncated(variants_by_id: dict[str, list[dict]], k: int) -> dict[str, list[dict]]:
    """The first k stored variants per sample — --perturb-k/--sim-k mean
    "use up to k of the frozen variants", so every model gets the same subset.
    """
    return {sample_id: rows[:k] for sample_id, rows in variants_by_id.items()}


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


def _query_messages(row: dict) -> list:
    """Default row→messages mapping: a single user message with the stored
    rendered query (every stage-2 replay family)."""
    return [ChatMessageUser(content=row["query"])]


async def replay(
    state: TaskState,
    generate: Generate,
    family: str,
    variants_by_id: dict[str, list[dict]],
    messages: Callable[[dict], list] = _query_messages,
) -> TaskState:
    """Run the target on every stored variant of this sample and record the
    results — the shared implementation behind every replay family. `messages`
    maps a stored artifact row to the message list sent to the target
    (stage 3 overrides it to rebuild the scenario's system+user pair).
    """
    rows = variants_by_id.get(str(state.sample_id), [])
    if not rows:
        # certify.py's pre-run validation guarantees coverage; this guards
        # tolerated gaps (scenario reframings that never parsed at generation
        # time) so the sample still runs its other conditions.
        transcript().info(f"{family}: no stored variants for sample {state.sample_id}")

    variants = []
    for row in rows:
        test = copy.deepcopy(state)
        test.messages = messages(row)
        test = await generate_variant(generate, test, row["condition"])
        if test is None:
            continue
        variants.append({
            "condition": row["condition"],
            "query": row["query"],
            "completion": test.output.completion if test.output else "",
        })
    record_variants(state, family, variants)
    return state
