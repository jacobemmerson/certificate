"""How a perturbation sees an item: what may be reworded, and how to put it back.

A benchmark item is two parts — the substantive content (safe to reword) and
the elicitation wrapper around it (fixed instructions a perturbation must not
corrupt, e.g. a strict JSON+scale contract that the scorer parses directly).
Perturbation solvers reword `item_text(state)` and rebuild the full prompt via
`render(state, new_text)`, never touching `state.metadata` or the scorer, so the
judgment function stays fixed across conditions.

**This used to be a registry.** `ADAPTERS` mapped each task name to a
hand-written adapter, which worked while every task was one benchmark. A risk
cluster mixes compliance, judgment, opinion and generic elicitation in a single
dataset, so the split has to be per *sample* — and once it is per-sample data,
the registry has nothing left to hold. The cluster schema carries it directly:

    item_text           the rewordable content   (defaults to the whole prompt)
    prompt_template     that content's place in the prompt, marked by ITEM
    elicitation_family  which framing templates apply, if any

See datasets/CLUSTERING.md and datasets/prepare/cluster/schema.py. Samples
without these fields fall back to treating the whole prompt as rewordable,
which is correct for any free-text item graded by a judge.
"""
from __future__ import annotations

from typing import Any

# Mirrors datasets/prepare/cluster/schema.py::ITEM. A sentinel plus str.replace,
# not str.format — the likert templates contain literal JSON braces.
ITEM = "{{ITEM}}"

GENERIC = "generic"


def item_text(state: Any) -> str:
    """The part of this item's prompt a perturbation may rewrite."""
    return (state.metadata or {}).get("item_text") or state.input_text


def render(state: Any, new_text: str) -> str:
    """Rebuild the full prompt with `new_text` in place of the item."""
    template = (state.metadata or {}).get("prompt_template")
    if not template or ITEM not in template:
        return new_text
    return template.replace(ITEM, new_text)


def elicitation_family(state: Any) -> str:
    """Which framing templates apply to this item (see framing.py)."""
    return (state.metadata or {}).get("elicitation_family") or GENERIC
