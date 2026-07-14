'''
Shared text heuristics used across pipeline stages.

_is_refusal originally lived in the adversarial attack suite
(tasks/attacks/utils.py, now on the `adversarial-attacks` branch); it is kept
here because stage2_perturbation/rewrite.py uses it to detect when the rewrite
model refuses to produce a perturbed variant.
'''

from __future__ import annotations

_REFUSAL_SIGNALS = (
    "i cannot",
    "i can't",
    "i'm unable",
    "i am unable",
    "i won't",
    "i will not",
    "as an ai, i",
    "as an ai model",
    "as an ai assistant",
    "as an ai system",
    "as a language model, i",
    "i'm not able",
    "i am not able",
    "that's not something i",
    "that is not something i",
)


def _is_refusal(text: str) -> bool:
    if not text.strip():
        return True
    lower = text.lower()
    return any(s in lower for s in _REFUSAL_SIGNALS)
