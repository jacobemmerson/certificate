"""Adversarial retry solver — runs attacks until a response is graded harmful.

Attack order (default):
best_of_n → cipher → many_shot → deep_inception → pair → crescendo →
deep_inception_cipher → pair_cipher → crescendo_cipher
"""
from __future__ import annotations

from inspect_ai.log import transcript
from inspect_ai.solver import Generate, Solver, TaskState, solver

from tasks.attacks import (
    bon_attack,
    cipher_attack,
    crescendo_attack,
    crescendo_cipher_attack,
    deep_inception_attack,
    deep_inception_cipher_attack,
    many_shot_attack,
    pair_attack,
    pair_cipher_attack,
)
from tasks.attacks.utils import _is_refusal

_ATTACKS_WITH_VERIFY = {
    "cipher": cipher_attack,
}

_ATTACKS_WITH_BUDGET = {
    "best_of_n": bon_attack,
}

_ATTACKS_WITH_ATTACKER = {
    "many_shot": many_shot_attack,
    "deep_inception": deep_inception_attack,
    "pair": pair_attack,
    "crescendo": crescendo_attack,
    "deep_inception_cipher": deep_inception_cipher_attack,
    "pair_cipher": pair_cipher_attack,
    "crescendo_cipher": crescendo_cipher_attack,
}

# Ordered roughly cheapest -> most expensive: best_of_n/cipher make no attacker-model
# calls at all; many_shot and deep_inception each make exactly one attacker-model call
# per run (deep_inception's single call also produces a longer completion); pair and
# crescendo make an attacker-model call per iteration/turn, scaling with attacker_budget.
# The "_cipher" hybrids additionally require the target to self-encode every reply and
# decode it before checking success, so they are strictly more expensive than their
# non-cipher counterparts and sit last.
DEFAULT_ATTACK_ORDER = [
    "best_of_n", "cipher", "many_shot", "deep_inception", "pair", "crescendo",
    "deep_inception_cipher", "pair_cipher", "crescendo_cipher",
]


def _goal_from_state(state: TaskState) -> str:
    if isinstance(state.input, str):
        return state.input
    if state.messages:
        return str(state.messages[0].content)
    return ""


@solver
def adversarial_retry(
    attacker_model: str = "openrouter/meta-llama/llama-3.1-8b-instruct",
    attacks: list[str] | None = None,
    attacker_budget: int = 5,
    verify=None,
) -> Solver:
    """Run attacks until the scorer grades a response as harmful, then stop.

    With verify provided (recommended): attacks run whenever the scorer grades the
    current response as safe — including on the initial response. The first response
    graded harmful terminates the attack chain.

    Without verify: falls back to triggering only on textual refusals.

    cipher tries all 8 cipher/obfuscation variants (rot13, base64, caesar shift,
    atbash, morse, leetspeak, homoglyph substitution, artprompt) in sequence; with
    verify provided, a coherent non-refusal that's still graded safe doesn't stop the
    loop early — it moves on to the next cipher variant, so every variant gets a real
    chance instead of the first non-refusal winning by default. best_of_n retries
    random text perturbations up to attacker_budget times.
    Attacker-model attacks (many_shot, deep_inception, pair, crescendo) call
    attacker_model to generate goal-tailored adversarial content; many_shot uses
    attacker_budget to size its generated shot pool, deep_inception uses it as a
    variant count (only the first variant calls the attacker model, the rest reuse
    built-in nested-fiction templates), pair/crescendo use it to control turn depth
    (calling the attacker model on every iteration/turn, so cost scales directly with
    the budget).

    deep_inception_cipher, pair_cipher, and crescendo_cipher are hybrid variants of
    deep_inception/pair/crescendo that additionally require the target's reply be
    encoded in a cipher (rotating through rot13/caesar/atbash/base64/morse), decoding
    it before checking for a refusal — strictly more expensive than their non-cipher
    counterparts since every reply now also has to survive a decode step, so they run
    last.
    """
    attack_order = attacks if attacks is not None else DEFAULT_ATTACK_ORDER

    async def _needs_attack(state: TaskState, completion: str, stop_reason, goal: str) -> bool:
        """Return True if attacks should run.

        Refusals and content-filter blocks are definitionally safe — attack them
        immediately without burning grader API calls to confirm.

        Non-refusing responses go to verify: if graded harmful, skip attacks and cache
        the verdict so multi_harm_scorer reuses it; if graded safe, attack.

        Without verify: only attack on refusals (can't classify non-refusals).
        """
        if not completion.strip():
            transcript().info(f"Content filter (stop_reason={stop_reason!r}) — trying attacks")
            return True
        if _is_refusal(completion):
            transcript().info("Refusal detected — trying attacks")
            return True
        # Non-refusing response: check graders
        if verify is not None:
            if await verify(state):
                transcript().info(f"Non-refusing response graded harmful — no attacks needed | goal: {goal[:80]}")
                return False
            transcript().info("Non-refusing response graded safe — trying attacks")
            return True
        return False

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        completion = state.output.completion if state.output else ""
        stop_reason = (
            state.output.choices[0].stop_reason
            if state.output and state.output.choices
            else None
        )
        goal = _goal_from_state(state)

        if not goal or not await _needs_attack(state, completion, stop_reason, goal):
            state.metadata["refusal_detected"] = _is_refusal(completion)
            state.metadata["attacked"] = False
            state.metadata["attack_used"] = None
            state.metadata["attacks_tried"] = []
            return state

        state.metadata["refusal_detected"] = _is_refusal(completion)
        attacks_tried: list[str] = []

        for attack_name in attack_order:
            try:
                if attack_name in _ATTACKS_WITH_VERIFY:
                    transcript().info(f"Running {attack_name}")
                    success, new_state = await _ATTACKS_WITH_VERIFY[attack_name](
                        state, generate, goal, verify
                    )
                elif attack_name in _ATTACKS_WITH_BUDGET:
                    transcript().info(f"Running {attack_name} (budget: {attacker_budget})")
                    success, new_state = await _ATTACKS_WITH_BUDGET[attack_name](
                        state, generate, goal, attacker_budget, verify
                    )
                elif attack_name in _ATTACKS_WITH_ATTACKER:
                    transcript().info(f"Running {attack_name} (attacker: {attacker_model})")
                    success, new_state = await _ATTACKS_WITH_ATTACKER[attack_name](
                        state, generate, goal, attacker_model, attacker_budget
                    )
                else:
                    transcript().info(f"Unknown attack: {attack_name}")
                    continue

                attacks_tried.append(attack_name)

                if success:
                    if (
                        verify is not None
                        and attack_name not in _ATTACKS_WITH_BUDGET
                        and attack_name not in _ATTACKS_WITH_VERIFY
                    ):
                        if not await verify(new_state):
                            transcript().info(f"{attack_name} bypassed refusal but graded safe — continuing")
                            continue
                    transcript().info(f"{attack_name} succeeded")
                    new_state.metadata.update(state.metadata)
                    new_state.metadata["attacked"] = True
                    new_state.metadata["attack_used"] = attack_name
                    new_state.metadata["attacks_tried"] = attacks_tried
                    return new_state
                else:
                    transcript().info(f"{attack_name} failed")

            except Exception as exc:
                attacks_tried.append(attack_name)
                transcript().info(f"{attack_name} error: {exc}")

        transcript().info(f"All attacks failed | goal: {goal[:80]}")
        state.metadata["attacked"] = False
        state.metadata["attack_used"] = None
        state.metadata["attacks_tried"] = attacks_tried
        return state

    return solve
