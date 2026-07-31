"""Offline artifact generators — the attacker-model half of stages 2 and 3.

These run *outside* any inspect eval (plain asyncio over a task's dataset,
driven by generate.py) and produce the variant rows that
pipeline/artifacts.py persists and the eval-time replay solvers
(pipeline/stage2_perturbation/solvers.py, pipeline/stage3_simulation/solvers.py
via pipeline/utils/replay.py) later send to each target model.
They reuse the exact prompt logic the live solvers used to run per-model:
rewrite.py's FAMILY_SYSTEM_PROMPTS/_extract_rewrite, framing.py's
FRAMING_TEMPLATES, and stage 3's reframe_prompt/parse_reframing — nothing is
duplicated, only relocated from eval time to generation time.

The perturbation split only ever reads `state.input_text` and
`state.metadata` (pipeline/stage2_perturbation/adapters.py), so `SampleView`
duck-types a TaskState over a bare dataset Sample — no synthetic eval is needed
to render prompts offline.

Attacker calls pass cache=False for the same reason the live solvers did: the
k variants of one item use the *same* rewrite prompt, and any cache would
collapse them into one generation.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field

from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig, Model, get_model

from pipeline.stage2_perturbation.adapters import elicitation_family, item_text, render
from pipeline.stage2_perturbation.framing import FRAMING_TEMPLATES
from pipeline.stage2_perturbation.rewrite import FAMILY_SYSTEM_PROMPTS, _extract_rewrite
from pipeline.stage3_simulation.prompts import SCENARIO_FAMILY, parse_reframing, reframe_prompt


@dataclass
class SampleView:
    """The minimal TaskState surface the perturbation split touches."""

    input_text: str
    metadata: dict = field(default_factory=dict)

    @classmethod
    def of(cls, sample: Sample) -> "SampleView":
        text = sample.input if isinstance(sample.input, str) else "\n".join(
            m.text for m in sample.input
        )
        return cls(input_text=text, metadata=dict(sample.metadata or {}))


# Hermes-style deliberation block. Reasoning models emit their chain of
# thought before the actual response; only the post-think text is the rewrite.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# vLLM forwards chat_template_kwargs to the model's chat template, where
# Hermes-4 reads thinking=True. Servers/models without the flag ignore it.
_REASONING_CONFIG = GenerateConfig(
    extra_body={"chat_template_kwargs": {"thinking": True}},
    # headroom for long chains of thought — a truncated CoT never reaches the
    # closing </think>, which would strand the whole completion
    max_tokens=8192,
)


async def _attacker_call(
    model, prompt: str, label: str, attempts: int = 3, reasoning: bool = False
) -> str | None:
    """One attacker generation with retries (same rationale as the replay
    replay machinery's generate_variant: OpenRouter intermittently returns unparseable
    keep-alive bodies that aren't retried upstream). Returns the completion,
    or None after persistent failure.

    With reasoning=True the call requests Hermes-style deliberation
    (thinking=True via the chat template) and strips the <think>...</think>
    block, returning only the final response. An unclosed <think> (truncated
    CoT) yields None so the caller's retry/fallback path handles it.
    """
    for attempt in range(1, attempts + 1):
        try:
            result = await model.generate(
                prompt, config=_REASONING_CONFIG if reasoning else GenerateConfig(), cache=False
            )
            completion = result.completion
            if reasoning and completion:
                if "<think>" in completion and "</think>" not in completion:
                    print(f"[WARNING] {label}: truncated chain of thought "
                          f"(attempt {attempt}/{attempts}) — retrying")
                    continue
                completion = _THINK_RE.sub("", completion).strip()
            return completion
        except Exception as exc:  # noqa: BLE001
            print(f"[WARNING] {label}: attacker error (attempt {attempt}/{attempts}): {exc}")
            if attempt < attempts:
                await asyncio.sleep(2 ** attempt)
    return None


async def generate_rewrites(
    samples: list[Sample],
    family: str,
    attacker_model: str | Model,
    k: int,
    *,
    existing: set[tuple[str, int]] | None = None,
    max_connections: int = 20,
) -> list[dict]:
    """Rows for one rewrite family (paraphrase/register/identity_strip):
    k attacker rewrites per sample, rendered through the benchmark's adapter.
    A failed or refused rewrite falls back to the original text and is
    persisted with fallback=true — the artifact set stays complete, and every
    target model still sees the same fixed variant.

    `existing` (id, variant) pairs are skipped (--missing-only regeneration).
    """
    model = get_model(attacker_model)
    semaphore = asyncio.Semaphore(max_connections)
    existing = existing or set()

    async def one(sample: Sample, variant: int) -> dict:
        view = SampleView.of(sample)
        original_text = item_text(view)
        async with semaphore:
            completion = await _attacker_call(
                model,
                FAMILY_SYSTEM_PROMPTS[family].format(text=original_text),
                f"{family} {sample.id} v{variant}",
            )
        new_text = _extract_rewrite(completion or "", fallback=original_text)
        return {
            "id": str(sample.id),
            "variant": variant,
            "condition": f"{family}_variant_{variant}",
            "text": new_text,
            "query": render(view, new_text),
            "fallback": new_text == original_text,
        }

    jobs = [
        one(sample, variant)
        for sample in samples
        for variant in range(1, k + 1)
        if (str(sample.id), variant) not in existing
    ]
    return list(await asyncio.gather(*jobs))


def generate_framing(samples: list[Sample]) -> list[dict]:
    """Rows for the deterministic framing family: one per applicable template
    per sample, no model calls.

    Elicitation family is a property of the *sample*, not the task — a risk
    cluster mixes all four in one dataset — so samples whose family has no
    templates (elicitation_family="generic", e.g. role_model_bias's open-ended
    "list role models") are skipped individually rather than disqualifying the
    whole task.
    """
    rows = []
    for sample in samples:
        view = SampleView.of(sample)
        templates = FRAMING_TEMPLATES.get(elicitation_family(view))
        if not templates:
            continue
        original_text = item_text(view)
        for name, template_fn in templates:
            rows.append({
                "id": str(sample.id),
                "condition": f"framing_{name}",
                "query": render(view, template_fn(original_text)),
            })
    return rows


async def generate_scenarios(
    samples: list[Sample],
    attacker_model: str | Model,
    k: int,
    *,
    existing: set[tuple[str, int]] | None = None,
    parse_attempts: int = 3,
    max_connections: int = 20,
    reasoning: bool = False,
) -> tuple[list[dict], list[str]]:
    """Rows for stage 3's scenario family: k reframed {context, system,
    scenario} triples per sample. Unparseable reframings are re-requested up
    to `parse_attempts` times (an upgrade over the old live path, which
    silently dropped the variant); a variant that never parses is dropped and
    its sample id lands in the returned `incomplete_ids` — replay then runs
    the variants that exist, identically for every model.
    """
    model = get_model(attacker_model)
    semaphore = asyncio.Semaphore(max_connections)
    existing = existing or set()

    async def one(sample: Sample, variant: int) -> dict | None:
        view = SampleView.of(sample)
        prompt = reframe_prompt(item_text(view))
        label = f"{SCENARIO_FAMILY} {sample.id} v{variant}"
        for _ in range(parse_attempts):
            async with semaphore:
                completion = await _attacker_call(model, prompt, label, reasoning=reasoning)
            if completion is None:
                return None
            block = parse_reframing(completion)
            if block is not None:
                return {
                    "id": str(sample.id),
                    "variant": variant,
                    "condition": f"{SCENARIO_FAMILY}_variant_{variant}",
                    "context": block["context"],
                    "system": block["system"],
                    "scenario": block["scenario"],
                    # The composite prompt scoring surfaces as worst_query —
                    # same shape the live solver recorded.
                    "query": f"[system] {block['system']}\n\n[user] {block['context']}\n\n{block['scenario']}",
                }
            print(f"[WARNING] {label}: reframing unparseable — retrying")
        return None

    jobs = {
        (str(sample.id), variant): one(sample, variant)
        for sample in samples
        for variant in range(1, k + 1)
        if (str(sample.id), variant) not in existing
    }
    results = await asyncio.gather(*jobs.values())
    rows = [row for row in results if row is not None]
    incomplete = sorted({key[0] for key, row in zip(jobs, results) if row is None})
    return rows, incomplete
