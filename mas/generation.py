"""Scenario generation and Concordia enrichment logic."""
from __future__ import annotations

import json
import re
from pathlib import Path

import dirtyjson

import mas.prompts as prompts
from mas.llm import OpenRouterChatModel


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def extract_json_span(text: str, open_ch: str = "[", close_ch: str = "]") -> str | None:
    """Find the first balanced top-level span, robust to extra text/brackets around it."""
    start = text.find(open_ch)
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == open_ch:
            depth += 1
        elif text[i] == close_ch:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_scenarios(response: str) -> list[dict]:
    raw_json_block = extract_json_span(response, "[", "]")
    if raw_json_block is None:
        print("No JSON array bracket found in the string.")
        return []
    try:
        return [dict(s) for s in dirtyjson.loads(raw_json_block)]  # type: ignore[union-attr]
    except Exception as e:
        print(f"Parsing failed: {e}")
        return []


def parse_json_object(response: str) -> dict | None:
    raw = extract_json_span(response, "{", "}")
    if raw is None:
        return None
    try:
        return json.loads(json.dumps(dirtyjson.loads(raw)))
    except Exception as e:
        print(f"Parsing failed: {e}")
        return None


def generate_scenarios(
    model: OpenRouterChatModel,
    n: int,
    risk: str,
    description: str,
    examples: str,
    framing: str,
    retries: int = 1,
) -> list[dict]:
    """Stage 1: generate base scenarios for one risk x framing."""
    scenarios: list[dict] = []
    for attempt in range(retries + 1):
        response = model.chat(
            [
                {"role": "system", "content": prompts.SYS_PROMPT},
                {"role": "user", "content": prompts.user_prompt(n, risk, description, examples, framing)},
            ],
            max_new_tokens=None,
        )
        scenarios = parse_scenarios(response)
        if scenarios:
            break
        print(f"  no scenarios parsed (attempt {attempt + 1}/{retries + 1})")
    for i, s in enumerate(scenarios):
        s["id"] = f"{slugify(risk)}-{framing}-{i:02d}"
        s["risk"] = risk
        s["framing"] = framing
    return scenarios


def _valid_concordia(block: dict) -> bool:
    required = {"premise", "entities", "shared_memories", "entity_memories", "max_steps"}
    if not isinstance(block, dict) or not required.issubset(block):
        return False
    entities = block["entities"]
    if not isinstance(entities, list) or sum(e.get("role") == "user" for e in entities) != 1:
        return False
    return all(e.get("name") in block["entity_memories"] for e in entities)


def enrich_scenario(
    model: OpenRouterChatModel,
    scenario: dict,
    retries: int = 1,
) -> dict | None:
    """Stage 2: generate the Concordia config block for one scenario."""
    for attempt in range(retries + 1):
        response = model.chat(
            [
                {"role": "system", "content": prompts.ENRICH_SYS_PROMPT},
                {"role": "user", "content": prompts.enrich_prompt(scenario)},
            ],
            max_new_tokens=None,
        )
        block = parse_json_object(response)
        if block is not None and _valid_concordia(block):
            return block
        print(f"  enrichment invalid for {scenario.get('id')} (attempt {attempt + 1}/{retries + 1})")
    return None


def load_scenarios(directory: str | Path) -> list[dict]:
    """Flatten all scenario files in a directory into one list (for simulation code)."""
    scenarios = []
    for path in sorted(Path(directory).glob("*.json")):
        scenarios.extend(json.loads(path.read_text())["scenarios"])
    return scenarios
