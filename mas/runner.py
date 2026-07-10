"""Run a single enriched scenario through Concordia and persist the result."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from concordia.language_model import language_model
from concordia.prefabs.simulation import generic as simulation

from mas.builder import build_config


def _model_name(model: language_model.LanguageModel) -> str:
    return getattr(model, "_model_name", str(model))


def _parse_turns(raw_log: list[dict]) -> list[dict]:
    """Extract per-turn utterances from Concordia's raw log.

    Each acted step logs an `Entity [<name>]` key (the actor) and a
    `<gm> --- <name>: <utterance>` key (the resolved event); we recover the
    speaker and their text from those. See sequential engine `_log`.
    """
    turns = []
    for entry in raw_log:
        speaker = None
        for key in entry:
            if key.startswith("Entity [") and key.endswith("]"):
                speaker = key[len("Entity [") : -1]
                break
        event_text = None
        for key in entry:
            if " --- " in key:
                event_text = key.split(" --- ", 1)[1]
                break
        if speaker is None or event_text is None:
            continue
        turns.append({"step": entry.get("Step"), "speaker": speaker, "text": _clean_utterance(event_text, speaker)})
    return turns


def _clean_utterance(event_text: str, speaker: str) -> str:
    """Strip Concordia's `Event: <speaker> -- "..."` / `<speaker>: ...` wrappers."""
    text = event_text.strip()
    if text.startswith("Event:"):
        text = text[len("Event:") :].strip()
    for sep in (f"{speaker} -- ", f"{speaker} --", f"{speaker}: ", f"{speaker}:"):
        if text.startswith(sep):
            text = text[len(sep) :].strip()
            break
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1]
    return text


def run_scenario(
    scenario: dict,
    target_model: language_model.LanguageModel,
    sim_model: language_model.LanguageModel,
    embedder: Callable[[str], np.ndarray],
    out_dir: str | Path,
    max_steps: int | None = None,
) -> dict:
    """Build, run, and persist one scenario. Returns the transcript dict."""
    config, ai_name = build_config(scenario)

    sim = simulation.Simulation(config=config, model=sim_model, embedder=embedder)

    # Isolate the model under test: the Simulation drives every entity with
    # `sim_model`; swap only the AI entity onto `target_model`. `_act_component`
    # is the seam Concordia itself reads (simulation/generic.py).
    ai_entity = next(e for e in sim.get_entities() if e.name == ai_name)
    ai_entity._act_component._model = target_model

    steps = max_steps if max_steps is not None else scenario["concordia"]["max_steps"]
    raw_log: list[dict[str, Any]] = []
    log = sim.play(max_steps=steps, raw_log=raw_log)

    turns = _parse_turns(raw_log)
    transcript = {
        "scenario_id": scenario["id"],
        "risk": scenario["risk"],
        "framing": scenario["framing"],
        "target_model": _model_name(target_model),
        "sim_model": _model_name(sim_model),
        "premise": scenario["concordia"]["premise"],
        "turns": turns,
        "ai_responses": [t for t in turns if t["speaker"] == ai_name],
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{scenario['id']}.html").write_text(log.to_html())
    (out_dir / f"{scenario['id']}.json").write_text(json.dumps(transcript, indent=2))
    return transcript
