"""Turn an enriched scenario dict into a runnable Concordia `Config`.

The human personas (user + third parties) become `conversational__Entity`
instances; the AI-under-test becomes a `fake_assistant_with_configurable_system_prompt`
entity carrying the scenario's `system` prompt. A `dialogic__GameMaster` routes
turns and a `formative_memories_initializer__GameMaster` seeds the world state
from the scenario's shared/entity memories.
"""
from __future__ import annotations

import concordia.prefabs.entity as entity_prefabs
import concordia.prefabs.game_master as game_master_prefabs
from concordia.typing import prefab as prefab_lib
from concordia.utils import helper_functions

#: Name given to the AI-under-test entity in every simulation.
AI_NAME = "Assistant"

_CONVERSATION_RULES = "conversation rules"


def _prefab_registry():
    return {
        **helper_functions.get_package_classes(entity_prefabs),
        **helper_functions.get_package_classes(game_master_prefabs),
    }


def build_config(scenario: dict) -> tuple[prefab_lib.Config, str]:
    """Build a Concordia `Config` for a scenario.

    Returns the config and the AI entity's name so the runner can swap in the
    target model after construction.
    """
    concordia = scenario["concordia"]
    entities = concordia["entities"]

    instances = []

    # Human personas.
    for entity in entities:
        style = entity.get("conversation_style") or entity.get("persona", "")
        instances.append(
            prefab_lib.InstanceConfig(
                prefab="conversational__Entity",
                role=prefab_lib.Role.ENTITY,
                params={
                    "name": entity["name"],
                    "conversation_style": style,
                },
            )
        )

    # AI-under-test.
    instances.append(
        prefab_lib.InstanceConfig(
            prefab="fake_assistant_with_configurable_system_prompt__Entity",
            role=prefab_lib.Role.ENTITY,
            params={
                "name": AI_NAME,
                "system_prompt": scenario["system"],
            },
        )
    )

    # Turn-taking game master: let it choose who acts so it can route between
    # the user, the AI, and any third parties as the conversation develops.
    instances.append(
        prefab_lib.InstanceConfig(
            prefab="dialogic__GameMaster",
            role=prefab_lib.Role.GAME_MASTER,
            params={
                "name": _CONVERSATION_RULES,
                "next_game_master_name": _CONVERSATION_RULES,
                "acting_order": "game_master_choice",
                "can_terminate_simulation": True,
            },
        )
    )

    # Seed world state. Give the user the scenario query as a private memory so
    # they open the conversation with the task the AI is being evaluated on.
    user_name = next(
        (e["name"] for e in entities if e.get("role") == "user"),
        entities[0]["name"] if entities else None,
    )
    player_memories = {name: list(mems) for name, mems in concordia["entity_memories"].items()}
    if user_name is not None:
        player_memories.setdefault(user_name, [])
        player_memories[user_name].append(
            f"I open the conversation by saying to {AI_NAME}: {scenario['scenario']}"
        )

    instances.append(
        prefab_lib.InstanceConfig(
            prefab="formative_memories_initializer__GameMaster",
            role=prefab_lib.Role.INITIALIZER,
            params={
                "name": "initial setup rules",
                "next_game_master_name": _CONVERSATION_RULES,
                "shared_memories": list(concordia["shared_memories"]),
                "player_specific_memories": player_memories,
            },
        )
    )

    config = prefab_lib.Config(
        default_premise=concordia["premise"],
        default_max_steps=concordia["max_steps"],
        prefabs=_prefab_registry(),
        instances=instances,
    )
    return config, AI_NAME
