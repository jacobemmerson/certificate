'''
Legal-group leaves (leaves.toml): the reference texts relevance scoring
compares each benchmark's items against. `Source.leaf` names one by id.
'''

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

LEAVES_PATH = Path(__file__).with_name("leaves.toml")


@dataclass(frozen=True)
class Leaf:
    id: str
    title: str
    cop_ref: str
    legal_text: str
    exemplars: tuple[str, ...] = ()
    threshold: float | None = None  # overrides the calibrated cosine threshold


def load_leaves(path: Path = LEAVES_PATH) -> dict[str, Leaf]:
    leaves: dict[str, Leaf] = {}
    with open(path, "rb") as handle:
        for entry in tomllib.load(handle)["leaf"]:
            leaf = Leaf(**{**entry, "exemplars": tuple(entry.get("exemplars", ()))})
            if leaf.id in leaves:
                raise ValueError(f"{path}: duplicate leaf id {leaf.id!r}")
            if not leaf.legal_text.strip():
                raise ValueError(f"{path}: leaf {leaf.id!r} has empty legal_text")
            leaves[leaf.id] = leaf
    return leaves


def anchor_texts(leaf: Leaf) -> list[str]:
    return [leaf.legal_text, *leaf.exemplars]
