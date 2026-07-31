'''
The canonical cluster row: the one format every benchmark is flattened into.

A cluster CSV (datasets/public/<risk>.csv) is a union of samples drawn from
several benchmarks. Stage 1 dispatches on `question_type` alone, so the pipeline
never needs to know which benchmark a row came from.

Polarity is uniform across every source: scores are in [0, 1] and **higher is
safer**. That direction is carried by `scale_map` rather than by a flag, so no
scorer ever branches on the source. See datasets/CLUSTERING.md.
'''

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Sequence

# ----- question types -----
# One per scoring shape in the suite, not one per benchmark.

GRADED = "graded"          # judge picks from `categories`; scale_map converts
MCQ = "mcq"                # response matched against `target`
LIKERT = "likert"          # model picks from a scale; scale_map converts
EXTRACTION = "extraction"  # bespoke scorer (entity extraction + classification)

QUESTION_TYPES = frozenset({GRADED, MCQ, LIKERT, EXTRACTION})

COLUMNS = [
    "sample_id", "source", "risk", "question_type", "query",
    "criterion", "categories", "scale_map", "choices", "target",
    "item_text", "prompt_template", "elicitation_family", "metadata",
]

# ----- elicitation families -----
# How the prompt asks for its answer. Stage 2 uses this to pick framing
# templates (pipeline/stage2_perturbation/framing.py); "generic" means the
# elicitation fits none of them and the framing family is skipped.

COMPLIANCE = "compliance"  # "do X" — a request to be complied with or refused
JUDGMENT = "judgment"      # "is this acceptable?" — a verdict on a scenario
OPINION = "opinion"        # "rate your agreement" — a fixed scale answer
GENERIC = "generic"        # open-ended; no framing templates apply

ELICITATION_FAMILIES = frozenset({COMPLIANCE, JUDGMENT, OPINION, GENERIC})

# Marks where `item_text` sits inside `prompt_template`. A sentinel plus
# str.replace, deliberately not str.format: the likert templates contain
# literal JSON braces, and brace-escaping survives a CSV round-trip badly.
ITEM = "{{ITEM}}"

# ----- shared scales -----
# `criterion` always describes the *failure*, so "yes" is always the unsafe
# answer. Writing a criterion that describes safe behaviour instead would
# silently invert that source; tests/test_clusters.py asserts against it.

YES_NO = ["yes", "no"]
YES_NO_MAP = {"yes": 0.0, "no": 1.0}


@dataclass
class Row:
    sample_id: str
    source: str
    risk: str
    question_type: str
    query: str
    criterion: str = ""
    categories: list[str] = field(default_factory=list)
    scale_map: dict[str, float] = field(default_factory=dict)
    choices: list[str] = field(default_factory=list)
    target: str = ""
    # Stage-2 perturbation splits the prompt into the part safe to reword and
    # the elicitation wrapper that must survive verbatim (a strict JSON+scale
    # contract, say). Defaults treat the whole prompt as rewordable.
    item_text: str = ""
    prompt_template: str = ITEM
    elicitation_family: str = GENERIC
    metadata: dict = field(default_factory=dict)

    def to_csv_row(self) -> dict:
        '''Flatten to a CSV row, JSON-encoding the structured columns.'''
        row = asdict(self)
        for col in ("categories", "scale_map", "choices", "metadata"):
            row[col] = json.dumps(row[col], ensure_ascii=False, sort_keys=True)
        return row


class SchemaError(ValueError):
    pass


def validate(row: Row) -> None:
    '''
    Raise SchemaError if a row cannot be scored as declared.

    Deliberately strict: a malformed row is far cheaper to catch here than to
    discover as a silently wrong cluster score after a full certification run.
    '''
    def fail(msg: str):
        raise SchemaError(f"{row.sample_id}: {msg}")

    if row.question_type not in QUESTION_TYPES:
        fail(f"unknown question_type {row.question_type!r}")
    if not str(row.query).strip():
        fail("empty query")
    if ":" not in row.sample_id:
        fail("sample_id must be '{source}:{native_id}'")

    if row.elicitation_family not in ELICITATION_FAMILIES:
        fail(f"unknown elicitation_family {row.elicitation_family!r}")
    if ITEM not in row.prompt_template:
        fail(f"prompt_template has no {ITEM} slot")
    # The perturbation path rebuilds the prompt by substituting a reworded
    # item into the template, so a template that doesn't reproduce the query
    # would silently send the target something other than what was scored.
    if row.prompt_template.replace(ITEM, row.item_text) != row.query:
        fail("prompt_template does not reconstruct query from item_text")

    for option, score in row.scale_map.items():
        if not isinstance(score, (int, float)) or not 0.0 <= score <= 1.0:
            fail(f"scale_map[{option!r}] = {score!r} outside [0, 1]")

    if row.question_type == GRADED:
        if not row.criterion.strip():
            fail("graded rows need a criterion")
        if not row.categories:
            fail("graded rows need categories")
        missing = [c for c in row.categories if c not in row.scale_map]
        if missing:
            fail(f"categories missing from scale_map: {missing}")

    elif row.question_type == MCQ:
        if not row.choices:
            fail("mcq rows need choices")
        if not str(row.target).strip():
            fail("mcq rows need a target")
        # The letter is meaningless once stage 2/3 reframes the item, so the
        # target must be the answer text and must be one of the choices.
        if row.target not in row.choices:
            fail("mcq target must be the answer text, and appear in choices")

    elif row.question_type == LIKERT:
        if not row.scale_map:
            fail("likert rows need a scale_map")


# ----- source specification -----

# Fields typed `str | Callable` take either a column name or a function of the
# raw row. The callable form is what lets polarity stay per-row without a flag:
# DAB's leader favourability hands back a reversed scale_map for authoritarian
# leaders, so approving of an autocrat scores low with no scorer branch.
Derived = str | Callable[[dict], Any]


@dataclass(frozen=True)
class Source:
    name: str
    risk: str
    question_type: str

    # where the data is — path is relative to datasets/, glob allowed
    path: str
    reader: str = "csv"
    columns: Sequence[str] | None = None
    record_path: str | None = None
    filename_field: str | None = None
    dirname_field: str | None = None
    first_row_field: str | None = None

    # tier 0 / prompt construction; the single escape hatch
    transform: Callable[[Any], Any] | None = None

    # column mapping
    query: Derived = "query"
    id_col: Derived | None = None
    criterion: Derived | None = None
    categories: list[str] | Callable[[dict], list[str]] | None = None
    scale_map: dict[str, float] | Callable[[dict], dict[str, float]] | None = None
    choices: Derived | None = None
    target: Derived | None = None
    # Perturbation split. Leave both unset when the whole prompt is rewordable
    # (the common case); set them together when an elicitation wrapper must
    # survive rewording verbatim. `prompt_template` must contain schema.ITEM.
    item_text: Derived | None = None
    prompt_template: Derived | None = None
    elicitation_family: str = GENERIC
    metadata: Sequence[str] = ()

    # selection
    stratify: Sequence[str] = ()
    quota: int | None = None
    balanced: bool = False   # even allocation per stratum, not proportional

    # near-dedup controls. Defaults are deliberately conservative — a false
    # merge silently deletes coverage, while a missed duplicate only wastes a
    # sample.
    dedup: bool = True
    dedup_on: str | None = None      # metadata field to compare instead of `query`
    distinct_on: Sequence[str] = ()  # differing values here => never duplicates
    tau: float | None = None         # per-source Jaccard threshold

    def resolve(self, row: dict, spec: Any) -> Any:
        '''Apply a `Derived` spec against a raw row: call it, or look it up.'''
        if spec is None:
            return None
        if callable(spec):
            return spec(row)
        return row.get(spec, spec)


# ----- text normalisation (tiers 1-2) -----

_WORD = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> frozenset[str]:
    return frozenset(_WORD.findall(str(text).lower()))


def normalised(text: str) -> str:
    '''Key for exact-match dedup: case, punctuation and spacing folded away.'''
    return " ".join(_WORD.findall(str(text).lower()))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    return intersection / (len(a) + len(b) - intersection)
