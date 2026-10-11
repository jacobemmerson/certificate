'''
The source registry.

Every benchmark is a Source in one of the modules here; the module split is
only for reading convenience and means nothing to the pipeline. A benchmark is
scored against the CoP leaves its annotators assigned it (annotations.json),
or its hand `leaves` when it has no annotation. The datasets are built per
leaf: LEAVES is every leaf at least one benchmark serves, and that leaf id is
the dataset name (datasets/public/<leaf>.csv). Grouping leaves into systemic
risks is a human decision taken later, outside this registry.
'''

import importlib
import pkgutil

from ..annotations import DEFAULT_RELEVANCE, load_annotations, manifest_name
from ..schema import SchemaError, Source

SOURCES: list[Source] = [
    source
    for info in sorted(pkgutil.iter_modules(__path__), key=lambda info: info.name)
    for source in getattr(importlib.import_module(f"{__name__}.{info.name}"), "SOURCES", [])
]


def source_leaves(source: Source, annotations: dict[str, dict[str, float]]) -> dict[str, float]:
    '''{leaf id: relevance} the source is scored against: its hand `leaves` when set, else its annotation.'''
    if source.leaves:
        return {leaf: DEFAULT_RELEVANCE for leaf in source.leaves}
    annotated = annotations.get(manifest_name(source.path))
    if annotated:
        return dict(annotated)
    raise SchemaError(f"{source.name}: no annotation in annotations.json and no `leaves`; "
                      "it cannot be scored")


_ANNOTATIONS = load_annotations()
LEAF_RELEVANCE: dict[str, dict[str, float]] = {
    source.name: source_leaves(source, _ANNOTATIONS) for source in SOURCES
}
LEAVES: list[str] = sorted({leaf for leaves in LEAF_RELEVANCE.values() for leaf in leaves})


def annotated(source: Source) -> bool:
    return not source.leaves and manifest_name(source.path) in _ANNOTATIONS


def for_leaf(leaf: str) -> list[tuple[Source, float]]:
    '''(source, relevance) for every source scored against `leaf`, in registration order.'''
    return [(source, LEAF_RELEVANCE[source.name][leaf])
            for source in SOURCES if leaf in LEAF_RELEVANCE[source.name]]
