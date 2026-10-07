'''
The source registry.

One module per systemic risk, each holding its Source entries and any transforms
they need side by side, so adding a benchmark is a single-file edit. A module
with a non-empty `SOURCES` list is a risk: its name is the risk name, and RISKS
is the sorted set of those names. A scaffold with no sources is not a risk yet. See datasets/README.md for the steps.
'''

import importlib
import pkgutil

from ..schema import Source

_MODULES = {
    info.name: module
    for info in pkgutil.iter_modules(__path__)
    if getattr(module := importlib.import_module(f"{__name__}.{info.name}"), "SOURCES", None)
}

RISKS: list[str] = sorted(_MODULES)

SOURCES: list[Source] = [source for risk in RISKS for source in _MODULES[risk].SOURCES]


def for_risk(risk: str) -> list[Source]:
    return [source for source in SOURCES if source.risk == risk]
