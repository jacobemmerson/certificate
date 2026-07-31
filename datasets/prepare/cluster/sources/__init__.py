'''
The source registry.

One module per systemic risk, each holding its Source entries and any transforms
they need side by side, so adding a benchmark is a single-file edit. See
datasets/README.md for the four steps.
'''

from . import cyber, democracy, loss_of_control, persuasion
from ..schema import Source

SOURCES: list[Source] = [
    *democracy.SOURCES,
    *persuasion.SOURCES,
    *cyber.SOURCES,
    *loss_of_control.SOURCES,
]

RISKS = ["democracy", "persuasion", "cyber", "loss_of_control"]


def for_risk(risk: str) -> list[Source]:
    return [source for source in SOURCES if source.risk == risk]
