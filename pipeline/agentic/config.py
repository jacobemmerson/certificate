"""Typed configuration for controlled agentic benchmark protocols."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from math import isfinite

PROTOCOL_VERSION = "1"


class Condition(str, Enum):
    """Supported benchmark-solving conditions."""

    C0 = "c0"
    C1 = "c1"
    C2 = "c2"
    C3 = "c3"
    C4 = "c4"


@dataclass(frozen=True)
class ModelRoles:
    """Inspect model-role names used for non-target reviews."""

    analyst: str = "agentic_analyst"
    critic: str = "agentic_critic"

    def __post_init__(self) -> None:
        if not self.analyst.strip() or not self.critic.strip():
            raise ValueError("analyst and critic role identifiers must be non-empty")
        if self.analyst == self.critic:
            raise ValueError("analyst and critic role identifiers must be distinct")


@dataclass(frozen=True)
class ProtocolBudget:
    """Finite executable limits for one protocol sample.

    All token fields meter output tokens. ``aggregate_output_tokens`` is the
    parent limit and ``final_output_tokens`` is reserved for the separate final
    submission stage. Per-call caps ensure that every possible pre-final call
    fits below the reserve.
    """

    aggregate_output_tokens: int
    final_output_tokens: int
    draft_output_tokens: int
    review_output_tokens: int
    consultation_output_tokens: int
    helper_output_tokens: int
    max_consultation_turns: int
    max_final_turns: int
    max_helper_calls: int
    time_seconds: float
    working_seconds: float

    def __post_init__(self) -> None:
        numeric_limits = {
            "aggregate_output_tokens": self.aggregate_output_tokens,
            "final_output_tokens": self.final_output_tokens,
            "draft_output_tokens": self.draft_output_tokens,
            "review_output_tokens": self.review_output_tokens,
            "consultation_output_tokens": self.consultation_output_tokens,
            "helper_output_tokens": self.helper_output_tokens,
            "max_consultation_turns": self.max_consultation_turns,
            "max_final_turns": self.max_final_turns,
            "time_seconds": self.time_seconds,
            "working_seconds": self.working_seconds,
        }
        invalid = [
            name
            for name, value in numeric_limits.items()
            if value <= 0 or not isfinite(value)
        ]
        if invalid:
            raise ValueError(
                f"protocol limits must be finite and positive: {', '.join(invalid)}"
            )
        integer_limits = {
            "aggregate_output_tokens": self.aggregate_output_tokens,
            "final_output_tokens": self.final_output_tokens,
            "draft_output_tokens": self.draft_output_tokens,
            "review_output_tokens": self.review_output_tokens,
            "consultation_output_tokens": self.consultation_output_tokens,
            "helper_output_tokens": self.helper_output_tokens,
            "max_consultation_turns": self.max_consultation_turns,
            "max_final_turns": self.max_final_turns,
            "max_helper_calls": self.max_helper_calls,
        }
        non_integer = [
            name
            for name, value in integer_limits.items()
            if not isinstance(value, int) or isinstance(value, bool)
        ]
        if non_integer:
            raise ValueError(
                f"protocol token, turn, and helper limits must be integers: "
                f"{', '.join(non_integer)}"
            )
        if not 0 <= self.max_helper_calls <= 2:
            raise ValueError("max_helper_calls must be between zero and two")
        if self.final_output_tokens > self.aggregate_output_tokens:
            raise ValueError("final output reserve exceeds aggregate output limit")

    @property
    def pre_final_output_tokens(self) -> int:
        return self.aggregate_output_tokens - self.final_output_tokens

    def validate_for(self, condition: Condition) -> None:
        """Validate the worst-case call allocation for ``condition``."""
        if condition in (Condition.C0, Condition.C1):
            required = 0
        elif condition in (Condition.C2, Condition.C3):
            required = self.draft_output_tokens + 2 * self.review_output_tokens
        else:
            required = (
                self.max_consultation_turns * self.consultation_output_tokens
                + self.max_helper_calls * self.helper_output_tokens
            )
        if required > self.pre_final_output_tokens:
            raise ValueError(
                f"{condition.value} pre-final caps ({required}) exceed the executable "
                f"pre-final budget ({self.pre_final_output_tokens})"
            )


_SHARED_BUDGET = ProtocolBudget(
    aggregate_output_tokens=4096,
    final_output_tokens=1024,
    draft_output_tokens=768,
    review_output_tokens=512,
    consultation_output_tokens=512,
    helper_output_tokens=512,
    max_consultation_turns=3,
    max_final_turns=2,
    max_helper_calls=2,
    time_seconds=120,
    working_seconds=120,
)

# Metadata used by benchmark scorers/metrics. Audit sentinel extraction ignores
# values already present in public messages (for example an HR scenario or a
# role-model nationality) while still tracking the benchmark-aware field policy.
_BENCHMARK_PRIVATE_METADATA_KEYS: dict[str, tuple[str, ...]] = {
    "history_no_push": ("True Version", "False Version", "Scenario"),
    "history_explicit_push": ("True Version", "False Version", "Scenario"),
    "fscale": ("fscale_variables", "category"),
    "leader_favorability": (
        "classification",
        "country",
        "status",
        "era",
        "yes_indicates",
        "category",
    ),
    "role_model_bias": ("nationality",),
    "social_harm_bench": (
        "category",
        "sub_topic",
        "type",
        "year",
        "ethnicity",
        "religion",
        "location",
    ),
    "udhr": (
        "limited_article",
        "limited_article_name",
        "actor",
        "affected_group",
        "severity",
        "state_of_emergency",
        "scenario_text",
    ),
    "udhr_individual": (
        "limited_article",
        "limited_article_name",
        "actor",
        "affected_group",
        "severity",
        "state_of_emergency",
        "scenario_text",
    ),
    "udhr_government": (
        "limited_article",
        "limited_article_name",
        "actor",
        "affected_group",
        "severity",
        "state_of_emergency",
        "scenario_text",
    ),
    "echr": (
        "limited_article",
        "limited_article_name",
        "actor",
        "affected_group",
        "severity",
        "state_of_emergency",
        "scenario_text",
    ),
    "echr_individual": (
        "limited_article",
        "limited_article_name",
        "actor",
        "affected_group",
        "severity",
        "state_of_emergency",
        "scenario_text",
    ),
    "echr_government": (
        "limited_article",
        "limited_article_name",
        "actor",
        "affected_group",
        "severity",
        "state_of_emergency",
        "scenario_text",
    ),
}


_C1_BUDGET = ProtocolBudget(
    aggregate_output_tokens=1536,
    final_output_tokens=1536,
    draft_output_tokens=1,
    review_output_tokens=1,
    consultation_output_tokens=1,
    helper_output_tokens=1,
    max_consultation_turns=1,
    max_final_turns=2,
    max_helper_calls=0,
    time_seconds=60,
    working_seconds=60,
)


@dataclass(frozen=True)
class AgenticConfig:
    """Complete immutable configuration for one condition wrapper."""

    condition: Condition
    budget: ProtocolBudget | None = None
    roles: ModelRoles = field(default_factory=ModelRoles)
    private_metadata_keys: tuple[str, ...] = ("private", "secret", "hidden")
    protocol_version: str = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.condition, Condition):
            object.__setattr__(self, "condition", Condition(self.condition))
        if self.condition is Condition.C0:
            return
        if self.budget is None:
            raise ValueError(f"{self.condition.value} requires finite protocol limits")
        self.budget.validate_for(self.condition)

    @classmethod
    def default(
        cls,
        condition: Condition | str,
        *,
        roles: ModelRoles | None = None,
        private_metadata_keys: tuple[str, ...] = ("private", "secret", "hidden"),
    ) -> "AgenticConfig":
        condition = Condition(condition)
        if condition is Condition.C0:
            budget = None
        elif condition is Condition.C1:
            budget = _C1_BUDGET
        else:
            budget = _SHARED_BUDGET
        return cls(
            condition=condition,
            budget=budget,
            roles=roles or ModelRoles(),
            private_metadata_keys=private_metadata_keys,
        )

    def with_budget_overrides(self, **overrides: int | float) -> "AgenticConfig":
        """Apply validated budget overrides through one shared resolution path."""
        if not overrides:
            return self
        if self.budget is None:
            raise ValueError("protocol budget options require condition c1-c4")
        return replace(self, budget=replace(self.budget, **overrides))

    def for_benchmark(self, base_task: str) -> "AgenticConfig":
        """Return this config with the benchmark's scorer-private keys added."""
        benchmark_keys = _BENCHMARK_PRIVATE_METADATA_KEYS.get(base_task, ())
        keys = tuple(dict.fromkeys((*self.private_metadata_keys, *benchmark_keys)))
        if keys == self.private_metadata_keys:
            return self
        return replace(self, private_metadata_keys=keys)

    @property
    def canonical_suffix(self) -> str:
        return (
            "" if self.condition is Condition.C0 else f"_agentic_{self.condition.value}"
        )

    def metadata(self, base_task: str) -> dict[str, object]:
        budget = self.budget
        return {
            "condition": self.condition.value,
            "protocol_version": self.protocol_version,
            "base_task": base_task,
            "roles": {
                "analyst": self.roles.analyst,
                "critic": self.roles.critic,
            },
            "budget": None
            if budget is None
            else {
                "aggregate_output_tokens": budget.aggregate_output_tokens,
                "final_output_tokens": budget.final_output_tokens,
                "max_consultation_turns": budget.max_consultation_turns,
                "max_final_turns": budget.max_final_turns,
                "max_helper_calls": budget.max_helper_calls,
                "time_seconds": budget.time_seconds,
                "working_seconds": budget.working_seconds,
            },
        }
