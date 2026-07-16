"""Sanitized pre-scoring process audit for agentic protocols."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from hashlib import sha256
import json
from math import isfinite
from typing import Any, Iterable

from inspect_ai.event import ModelEvent, SampleLimitEvent, SpanBeginEvent, ToolEvent
from inspect_ai.log import transcript

from pipeline.agentic.config import AgenticConfig, Condition

AUDIT_METADATA_KEY = "agentic_protocol_audit"


@dataclass
class ProtocolTrace:
    """Ephemeral protocol facts; raw payloads are never persisted."""

    submit_payloads: list[str] = field(default_factory=list)
    helper_launches: list[str] = field(default_factory=list)
    finish_count: int = 0
    violations: list[str] = field(default_factory=list)
    termination_reason: str = "incomplete"
    caught_error: str | None = None
    accepted_payload: str | None = None

    def violate(self, code: str) -> None:
        if code not in self.violations:
            self.violations.append(code)


def event_offset() -> int:
    """Return the current transcript offset for a pre-scoring snapshot."""
    return len(transcript().events)


def _jsonable(value: Any) -> Any:
    """Convert known event values for inspection, failing closed on unknowns."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported audit payload type: {type(value).__name__}")


def _is_token_character(character: str) -> bool:
    return character == "_" or character.isalnum()


def _string_contains(text: str, sentinel: str) -> bool:
    """Match short sentinels as scalars/tokens, not arbitrary substrings."""
    if not sentinel:
        return False
    if text == sentinel or len(sentinel) >= 12:
        return sentinel in text

    start = 0
    while (index := text.find(sentinel, start)) >= 0:
        end = index + len(sentinel)
        left_boundary = index == 0 or not _is_token_character(text[index - 1])
        right_boundary = end == len(text) or not _is_token_character(text[end])
        if left_boundary and right_boundary:
            return True
        start = index + 1
    return False


PrivateScalar = str | int | float | bool


def _sentinel_text(sentinel: PrivateScalar) -> str:
    if isinstance(sentinel, str):
        return sentinel
    return json.dumps(sentinel, ensure_ascii=False, allow_nan=False)


def _contains(value: Any, sentinel: PrivateScalar) -> bool:
    """Inspect a payload for a sentinel; serialization errors propagate."""
    inspected = _jsonable(value)
    sentinel_text = _sentinel_text(sentinel)
    # Keep validation equivalent to persisted JSON inspection: unsupported
    # values and non-finite numbers must invalidate rather than evade the audit.
    json.dumps(inspected, ensure_ascii=False, sort_keys=True, allow_nan=False)

    def visit(item: Any) -> bool:
        if isinstance(item, str):
            return _string_contains(item, sentinel_text)
        if type(item) is type(sentinel) and item == sentinel:
            return True
        if isinstance(item, dict):
            return any(
                _string_contains(str(key), sentinel_text) or visit(child)
                for key, child in item.items()
            )
        if isinstance(item, list):
            return any(visit(child) for child in item)
        return False

    return visit(inspected)


def _secret_sentinels(
    state: Any,
    config: AgenticConfig,
    public_messages: Any | None = None,
) -> tuple[PrivateScalar, ...]:
    """Collect every non-public private scalar for ephemeral comparison."""
    values: list[PrivateScalar] = []
    target = getattr(state, "target", None)
    target_values = getattr(target, "target", None)
    if isinstance(target_values, (list, tuple)):
        for value in target_values:
            if not isinstance(value, (str, int, float, bool)):
                raise TypeError("unsupported target scalar type")
            values.append(value)
    else:
        target_text = getattr(target, "text", None)
        if isinstance(target_text, (str, int, float, bool)):
            values.append(target_text)
        elif target_text is not None:
            raise TypeError("unsupported target scalar type")

    metadata = getattr(state, "metadata", {}) or {}
    private_keys = {key.casefold() for key in config.private_metadata_keys}

    def collect(value: Any) -> None:
        if isinstance(value, (str, int, float, bool)):
            values.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)
        elif hasattr(value, "model_dump"):
            collect(value.model_dump(mode="json"))
        elif value is not None:
            raise TypeError(
                f"unsupported private metadata type: {type(value).__name__}"
            )

    for key, value in metadata.items():
        if str(key).casefold() in private_keys:
            collect(value)

    # Only the initial public projection can declassify a duplicate. This keeps
    # short private labels auditable while avoiding false positives for fields
    # intentionally rendered into the benchmark prompt (for example HR
    # scenario_text).
    if public_messages is None:
        public_messages = getattr(state, "messages", []) or []
    candidates = {
        (type(value), value): value
        for value in values
        if value != "" and value is not None
    }
    return tuple(
        value for value in candidates.values() if not _contains(public_messages, value)
    )


def _phase_for_span(
    span_id: str | None,
    spans: dict[str, SpanBeginEvent],
) -> str | None:
    while span_id is not None:
        event = spans.get(span_id)
        if event is None:
            return None
        if event.name.startswith("agentic:"):
            return event.name
        span_id = event.parent_id
    return None


def _empty_usage() -> dict[str, int | float | None]:
    return {
        "model_events": 0,
        "events_with_usage": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "reasoning_tokens": 0,
        "input_tokens_cache_write": 0,
        "input_tokens_cache_read": 0,
        "total_cost": 0.0,
        "events_with_cost": 0,
    }


def _usage_summary(events: Iterable[ModelEvent]) -> dict[str, int | float | None]:
    result = _empty_usage()
    for event in events:
        result["model_events"] = int(result["model_events"] or 0) + 1
        usage = event.output.usage
        if usage is None:
            continue
        result["events_with_usage"] = int(result["events_with_usage"] or 0) + 1
        for usage_field in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "reasoning_tokens",
            "input_tokens_cache_write",
            "input_tokens_cache_read",
        ):
            value = getattr(usage, usage_field, None)
            if value is not None:
                result[usage_field] = int(result[usage_field] or 0) + int(value)
        if usage.total_cost is not None:
            if not isfinite(usage.total_cost):
                raise ValueError("non-finite model cost in audit")
            result["total_cost"] = float(result["total_cost"] or 0) + float(
                usage.total_cost
            )
            result["events_with_cost"] = int(result["events_with_cost"] or 0) + 1

    event_count = int(result["model_events"] or 0)
    usage_count = int(result["events_with_usage"] or 0)
    cost_count = int(result["events_with_cost"] or 0)
    result["usage_complete"] = event_count > 0 and usage_count == event_count
    result["cost_complete"] = event_count > 0 and cost_count == event_count
    if cost_count == 0:
        result["total_cost"] = None
    return result


def _group_usage(
    model_events: list[ModelEvent],
    key_for_event,
) -> dict[str, dict[str, int | float | None]]:
    grouped: dict[str, list[ModelEvent]] = {}
    for event in model_events:
        grouped.setdefault(str(key_for_event(event)), []).append(event)
    return {key: _usage_summary(events) for key, events in sorted(grouped.items())}


def _tool_producers(
    events: list[Any],
    spans: dict[str, SpanBeginEvent],
) -> dict[str, tuple[int, str, str | None]]:
    producers: dict[str, tuple[int, str, str | None]] = {}
    for index, event in enumerate(events):
        if not isinstance(event, ModelEvent) or not event.output.choices:
            continue
        for call in event.output.message.tool_calls or []:
            if call.id in producers:
                raise ValueError("duplicate tool-call event identifier")
            producers[call.id] = (
                index,
                call.function,
                _phase_for_span(event.span_id, spans),
            )
    return producers


def _validate_tool_boundaries(
    *,
    events: list[Any],
    spans: dict[str, SpanBeginEvent],
    tool_events: list[ToolEvent],
    trace: ProtocolTrace,
) -> dict[str, int]:
    """Validate each executed tool against its producing model turn."""
    producers = _tool_producers(events, spans)
    indexes = {id(event): index for index, event in enumerate(events)}
    tool_indexes: dict[str, int] = {}
    for event in tool_events:
        producer = producers.get(event.id)
        tool_index = indexes[id(event)]
        if producer is None:
            trace.violate("tool_without_model_turn")
            continue
        producer_index, function, _phase = producer
        if function != event.function or producer_index >= tool_index:
            trace.violate("invalid_tool_turn_boundary")
        tool_indexes[event.id] = tool_index
    return tool_indexes


def _validate_fixed_identity(
    *,
    condition: Condition,
    config: AgenticConfig,
    model_events: list[ModelEvent],
    spans: dict[str, SpanBeginEvent],
    trace: ProtocolTrace,
) -> int:
    expected = (
        f"agentic:{condition.value}:draft",
        f"agentic:{condition.value}:review_a",
        f"agentic:{condition.value}:review_b",
    )
    phase_events = {
        phase: [
            event
            for event in model_events
            if _phase_for_span(event.span_id, spans) == phase
        ]
        for phase in expected
    }
    if any(len(events) != 1 for events in phase_events.values()):
        trace.violate("fixed_phase_count")

    reviewer_count = len(phase_events[expected[1]]) + len(phase_events[expected[2]])
    if condition is Condition.C3:
        identities = (
            (expected[1], config.roles.analyst),
            (expected[2], config.roles.critic),
        )
        for phase, role in identities:
            if len(phase_events[phase]) != 1 or phase_events[phase][0].role != role:
                trace.violate("fixed_reviewer_identity")
    else:
        for phase in expected[1:]:
            if any(
                event.role in (config.roles.analyst, config.roles.critic)
                for event in phase_events[phase]
            ):
                trace.violate("self_review_identity")
    return reviewer_count


def _validate_c4_causality(
    *,
    events: list[Any],
    model_events: list[ModelEvent],
    tool_events: list[ToolEvent],
    spans: dict[str, SpanBeginEvent],
    config: AgenticConfig,
    trace: ProtocolTrace,
) -> tuple[int, list[str]]:
    helper_role = {
        "agentic_analyst": config.roles.analyst,
        "agentic_critic": config.roles.critic,
    }
    helpers = [event for event in tool_events if event.function in helper_role]
    helper_names = [event.function for event in helpers]
    budget = config.budget
    assert budget is not None
    if len(helpers) > budget.max_helper_calls:
        trace.violate("helper_cap")
    if helper_names != trace.helper_launches:
        trace.violate("helper_trace_mismatch")

    indexes = {id(event): index for index, event in enumerate(events)}
    producers = _tool_producers(events, spans)

    def is_only_call(
        producer: tuple[int, str, str | None] | None,
        expected_function: str,
    ) -> bool:
        if producer is None:
            return False
        event = events[producer[0]]
        calls = event.output.message.tool_calls or []
        return (
            len(calls) == 1
            and calls[0].function == expected_function
            and calls[0].id in producers
        )

    previous_completion = -1
    for ordinal, helper_event in enumerate(helpers, start=1):
        producer = producers.get(helper_event.id)
        helper_index = indexes[id(helper_event)]
        expected_phase = f"agentic:c4:helper_{ordinal}:{helper_event.function}"
        nested = [
            event
            for event in model_events
            if _phase_for_span(event.span_id, spans) == expected_phase
        ]
        if len(nested) != 1 or nested[0].role != helper_role[helper_event.function]:
            trace.violate("helper_role_identity")
            nested_index = None
        else:
            nested_index = indexes[id(nested[0])]

        valid_chain = (
            producer is not None
            and producer[2] == "agentic:c4:consultation"
            and is_only_call(producer, helper_event.function)
            and _phase_for_span(helper_event.span_id, spans) == expected_phase
            and nested_index is not None
            and previous_completion < producer[0] < helper_index < nested_index
        )
        if not valid_chain:
            trace.violate("helper_causal_order")
        if nested_index is not None:
            previous_completion = max(previous_completion, nested_index)

    finish_events = [
        event for event in tool_events if event.function == "finish_consultation"
    ]
    submit_events = [event for event in tool_events if event.function == "submit"]
    if len(finish_events) != 1:
        trace.violate("finish_count")
    if len(submit_events) != 1:
        trace.violate("submit_count")

    finish_index = -1
    if finish_events:
        finish = finish_events[0]
        finish_producer = producers.get(finish.id)
        finish_index = indexes[id(finish)]
        valid_finish_chain = (
            finish_producer is not None
            and finish_producer[2] == "agentic:c4:consultation"
            and is_only_call(finish_producer, "finish_consultation")
            and _phase_for_span(finish.span_id, spans) == "agentic:c4:consultation"
            and previous_completion < finish_producer[0] < finish_index
        )
        if not valid_finish_chain:
            trace.violate("finish_causal_order")

    if submit_events:
        submit = submit_events[0]
        submit_producer = producers.get(submit.id)
        submit_index = indexes[id(submit)]
        valid_submit_chain = (
            submit_producer is not None
            and submit_producer[2] == "agentic:c4:final_submit"
            and is_only_call(submit_producer, "submit")
            and _phase_for_span(submit.span_id, spans) == "agentic:c4:final_submit"
            and finish_index < submit_producer[0] < submit_index
        )
        if not valid_submit_chain:
            trace.violate("submit_causal_order")

    return len(helpers), helper_names


def sanitized_audit(
    *,
    state: Any,
    config: AgenticConfig,
    trace: ProtocolTrace,
    start: int,
    public_messages: Any | None = None,
) -> dict[str, Any]:
    """Build a content-free audit from solving events before scoring starts."""
    events = list(transcript().events[start:])
    spans = {event.id: event for event in events if isinstance(event, SpanBeginEvent)}
    model_events = [event for event in events if isinstance(event, ModelEvent)]
    tool_events = [event for event in events if isinstance(event, ToolEvent)]
    limit_events = [event for event in events if isinstance(event, SampleLimitEvent)]

    phase_counts: Counter[str] = Counter()
    for event in model_events:
        phase = _phase_for_span(event.span_id, spans)
        phase_counts[phase or "unlabelled"] += 1

    tool_counts = Counter(event.function for event in tool_events)
    event_ids = [
        event.uuid
        for event in events
        if isinstance(event, (ModelEvent, ToolEvent)) and event.uuid is not None
    ]
    role_counts = Counter(event.role or "target" for event in model_events)

    sentinels = _secret_sentinels(state, config, public_messages)
    hidden_leakage = any(
        _contains((event.input, event.tools, event.output), sentinel)
        for event in model_events
        for sentinel in sentinels
    ) or any(
        _contains((event.arguments, event.result), sentinel)
        for event in tool_events
        for sentinel in sentinels
    )
    if hidden_leakage:
        trace.violate("hidden_data_in_solving_payload")

    tool_failures = sum(
        1 for event in tool_events if event.error is not None or event.failed is True
    )
    if tool_failures:
        trace.violate("tool_failure")
    if limit_events and trace.termination_reason != "limit":
        trace.termination_reason = "limit"

    _validate_tool_boundaries(
        events=events,
        spans=spans,
        tool_events=tool_events,
        trace=trace,
    )

    submit_events = [event for event in tool_events if event.function == "submit"]
    submit_count = len(submit_events)
    exact_payload = (
        submit_count == 1
        and len(trace.submit_payloads) == 1
        and trace.accepted_payload is not None
        and trace.submit_payloads[0] == trace.accepted_payload
    )
    if submit_count != 1 or len(trace.submit_payloads) != 1:
        trace.violate("submit_count")
    if submit_count == 1 and not exact_payload:
        trace.violate("submit_payload_mismatch")

    condition = config.condition
    fixed_reviewer_count = 0
    dynamic_helper_count = 0
    helper_roles: list[str] = []
    if condition in (Condition.C2, Condition.C3):
        fixed_reviewer_count = _validate_fixed_identity(
            condition=condition,
            config=config,
            model_events=model_events,
            spans=spans,
            trace=trace,
        )
    if condition is Condition.C4:
        consultation = sum(
            1
            for event in model_events
            if _phase_for_span(event.span_id, spans) == "agentic:c4:consultation"
            and event.role not in (config.roles.analyst, config.roles.critic)
        )
        budget = config.budget
        assert budget is not None
        if not 1 <= consultation <= budget.max_consultation_turns:
            trace.violate("consultation_turn_count")
        dynamic_helper_count, helper_roles = _validate_c4_causality(
            events=events,
            model_events=model_events,
            tool_events=tool_events,
            spans=spans,
            config=config,
            trace=trace,
        )
        if trace.finish_count != 1:
            trace.violate("finish_count")

    final_phase = f"agentic:{condition.value}:final_submit"
    budget = config.budget
    assert budget is not None
    final_turns = phase_counts[final_phase]
    if final_turns == 0:
        trace.violate("missing_final_phase")
    elif final_turns > budget.max_final_turns:
        trace.violate("final_phase_turn_count")

    aggregate_usage = _usage_summary(model_events)
    phase_usage = _group_usage(
        model_events,
        lambda event: _phase_for_span(event.span_id, spans) or "unlabelled",
    )
    role_usage = _group_usage(model_events, lambda event: event.role or "target")
    usage_complete = bool(aggregate_usage["usage_complete"])
    if usage_complete:
        aggregate_output = int(aggregate_usage["output_tokens"] or 0)
        within_ceiling = aggregate_output <= budget.aggregate_output_tokens
        usage_status = "verified" if within_ceiling else "exceeded"
        if not within_ceiling:
            trace.violate("aggregate_output_ceiling_exceeded")
    else:
        within_ceiling = None
        usage_status = "unknown"

    valid = (
        exact_payload
        and not trace.violations
        and trace.caught_error is None
        and not limit_events
        and trace.termination_reason == "submitted"
    )
    payload_hash = (
        sha256(trace.accepted_payload.encode("utf-8")).hexdigest()
        if trace.accepted_payload is not None
        else None
    )
    audit = {
        "version": config.protocol_version,
        "condition": condition.value,
        "valid": valid,
        "termination_reason": trace.termination_reason,
        "violations": sorted(trace.violations),
        "submit_count": submit_count,
        "submit_payload_sha256": payload_hash,
        "submit_payload_exact": exact_payload,
        "dynamic_helper_count": dynamic_helper_count,
        # Backward-compatible summary field; C3 fixed reviewers were never
        # dynamic helpers and are now reported separately below.
        "helper_count": dynamic_helper_count,
        "fixed_reviewer_count": fixed_reviewer_count,
        "helper_roles": helper_roles,
        "finish_count": trace.finish_count,
        "model_event_count": len(model_events),
        "tool_event_count": len(tool_events),
        "tool_failure_count": tool_failures,
        "limit_event_count": len(limit_events),
        "phase_model_counts": dict(sorted(phase_counts.items())),
        "role_model_counts": dict(sorted(role_counts.items())),
        "tool_counts": dict(sorted(tool_counts.items())),
        "solving_event_ids": event_ids,
        "hidden_leakage_detected": hidden_leakage,
        "caught_error_type": trace.caught_error,
        "usage_status": usage_status,
        "usage_verifiable": usage_complete,
        "usage_within_configured_ceiling": within_ceiling,
        "configured_output_token_ceiling": budget.aggregate_output_tokens,
        "aggregate_usage": aggregate_usage,
        "phase_usage": phase_usage,
        "role_usage": role_usage,
    }
    # Persistence uses JSON, so validate the exact sanitized record now. An
    # unexpected value must become audit_error rather than a silent valid run.
    json.dumps(audit, sort_keys=True, allow_nan=False)
    return audit


def audit_failure(config: AgenticConfig, error: BaseException) -> dict[str, Any]:
    """Return a deterministic invalid record if audit construction itself fails."""
    return {
        "version": config.protocol_version,
        "condition": config.condition.value,
        "valid": False,
        "termination_reason": "audit_error",
        "violations": ["audit_error"],
        "usage_status": "unknown",
        "usage_verifiable": False,
        "usage_within_configured_ceiling": None,
        "caught_error_type": type(error).__name__,
    }
