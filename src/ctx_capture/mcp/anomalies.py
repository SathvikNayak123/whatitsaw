"""v1 anomaly detection. See docs/DESIGN.md MCP surface spec — find_context_anomalies.

Single source of truth: both the find_context_anomalies tool and list_traces' has_anomalies
field call detect_anomalies() so the two never disagree about what counts as an anomaly.

Four anomaly types, all derivable from data already in the schema (no extra config needed):
  - truncation: a TruncationEvent recorded on the step
  - tool_result_mismatch: a tool call whose result_as_inserted differs from result_as_returned
    (checked directly against the tool_calls, independent of whether a TruncationEvent was
    also recorded, since a future capture path could diff without recording one)
  - budget_overflow: a model call's completion_tokens hit params["max_tokens"] — the model
    likely got cut off by the token budget
  - dropped_message: a step's message count dropped versus the previous step, with no
    truncation event to explain it — a signal of context lost outside the capture SDK's view
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel

from ctx_capture.schema import Trace

AnomalyType = Literal["truncation", "budget_overflow", "dropped_message", "tool_result_mismatch"]
ALL_ANOMALY_TYPES: tuple[AnomalyType, ...] = (
    "truncation",
    "budget_overflow",
    "dropped_message",
    "tool_result_mismatch",
)


class Anomaly(BaseModel):
    step_index: int
    type: AnomalyType
    severity: Literal["low", "medium", "high"]
    detail: str
    byte_delta: int = 0


def _severity_for_shrink(original_bytes: int, new_bytes: int) -> Literal["low", "medium", "high"]:
    if original_bytes <= 0:
        return "low"
    shrink_ratio = (original_bytes - new_bytes) / original_bytes
    if shrink_ratio >= 0.5:
        return "high"
    if shrink_ratio >= 0.15:
        return "medium"
    return "low"


def detect_anomalies(trace: Trace, types: tuple[AnomalyType, ...] = ALL_ANOMALY_TYPES) -> list[Anomaly]:
    anomalies: list[Anomaly] = []
    wanted = set(types)
    prev_message_count: int | None = None

    for step in trace.steps:
        if "truncation" in wanted:
            for event in step.truncation_events:
                anomalies.append(
                    Anomaly(
                        step_index=step.step_index,
                        type="truncation",
                        severity=_severity_for_shrink(event.original_size_bytes, event.truncated_size_bytes),
                        detail=f"{event.location} truncated ({event.strategy}, detected {event.detected_by})",
                        byte_delta=event.truncated_size_bytes - event.original_size_bytes,
                    )
                )

        if "tool_result_mismatch" in wanted:
            for tool_call in step.tool_calls:
                returned_json = json.dumps(tool_call.result_as_returned, sort_keys=True, default=str)
                inserted_json = json.dumps(tool_call.result_as_inserted, sort_keys=True, default=str)
                if returned_json != inserted_json:
                    anomalies.append(
                        Anomaly(
                            step_index=step.step_index,
                            type="tool_result_mismatch",
                            severity=_severity_for_shrink(len(returned_json), len(inserted_json)),
                            detail=f"tool_call {tool_call.tool_call_id} ({tool_call.tool_name}) "
                            "result_as_inserted differs from result_as_returned",
                            byte_delta=len(inserted_json.encode()) - len(returned_json.encode()),
                        )
                    )

        if "budget_overflow" in wanted and step.model_call is not None:
            max_tokens = step.model_call.params.get("max_tokens")
            completion_tokens = step.model_call.token_counts.completion_tokens
            if isinstance(max_tokens, (int, float)) and completion_tokens >= max_tokens > 0:
                anomalies.append(
                    Anomaly(
                        step_index=step.step_index,
                        type="budget_overflow",
                        severity="high",
                        detail=(
                            f"completion hit max_tokens budget "
                            f"({completion_tokens}/{int(max_tokens)} tokens)"
                        ),
                        byte_delta=0,
                    )
                )

        if "dropped_message" in wanted and step.model_call is not None:
            message_count = len(step.model_call.messages)
            if (
                prev_message_count is not None
                and message_count < prev_message_count
                and not step.truncation_events
            ):
                anomalies.append(
                    Anomaly(
                        step_index=step.step_index,
                        type="dropped_message",
                        severity="medium",
                        detail=(
                            f"message count dropped from {prev_message_count} to {message_count} "
                            "with no recorded truncation event"
                        ),
                        byte_delta=0,
                    )
                )
            prev_message_count = message_count

    return anomalies
