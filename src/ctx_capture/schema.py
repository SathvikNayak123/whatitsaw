"""Versioned trace/context schema. See docs/DESIGN.md "The schema" for the design rationale.

Schema version: 1.0. Minor versions are additive-only (new optional fields only); a major
version bump requires a migration. `messages` and `response` are intentionally opaque
passthrough (provider-native dicts, not modeled field-by-field) to preserve byte-exact replay.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"


class UnsupportedSchemaVersionError(ValueError):
    """Raised when a stored trace's major schema_version doesn't match this server's supported
    major version. See docs/DESIGN.md "The schema": servers refuse (not silently coerce) traces
    whose major version they don't support."""


def _major_version(schema_version: str) -> str:
    return schema_version.split(".", 1)[0]


def assert_schema_version_supported(schema_version: str) -> None:
    supported = _major_version(SCHEMA_VERSION)
    if _major_version(schema_version) != supported:
        raise UnsupportedSchemaVersionError(
            f"trace has schema_version {schema_version!r} (major version "
            f"{_major_version(schema_version)}), but this server only supports major version "
            f"{supported}.x — see docs/DESIGN.md 'The schema' for the versioning/migration policy"
        )


class TokenCounts(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


class TruncationEvent(BaseModel):
    location: str
    original_size_bytes: int
    truncated_size_bytes: int
    strategy: Literal["middle-out", "tail-cut", "summarized", "unknown"] = "unknown"
    detected_by: Literal["capture-sdk", "inferred"] = "inferred"


class ToolCall(BaseModel):
    tool_call_id: str
    tool_name: str
    # args_raw / result_as_* are captured as their JSON representation, not as live Python
    # objects: they are persisted through a JSON round-trip (pydantic model_dump_json ->
    # model_validate_json), so non-JSON-native types are coerced deterministically — tuples and
    # sets become arrays, bytes become a string. This is faithful capture of "what the model
    # saw", because tool arguments arrive *from* the model as JSON and tool results are inserted
    # *back into* the message array as JSON; the Python-only tuple/set/bytes distinction never
    # crosses the model boundary. It is a specified contract, not a silent lossy transform — the
    # opaque byte-exact guarantee applies to the provider-native `messages`/`response` payloads
    # (ModelCall), which are already JSON. See docs/DESIGN.md "The schema" design notes.
    args_raw: dict[str, Any] = Field(
        description="Tool arguments by parameter name (positional args bound to their names), "
        "captured as their JSON representation."
    )
    result_as_returned: Any = Field(
        description="Tool return value pre-truncation, captured as its JSON representation "
        "(non-JSON-native types coerced deterministically: tuple/set -> array, bytes -> string)."
    )
    result_as_inserted: Any = Field(
        default=None,
        description="The value actually inserted into the next model call's messages, post any "
        "framework truncation; captured as its JSON representation.",
    )
    started_at: datetime
    ended_at: datetime | None = None
    error: str | None = None


class ModelCall(BaseModel):
    provider: str
    model: str
    params: dict[str, Any] = Field(default_factory=dict)
    messages: list[dict[str, Any]]
    request_id: str | None = None
    response: dict[str, Any] = Field(default_factory=dict)
    token_counts: TokenCounts = Field(default_factory=TokenCounts)
    cost_usd: float | None = None
    latency_ms: float | None = None
    raw_request_bytes_ref: str | None = None
    raw_response_bytes_ref: str | None = None


class Step(BaseModel):
    step_index: int
    step_id: str = Field(default_factory=lambda: str(uuid4()))
    started_at: datetime
    ended_at: datetime | None = None
    model_call: ModelCall | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    truncation_events: list[TruncationEvent] = Field(default_factory=list)


class Trace(BaseModel):
    schema_version: str = SCHEMA_VERSION
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    agent_name: str | None = None
    agent_version: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    steps: list[Step] = Field(default_factory=list)
