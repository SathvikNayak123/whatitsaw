"""MCP tool I/O DTOs. Separate from ctx_capture.schema (the versioned trace schema) because
these are MCP-surface response shapes, not the persisted, versioned trace format — see
docs/DESIGN.md "MCP surface spec". FastMCP derives each tool's outputSchema and
structuredContent automatically from these return-type annotations.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from ctx_capture.mcp.anomalies import Anomaly
from ctx_capture.schema import TokenCounts


class TraceSummary(BaseModel):
    trace_id: str
    agent_name: str | None
    created_at: str
    step_count: int
    total_tokens: int
    total_cost_usd: float
    has_anomalies: bool


class ListTracesResult(BaseModel):
    traces: list[TraceSummary]
    next_cursor: str | None = None


class ModelCallView(BaseModel):
    provider: str
    model: str
    params: dict[str, Any]
    messages: list[dict[str, Any]]
    request_id: str | None
    response: dict[str, Any]
    token_counts: TokenCounts
    cost_usd: float | None
    latency_ms: float | None


class GetStepContextResult(BaseModel):
    trace_id: str
    step_index: int
    model_call: ModelCallView
    total_message_count: int
    truncated: bool
    resource_uri: str | None = None
    continuation_cursor: str | None = None


class MessageEntry(BaseModel):
    index: int
    message: dict[str, Any]


class ChangedMessageEntry(BaseModel):
    index: int
    before: dict[str, Any]
    after: dict[str, Any]


class TokenDelta(BaseModel):
    prompt: int
    completion: int


class DiffStepContextsResult(BaseModel):
    trace_id: str
    step_a: int
    step_b: int
    added_messages: list[MessageEntry]
    removed_messages: list[MessageEntry]
    changed_messages: list[ChangedMessageEntry]
    token_delta: TokenDelta
    truncated: bool = False


class FindAnomaliesResult(BaseModel):
    trace_id: str
    anomalies: list[Anomaly]
    count: int


class TokenAccountingBreakdownItem(BaseModel):
    key: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    message_count: int | None = None


class GetTokenAccountingResult(BaseModel):
    trace_id: str
    total_prompt_tokens: int
    total_completion_tokens: int
    total_cost_usd: float
    breakdown: list[TokenAccountingBreakdownItem]
