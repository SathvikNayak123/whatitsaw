"""Storage repository interface.

A Postgres-backed repository slots in by implementing this same ABC against the same
traces/steps table shape (see docs/DESIGN.md, storage decision) — no other layer depends on
SQLite specifically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ctx_capture.schema import Step, Trace


@dataclass
class TraceSummaryRow:
    """A lightweight, mechanically-computed listing row — no anomaly semantics here (that's the
    MCP layer's job, via ctx_capture.mcp.anomalies), just facts SQL can aggregate cheaply."""

    trace_id: str
    agent_name: str | None
    created_at: str
    step_count: int
    total_tokens: int
    total_cost_usd: float


class TraceRepository(ABC):
    @abstractmethod
    def save(self, trace: Trace) -> None: ...

    @abstractmethod
    def get_trace(self, trace_id: str) -> Trace: ...

    @abstractmethod
    def get_step(self, trace_id: str, step_index: int) -> Step: ...

    @abstractmethod
    def list_traces(
        self,
        *,
        agent_name: str | None = None,
        since: str | None = None,
        until: str | None = None,
        tags: dict[str, Any] | None = None,
        limit: int = 20,
        cursor: str | None = None,
    ) -> tuple[list[TraceSummaryRow], str | None]:
        """Cursor-paginated, filtered trace listing. Returns (page, next_cursor)."""
        ...
