"""FastMCP server exposing the 5 designed tools + trace:// resources. See docs/DESIGN.md
"MCP surface spec". Every tool response is capped at `max_response_bytes` (default 50 KB);
anything that would exceed it comes back as a bounded, labeled page rather than being silently
dropped — see ctx_capture.mcp.pagination.
"""

from __future__ import annotations

import importlib.metadata
from typing import Any

from mcp.server.fastmcp import FastMCP

from ctx_capture.mcp.anomalies import ALL_ANOMALY_TYPES, AnomalyType, detect_anomalies
from ctx_capture.mcp.models import (
    ChangedMessageEntry,
    DiffStepContextsResult,
    FindAnomaliesResult,
    GetStepContextResult,
    GetTokenAccountingResult,
    ListTracesResult,
    MessageEntry,
    ModelCallView,
    TokenAccountingBreakdownItem,
    TokenDelta,
    TraceSummary,
)
from ctx_capture.mcp.pagination import canonical_size, paginate_items
from ctx_capture.schema import TokenCounts
from ctx_capture.storage.repository import TraceRepository, TraceSummaryRow

DEFAULT_MAX_RESPONSE_BYTES = 50_000
_HAS_ANOMALIES_MAX_SCAN = 1000


def _to_summary(repo: TraceRepository, row: TraceSummaryRow) -> TraceSummary:
    trace = repo.get_trace(row.trace_id)
    anomalies = detect_anomalies(trace)
    return TraceSummary(
        trace_id=row.trace_id,
        agent_name=row.agent_name,
        created_at=row.created_at,
        step_count=row.step_count,
        total_tokens=row.total_tokens,
        total_cost_usd=row.total_cost_usd,
        has_anomalies=len(anomalies) > 0,
    )


def _shrink_diff_to_fit(result: DiffStepContextsResult, cap: int) -> DiffStepContextsResult:
    result = result.model_copy(deep=True)
    while canonical_size(result.model_dump(mode="json")) > cap and (
        result.added_messages or result.removed_messages or result.changed_messages
    ):
        result.truncated = True
        if result.changed_messages and len(result.changed_messages) >= max(
            len(result.added_messages), len(result.removed_messages)
        ):
            result.changed_messages.pop()
        elif result.added_messages and len(result.added_messages) >= len(result.removed_messages):
            result.added_messages.pop()
        elif result.removed_messages:
            result.removed_messages.pop()
        else:
            result.changed_messages.pop()
    return result


def create_server(
    repo: TraceRepository,
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
    name: str = "ctx-capture",
) -> FastMCP:
    mcp = FastMCP(name=name, instructions="Query exactly what an LLM agent saw at every step.")
    # FastMCP has no public `version` kwarg in this SDK version; without this, connecting clients
    # see the `mcp` library's own version instead of ctx-capture's (Server.version, if unset,
    # falls back to pkg_version("mcp")).
    mcp._mcp_server.version = importlib.metadata.version("ctx-capture")

    @mcp.tool()
    def list_traces(
        agent_name: str | None = None,
        since: str | None = None,
        until: str | None = None,
        has_anomalies: bool | None = None,
        tags: dict[str, Any] | None = None,
        limit: int = 20,
        cursor: str | None = None,
    ) -> ListTracesResult:
        """List captured traces, optionally filtered by agent_name/since/until/has_anomalies/tags.
        Cursor-paginated: pass back `next_cursor` to get the next page."""
        limit = max(1, min(limit, 200))

        if has_anomalies is None:
            rows, next_cursor = repo.list_traces(
                agent_name=agent_name, since=since, until=until, tags=tags, limit=limit, cursor=cursor
            )
            return ListTracesResult(traces=[_to_summary(repo, r) for r in rows], next_cursor=next_cursor)

        # has_anomalies requires evaluating each candidate trace, so scan one row at a time (each
        # step's cursor is exact) up to a bound rather than risk skipping traces at batch edges.
        summaries: list[TraceSummary] = []
        scan_cursor = cursor
        for _ in range(_HAS_ANOMALIES_MAX_SCAN):
            if len(summaries) >= limit:
                break
            rows, scan_cursor = repo.list_traces(
                agent_name=agent_name, since=since, until=until, tags=tags, limit=1, cursor=scan_cursor
            )
            if not rows:
                scan_cursor = None
                break
            summary = _to_summary(repo, rows[0])
            if summary.has_anomalies == has_anomalies:
                summaries.append(summary)
        next_cursor = scan_cursor if len(summaries) >= limit else None
        return ListTracesResult(traces=summaries, next_cursor=next_cursor)

    @mcp.tool()
    def get_step_context(
        trace_id: str,
        step_index: int,
        max_bytes: int | None = None,
        cursor: str | None = None,
    ) -> GetStepContextResult:
        """Return the exact reconstructed model input for one step — the byte-exact `messages`
        array as sent, verbatim. Paginated by whole messages when the step is larger than
        max_bytes; pass back `continuation_cursor` to fetch the rest, or read `resource_uri` for
        the full, unpaginated step."""
        cap = max_bytes or max_response_bytes
        step = repo.get_step(trace_id, step_index)
        if step.model_call is None:
            raise ValueError(f"step {step_index} of trace {trace_id} has no model call")

        start = int(cursor) if cursor else 0
        messages = step.model_call.messages
        page, resume_index = paginate_items(messages, start, cap)

        model_call_view = ModelCallView(
            provider=step.model_call.provider,
            model=step.model_call.model,
            params=step.model_call.params,
            messages=page,
            request_id=step.model_call.request_id,
            response=step.model_call.response,
            token_counts=step.model_call.token_counts,
            cost_usd=step.model_call.cost_usd,
            latency_ms=step.model_call.latency_ms,
        )
        truncated = resume_index is not None
        return GetStepContextResult(
            trace_id=trace_id,
            step_index=step_index,
            model_call=model_call_view,
            total_message_count=len(messages),
            truncated=truncated,
            resource_uri=f"trace://{trace_id}/step/{step_index}" if truncated else None,
            continuation_cursor=str(resume_index) if resume_index is not None else None,
        )

    @mcp.tool()
    def diff_step_contexts(
        trace_id: str,
        step_a: int,
        step_b: int,
        diff_type: str = "messages",
        max_bytes: int | None = None,
    ) -> DiffStepContextsResult:
        """Diff the reconstructed model input between two steps of the same trace."""
        cap = max_bytes or max_response_bytes
        a = repo.get_step(trace_id, step_a)
        b = repo.get_step(trace_id, step_b)

        messages_a = a.model_call.messages if a.model_call else []
        messages_b = b.model_call.messages if b.model_call else []

        added: list[MessageEntry] = []
        removed: list[MessageEntry] = []
        changed: list[ChangedMessageEntry] = []

        common_len = min(len(messages_a), len(messages_b))
        for i in range(common_len):
            if messages_a[i] != messages_b[i]:
                changed.append(ChangedMessageEntry(index=i, before=messages_a[i], after=messages_b[i]))
        for i in range(common_len, len(messages_b)):
            added.append(MessageEntry(index=i, message=messages_b[i]))
        for i in range(common_len, len(messages_a)):
            removed.append(MessageEntry(index=i, message=messages_a[i]))

        tokens_a = a.model_call.token_counts if a.model_call else TokenCounts()
        tokens_b = b.model_call.token_counts if b.model_call else TokenCounts()
        token_delta = TokenDelta(
            prompt=tokens_b.prompt_tokens - tokens_a.prompt_tokens,
            completion=tokens_b.completion_tokens - tokens_a.completion_tokens,
        )

        result = DiffStepContextsResult(
            trace_id=trace_id,
            step_a=step_a,
            step_b=step_b,
            added_messages=added,
            removed_messages=removed,
            changed_messages=changed,
            token_delta=token_delta,
            truncated=False,
        )
        if canonical_size(result.model_dump(mode="json")) > cap:
            result = _shrink_diff_to_fit(result, cap)
        return result

    @mcp.tool()
    def find_context_anomalies(
        trace_id: str,
        types: list[AnomalyType] | None = None,
    ) -> FindAnomaliesResult:
        """Find truncations, token-budget overflows, dropped messages, and tool-result mismatches
        across a trace."""
        trace = repo.get_trace(trace_id)
        wanted = tuple(types) if types else ALL_ANOMALY_TYPES
        anomalies = detect_anomalies(trace, types=wanted)
        return FindAnomaliesResult(trace_id=trace_id, anomalies=anomalies, count=len(anomalies))

    @mcp.tool()
    def get_token_accounting(
        trace_id: str,
        group_by: str = "step",
    ) -> GetTokenAccountingResult:
        """Token/cost accounting for a trace, grouped by step (default), tool, or role."""
        trace = repo.get_trace(trace_id)
        calls = [s.model_call for s in trace.steps if s.model_call is not None]
        total_prompt = sum(c.token_counts.prompt_tokens for c in calls)
        total_completion = sum(c.token_counts.completion_tokens for c in calls)
        total_cost = sum(c.cost_usd or 0.0 for c in calls)

        breakdown: list[TokenAccountingBreakdownItem] = []
        if group_by == "step":
            for step in trace.steps:
                if step.model_call is None:
                    continue
                tc = step.model_call.token_counts
                breakdown.append(
                    TokenAccountingBreakdownItem(
                        key=str(step.step_index),
                        prompt_tokens=tc.prompt_tokens,
                        completion_tokens=tc.completion_tokens,
                        total_tokens=tc.total_tokens,
                        cost_usd=step.model_call.cost_usd or 0.0,
                    )
                )
        elif group_by == "tool":
            by_tool: dict[str, TokenAccountingBreakdownItem] = {}
            for step in trace.steps:
                if step.model_call is None or not step.tool_calls:
                    continue
                tc = step.model_call.token_counts
                for tool_call in step.tool_calls:
                    item = by_tool.setdefault(
                        tool_call.tool_name, TokenAccountingBreakdownItem(key=tool_call.tool_name)
                    )
                    item.prompt_tokens += tc.prompt_tokens
                    item.completion_tokens += tc.completion_tokens
                    item.total_tokens += tc.total_tokens
                    item.cost_usd += step.model_call.cost_usd or 0.0
            breakdown = list(by_tool.values())
        elif group_by == "role":
            # messages are cumulative per step (the full array as sent), so only the last step's
            # messages need counting — summing across steps would multiply-count history.
            last_call = next((c for c in reversed(calls)), None)
            by_role: dict[str, int] = {}
            if last_call is not None:
                for message in last_call.messages:
                    role = str(message.get("role", "unknown"))
                    by_role[role] = by_role.get(role, 0) + 1
            breakdown = [
                TokenAccountingBreakdownItem(key=role, message_count=count)
                for role, count in sorted(by_role.items())
            ]
        else:
            raise ValueError(f"unknown group_by: {group_by!r} (expected 'step', 'tool', or 'role')")

        return GetTokenAccountingResult(
            trace_id=trace_id,
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
            total_cost_usd=total_cost,
            breakdown=breakdown,
        )

    @mcp.resource("trace://{trace_id}")
    def trace_resource(trace_id: str) -> dict[str, Any]:
        """Trace metadata + step index, for browsing clients."""
        trace = repo.get_trace(trace_id)
        return {
            "trace_id": trace.trace_id,
            "schema_version": trace.schema_version,
            "agent_name": trace.agent_name,
            "agent_version": trace.agent_version,
            "created_at": trace.created_at.isoformat(),
            "metadata": trace.metadata,
            "steps": [
                {
                    "step_index": s.step_index,
                    "step_id": s.step_id,
                    "has_model_call": s.model_call is not None,
                    "tool_call_count": len(s.tool_calls),
                    "truncation_event_count": len(s.truncation_events),
                }
                for s in trace.steps
            ],
        }

    @mcp.resource("trace://{trace_id}/step/{step_index}")
    def trace_step_resource(trace_id: str, step_index: int):
        """Full, byte-exact step detail — the large payload lives here for clients that want to
        read it directly rather than through a size-capped tool call."""
        return repo.get_step(trace_id, step_index)

    return mcp
