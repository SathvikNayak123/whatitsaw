"""Size-limit behavior: responses never silently exceed their byte cap — they come back
labeled `truncated: true` with a way to get the rest. See docs/DESIGN.md "Pagination and size
limits": "an MCP server that blows up its client's context window is an irony we will not ship."
"""

from __future__ import annotations

from ctx_capture.mcp.pagination import canonical_size
from tests.mcp.helpers import mcp_client


async def test_default_cap_triggers_on_oversized_payload(mcp_server, oversized_default_cap_trace_id):
    async with mcp_client(mcp_server) as session:
        result = await session.call_tool(
            "get_step_context", {"trace_id": oversized_default_cap_trace_id, "step_index": 0}
        )
        sc = result.structuredContent
        assert sc["total_message_count"] == 400
        assert sc["truncated"] is True
        assert sc["continuation_cursor"] is not None
        assert sc["resource_uri"] == f"trace://{oversized_default_cap_trace_id}/step/0"
        assert len(sc["model_call"]["messages"]) < 400
        assert canonical_size(sc["model_call"]["messages"]) <= 50_000


async def test_max_bytes_override_is_respected(mcp_server, large_message_trace_id):
    async with mcp_client(mcp_server) as session:
        result = await session.call_tool(
            "get_step_context",
            {"trace_id": large_message_trace_id, "step_index": 0, "max_bytes": 500},
        )
        sc = result.structuredContent
        assert sc["truncated"] is True
        assert canonical_size(sc["model_call"]["messages"]) <= 500
        assert len(sc["model_call"]["messages"]) < 50


async def test_small_payload_is_not_marked_truncated(mcp_server, seeded_trace_id):
    async with mcp_client(mcp_server) as session:
        result = await session.call_tool(
            "get_step_context", {"trace_id": seeded_trace_id, "step_index": 0}
        )
        sc = result.structuredContent
        assert sc["truncated"] is False
        assert sc["continuation_cursor"] is None
        assert sc["resource_uri"] is None


async def test_diff_step_contexts_caps_oversized_diff(small_cap_mcp_server, oversized_default_cap_trace_id, repo):
    # Build a second, disjoint large-message step in the same trace so the diff itself is huge.
    from ctx_capture.capture import TraceRecorder

    from tests.fixtures.fake_client import FakeOpenAIClient
    from tests.fixtures.toy_agent import build_single_step_response, run_single_step_agent

    recorder = TraceRecorder(agent_name="pager")
    messages_a = [{"role": "user", "content": "a" * 200} for _ in range(60)]
    messages_b = [{"role": "user", "content": "b" * 200} for _ in range(60)]
    client = FakeOpenAIClient([build_single_step_response(), build_single_step_response()])
    capturing_client = recorder.wrap_client(client)
    run_single_step_agent(recorder, capturing_client, messages=messages_a)
    run_single_step_agent(recorder, capturing_client, messages=messages_b)
    repo.save(recorder.trace)

    async with mcp_client(small_cap_mcp_server) as session:
        result = await session.call_tool(
            "diff_step_contexts",
            {"trace_id": recorder.trace.trace_id, "step_a": 0, "step_b": 1},
        )
        assert result.isError is False
        sc = result.structuredContent
        assert sc["truncated"] is True
        assert canonical_size(sc) <= 300 * 4  # bounded, not unbounded — generous slack for wrapper fields
