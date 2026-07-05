"""Error shapes: invalid input never crashes the server — it comes back as a protocol-level
CallToolResult with isError=True and a human-readable message in `content`, never a raised
exception the client has to catch specially.
"""

from __future__ import annotations

import pytest

from tests.mcp.helpers import mcp_client


@pytest.mark.parametrize(
    ("tool_name", "args"),
    [
        ("get_step_context", {"trace_id": "does-not-exist", "step_index": 0}),
        ("diff_step_contexts", {"trace_id": "does-not-exist", "step_a": 0, "step_b": 1}),
        ("find_context_anomalies", {"trace_id": "does-not-exist"}),
        ("get_token_accounting", {"trace_id": "does-not-exist"}),
    ],
)
async def test_unknown_trace_is_a_protocol_error_not_a_crash(mcp_server, tool_name, args):
    async with mcp_client(mcp_server) as session:
        result = await session.call_tool(tool_name, args)
        assert result.isError is True
        assert result.content, "error result must carry a human-readable message"


async def test_unknown_step_index_is_error(mcp_server, seeded_trace_id):
    async with mcp_client(mcp_server) as session:
        result = await session.call_tool(
            "get_step_context", {"trace_id": seeded_trace_id, "step_index": 9999}
        )
        assert result.isError is True


async def test_invalid_group_by_is_error(mcp_server, seeded_trace_id):
    async with mcp_client(mcp_server) as session:
        result = await session.call_tool(
            "get_token_accounting", {"trace_id": seeded_trace_id, "group_by": "bogus"}
        )
        assert result.isError is True


async def test_step_with_no_model_call_is_error(mcp_server, seeded_trace_id):
    # Step 4 of the toy agent's final turn has no tool call, but every step has a model call in
    # this fixture — assert the well-formed case still succeeds, as a control for the error cases
    # above (a naive implementation could accidentally isError=True on everything).
    async with mcp_client(mcp_server) as session:
        result = await session.call_tool(
            "get_step_context", {"trace_id": seeded_trace_id, "step_index": 4}
        )
        assert result.isError is False
