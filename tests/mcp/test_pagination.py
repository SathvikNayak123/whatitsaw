"""Pagination behavior: list_traces cursor pagination covers every trace exactly once;
get_step_context's continuation cursor reconstructs the full messages array with nothing
silently dropped. See docs/DESIGN.md "Pagination and size limits".
"""

from __future__ import annotations

from tests.mcp.helpers import mcp_client


async def test_list_traces_cursor_pagination_covers_everything_once(mcp_server, five_trace_ids):
    seen: list[str] = []
    cursor = None
    async with mcp_client(mcp_server) as session:
        for _ in range(10):
            result = await session.call_tool("list_traces", {"limit": 2, "cursor": cursor})
            sc = result.structuredContent
            seen.extend(t["trace_id"] for t in sc["traces"])
            cursor = sc["next_cursor"]
            if cursor is None:
                break
        else:
            raise AssertionError("pagination did not terminate within 10 pages")

    assert sorted(seen) == sorted(five_trace_ids)
    assert len(seen) == len(set(seen)), "pagination returned a duplicate trace"


async def test_list_traces_limit_is_honored_per_page(mcp_server, five_trace_ids):
    async with mcp_client(mcp_server) as session:
        result = await session.call_tool("list_traces", {"limit": 2})
        traces = result.structuredContent["traces"]
        assert len(traces) == 2
        assert result.structuredContent["next_cursor"] is not None


async def test_get_step_context_continuation_reconstructs_full_messages(
    small_cap_mcp_server, large_message_trace_id
):
    expected = [{"role": "user", "content": "x" * 200} for _ in range(50)]

    collected: list[dict] = []
    cursor = None
    pages = 0
    async with mcp_client(small_cap_mcp_server) as session:
        for _ in range(200):
            result = await session.call_tool(
                "get_step_context",
                {"trace_id": large_message_trace_id, "step_index": 0, "cursor": cursor},
            )
            sc = result.structuredContent
            collected.extend(sc["model_call"]["messages"])
            pages += 1
            cursor = sc["continuation_cursor"]
            if not sc["truncated"]:
                break
        else:
            raise AssertionError("pagination did not terminate within 200 pages")

    assert pages > 1, "test cap was too generous to actually exercise pagination"
    assert collected == expected, "reconstructed messages differ from what was captured — silent data loss"
