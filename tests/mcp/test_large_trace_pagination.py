"""Exit-criterion test: a single trace whose step payload is bigger than a real context window
(~1MB of message content, comparable to a ~200K-token context) must be servable through
get_step_context's pagination without ever silently dropping data — every message makes it back
to the client across however many capped pages it takes. See docs/DESIGN.md "Pagination and
size limits": "an MCP server that blows up its client's context window is an irony we will not
ship."
"""

from __future__ import annotations

from ctx_capture.capture import TraceRecorder
from ctx_capture.mcp.pagination import canonical_size
from ctx_capture.storage.sqlite_repo import SQLiteTraceRepository
from tests.fixtures.fake_client import FakeOpenAIClient
from tests.fixtures.toy_agent import build_single_step_response, run_single_step_agent
from tests.mcp.helpers import mcp_client

N_MESSAGES = 2000
CONTENT_BYTES_PER_MESSAGE = 500  # ~1MB total: comparable to a ~200K-token context window


def _build_context_window_sized_trace(repo) -> str:
    recorder = TraceRecorder(agent_name="huge-context-agent")
    messages = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg-{i}:" + ("y" * CONTENT_BYTES_PER_MESSAGE)}
        for i in range(N_MESSAGES)
    ]
    client = FakeOpenAIClient([build_single_step_response()])
    capturing_client = recorder.wrap_client(client)
    run_single_step_agent(recorder, capturing_client, messages=messages)
    repo.save(recorder.trace)
    return recorder.trace.trace_id, messages


async def test_context_window_sized_step_served_via_pagination_without_silent_truncation(tmp_path):
    from ctx_capture.mcp.server import create_server

    repo = SQLiteTraceRepository(str(tmp_path / "huge.db"))
    trace_id, expected_messages = _build_context_window_sized_trace(repo)
    total_payload_bytes = canonical_size(expected_messages)
    assert total_payload_bytes > 1_000_000, "fixture isn't actually context-window-sized"

    mcp = create_server(repo, max_response_bytes=50_000)  # the documented default cap

    collected: list[dict] = []
    cursor = None
    pages = 0
    async with mcp_client(mcp) as session:
        for _ in range(200):
            result = await session.call_tool(
                "get_step_context", {"trace_id": trace_id, "step_index": 0, "cursor": cursor}
            )
            assert result.isError is False
            sc = result.structuredContent
            page_messages = sc["model_call"]["messages"]

            # Every single page must itself respect the cap — no page silently blows past it.
            assert canonical_size(page_messages) <= 50_000

            collected.extend(page_messages)
            pages += 1
            cursor = sc["continuation_cursor"]
            if not sc["truncated"]:
                break
        else:
            raise AssertionError("did not finish paginating a ~1MB step within 200 pages")

    assert pages >= 20, f"expected many pages for a ~{total_payload_bytes} byte payload, got {pages}"
    assert len(collected) == N_MESSAGES
    assert collected == expected_messages, "reconstructed messages differ from what was captured"
