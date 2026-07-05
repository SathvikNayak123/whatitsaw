from __future__ import annotations

import pytest

from ctx_capture.capture import TraceRecorder
from ctx_capture.mcp.server import create_server
from ctx_capture.storage.sqlite_repo import SQLiteTraceRepository
from tests.fixtures.fake_client import FakeOpenAIClient
from tests.fixtures.toy_agent import (
    build_canned_responses,
    build_single_step_response,
    run_single_step_agent,
    run_toy_agent,
    tool_impl,
)


@pytest.fixture
def repo(tmp_path):
    return SQLiteTraceRepository(str(tmp_path / "test.db"))


@pytest.fixture
def mcp_server(repo):
    return create_server(repo, max_response_bytes=50_000)


@pytest.fixture
def small_cap_mcp_server(repo):
    return create_server(repo, max_response_bytes=300)


@pytest.fixture
def seeded_trace_id(repo) -> str:
    recorder = TraceRecorder(agent_name="toy-agent")
    fake_client = FakeOpenAIClient(build_canned_responses())
    capturing_client = recorder.wrap_client(fake_client, provider="toy-provider")
    run_toy_agent(recorder, capturing_client, tool_impl)
    repo.save(recorder.trace)
    return recorder.trace.trace_id


@pytest.fixture
def five_trace_ids(repo) -> list[str]:
    ids = []
    for _ in range(5):
        recorder = TraceRecorder(agent_name="pager")
        client = FakeOpenAIClient([build_single_step_response()])
        capturing_client = recorder.wrap_client(client)
        run_single_step_agent(recorder, capturing_client)
        repo.save(recorder.trace)
        ids.append(recorder.trace.trace_id)
    return ids


def _make_message_trace(repo, n_messages: int, content_size: int) -> str:
    recorder = TraceRecorder(agent_name="pager")
    messages = [{"role": "user", "content": "x" * content_size} for _ in range(n_messages)]
    client = FakeOpenAIClient([build_single_step_response()])
    capturing_client = recorder.wrap_client(client)
    run_single_step_agent(recorder, capturing_client, messages=messages)
    repo.save(recorder.trace)
    return recorder.trace.trace_id


@pytest.fixture
def large_message_trace_id(repo) -> str:
    """~10.5KB across 50 messages — enough to force several pages under a small cap."""
    return _make_message_trace(repo, n_messages=50, content_size=200)


@pytest.fixture
def oversized_default_cap_trace_id(repo) -> str:
    """~84KB — exceeds the *default* 50KB cap, unlike large_message_trace_id."""
    return _make_message_trace(repo, n_messages=400, content_size=200)
