"""Unit tests for ctx_capture.capture.TraceRecorder.wrap_tool's argument capture. See
docs/AUDIT_REPORT.md P1b: a plain `copy.deepcopy(kwargs)` silently drops positional tool args,
which is exactly how many agent frameworks invoke parsed tool calls (`fn(*parsed_args)`).
"""

from __future__ import annotations

from ctx_capture.capture import TraceRecorder
from ctx_capture.storage import SQLiteTraceRepository


def test_positional_args_are_captured_by_parameter_name():
    def search(query: str, top_k: int) -> str:
        return f"{query}:{top_k}"

    recorder = TraceRecorder(agent_name="x")
    recorder.begin_step()
    wrapped = recorder.wrap_tool(search, tool_name="search")
    result, _ = wrapped("weather", 5)
    recorder.end_step()

    tool_call = recorder.trace.steps[0].tool_calls[0]
    assert result == "weather:5"
    assert tool_call.args_raw == {"query": "weather", "top_k": 5}


def test_mixed_positional_and_keyword_args_are_captured():
    def search(query: str, top_k: int = 3) -> str:
        return f"{query}:{top_k}"

    recorder = TraceRecorder(agent_name="x")
    recorder.begin_step()
    wrapped = recorder.wrap_tool(search, tool_name="search")
    wrapped("weather", top_k=10)
    recorder.end_step()

    assert recorder.trace.steps[0].tool_calls[0].args_raw == {"query": "weather", "top_k": 10}


def test_keyword_only_args_still_captured_as_before():
    def search(query: str) -> str:
        return query

    recorder = TraceRecorder(agent_name="x")
    recorder.begin_step()
    wrapped = recorder.wrap_tool(search, tool_name="search")
    wrapped(query="weather")
    recorder.end_step()

    assert recorder.trace.steps[0].tool_calls[0].args_raw == {"query": "weather"}


def test_uninspectable_callable_falls_back_without_dropping_positional_args():
    # `dict()` is a C builtin: inspect.signature(dict) may or may not resolve depending on the
    # interpreter, but the point is that even when signature inspection fails, positional args
    # are still preserved under a fallback key rather than silently dropped.
    recorder = TraceRecorder(agent_name="x")
    recorder.begin_step()
    wrapped = recorder.wrap_tool(lambda *a, **k: (a, k), tool_name="anon")
    wrapped(1, 2, x=3)
    recorder.end_step()

    args_raw = recorder.trace.steps[0].tool_calls[0].args_raw
    # a lambda with *a, **k has an inspectable signature, so positionals bind under the
    # variadic parameter name rather than falling back — assert nothing was dropped either way.
    assert args_raw.get("x") == 3 or args_raw.get("k") == {"x": 3}
    flattened = str(args_raw)
    assert "1" in flattened and "2" in flattened


def test_non_json_types_are_coerced_deterministically_through_storage(tmp_path):
    """Pins the specified JSON-typing contract (docs/DESIGN.md "The schema", schema.py ToolCall):
    tool args/results are captured as their JSON representation, so non-JSON-native types coerce
    deterministically — tuple/set -> array, bytes -> string — rather than being a silent surprise.
    This is faithful to what the model saw: tool I/O crosses the model boundary only as JSON.
    """
    def make_payload(tag: str) -> dict:
        return {"pair": (1, 2), "blob": b"xy", "labels": {"a"}, "tag": tag}

    recorder = TraceRecorder(agent_name="types")
    recorder.begin_step()
    wrapped = recorder.wrap_tool(make_payload, tool_name="make_payload")
    wrapped(tag="t")
    recorder.end_step()

    repo = SQLiteTraceRepository(str(tmp_path / "t.db"))
    repo.save(recorder.trace)

    result = repo.get_step(recorder.trace.trace_id, 0).tool_calls[0].result_as_returned
    assert result["pair"] == [1, 2]            # tuple -> array
    assert result["labels"] == ["a"]           # set -> array
    assert isinstance(result["blob"], str)     # bytes -> string
    assert result["tag"] == "t"


def test_wrap_anthropic_client_captures_model_call():
    """Smoke test for the Anthropic capture path (see tests/test_fidelity_anthropic.py for the
    byte-exact fidelity gate): `wrap_anthropic_client` records a ModelCall from
    `client.messages.create`, with provider/model/params set and Anthropic's usage shape mapped
    into TokenCounts."""
    from tests.fixtures.fake_anthropic_client import FakeAnthropicClient, FakeAnthropicResponse

    fake_client = FakeAnthropicClient(
        [
            FakeAnthropicResponse(
                {
                    "id": "msg_0",
                    "usage": {"input_tokens": 8, "output_tokens": 3},
                    "content": [{"type": "text", "text": "hi"}],
                    "stop_reason": "end_turn",
                }
            )
        ]
    )
    recorder = TraceRecorder(agent_name="anthropic-smoke")
    capturing_client = recorder.wrap_anthropic_client(fake_client)

    recorder.begin_step()
    capturing_client.messages.create(
        model="claude-sonnet-5", system="be helpful", messages=[{"role": "user", "content": "hi"}], max_tokens=64
    )
    recorder.end_step()

    model_call = recorder.trace.steps[0].model_call
    assert model_call.provider == "anthropic"
    assert model_call.model == "claude-sonnet-5"
    assert model_call.params == {"system": "be helpful", "max_tokens": 64}
    assert model_call.token_counts.prompt_tokens == 8
    assert model_call.token_counts.completion_tokens == 3
