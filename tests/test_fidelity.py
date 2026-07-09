"""THE fidelity test. See docs/DESIGN.md "Capture-fidelity acceptance test".

The capture SDK's recorded messages for a step, after canonical JSON serialization, must be
byte-identical to what actually left application code for that call, as observed by an
independent test-only interception point (FakeOpenAIClient.call_log — a recording point separate
from ctx_capture's own capture wrapper). This test is CI-blocking forever: it is the schema's
acceptance test, not an optional check.

Comparison is intentionally order-SENSITIVE (no `sort_keys`): a reconstruction that silently
reorders a message's keys is not byte-identical to what was sent, even though it would compare
equal as a Python dict. `sort_keys=True` would hide exactly that class of bug.
"""

from __future__ import annotations

import json
from typing import Any

from ctx_capture.capture import TraceRecorder
from ctx_capture.storage import SQLiteTraceRepository
from tests.fixtures.fake_client import FakeOpenAIClient, FakeResponse
from tests.fixtures.toy_agent import build_canned_responses, run_toy_agent, tool_impl


def _canonical(obj: Any) -> str:
    return json.dumps(obj, default=str)


def test_capture_fidelity_byte_identical(tmp_path):
    recorder = TraceRecorder(agent_name="toy-agent")
    fake_client = FakeOpenAIClient(build_canned_responses())
    capturing_client = recorder.wrap_client(fake_client, provider="toy-provider")

    run_toy_agent(recorder, capturing_client, tool_impl)

    repo = SQLiteTraceRepository(str(tmp_path / "trace.db"))
    repo.save(recorder.trace)

    ground_truth = fake_client.call_log
    assert len(ground_truth) == 5

    for step_index in range(5):
        stored_step = repo.get_step(recorder.trace.trace_id, step_index)
        reconstructed_messages = stored_step.model_call.messages
        actual_sent_messages = ground_truth[step_index]["messages"]
        assert _canonical(reconstructed_messages) == _canonical(actual_sent_messages), (
            f"step {step_index}: context reconstructed from storage is not byte-identical "
            "to what was actually sent to the model"
        )


def test_truncation_captured_as_returned_vs_as_inserted(tmp_path):
    recorder = TraceRecorder(agent_name="toy-agent")
    fake_client = FakeOpenAIClient(build_canned_responses())
    capturing_client = recorder.wrap_client(fake_client, provider="toy-provider")

    run_toy_agent(recorder, capturing_client, tool_impl)

    repo = SQLiteTraceRepository(str(tmp_path / "trace.db"))
    repo.save(recorder.trace)
    trace_id = recorder.trace.trace_id

    # Step 1's tool call fetched an oversized result that was truncated before reinsertion.
    truncated_step = repo.get_step(trace_id, 1)
    assert len(truncated_step.truncation_events) == 1
    tc = truncated_step.tool_calls[0]
    assert tc.result_as_returned != tc.result_as_inserted
    assert len(tc.result_as_returned) == 5000
    assert tc.result_as_inserted.endswith("...[truncated]")

    # Every other step's tool call was inserted unchanged: no truncation event.
    for step_index in (0, 2, 3):
        step = repo.get_step(trace_id, step_index)
        assert step.truncation_events == []
        assert step.tool_calls[0].result_as_returned == step.tool_calls[0].result_as_inserted

    # Step 4 has no tool call at all (final answer).
    final_step = repo.get_step(trace_id, 4)
    assert final_step.tool_calls == []


def test_capture_fidelity_multiblock_and_image_content(tmp_path):
    """Fixture-matrix entries the DESIGN acceptance test mandates that the 5-step toy agent above
    doesn't exercise: a message whose `content` is multi-block (text + tool_use + tool_result
    mixed in one array) and a message carrying an image content block. Byte-exact reconstruction
    must hold for these provider-native shapes too, not just plain strings.
    """
    recorder = TraceRecorder(agent_name="multiblock-agent")
    fake_client = FakeOpenAIClient(
        [
            FakeResponse(
                {
                    "id": "resp_0",
                    "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
                    "choices": [{"message": {"role": "assistant", "content": "ok", "tool_calls": None}}],
                }
            )
        ]
    )
    capturing_client = recorder.wrap_client(fake_client, provider="toy-provider")

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a helpful test agent."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Look at this image and the prior tool result."},
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgoAAAANS=="},
                },
                {"type": "tool_use", "id": "call_0", "name": "search", "input": {"query": "q0"}},
                {
                    "type": "tool_result",
                    "tool_use_id": "call_0",
                    "content": [{"type": "text", "text": "result for q0"}],
                },
            ],
        },
    ]

    recorder.begin_step()
    capturing_client.chat.completions.create(model="toy-model", messages=messages, temperature=0)
    recorder.end_step()

    repo = SQLiteTraceRepository(str(tmp_path / "trace.db"))
    repo.save(recorder.trace)

    reconstructed = repo.get_step(recorder.trace.trace_id, 0).model_call.messages
    actual_sent = fake_client.call_log[0]["messages"]
    assert _canonical(reconstructed) == _canonical(actual_sent), (
        "multi-block/image message content was not reconstructed byte-identically"
    )
