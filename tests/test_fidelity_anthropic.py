"""The Anthropic-path fidelity gate — same acceptance test as tests/test_fidelity.py, run against
`wrap_anthropic_client` / `client.messages.create` instead of the OpenAI-compatible path. See
docs/DESIGN.md "Capture-fidelity acceptance test": each supported capture path gets this gate.

Comparison is intentionally order-SENSITIVE (no `sort_keys`) for the same reason as the OpenAI
fidelity test: a reconstruction that silently reorders a message's keys is not byte-identical to
what was sent, even though it would compare equal as a Python dict.
"""

from __future__ import annotations

import json
from typing import Any

from ctx_capture.capture import TraceRecorder
from ctx_capture.storage import SQLiteTraceRepository
from tests.fixtures.fake_anthropic_client import FakeAnthropicClient, FakeAnthropicResponse


def _canonical(obj: Any) -> str:
    return json.dumps(obj, default=str)


def test_anthropic_capture_fidelity_byte_identical_text_and_tool_use(tmp_path):
    """Plain text turn, then a tool_use/tool_result turn — Anthropic-native content-block shapes,
    with `system` as a top-level param (not folded into the messages array)."""
    recorder = TraceRecorder(agent_name="anthropic-agent")
    fake_client = FakeAnthropicClient(
        [
            FakeAnthropicResponse(
                {
                    "id": "msg_0",
                    "usage": {"input_tokens": 15, "output_tokens": 6},
                    "content": [
                        {"type": "tool_use", "id": "toolu_0", "name": "search", "input": {"query": "q0"}}
                    ],
                    "stop_reason": "tool_use",
                }
            ),
            FakeAnthropicResponse(
                {
                    "id": "msg_1",
                    "usage": {
                        "input_tokens": 30,
                        "output_tokens": 10,
                        "cache_read_input_tokens": 12,
                        "cache_creation_input_tokens": 4,
                    },
                    "content": [{"type": "text", "text": "final answer"}],
                    "stop_reason": "end_turn",
                }
            ),
        ]
    )
    capturing_client = recorder.wrap_anthropic_client(fake_client)

    messages: list[dict[str, Any]] = [{"role": "user", "content": "What's the weather signal say?"}]

    recorder.begin_step()
    capturing_client.messages.create(
        model="claude-sonnet-5",
        system="You are a helpful test agent.",
        messages=messages,
        max_tokens=512,
    )
    recorder.end_step()

    messages.append(
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "toolu_0", "name": "search", "input": {"query": "q0"}}],
        }
    )
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_0", "content": [{"type": "text", "text": "result for q0"}]}
            ],
        }
    )

    recorder.begin_step()
    capturing_client.messages.create(
        model="claude-sonnet-5",
        system="You are a helpful test agent.",
        messages=messages,
        max_tokens=512,
    )
    recorder.end_step()

    repo = SQLiteTraceRepository(str(tmp_path / "trace.db"))
    repo.save(recorder.trace)
    trace_id = recorder.trace.trace_id

    ground_truth = fake_client.call_log
    assert len(ground_truth) == 2

    for step_index in range(2):
        stored_step = repo.get_step(trace_id, step_index)
        reconstructed_messages = stored_step.model_call.messages
        actual_sent_messages = ground_truth[step_index]["messages"]
        assert _canonical(reconstructed_messages) == _canonical(actual_sent_messages), (
            f"step {step_index}: Anthropic messages reconstructed from storage is not "
            "byte-identical to what was actually sent to the model"
        )
        # `system` must land in params, not be folded into the messages array — the thing that
        # keeps the messages-array byte-exactness honest for Anthropic's wire shape.
        assert "system" not in [m.get("role") for m in reconstructed_messages]
        assert stored_step.model_call.params["system"] == "You are a helpful test agent."

    # Token mapping: Anthropic's input_tokens/output_tokens/cache_*_input_tokens -> ctx-capture's
    # provider-neutral TokenCounts fields.
    step_0 = repo.get_step(trace_id, 0)
    assert step_0.model_call.token_counts.prompt_tokens == 15
    assert step_0.model_call.token_counts.completion_tokens == 6
    assert step_0.model_call.token_counts.total_tokens == 21

    step_1 = repo.get_step(trace_id, 1)
    assert step_1.model_call.token_counts.prompt_tokens == 30
    assert step_1.model_call.token_counts.completion_tokens == 10
    assert step_1.model_call.token_counts.cache_read_tokens == 12
    assert step_1.model_call.token_counts.cache_write_tokens == 4


def test_anthropic_capture_fidelity_multiblock_and_image_content(tmp_path):
    """Fixture-matrix entries the DESIGN acceptance test mandates: a message whose `content` is
    multi-block (text + tool_use + tool_result mixed in one array) and a message carrying an
    image content block, in Anthropic's native block shape (base64 source object, not a URL)."""
    recorder = TraceRecorder(agent_name="anthropic-multiblock-agent")
    fake_client = FakeAnthropicClient(
        [
            FakeAnthropicResponse(
                {
                    "id": "msg_0",
                    "usage": {"input_tokens": 20, "output_tokens": 5},
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                }
            )
        ]
    )
    capturing_client = recorder.wrap_anthropic_client(fake_client)

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Look at this image and the prior tool result."},
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgoAAAANS=="},
                },
                {"type": "tool_use", "id": "toolu_0", "name": "search", "input": {"query": "q0"}},
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_0",
                    "content": [{"type": "text", "text": "result for q0"}],
                },
            ],
        }
    ]

    recorder.begin_step()
    capturing_client.messages.create(
        model="claude-sonnet-5", system="You are a helpful test agent.", messages=messages, max_tokens=512
    )
    recorder.end_step()

    repo = SQLiteTraceRepository(str(tmp_path / "trace.db"))
    repo.save(recorder.trace)

    reconstructed = repo.get_step(recorder.trace.trace_id, 0).model_call.messages
    actual_sent = fake_client.call_log[0]["messages"]
    assert _canonical(reconstructed) == _canonical(actual_sent), (
        "Anthropic multi-block/image message content was not reconstructed byte-identically"
    )
