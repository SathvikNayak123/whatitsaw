"""A toy 5-step tool-using agent used by the fidelity test. Step 1 fetches an oversized tool
result that gets truncated before reinsertion, exercising result_as_returned vs
result_as_inserted and truncation-event detection.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from tests.fixtures.fake_client import FakeResponse

OVERSIZED_QUERY = "q1"


def tool_impl(query: str) -> str:
    if query == OVERSIZED_QUERY:
        return "x" * 5000
    return f"result for {query}"


def _assistant_tool_call_response(step: int) -> FakeResponse:
    return FakeResponse(
        {
            "id": f"resp_{step}",
            "usage": {"prompt_tokens": 10 + step, "completion_tokens": 5, "total_tokens": 15 + step},
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"call_{step}",
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    "arguments": json.dumps({"query": f"q{step}"}),
                                },
                            }
                        ],
                    }
                }
            ],
        }
    )


def _assistant_final_response(step: int) -> FakeResponse:
    return FakeResponse(
        {
            "id": f"resp_{step}",
            "usage": {"prompt_tokens": 10 + step, "completion_tokens": 8, "total_tokens": 18 + step},
            "choices": [
                {"message": {"role": "assistant", "content": "final answer", "tool_calls": None}}
            ],
        }
    )


def build_single_step_response() -> FakeResponse:
    return FakeResponse(
        {
            "id": "r0",
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            "choices": [{"message": {"role": "assistant", "content": "ok", "tool_calls": None}}],
        }
    )


def run_single_step_agent(recorder: Any, capturing_client: Any, messages: list[dict] | None = None) -> None:
    """Records exactly one model-call step. Used by tests that need many small traces or one
    step with a specific, large `messages` array, without running the full 5-step loop."""
    recorder.begin_step()
    capturing_client.chat.completions.create(
        model="toy-model",
        messages=messages or [{"role": "user", "content": "hi"}],
        temperature=0,
    )
    recorder.end_step()


def build_canned_responses() -> list[FakeResponse]:
    return [
        _assistant_tool_call_response(0),
        _assistant_tool_call_response(1),
        _assistant_tool_call_response(2),
        _assistant_tool_call_response(3),
        _assistant_final_response(4),
    ]


def run_toy_agent(recorder: Any, capturing_client: Any, tool_fn: Callable[..., str]) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": "You are a helpful test agent."}]

    for step in range(5):
        recorder.begin_step()
        response = capturing_client.chat.completions.create(model="toy-model", messages=messages, temperature=0)
        choice_message = response["choices"][0]["message"]
        messages.append(
            {
                "role": "assistant",
                "content": choice_message.get("content"),
                "tool_calls": choice_message.get("tool_calls"),
            }
        )

        tool_calls = choice_message.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                args = json.loads(tc["function"]["arguments"])
                wrapped_tool = recorder.wrap_tool(tool_fn, tool_name=tc["function"]["name"])
                result, tool_call_id = wrapped_tool(**args)

                # Simulate a framework that truncates an oversized tool result before
                # reinserting it into the message history.
                inserted = result[:100] + "...[truncated]" if args["query"] == OVERSIZED_QUERY else result
                recorder.record_insertion(tool_call_id, inserted)

                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": inserted})

        recorder.end_step()

    return messages
