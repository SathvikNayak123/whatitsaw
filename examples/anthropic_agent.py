"""A small tool-using agent instrumented against the Anthropic capture path
(`recorder.wrap_anthropic_client`) — the counterpart to examples/deep_research_agent.py, which
exercises the OpenAI-compatible path. Point `recorder.wrap_anthropic_client(...)` at a real
`anthropic.Anthropic()` client instead and the rest of this file is unchanged; that's the point
of the wrapper.

The model here is a deterministic, offline simulation (no API key required to run this) — clearly
labeled as such rather than pretending to be a live call.

Run: python examples/anthropic_agent.py [db_path]
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ctx_capture.capture import TraceRecorder  # noqa: E402
from ctx_capture.storage.sqlite_repo import SQLiteTraceRepository  # noqa: E402

SYSTEM_PROMPT = "You are a research assistant. Use the search tool, then answer concisely."

# A deliberately oversized tool result, to exercise a real truncation event the way an agent
# framework's own context-management step would trim an oversized tool result before reinserting
# it into history.
_OVERSIZED_RESULT = "relevant finding. " * 400


def web_search(query: str) -> str:
    if query == "deep-dive":
        return _OVERSIZED_RESULT
    return f"top result for {query!r}"


class _SimulatedResponse(dict):
    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        return copy.deepcopy(dict(self))


def _tool_use(step: int, query: str) -> _SimulatedResponse:
    return _SimulatedResponse(
        {
            "id": f"msg_{step}",
            "usage": {"input_tokens": 40 + step * 5, "output_tokens": 12},
            "content": [{"type": "tool_use", "id": f"toolu_{step}", "name": "web_search", "input": {"query": query}}],
            "stop_reason": "tool_use",
        }
    )


def _final_answer(step: int, text: str) -> _SimulatedResponse:
    return _SimulatedResponse(
        {
            "id": f"msg_{step}",
            "usage": {"input_tokens": 40 + step * 5, "output_tokens": 30},
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
        }
    )


class _SimulatedAnthropicClient:
    """Deterministic, offline stand-in for `anthropic.Anthropic()` — see module docstring."""

    def __init__(self, responses: list[_SimulatedResponse]) -> None:
        self._responses = responses
        self._i = 0
        self.messages = self

    def create(self, *, model: str, messages: list[dict[str, Any]], **params: Any) -> _SimulatedResponse:
        response = self._responses[self._i]
        self._i += 1
        return response


def run(db_path: str) -> str:
    responses = [
        _tool_use(0, "context observability tools"),
        _tool_use(1, "deep-dive"),
        _final_answer(2, "Synthesis: byte-exact per-step capture is the differentiator."),
    ]

    recorder = TraceRecorder(agent_name="anthropic-research-agent", agent_version="0.1.0")
    client = _SimulatedAnthropicClient(responses)
    capturing_client = recorder.wrap_anthropic_client(client)

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Research context-window observability tooling and summarize."}
    ]

    for step in range(3):
        recorder.begin_step()
        response = capturing_client.messages.create(
            model="claude-sonnet-5", system=SYSTEM_PROMPT, messages=messages, max_tokens=512
        )
        content = response["content"]
        messages.append({"role": "assistant", "content": content})

        for block in content:
            if block.get("type") != "tool_use":
                continue
            wrapped_tool = recorder.wrap_tool(web_search, tool_name="web_search")
            result, tool_call_id = wrapped_tool(query=block["input"]["query"])

            # Simulate a framework that truncates an oversized tool result before reinserting it.
            inserted = result if len(result) <= 300 else result[:300] + " ...[truncated]"
            recorder.record_insertion(tool_call_id, inserted)
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": block["id"], "content": [{"type": "text", "text": inserted}]}
                    ],
                }
            )

        recorder.end_step()

    repo = SQLiteTraceRepository(db_path)
    repo.save(recorder.trace)
    repo._conn.close()
    return recorder.trace.trace_id


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "anthropic_agent_trace.db"
    trace_id = run(path)
    print(f"captured trace {trace_id} -> {path}")
