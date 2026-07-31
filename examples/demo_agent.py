"""The smallest complete instrumentation example: a 3-step trip-planner agent (search twice,
then answer), captured end to end with the ctx_capture SDK. Start here if you're wiring up your
own agent; examples/deep_research_agent.py is the longer version that also exercises truncation
and budget-overflow anomalies.

The model client here is a deterministic, offline simulation (no API key required) — swap in any
real OpenAI-compatible client via `recorder.wrap_client(...)` and the rest of this file is
unchanged. It's the same trip-planner script as trace-replay's demo agent, so a trace captured
here looks familiar if you graduate to replay later.

Run: python examples/demo_agent.py [db_path]
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ctx_capture.capture import TraceRecorder  # noqa: E402
from ctx_capture.storage.sqlite_repo import SQLiteTraceRepository  # noqa: E402


def search(query: str) -> str:
    """The agent's one tool — a canned offline search."""
    corpus = {
        "best hiking trails colorado": "Top results: Mount Elbert, Bear Lake Trailhead, Quandary Peak.",
        "mount elbert permit requirements": "Mount Elbert requires no permit; parking fills by 7am in summer.",
    }
    return corpus.get(query.lower(), f"No indexed results for {query!r}.")


class _SimulatedResponse(dict):
    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        return copy.deepcopy(dict(self))


def _tool_call_response(step: int, query: str) -> _SimulatedResponse:
    return _SimulatedResponse(
        {
            "id": f"resp_{step}",
            "usage": {"prompt_tokens": 30 + step * 10, "completion_tokens": 8, "total_tokens": 38 + step * 10},
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"call_{step}",
                                "type": "function",
                                "function": {"name": "search", "arguments": json.dumps({"query": query})},
                            }
                        ],
                    }
                }
            ],
        }
    )


def _final_response(step: int, text: str) -> _SimulatedResponse:
    return _SimulatedResponse(
        {
            "id": f"resp_{step}",
            "usage": {"prompt_tokens": 30 + step * 10, "completion_tokens": 20, "total_tokens": 50 + step * 10},
            "choices": [{"message": {"role": "assistant", "content": text, "tool_calls": None}}],
        }
    )


class _SimulatedClient:
    """Deterministic, offline stand-in for an OpenAI-compatible client — see module docstring."""

    def __init__(self, responses: list[_SimulatedResponse]) -> None:
        self._responses = responses
        self._i = 0
        self.chat = self

    @property
    def completions(self):
        return self

    def create(self, *, model: str, messages: list[dict[str, Any]], **params: Any) -> _SimulatedResponse:
        response = self._responses[self._i]
        self._i += 1
        return response


def run(db_path: str) -> str:
    responses = [
        _tool_call_response(0, "best hiking trails colorado"),
        _tool_call_response(1, "mount elbert permit requirements"),
        _final_response(2, "Recommend Mount Elbert: no permit needed, arrive before 7am for parking."),
    ]

    # 1. Create a recorder and wrap the client — call sites below are unchanged.
    recorder = TraceRecorder(agent_name="demo-trip-agent", agent_version="0.1.0")
    capturing_client = recorder.wrap_client(_SimulatedClient(responses), provider="simulated")

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a trip-planning agent."},
        {"role": "user", "content": "Plan me a Colorado hike for this weekend."},
    ]

    # 2. Bracket each model call with begin_step/end_step.
    for step in range(3):
        recorder.begin_step()
        response = capturing_client.chat.completions.create(model="demo-model", messages=messages, temperature=0)
        choice = response["choices"][0]["message"]
        messages.append({"role": "assistant", "content": choice.get("content"), "tool_calls": choice.get("tool_calls")})

        # 3. Wrap tool calls, and record what actually went back into `messages` afterward
        #    (here they're identical; a framework that truncates would pass the truncated value).
        for tool_call in choice.get("tool_calls") or []:
            args = json.loads(tool_call["function"]["arguments"])
            wrapped_tool = recorder.wrap_tool(search, tool_name=tool_call["function"]["name"])
            result, tool_call_id = wrapped_tool(**args)
            recorder.record_insertion(tool_call_id, result)
            messages.append({"role": "tool", "tool_call_id": tool_call["id"], "content": result})

        recorder.end_step()

    # 4. Save the trace — this file is what the MCP server's --db flag points at.
    repo = SQLiteTraceRepository(db_path)
    repo.save(recorder.trace)
    repo._conn.close()
    return recorder.trace.trace_id


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "demo_trip_agent.db"
    trace_id = run(path)
    print(f"captured trace {trace_id} -> {path}")
