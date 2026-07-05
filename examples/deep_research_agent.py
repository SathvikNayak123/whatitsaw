"""A small multi-step "deep research" agent, instrumented with the ctx_capture SDK end to end —
this is the project dogfooding its own instrumentation, not a test fixture. See docs/proof/ for
a captured run and the MCP queries pulled from it.

The model and search backend here are a deterministic, offline simulation (no API key is
required to run this) — clearly labeled as such rather than pretending to be a live call. Point
`recorder.wrap_client(...)` at any real OpenAI-compatible client instead and the rest of this
file is unchanged; that's the point of the wrapper.

Run: python examples/deep_research_agent.py [db_path]
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

MID_RUN_STEP_INDEX = 3  # the fetch_document/truncation step, literally the middle of 7 steps (0-6)

RESEARCH_QUESTION = (
    "Summarize the current state of context-window observability tooling for LLM agents, "
    "and identify the biggest open gap."
)

_SEARCH_CORPUS = {
    "llm agent observability tools 2026": (
        "Most agent observability tools (Langfuse, LangSmith, Helicone) render trace trees: "
        "spans, durations, nested calls. They're built around 'what happened', displayed for a "
        "human in a dashboard."
    ),
    "context window truncation failure modes": (
        "Common failure modes: tool results silently truncated by the framework before reaching "
        "the model; token-budget overflows that drop messages from the middle of history; "
        "retries that duplicate a tool call the model never saw resolve."
    ),
    "open problems context observability": (
        "The open gap: existing tools show trace structure but rarely capture the exact "
        "byte-for-byte model input at a given step, especially the difference between what a "
        "tool returned and what the framework actually inserted after truncation."
    ),
}

# A deliberately oversized "source document" to exercise a real truncation event, the way a
# framework's own context-management step would summarize an oversized tool result before
# reinserting it into history.
_WHITEPAPER = (
    "Context Window Observability: A Survey. " + ("Prior work on trace visualization. " * 300)
)


def web_search(query: str) -> str:
    return _SEARCH_CORPUS.get(query.lower(), f"No indexed results for {query!r}.")


def fetch_document(url: str) -> str:
    return _WHITEPAPER


class _SimulatedResponse(dict):
    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        return copy.deepcopy(dict(self))


def _assistant_text(step: int, content: str) -> _SimulatedResponse:
    return _SimulatedResponse(
        {
            "id": f"resp_{step}",
            "usage": {"prompt_tokens": 80 + step * 10, "completion_tokens": 40, "total_tokens": 120 + step * 10},
            "choices": [{"message": {"role": "assistant", "content": content, "tool_calls": None}}],
        }
    )


def _assistant_tool_call(step: int, tool_name: str, args: dict[str, Any], completion_tokens: int = 20) -> _SimulatedResponse:
    return _SimulatedResponse(
        {
            "id": f"resp_{step}",
            "usage": {
                "prompt_tokens": 80 + step * 10,
                "completion_tokens": completion_tokens,
                "total_tokens": 80 + step * 10 + completion_tokens,
            },
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"call_{step}",
                                "type": "function",
                                "function": {"name": tool_name, "arguments": json.dumps(args)},
                            }
                        ],
                    }
                }
            ],
        }
    )


class _SimulatedResearchClient:
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
        _assistant_text(
            0,
            "Plan: (1) survey existing tools, (2) research common failure modes, "
            "(3) read a detailed source on the topic, (4) identify the open gap, (5) synthesize.",
        ),
        _assistant_tool_call(1, "web_search", {"query": "LLM agent observability tools 2026"}),
        _assistant_tool_call(2, "web_search", {"query": "context window truncation failure modes"}),
        _assistant_tool_call(3, "fetch_document", {"url": "https://example.org/context-observability-survey"}),
        _assistant_tool_call(4, "web_search", {"query": "open problems context observability"}),
        # Step 5 deliberately sets a low max_tokens the completion hits, to exercise the
        # budget_overflow anomaly on a real captured trace, not just a unit-test fixture.
        _assistant_text(5, "Reflecting on findings so far before final synthesis..." + "x" * 200),
        _assistant_text(
            6,
            "Synthesis: agent observability tools today show trace structure well but rarely "
            "capture byte-exact model input per step, or the gap between a tool's raw result and "
            "what actually got inserted after truncation. That reconstruction gap is the biggest "
            "open problem — which is exactly what ctx-capture is for.",
        ),
    ]

    recorder = TraceRecorder(agent_name="deep-research-agent", agent_version="0.1.0")
    client = _SimulatedResearchClient(responses)
    capturing_client = recorder.wrap_client(client, provider="simulated-research-model")

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a research agent. Investigate thoroughly, then synthesize."},
        {"role": "user", "content": RESEARCH_QUESTION},
    ]

    tool_impls = {"web_search": web_search, "fetch_document": fetch_document}

    for step in range(7):
        recorder.begin_step()
        params: dict[str, Any] = {"temperature": 0.2}
        if step == 5:
            params["max_tokens"] = 30  # low enough that this step's completion_tokens hits it

        response = capturing_client.chat.completions.create(model="deep-research-v1", messages=messages, **params)
        choice = response["choices"][0]["message"]
        messages.append(
            {"role": "assistant", "content": choice.get("content"), "tool_calls": choice.get("tool_calls")}
        )

        for tool_call in choice.get("tool_calls") or []:
            name = tool_call["function"]["name"]
            args = json.loads(tool_call["function"]["arguments"])
            wrapped_tool = recorder.wrap_tool(tool_impls[name], tool_name=name)
            result, tool_call_id = wrapped_tool(**args)

            # A real framework would summarize/truncate an oversized tool result before
            # reinserting it into history — simulate that context-management step here.
            inserted = result if len(result) <= 500 else result[:500] + " ...[truncated by agent]"
            recorder.record_insertion(tool_call_id, inserted)
            messages.append({"role": "tool", "tool_call_id": tool_call["id"], "content": inserted})

        recorder.end_step()

    repo = SQLiteTraceRepository(db_path)
    repo.save(recorder.trace)
    repo._conn.close()
    return recorder.trace.trace_id


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "deep_research_trace.db"
    trace_id = run(path)
    print(f"captured trace {trace_id} -> {path}")
