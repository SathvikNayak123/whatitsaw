"""A minimal stand-in for an OpenAI-compatible client, shaped like `client.chat.completions.create`.

`call_log` is the fidelity test's independent ground truth: it records exactly what left
application code at the client boundary, via a code path separate from ctx_capture's own
capture wrapper (see docs/DESIGN.md "Capture-fidelity acceptance test").
"""

from __future__ import annotations

import copy
from typing import Any


class FakeResponse(dict):
    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        return copy.deepcopy(dict(self))


class _FakeCompletions:
    def __init__(self, canned_responses: list[FakeResponse]) -> None:
        self._canned = canned_responses
        self._call_index = 0
        self.call_log: list[dict[str, Any]] = []

    def create(self, *, model: str, messages: list[dict[str, Any]], **params: Any) -> FakeResponse:
        self.call_log.append(
            {
                "model": model,
                "messages": copy.deepcopy(messages),
                "params": copy.deepcopy(params),
            }
        )
        response = self._canned[self._call_index]
        self._call_index += 1
        return response


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions


class FakeOpenAIClient:
    def __init__(self, canned_responses: list[FakeResponse]) -> None:
        self.chat = _FakeChat(_FakeCompletions(canned_responses))

    @property
    def call_log(self) -> list[dict[str, Any]]:
        return self.chat.completions.call_log
