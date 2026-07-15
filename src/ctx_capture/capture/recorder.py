"""Instrumentation SDK: wraps an OpenAI-compatible client (`client.chat.completions.create`), an
Anthropic client (`client.messages.create`), and generic tool functions to build a Trace as an
agent runs. See docs/DESIGN.md "Capture mechanism" for why this is SDK-first (wrapping the actual
Python objects the agent code passes) rather than OTel ingestion or a proxy. Both client wrappers
are duck-typed against the shape of the call, not a hard dependency on either provider's SDK
package.
"""

from __future__ import annotations

import copy
import inspect
import json
import time
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import uuid4

from ctx_capture.schema import ModelCall, Step, TokenCounts, ToolCall, Trace, TruncationEvent


def _bind_args(fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Capture the exact args passed to a tool call, positional or keyword, by parameter name —
    a plain `copy.deepcopy(kwargs)` silently drops positional args, which most tool-calling code
    (agent frameworks invoke tools with `fn(*parsed_args)` as often as by keyword) actually uses.
    Falls back to keyword-only capture, tagging any positionals under `_positional_args` so
    nothing is silently dropped, if `fn`'s signature can't be inspected (e.g. a C builtin)."""
    try:
        bound = inspect.signature(fn).bind(*args, **kwargs)
        return copy.deepcopy(dict(bound.arguments))
    except (TypeError, ValueError):
        raw = dict(kwargs)
        if args:
            raw["_positional_args"] = list(args)
        return copy.deepcopy(raw)


def _to_plain(obj: Any) -> dict[str, Any]:
    """Best-effort, verbatim conversion of a provider SDK response object to a plain dict."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if hasattr(obj, "dict"):
        return obj.dict()
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    raise TypeError(f"don't know how to capture response of type {type(obj)!r}")


def _openai_tokens(usage: dict[str, Any]) -> TokenCounts:
    return TokenCounts(
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        total_tokens=usage.get("total_tokens", 0),
        cache_read_tokens=usage.get("cache_read_tokens", 0),
        cache_write_tokens=usage.get("cache_write_tokens", 0),
    )


def _anthropic_tokens(usage: dict[str, Any]) -> TokenCounts:
    # Anthropic's usage block uses input_tokens/output_tokens (not prompt/completion_tokens) and
    # splits cache accounting into cache_read_input_tokens / cache_creation_input_tokens. This is
    # a read of the provider's own metadata into ctx-capture's TokenCounts field, not a reshape of
    # the opaque `messages`/`response` payloads — same kind of normalization the OpenAI path
    # already does for its own usage shape.
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    return TokenCounts(
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        total_tokens=usage.get("total_tokens", input_tokens + output_tokens),
        cache_read_tokens=usage.get("cache_read_input_tokens", 0),
        cache_write_tokens=usage.get("cache_creation_input_tokens", 0),
    )


def _record_call(
    recorder: "TraceRecorder",
    real_call: Callable[..., Any],
    provider: str,
    token_fn: Callable[[dict[str, Any]], TokenCounts],
    *,
    model: str,
    messages: list[dict[str, Any]],
    params: dict[str, Any],
) -> Any:
    # Snapshot before the call: this is the exact input the model receives, captured before any
    # downstream code has a chance to mutate the same list/dict objects.
    messages_snapshot = copy.deepcopy(messages)
    params_snapshot = copy.deepcopy(params)

    started_at = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    response = real_call(model=model, messages=messages, **params)
    latency_ms = (time.perf_counter() - t0) * 1000

    response_plain = _to_plain(response)
    usage = response_plain.get("usage") or {}

    model_call = ModelCall(
        provider=provider,
        model=model,
        params=params_snapshot,
        messages=messages_snapshot,
        request_id=response_plain.get("id"),
        response=response_plain,
        token_counts=token_fn(usage),
        latency_ms=latency_ms,
    )
    recorder._record_model_call(model_call, started_at)
    return response


class _ChatCompletions:
    def __init__(self, real_create: Callable[..., Any], recorder: "TraceRecorder", provider: str) -> None:
        self._real_create = real_create
        self._recorder = recorder
        self._provider = provider

    def create(self, *, model: str, messages: list[dict[str, Any]], **params: Any) -> Any:
        return _record_call(
            self._recorder, self._real_create, self._provider, _openai_tokens,
            model=model, messages=messages, params=params,
        )


class _Chat:
    def __init__(self, completions: _ChatCompletions) -> None:
        self.completions = completions


class CapturingClient:
    """Wraps an OpenAI-compatible client so `client.chat.completions.create(...)` calls are
    captured transparently; call sites don't otherwise change."""

    def __init__(self, client: Any, recorder: "TraceRecorder", provider: str = "openai-compatible") -> None:
        self.chat = _Chat(_ChatCompletions(client.chat.completions.create, recorder, provider))


class _Messages:
    def __init__(self, real_create: Callable[..., Any], recorder: "TraceRecorder", provider: str) -> None:
        self._real_create = real_create
        self._recorder = recorder
        self._provider = provider

    def create(self, *, model: str, messages: list[dict[str, Any]], **params: Any) -> Any:
        # `system` is a top-level wire param on Anthropic's Messages API, not a message — keeping
        # it in `params` (rather than folding it into `messages`) is what preserves byte-exact
        # capture of the messages array actually sent.
        return _record_call(
            self._recorder, self._real_create, self._provider, _anthropic_tokens,
            model=model, messages=messages, params=params,
        )


class AnthropicCapturingClient:
    """Wraps an Anthropic client so `client.messages.create(...)` calls are captured
    transparently; call sites don't otherwise change. Duck-typed like `CapturingClient` — no hard
    dependency on the `anthropic` package."""

    def __init__(self, client: Any, recorder: "TraceRecorder", provider: str = "anthropic") -> None:
        self.messages = _Messages(client.messages.create, recorder, provider)


class TraceRecorder:
    """Builds a single Trace as an agent runs. One recorder per trace."""

    def __init__(self, agent_name: str | None = None, agent_version: str | None = None) -> None:
        self._trace = Trace(agent_name=agent_name, agent_version=agent_version)
        self._current_step: Step | None = None
        self._pending_tool_calls: dict[str, tuple[int, ToolCall]] = {}

    @property
    def trace(self) -> Trace:
        return self._trace

    def wrap_client(self, client: Any, provider: str = "openai-compatible") -> CapturingClient:
        return CapturingClient(client, self, provider)

    def wrap_anthropic_client(self, client: Any, provider: str = "anthropic") -> AnthropicCapturingClient:
        return AnthropicCapturingClient(client, self, provider)

    def wrap_tool(self, fn: Callable[..., Any], tool_name: str | None = None) -> Callable[..., tuple[Any, str]]:
        """Wrap a tool function. The wrapped call returns `(result, tool_call_id)` — pass
        `tool_call_id` to `record_insertion` once the caller knows what actually went back into
        the next model call's messages, so pre/post-truncation results can both be captured."""
        name = tool_name or getattr(fn, "__name__", "tool")

        def wrapped(*args: Any, **kwargs: Any) -> tuple[Any, str]:
            if self._current_step is None:
                raise RuntimeError("wrap_tool called outside of an active step (call begin_step() first)")
            current_step = self._current_step

            args_raw = _bind_args(fn, args, kwargs)
            started_at = datetime.now(timezone.utc)
            result = fn(*args, **kwargs)
            ended_at = datetime.now(timezone.utc)

            tool_call_id = str(uuid4())
            tool_call = ToolCall(
                tool_call_id=tool_call_id,
                tool_name=name,
                args_raw=args_raw,
                result_as_returned=copy.deepcopy(result),
                result_as_inserted=copy.deepcopy(result),
                started_at=started_at,
                ended_at=ended_at,
            )
            current_step.tool_calls.append(tool_call)
            self._pending_tool_calls[tool_call_id] = (current_step.step_index, tool_call)
            return result, tool_call_id

        return wrapped

    def record_insertion(self, tool_call_id: str, inserted_value: Any) -> None:
        """Record what was actually inserted into the next model call's messages for this tool
        call. If it differs from `result_as_returned`, infers a truncation event (the SDK can't
        know the framework's truncation strategy, so `detected_by` is always "inferred" here)."""
        step_index, tool_call = self._pending_tool_calls.pop(tool_call_id)
        tool_call.result_as_inserted = inserted_value

        returned_json = json.dumps(tool_call.result_as_returned, sort_keys=True, default=str)
        inserted_json = json.dumps(inserted_value, sort_keys=True, default=str)
        if returned_json != inserted_json:
            self._trace.steps[step_index].truncation_events.append(
                TruncationEvent(
                    location=f"tool_call:{tool_call_id}",
                    original_size_bytes=len(returned_json.encode()),
                    truncated_size_bytes=len(inserted_json.encode()),
                    strategy="unknown",
                    detected_by="inferred",
                )
            )

    def begin_step(self) -> Step:
        step = Step(step_index=len(self._trace.steps), started_at=datetime.now(timezone.utc))
        self._trace.steps.append(step)
        self._current_step = step
        return step

    def end_step(self) -> None:
        if self._current_step is not None:
            self._current_step.ended_at = datetime.now(timezone.utc)
        self._current_step = None

    def _record_model_call(self, model_call: ModelCall, started_at: datetime) -> None:
        if self._current_step is None:
            raise RuntimeError("model call captured outside of an active step (call begin_step() first)")
        self._current_step.model_call = model_call
