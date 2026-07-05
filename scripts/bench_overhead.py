"""Capture-overhead micro-benchmark: added latency per instrumented model call, wrapped vs.
unwrapped, against an in-process fake client (isolates SDK overhead from network/provider
latency). Writes docs/RESULTS.md. Run: python scripts/bench_overhead.py
"""

from __future__ import annotations

import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ctx_capture.capture import TraceRecorder

N = 2000


class _FakeCompletions:
    def create(self, *, model: str, messages: list[dict[str, Any]], **params: Any) -> dict[str, Any]:
        return {
            "id": "r",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        }


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeClient:
    def __init__(self) -> None:
        self.chat = _FakeChat()


def _messages() -> list[dict[str, Any]]:
    return [{"role": "system", "content": "s"}, {"role": "user", "content": "hello world"}]


def bench_unwrapped(n: int) -> list[float]:
    client = _FakeClient()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        client.chat.completions.create(model="m", messages=_messages(), temperature=0)
        times.append((time.perf_counter() - t0) * 1000)
    return times


def bench_wrapped(n: int) -> list[float]:
    client = _FakeClient()
    recorder = TraceRecorder(agent_name="bench")
    wrapped = recorder.wrap_client(client)
    times = []
    for _ in range(n):
        recorder.begin_step()
        t0 = time.perf_counter()
        wrapped.chat.completions.create(model="m", messages=_messages(), temperature=0)
        times.append((time.perf_counter() - t0) * 1000)
        recorder.end_step()
    return times


def _stats(times: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "p95_ms": statistics.quantiles(times, n=100)[94],
    }


def main() -> None:
    unwrapped = bench_unwrapped(N)
    wrapped = bench_wrapped(N)
    u, w = _stats(unwrapped), _stats(wrapped)
    overhead_mean_ms = w["mean_ms"] - u["mean_ms"]
    overhead_median_ms = w["median_ms"] - u["median_ms"]

    report = f"""# ctx-capture — capture-overhead benchmark

Measured: {datetime.now(timezone.utc).isoformat()}
Python: {sys.version.split()[0]} ({platform.system()} {platform.release()})
Iterations: {N} calls per arm, in-process fake client (isolates SDK overhead from network/provider latency)

| | unwrapped (ms) | wrapped (ms) |
|---|---|---|
| mean | {u["mean_ms"]:.4f} | {w["mean_ms"]:.4f} |
| median | {u["median_ms"]:.4f} | {w["median_ms"]:.4f} |
| p95 | {u["p95_ms"]:.4f} | {w["p95_ms"]:.4f} |

**Added overhead per instrumented call: {overhead_mean_ms:.4f} ms mean / {overhead_median_ms:.4f} ms median.**

Config: default capture path (deepcopy of messages/params on every call, no redaction hook, no
raw-bytes capture). Measures the in-memory capture wrapper only, not the storage write path.

Reproduce: `python scripts/bench_overhead.py`
"""
    Path("docs/RESULTS.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
