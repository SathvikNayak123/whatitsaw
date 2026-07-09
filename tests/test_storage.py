"""SQLiteTraceRepository behavior not covered by the fidelity/MCP suites: refusing traces whose
major schema_version this server doesn't support. See docs/DESIGN.md "The schema": servers
refuse (not silently coerce) traces with an unsupported major version.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ctx_capture.schema import ModelCall, Step, Trace, UnsupportedSchemaVersionError
from ctx_capture.storage.sqlite_repo import SQLiteTraceRepository


def _make_trace(schema_version: str) -> Trace:
    step = Step(
        step_index=0,
        started_at=datetime.now(timezone.utc),
        model_call=ModelCall(provider="p", model="m", messages=[{"role": "user", "content": "hi"}]),
    )
    return Trace(schema_version=schema_version, agent_name="future-agent", steps=[step])


def test_get_trace_refuses_unsupported_major_version(tmp_path):
    repo = SQLiteTraceRepository(str(tmp_path / "t.db"))
    trace = _make_trace("2.0")
    repo.save(trace)

    with pytest.raises(UnsupportedSchemaVersionError):
        repo.get_trace(trace.trace_id)


def test_get_step_refuses_unsupported_major_version(tmp_path):
    repo = SQLiteTraceRepository(str(tmp_path / "t.db"))
    trace = _make_trace("2.0")
    repo.save(trace)

    with pytest.raises(UnsupportedSchemaVersionError):
        repo.get_step(trace.trace_id, 0)


def test_get_trace_accepts_same_major_minor_version(tmp_path):
    repo = SQLiteTraceRepository(str(tmp_path / "t.db"))
    trace = _make_trace("1.9")  # a hypothetical later 1.x minor — must still be readable
    repo.save(trace)

    restored = repo.get_trace(trace.trace_id)
    assert restored.schema_version == "1.9"
