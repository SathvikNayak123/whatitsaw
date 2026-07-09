from datetime import datetime, timezone

import pytest

from ctx_capture.schema import (
    SCHEMA_VERSION,
    ModelCall,
    Step,
    Trace,
    UnsupportedSchemaVersionError,
    assert_schema_version_supported,
)


def test_trace_round_trips_through_json():
    step = Step(
        step_index=0,
        started_at=datetime.now(timezone.utc),
        model_call=ModelCall(
            provider="toy-provider",
            model="toy-model",
            messages=[{"role": "user", "content": "hi"}],
        ),
    )
    trace = Trace(agent_name="toy-agent", steps=[step])

    restored = Trace.model_validate_json(trace.model_dump_json())

    assert restored == trace


def test_same_major_version_is_supported():
    assert_schema_version_supported(SCHEMA_VERSION)
    assert_schema_version_supported("1.0")
    assert_schema_version_supported("1.7")  # a later 1.x minor is still readable per policy


def test_different_major_version_is_refused():
    with pytest.raises(UnsupportedSchemaVersionError):
        assert_schema_version_supported("2.0")
