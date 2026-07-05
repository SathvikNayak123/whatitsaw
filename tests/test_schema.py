from datetime import datetime, timezone

from ctx_capture.schema import ModelCall, Step, Trace


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
