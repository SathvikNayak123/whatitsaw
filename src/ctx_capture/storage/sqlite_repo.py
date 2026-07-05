from __future__ import annotations

import base64
import json
import sqlite3
from typing import Any

from ctx_capture.schema import Step, Trace
from ctx_capture.storage.repository import TraceRepository, TraceSummaryRow

_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS traces (
    trace_id TEXT PRIMARY KEY,
    schema_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    agent_name TEXT,
    agent_version TEXT,
    metadata_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS steps (
    trace_id TEXT NOT NULL,
    step_index INTEGER NOT NULL,
    step_json TEXT NOT NULL,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0.0,
    PRIMARY KEY (trace_id, step_index)
);

CREATE INDEX IF NOT EXISTS idx_traces_created_at ON traces (created_at, trace_id);
"""


def _encode_cursor(created_at: str, trace_id: str) -> str:
    raw = f"{created_at}|{trace_id}".encode()
    return base64.urlsafe_b64encode(raw).decode()


def _decode_cursor(cursor: str) -> tuple[str, str]:
    raw = base64.urlsafe_b64decode(cursor.encode()).decode()
    created_at, trace_id = raw.split("|", 1)
    return created_at, trace_id


class SQLiteTraceRepository(TraceRepository):
    def __init__(self, path: str = "ctx_capture.db") -> None:
        self._conn = sqlite3.connect(path)
        self._conn.executescript(_SCHEMA_DDL)
        self._conn.commit()

    def save(self, trace: Trace) -> None:
        with self._conn:
            self._conn.execute(
                """INSERT OR REPLACE INTO traces
                   (trace_id, schema_version, created_at, agent_name, agent_version, metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    trace.trace_id,
                    trace.schema_version,
                    trace.created_at.isoformat(),
                    trace.agent_name,
                    trace.agent_version,
                    json.dumps(trace.metadata),
                ),
            )
            self._conn.executemany(
                """INSERT OR REPLACE INTO steps
                   (trace_id, step_index, step_json, total_tokens, cost_usd)
                   VALUES (?, ?, ?, ?, ?)""",
                [
                    (
                        trace.trace_id,
                        step.step_index,
                        step.model_dump_json(),
                        step.model_call.token_counts.total_tokens if step.model_call else 0,
                        (step.model_call.cost_usd or 0.0) if step.model_call else 0.0,
                    )
                    for step in trace.steps
                ],
            )

    def get_trace(self, trace_id: str) -> Trace:
        row = self._conn.execute(
            "SELECT schema_version, created_at, agent_name, agent_version, metadata_json "
            "FROM traces WHERE trace_id = ?",
            (trace_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"no trace {trace_id}")
        schema_version, created_at, agent_name, agent_version, metadata_json = row

        step_rows = self._conn.execute(
            "SELECT step_json FROM steps WHERE trace_id = ? ORDER BY step_index",
            (trace_id,),
        ).fetchall()
        steps = [Step.model_validate_json(r[0]) for r in step_rows]

        return Trace(
            schema_version=schema_version,
            trace_id=trace_id,
            created_at=created_at,
            agent_name=agent_name,
            agent_version=agent_version,
            metadata=json.loads(metadata_json),
            steps=steps,
        )

    def get_step(self, trace_id: str, step_index: int) -> Step:
        row = self._conn.execute(
            "SELECT step_json FROM steps WHERE trace_id = ? AND step_index = ?",
            (trace_id, step_index),
        ).fetchone()
        if row is None:
            raise KeyError(f"no step {step_index} for trace {trace_id}")
        return Step.model_validate_json(row[0])

    def list_traces(
        self,
        *,
        agent_name: str | None = None,
        since: str | None = None,
        until: str | None = None,
        tags: dict[str, Any] | None = None,
        limit: int = 20,
        cursor: str | None = None,
    ) -> tuple[list[TraceSummaryRow], str | None]:
        limit = max(1, min(limit, 200))
        conditions: list[str] = []
        params: list[Any] = []

        if agent_name is not None:
            conditions.append("t.agent_name = ?")
            params.append(agent_name)
        if since is not None:
            conditions.append("t.created_at >= ?")
            params.append(since)
        if until is not None:
            conditions.append("t.created_at <= ?")
            params.append(until)
        if tags:
            for key, value in tags.items():
                conditions.append("json_extract(t.metadata_json, '$.' || ?) = ?")
                params.extend([key, value])
        if cursor is not None:
            cursor_created_at, cursor_trace_id = _decode_cursor(cursor)
            conditions.append("(t.created_at > ? OR (t.created_at = ? AND t.trace_id > ?))")
            params.extend([cursor_created_at, cursor_created_at, cursor_trace_id])

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = self._conn.execute(
            f"""
            SELECT t.trace_id, t.agent_name, t.created_at,
                   COUNT(s.step_index) AS step_count,
                   COALESCE(SUM(s.total_tokens), 0) AS total_tokens,
                   COALESCE(SUM(s.cost_usd), 0.0) AS total_cost_usd
            FROM traces t
            LEFT JOIN steps s ON s.trace_id = t.trace_id
            {where}
            GROUP BY t.trace_id
            ORDER BY t.created_at ASC, t.trace_id ASC
            LIMIT ?
            """,
            (*params, limit + 1),
        ).fetchall()

        has_more = len(rows) > limit
        page = rows[:limit]
        next_cursor = _encode_cursor(page[-1][2], page[-1][0]) if has_more and page else None

        summaries = [
            TraceSummaryRow(
                trace_id=r[0],
                agent_name=r[1],
                created_at=r[2],
                step_count=r[3],
                total_tokens=r[4],
                total_cost_usd=r[5],
            )
            for r in page
        ]
        return summaries, next_cursor
