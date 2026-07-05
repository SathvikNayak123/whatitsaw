"""Allow-nothing auth for streamable HTTP mode: no request reaches the MCP app without the
correct bearer token. See docs/DESIGN.md "Transport".
"""

from __future__ import annotations

from starlette.testclient import TestClient

from ctx_capture.mcp.auth import build_http_app
from ctx_capture.mcp.server import create_server
from ctx_capture.storage.sqlite_repo import SQLiteTraceRepository

TOKEN = "s3cr3t-test-token"


def _app(tmp_path):
    repo = SQLiteTraceRepository(str(tmp_path / "auth_test.db"))
    mcp = create_server(repo)
    return build_http_app(mcp, bearer_token=TOKEN)


def test_missing_authorization_header_is_rejected(tmp_path):
    # FastMCP's streamable-http session manager requires ASGI lifespan startup (a running task
    # group) before it will handle any request — TestClient only sends lifespan events when used
    # as a context manager.
    with TestClient(_app(tmp_path)) as client:
        response = client.post("/mcp", json={})
        assert response.status_code == 401


def test_wrong_token_is_rejected(tmp_path):
    with TestClient(_app(tmp_path)) as client:
        response = client.post("/mcp", json={}, headers={"Authorization": "Bearer wrong-token"})
        assert response.status_code == 401


def test_malformed_authorization_header_is_rejected(tmp_path):
    with TestClient(_app(tmp_path)) as client:
        response = client.post("/mcp", json={}, headers={"Authorization": TOKEN})  # missing "Bearer "
        assert response.status_code == 401


def test_correct_token_passes_the_auth_layer(tmp_path):
    with TestClient(_app(tmp_path)) as client:
        response = client.post("/mcp", json={}, headers={"Authorization": f"Bearer {TOKEN}"})
        # Not necessarily 200 — an empty body isn't a valid MCP request — but it must not be
        # rejected by the auth layer specifically.
        assert response.status_code != 401
