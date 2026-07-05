"""Bearer-token auth for streamable HTTP mode. See docs/DESIGN.md "Transport": bearer token is
the minimum viable gate against a wide-open remote server; full OAuth is roadmap, not v1, so this
deliberately doesn't pull in the SDK's OAuthAuthorizationServerProvider machinery.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class BearerAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, token: str) -> None:
        super().__init__(app)
        self._expected = f"Bearer {token}"

    async def dispatch(self, request: Request, call_next):
        if request.headers.get("authorization") != self._expected:
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)


def build_http_app(mcp: FastMCP, bearer_token: str) -> Starlette:
    """Streamable HTTP ASGI app for `mcp`, gated by a static bearer token on every request."""
    app = mcp.streamable_http_app()
    app.add_middleware(BearerAuthMiddleware, token=bearer_token)
    return app
