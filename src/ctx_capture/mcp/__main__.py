"""Run the ctx-capture MCP server. See docs/DESIGN.md "Transport".

  python -m ctx_capture.mcp                          # stdio, local dev (primary distribution)
  python -m ctx_capture.mcp --transport http --port 8000 --bearer-token secret
"""

from __future__ import annotations

import argparse
import os
import sys

from ctx_capture.mcp.auth import build_http_app
from ctx_capture.mcp.server import DEFAULT_MAX_RESPONSE_BYTES, create_server
from ctx_capture.storage.sqlite_repo import SQLiteTraceRepository


def main() -> None:
    parser = argparse.ArgumentParser(prog="ctx-capture")
    parser.add_argument("--db", default="ctx_capture.db", help="SQLite database path")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--bearer-token",
        default=os.environ.get("CTX_CAPTURE_BEARER_TOKEN"),
        help="required for --transport http; also read from CTX_CAPTURE_BEARER_TOKEN",
    )
    parser.add_argument("--max-response-bytes", type=int, default=DEFAULT_MAX_RESPONSE_BYTES)
    args = parser.parse_args()

    repo = SQLiteTraceRepository(args.db)
    mcp = create_server(repo, max_response_bytes=args.max_response_bytes)

    if args.transport == "stdio":
        mcp.run(transport="stdio")
        return

    if not args.bearer_token:
        print("error: --transport http requires --bearer-token or CTX_CAPTURE_BEARER_TOKEN", file=sys.stderr)
        raise SystemExit(2)

    import uvicorn

    app = build_http_app(mcp, args.bearer_token)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
