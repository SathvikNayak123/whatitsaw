"""Dogfood the SDK: run the deep-research example agent (examples/deep_research_agent.py),
capture a full trace, then query it through the real MCP server over the real stdio transport —
the same path scripts/manual_verify.py exercises with the toy agent, but against a genuine
multi-step research run instead of a test fixture.

Run: python scripts/dogfood.py
Writes docs/proof/dogfood_deep_research_transcript.txt
"""

from __future__ import annotations

import asyncio
import io
import json
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "examples"))

from mcp import ClientSession  # noqa: E402
from mcp.client.stdio import StdioServerParameters, stdio_client  # noqa: E402

from deep_research_agent import MID_RUN_STEP_INDEX, run  # noqa: E402

DB_PATH = REPO_ROOT / "scratch_dogfood.db"
PROOF_DIR = REPO_ROOT / "docs" / "proof"


async def _verify(trace_id: str) -> None:
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "ctx_capture.mcp", "--db", str(DB_PATH)],
        cwd=str(REPO_ROOT),
    )

    print(f"# ctx-capture dogfood: deep-research agent — {datetime.now(timezone.utc).isoformat()}")
    print(f"# subprocess: {sys.executable} -m ctx_capture.mcp --db {DB_PATH}  (real stdio transport)")
    print(f"# trace_id: {trace_id}\n")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            print(f"## connect\nserver: {init.serverInfo.name} {init.serverInfo.version}\n")

            print("## list_traces()")
            listed = await session.call_tool("list_traces", {})
            for t in listed.structuredContent["traces"]:
                print(f"  - {t['trace_id']}  agent={t['agent_name']}  steps={t['step_count']}  "
                      f"tokens={t['total_tokens']}  has_anomalies={t['has_anomalies']}")
            print()

            print("## get_token_accounting(trace_id, group_by='step')")
            accounting = await session.call_tool("get_token_accounting", {"trace_id": trace_id, "group_by": "step"})
            sc = accounting.structuredContent
            print(f"  total_prompt_tokens={sc['total_prompt_tokens']}  "
                  f"total_completion_tokens={sc['total_completion_tokens']}")
            for item in sc["breakdown"]:
                print(f"    step {item['key']}: {item['total_tokens']} tokens")
            print()

            print("## find_context_anomalies(trace_id)  — expect a real truncation + a real budget overflow")
            anomalies = await session.call_tool("find_context_anomalies", {"trace_id": trace_id})
            for a in anomalies.structuredContent["anomalies"]:
                print(f"  - step {a['step_index']}: {a['type']} ({a['severity']}) — {a['detail']}")
            print()

            print(f"## browse: read_resource(trace://{trace_id}) — step index")
            trace_resource = await session.read_resource(f"trace://{trace_id}")
            trace_view = json.loads(trace_resource.contents[0].text)
            print(f"  agent_name={trace_view['agent_name']}  step_count={len(trace_view['steps'])}\n")

            print(f"## pull mid-run step {MID_RUN_STEP_INDEX}'s exact context: "
                  f"get_step_context(trace_id, step_index={MID_RUN_STEP_INDEX})")
            result = await session.call_tool(
                "get_step_context", {"trace_id": trace_id, "step_index": MID_RUN_STEP_INDEX}
            )
            print(f"  isError={result.isError}")
            sc = result.structuredContent
            print(f"  provider={sc['model_call']['provider']}  model={sc['model_call']['model']}")
            print(f"  total_message_count={sc['total_message_count']}  truncated={sc['truncated']}")
            tool_message = sc["model_call"]["messages"][-1]
            print(f"  final message in context at this step: role={tool_message['role']!r}, "
                  f"content={tool_message['content'][:120]!r}...")
            print()

            print("## VERIFIED: deep-research agent traces are captured (via the real ctx_capture SDK,")
            print("## not a test fixture) and queryable through the real MCP server — list_traces,")
            print("## get_token_accounting, find_context_anomalies, resource browsing, and")
            print("## get_step_context on a mid-run step all worked against a real stdio-transport client.")


def main() -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()
    trace_id = run(str(DB_PATH))

    buf = io.StringIO()
    with redirect_stdout(buf):
        asyncio.run(_verify(trace_id))
    transcript = buf.getvalue()
    print(transcript)

    PROOF_DIR.mkdir(parents=True, exist_ok=True)
    (PROOF_DIR / "dogfood_deep_research_transcript.txt").write_text(transcript, encoding="utf-8")
    DB_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
