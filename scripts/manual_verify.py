"""Manual verification against a real MCP client, over the real stdio transport (a subprocess of
`python -m ctx_capture.mcp`, exactly as Claude Desktop/Code would spawn it) — not the in-memory
test harness used by the contract tests. Seeds a 15-step trace, connects, lists tools/resources,
browses the trace resource, and pulls step 12's exact reconstructed context.

Run: python scripts/manual_verify.py
Writes a full transcript to docs/proof/manual_verification_transcript.txt
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
sys.path.insert(0, str(REPO_ROOT))  # so `tests.fixtures` (test-only fake client) is importable

from mcp import ClientSession  # noqa: E402
from mcp.client.stdio import StdioServerParameters, stdio_client  # noqa: E402

from ctx_capture.capture import TraceRecorder  # noqa: E402
from ctx_capture.storage.sqlite_repo import SQLiteTraceRepository  # noqa: E402
from tests.fixtures.fake_client import FakeOpenAIClient, FakeResponse  # noqa: E402

DB_PATH = REPO_ROOT / "scratch_manual_verify.db"
PROOF_DIR = REPO_ROOT / "docs" / "proof"


def _seed_trace() -> str:
    """A 15-step trace so step index 12 exists, with a tool call + truncation along the way."""
    if DB_PATH.exists():
        DB_PATH.unlink()
    repo = SQLiteTraceRepository(str(DB_PATH))
    recorder = TraceRecorder(agent_name="manual-verify-agent", agent_version="1.0")

    responses = []
    for i in range(15):
        if i == 5:
            responses.append(
                FakeResponse(
                    {
                        "id": f"resp_{i}",
                        "usage": {"prompt_tokens": 20 + i, "completion_tokens": 5, "total_tokens": 25 + i},
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": None,
                                    "tool_calls": [
                                        {
                                            "id": f"call_{i}",
                                            "type": "function",
                                            "function": {"name": "search", "arguments": json.dumps({"query": "big"})},
                                        }
                                    ],
                                }
                            }
                        ],
                    }
                )
            )
        else:
            responses.append(
                FakeResponse(
                    {
                        "id": f"resp_{i}",
                        "usage": {"prompt_tokens": 20 + i, "completion_tokens": 5, "total_tokens": 25 + i},
                        "choices": [{"message": {"role": "assistant", "content": f"step {i} answer", "tool_calls": None}}],
                    }
                )
            )

    client = FakeOpenAIClient(responses)
    capturing_client = recorder.wrap_client(client, provider="toy-provider")

    messages: list[dict] = [{"role": "system", "content": "You are the manual-verification test agent."}]
    for i in range(15):
        recorder.begin_step()
        response = capturing_client.chat.completions.create(model="toy-model", messages=messages, temperature=0)
        choice = response["choices"][0]["message"]
        messages.append({"role": "assistant", "content": choice.get("content"), "tool_calls": choice.get("tool_calls")})

        if choice.get("tool_calls"):
            wrapped_tool = recorder.wrap_tool(lambda query: "z" * 5000, tool_name="search")
            result, tool_call_id = wrapped_tool(query="big")
            inserted = result[:80] + "...[truncated]"
            recorder.record_insertion(tool_call_id, inserted)
            messages.append({"role": "tool", "tool_call_id": f"call_{i}", "content": inserted})

        recorder.end_step()

    repo.save(recorder.trace)
    repo._conn.close()  # release the file lock before the subprocess opens its own connection
    return recorder.trace.trace_id


async def _verify(trace_id: str) -> None:
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "ctx_capture.mcp", "--db", str(DB_PATH)],
        cwd=str(REPO_ROOT),
    )

    print(f"# ctx-capture manual MCP verification — {datetime.now(timezone.utc).isoformat()}")
    print(f"# subprocess: {sys.executable} -m ctx_capture.mcp --db {DB_PATH}  (real stdio transport)")
    print(f"# trace_id: {trace_id}\n")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            print(f"## connect\nserver: {init.serverInfo.name} {init.serverInfo.version}\n")

            tools = (await session.list_tools()).tools
            print(f"## list_tools ({len(tools)})")
            for t in tools:
                print(f"  - {t.name}: outputSchema={'yes' if t.outputSchema else 'NO'}")
            print()

            templates = (await session.list_resource_templates()).resourceTemplates
            print(f"## list_resource_templates ({len(templates)})")
            for t in templates:
                print(f"  - {t.uriTemplate}")
            print()

            print(f"## browse: read_resource(trace://{trace_id})")
            trace_resource = await session.read_resource(f"trace://{trace_id}")
            trace_view = json.loads(trace_resource.contents[0].text)
            print(f"  agent_name={trace_view['agent_name']}  step_count={len(trace_view['steps'])}")
            print(f"  step 12 summary: {trace_view['steps'][12]}\n")

            print(f"## pull step 12's exact context: get_step_context(trace_id, step_index=12)")
            result = await session.call_tool(
                "get_step_context", {"trace_id": trace_id, "step_index": 12}
            )
            print(f"  isError={result.isError}")
            sc = result.structuredContent
            print(f"  provider={sc['model_call']['provider']}  model={sc['model_call']['model']}")
            print(f"  total_message_count={sc['total_message_count']}  truncated={sc['truncated']}")
            print(f"  message[0]={sc['model_call']['messages'][0]}")
            print(f"  message[-1]={sc['model_call']['messages'][-1]}")
            print(f"  token_counts={sc['model_call']['token_counts']}\n")

            print("## find_context_anomalies(trace_id)")
            anomalies = await session.call_tool("find_context_anomalies", {"trace_id": trace_id})
            for a in anomalies.structuredContent["anomalies"]:
                print(f"  - step {a['step_index']}: {a['type']} ({a['severity']}) — {a['detail']}")
            print()

            print("## VERIFIED: real stdio-transport MCP client connected, browsed a trace resource,")
            print("## and pulled step 12's byte-exact reconstructed context via get_step_context.")


def main() -> None:
    trace_id = _seed_trace()
    buf = io.StringIO()
    with redirect_stdout(buf):
        asyncio.run(_verify(trace_id))
    transcript = buf.getvalue()
    print(transcript)

    PROOF_DIR.mkdir(parents=True, exist_ok=True)
    (PROOF_DIR / "manual_verification_transcript.txt").write_text(transcript, encoding="utf-8")
    DB_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
