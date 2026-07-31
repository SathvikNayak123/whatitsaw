# ctx-capture

**See exactly what your LLM agent saw, at any step, byte-for-byte.**

An MCP server for context-window observability. It captures the exact model input at every step
of an agent run — including what a tool actually returned versus what got inserted into the next
call after truncation — and lets any MCP client query it directly.

Agent failures are usually context failures, not reasoning failures: a tool result got truncated
before it reached the model, a token budget overflowed and the framework quietly cut messages
from the middle, a retry duplicated a call the model never saw resolve. Answering "what did the
model actually see at step 12" today means archaeology across app logs, framework debug output,
and provider dashboards. ctx-capture exists so that question has a direct answer.

## Features

- **Byte-exact per-step capture** — the exact provider-native message array as sent, not a
  reshaped/normalized approximation.
- **Pre- and post-truncation tool results** — `result_as_returned` vs. `result_as_inserted`, so
  truncation is detectable, not just guessed at.
- **Two capture paths**: OpenAI-compatible `chat.completions` clients and Anthropic's Messages
  API, both duck-typed (no hard SDK dependency) and both gated by the same fidelity test.
- **5 MCP tools + 2 resources** — `list_traces`, `get_step_context`, `diff_step_contexts`,
  `find_context_anomalies`, `get_token_accounting`, plus `trace://` resources for browsing
  clients. All 5 tools declare `outputSchema` and return `structuredContent`.
- **Size-capped, paginated responses** — every tool response is bounded (default 50 KB); an
  oversized payload gets a `resource_uri` pointer instead of ever blowing up a client's context.
- **SQLite by default, zero external dependency** — `pip install ctx-capture` gives you a working
  MCP server with nothing else to stand up.
- **CI-blocking fidelity test** — the schema's acceptance test, not an optional check: captured
  `messages` must be byte-identical, after canonical JSON serialization, to what actually left
  application code.

## Setup

### Prerequisites

- **Python ≥ 3.10.**
- **An agent to instrument** — anything that calls an OpenAI-compatible `chat.completions`
  client or Anthropic's Messages API. Both capture paths are duck-typed, so no specific SDK
  version is required.
- **Any MCP client to query with** — Claude Code, Claude Desktop, Cursor, etc.
- **Nothing else** — storage is a local SQLite file, created on first save. No database server,
  no API key (the server itself never calls a model).

Note the split: the *query* side is plug-and-play (any MCP client, zero code), but the *capture*
side requires instrumenting your agent (step 2) — only code inside the agent process can see the
exact bytes sent to the model. An MCP server pointed at an empty DB has nothing to serve.

### 1. Install

```bash
pip install ctx-capture
# or, without installing: uvx ctx-capture --db my_agent.db
```

### 2. Instrument your agent

A few lines, either capture path:

```python
from ctx_capture.capture import TraceRecorder
from ctx_capture.storage import SQLiteTraceRepository

recorder = TraceRecorder(agent_name="my-agent")

# OpenAI-compatible client — call sites unchanged
client = recorder.wrap_client(my_openai_compatible_client)

# ...or an Anthropic client — call sites unchanged
client = recorder.wrap_anthropic_client(my_anthropic_client)

recorder.begin_step()
response = client.chat.completions.create(model="...", messages=messages)  # or client.messages.create(...)
recorder.end_step()

SQLiteTraceRepository("my_agent.db").save(recorder.trace)
```

To also capture tool calls (and detect truncation), wrap each tool and record what actually went
back into the message history:

```python
wrapped_tool = recorder.wrap_tool(my_search_fn, tool_name="search")
result, tool_call_id = wrapped_tool(query="...")
# ...after your framework truncates/summarizes `result` into what it puts in messages:
recorder.record_insertion(tool_call_id, inserted_value)
```

### 3. Record a run

Run your instrumented agent normally — each run saves a trace into the DB. Don't have an agent
handy? [examples/demo_agent.py](examples/demo_agent.py) is a complete, minimal worked example: a
3-step trip-planner agent (search twice, then answer) instrumented end to end, with an offline
simulated model so no API key is needed:

```bash
python examples/demo_agent.py my_agent.db
# captured trace <trace_id> -> my_agent.db
```

([examples/deep_research_agent.py](examples/deep_research_agent.py) is the longer version that
also exercises a real truncation event and a budget-overflow anomaly.)

### 4. Run the MCP server

SQLite-backed, zero config — point it at the same DB your agent writes to:

```bash
python -m ctx_capture.mcp --db my_agent.db
```

For a shared/remote deployment: `ctx-capture --transport http --port 8000 --bearer-token
<token>` (stdio is the default, HTTP is opt-in).

### 5. Connect an MCP client

Claude Desktop (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "ctx-capture": {
      "command": "ctx-capture",
      "args": ["--db", "/absolute/path/to/my_agent.db"]
    }
  }
}
```

Claude Code (VSCode / CLI) — same block in a `.mcp.json` at your project root, then reload the
window so it picks the server up.

### 6. Query

From your MCP client, in plain language: "list recent traces for my-agent" or "show me exactly
what the model saw at step 12" — the client calls `list_traces` / `get_step_context` for you.
(With the demo trace above, try "any context anomalies in the last demo-trip-agent trace?")

## Tool reference

| Tool | Answers |
|---|---|
| `list_traces(agent_name?, since?, until?, has_anomalies?, tags?, limit?, cursor?)` | "What traces have been captured?" — cursor-paginated. |
| `get_step_context(trace_id, step_index, max_bytes?, cursor?)` | "What did the model see at step N, exactly?" — the byte-exact reconstructed input. |
| `diff_step_contexts(trace_id, step_a, step_b)` | "What changed in the model's context between two steps?" |
| `find_context_anomalies(trace_id, types?)` | "Where did something go wrong?" — truncations, budget overflows, dropped messages, tool-result mismatches. |
| `get_token_accounting(trace_id, group_by?)` | "Where did the tokens/cost go?" — grouped by step, tool, or role. |

Plus 2 resources for browsing clients: `trace://{trace_id}` (metadata + step index) and
`trace://{trace_id}/step/{step_index}` (full, unpaginated step detail).

## The fidelity test

The project's one non-negotiable test, CI-blocking forever
([tests/test_fidelity.py](tests/test_fidelity.py),
[tests/test_fidelity_anthropic.py](tests/test_fidelity_anthropic.py)): capture a running agent's
model input, reconstruct it from storage, and assert the reconstructed `messages` array is
**byte-identical**, after canonical JSON serialization, to what an independent observation point
saw actually leave application code. If this test can't pass, the schema has failed at the one
thing it exists to do.

Capture overhead is measured, not assumed: **~0.017ms added per instrumented model call**
(in-process benchmark, wrapper only, reproduce with `python scripts/bench_overhead.py`).

## ctx-capture vs. Langfuse / LangSmith

| | ctx-capture | Langfuse / LangSmith |
|---|---|---|
| Core question | "What did the model see, byte-for-byte, at step N?" | "What happened across this run?" |
| Fidelity | Provider-native messages captured verbatim; tool results captured both pre- and post-truncation | Traces are typically reshaped for display; framework-level truncation is usually invisible |
| Interface | MCP tools/resources — built for an agent or MCP client to query directly | Web dashboard + SDK — built for a human browsing a UI |
| Evals | Not a goal (see Non-goals) | Core feature |
| Dashboards/UI | None — MCP clients are the UI | Mature, first-class |
| Multi-agent trace stitching | Not in v1 | Yes |
| Storage | SQLite (zero-dep) or Postgres, self-hosted | Hosted or self-hosted, Postgres-backed service |

**Use Langfuse/LangSmith** for broad observability, evals, cost dashboards, and team-wide trace
review. **Use ctx-capture** when you need to prove, precisely, what one model call actually saw —
usually while debugging a specific failure, from inside an MCP-capable client.

## Non-goals

- **No dashboards/UI** — MCP clients are the UI; we're not rebuilding Langfuse's trace viewer.
- **No eval features** — evals are a mature, separate category with a different workflow.
- **No multi-agent trace stitching in v1** — single-agent step fidelity first.
- **No prompt management** — that's a build-time concern; this is runtime observability.

## Schema stability

Every trace is written with a pinned `schema_version`. Within a major version, changes are
**additive-only** — new optional fields, never a repurposed or removed one — so any `1.x` reader
can read any `1.x` trace. A breaking change requires a major version bump and a migration script;
servers refuse (not silently coerce) traces with an unsupported major version.

## Roadmap

- **OTel span-ingestion adapter** — an additive way to bring in traces from OTel GenAI
  instrumentation, without making it the primary (lower-fidelity) capture path.
- **Redaction hook** — an opt-in `redact(message) -> message` hook at capture time. Not yet
  built; until it is, treat captured trace data as sensitive by default (it faithfully contains
  whatever the agent saw, PII/secrets included).
- **Postgres backend** — the `TraceRepository` interface is ready for it; the implementation
  isn't written yet.
- **OAuth for HTTP transport** — the current HTTP auth is a static bearer token, which is the
  minimum viable gate, not the final answer for multi-tenant deployments.

## License

MIT — see [LICENSE](LICENSE).
