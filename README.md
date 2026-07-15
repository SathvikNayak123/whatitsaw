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

```bash
pip install ctx-capture
# or, without installing: uvx ctx-capture --db my_agent.db
```

Run the server (SQLite-backed, zero config):

```bash
python -m ctx_capture.mcp --db my_agent.db
```

Point an MCP client at it — e.g. Claude Desktop's `claude_desktop_config.json`:

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

Instrument your agent — a few lines, either capture path:

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

Then, from your MCP client: "list recent traces for my-agent" or "show me exactly what the model
saw at step 12" — the client calls `list_traces` / `get_step_context` for you.

For a shared/remote deployment: `ctx-capture --transport http --port 8000 --bearer-token
<token>` (see [docs/DESIGN.md § Transport](docs/DESIGN.md) for why stdio is the default and HTTP
is opt-in).

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
(in-process benchmark, wrapper only — see [docs/RESULTS.md](docs/RESULTS.md), reproduce with
`python scripts/bench_overhead.py`).

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

See [docs/DESIGN.md § Non-goals](docs/DESIGN.md) for the reasoning behind each.

## Schema stability

Every trace is written with a pinned `schema_version`. Within a major version, changes are
**additive-only** — new optional fields, never a repurposed or removed one — so any `1.x` reader
can read any `1.x` trace. A breaking change requires a major version bump and a migration script;
servers refuse (not silently coerce) traces with an unsupported major version. See
[docs/DESIGN.md § The schema](docs/DESIGN.md) for the full schema and versioning rationale.

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
