# ctx-capture

**See exactly what your LLM agent saw, at any step, byte-for-byte.**

An MCP server for context-window observability. It captures the exact model input at every step
of an agent run — including what a tool actually returned versus what got inserted into the next
call after truncation — and lets any MCP client query it directly.

## The pain

Agent failures are usually context failures, not reasoning failures: a tool result got truncated
before it reached the model, a token budget overflowed and the framework quietly cut messages
from the middle, a retry duplicated a call the model never saw resolve. Today, answering "what
did the model actually see at step 12" means archaeology across app logs, framework debug output,
and provider dashboards — ctx-capture exists so that question has a direct answer.

## 60-second quickstart

Not yet published to PyPI (tracked in [docs/registry-submission.md](docs/registry-submission.md))
— `uvx` runs it straight from this repo instead via `--from git+...`, no `pip install` or clone
needed. First run downloads and builds from source (~15s on a cold cache, verified); every run
after that is instant from `uv`'s cache.

```bash
# 1. Run the server (SQLite-backed, zero config)
uvx --from git+https://github.com/SathvikNayak123/ctx-capture ctx-capture --db my_agent.db

# 2. Point an MCP client at it — e.g. Claude Desktop's claude_desktop_config.json:
{
  "mcpServers": {
    "ctx-capture": {
      "command": "uvx",
      "args": [
        "--from", "git+https://github.com/SathvikNayak123/ctx-capture",
        "ctx-capture", "--db", "/absolute/path/to/my_agent.db"
      ]
    }
  }
}

# 3. Instrument your agent (a few lines, works with any OpenAI-compatible client)
```
```python
from ctx_capture.capture import TraceRecorder
from ctx_capture.storage import SQLiteTraceRepository

recorder = TraceRecorder(agent_name="my-agent")
client = recorder.wrap_client(my_openai_compatible_client)  # unchanged call sites

recorder.begin_step()
response = client.chat.completions.create(model="...", messages=messages)
recorder.end_step()

SQLiteTraceRepository("my_agent.db").save(recorder.trace)
```

Then, from your MCP client: "list recent traces for my-agent" or "show me exactly what the model
saw at step 12" — the client calls `list_traces` / `get_step_context` for you.

No PyPI account, no server to stand up beyond the one command above — `--db` defaults to
`ctx_capture.db` in the current directory if you omit it. For a shared/remote deployment, run
`uvx --from git+https://github.com/SathvikNayak123/ctx-capture ctx-capture --transport http --port 8000 --bearer-token <token>`
instead (see [docs/DESIGN.md § Transport](docs/DESIGN.md) for why stdio is the default and HTTP
is opt-in).

Working from a local clone instead? `pip install -e .` then `python -m ctx_capture.mcp --db
my_agent.db` does the same thing without going through `uv`.

## Tool reference

All 5 tools declare `outputSchema` and return `structuredContent`; every response is capped
(default 50KB) and paginated rather than ever silently truncated — see
[docs/DESIGN.md § Pagination and size limits](docs/DESIGN.md).

| Tool | Answers |
|---|---|
| `list_traces(agent_name?, since?, until?, has_anomalies?, tags?, limit?, cursor?)` | "What traces have been captured?" — cursor-paginated. |
| `get_step_context(trace_id, step_index, max_bytes?, cursor?)` | "What did the model see at step N, exactly?" — the byte-exact reconstructed input. |
| `diff_step_contexts(trace_id, step_a, step_b)` | "What changed in the model's context between two steps?" |
| `find_context_anomalies(trace_id, types?)` | "Where did something go wrong?" — truncations, budget overflows, dropped messages, tool-result mismatches. |
| `get_token_accounting(trace_id, group_by?)` | "Where did the tokens/cost go?" — grouped by step, tool, or role. |

Plus 2 resources for browsing clients: `trace://{trace_id}` (metadata + step index) and
`trace://{trace_id}/step/{step_index}` (full, unpaginated step detail).

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

## Schema stability promise

Every trace is written with a pinned `schema_version`. Within a major version, changes are
**additive-only** — new optional fields, never a repurposed or removed one — so any `1.x` reader
can read any `1.x` trace. A breaking change requires a major version bump and a migration script;
servers refuse (not silently coerce) traces with an unsupported major version. See
[docs/DESIGN.md § The schema](docs/DESIGN.md) for the full schema and versioning rationale.

## The fidelity test

The project's one non-negotiable test, CI-blocking forever ([tests/test_fidelity.py](tests/test_fidelity.py)):
capture a running agent's model input, reconstruct it from storage, and assert the reconstructed
`messages` array is **byte-identical**, after canonical JSON serialization, to what an independent
observation point saw actually leave application code. If this test can't pass, the schema has
failed at the one thing it exists to do.

Capture overhead is measured, not assumed: **~0.017ms added per instrumented model call**
(in-process benchmark, wrapper only — see [docs/RESULTS.md](docs/RESULTS.md), reproduce with
`python scripts/bench_overhead.py`).

## Non-goals

- **No dashboards/UI** — MCP clients are the UI; we're not rebuilding Langfuse's trace viewer.
- **No eval features** — evals are a mature, separate category with a different workflow.
- **No multi-agent trace stitching in v1** — single-agent step fidelity first.
- **No prompt management** — that's a build-time concern; this is runtime observability.

See [docs/DESIGN.md § Non-goals](docs/DESIGN.md) for the reasoning behind each.

## Roadmap

- **Anthropic Messages API adapter** — capture today wraps OpenAI-compatible
  `chat.completions`-shaped clients only; a `client.messages.create` adapter for Anthropic's
  wire shape is not yet built. Until it ships, "framework-agnostic" (works regardless of which
  agent framework orchestrates the calls) should not be read as "provider-agnostic."
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
