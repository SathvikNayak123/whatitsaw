# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/). Schema versioning has its own, stricter promise —
see [docs/DESIGN.md § The schema](docs/DESIGN.md#the-schema): schema `1.x` is additive-only, and
any 1.x reader can read any 1.x trace.

## [0.2.0] - 2026-07-15

### Added

- **Anthropic Messages API capture path** (`TraceRecorder.wrap_anthropic_client`, wraps
  `client.messages.create`): duck-typed like the existing OpenAI-compatible wrapper, no hard
  dependency on the `anthropic` package. `system` is captured in `params` (a top-level wire param
  on Anthropic's API, not a message), keeping the `messages` array byte-exact. Anthropic's
  `input_tokens`/`output_tokens`/`cache_read_input_tokens`/`cache_creation_input_tokens` usage
  shape is mapped into the schema's provider-neutral `TokenCounts`.
- A second, independent capture-fidelity gate for this path
  (`tests/test_fidelity_anthropic.py`), covering the same fixture matrix as the OpenAI path: plain
  text, `tool_use`/`tool_result` blocks, multi-block mixed content, and image content blocks.
- `examples/anthropic_agent.py`, an Anthropic-path counterpart to
  `examples/deep_research_agent.py`.

### Changed

- The Anthropic Messages API adapter is no longer a roadmap item — "framework-agnostic" now
  covers both OpenAI-compatible and Anthropic capture paths. See docs/DESIGN.md § Capture
  mechanism and § Capture-fidelity acceptance test.

## [0.1.0] - 2026-07-05

Initial release.

### Added

- Versioned trace/context schema (`schema_version: "1.0"`) as pydantic models, with a checked-in
  JSON Schema export at `schema/v1.0.json`.
- Instrumentation SDK (`ctx_capture.capture.TraceRecorder`): a wrapper for OpenAI-compatible
  `client.chat.completions.create` calls and a decorator for tool functions, capturing the exact
  message array sent, provider response, token counts, and both `result_as_returned` and
  `result_as_inserted` for tool calls (so truncation is detectable, not just guessed at).
- SQLite storage backend (`ctx_capture.storage.SQLiteTraceRepository`) behind a `TraceRepository`
  interface a Postgres backend can implement later without touching callers.
- MCP server (`ctx_capture.mcp.create_server`) exposing 5 tools — `list_traces`,
  `get_step_context`, `diff_step_contexts`, `find_context_anomalies`, `get_token_accounting` —
  and 2 resource templates (`trace://{trace_id}`, `trace://{trace_id}/step/{step_index}`), over
  stdio and streamable-HTTP transports, with bearer-token auth on HTTP.
- v1 anomaly detection: `truncation`, `tool_result_mismatch`, `budget_overflow`,
  `dropped_message`.
- Byte-size pagination for oversized tool responses (default 50KB cap, configurable), so no tool
  response is ever silently unbounded.
- The capture-fidelity acceptance test (CI-blocking): captured `messages` for a step must be
  byte-identical, after canonical JSON serialization, to what actually left application code.
- Capture-overhead benchmark: ~0.017ms added per instrumented call (in-process, wrapper only —
  see [docs/RESULTS.md](docs/RESULTS.md), reproduce with `python scripts/bench_overhead.py`).
- `ctx-capture` console-script entry point (`python -m ctx_capture.mcp` / `uvx ctx-capture`).

### Known limitations (tracked for a future release)

- No Postgres implementation yet (interface is ready; see `TraceRepository`).
- No OTel span-ingestion adapter yet (SDK-first capture only — see docs/DESIGN.md § Capture
  mechanism for why this is the primary path, not a gap).
- No redaction hook implemented yet (documented as an opt-in mitigation in docs/DESIGN.md §
  Risks; not yet built — treat captured trace data as sensitive by default until it is).
- HTTP transport auth is a static bearer token; OAuth is roadmap, not implemented.
