"""Per-tool contract: every tool declares outputSchema, and every successful call's
structuredContent validates against it. See docs/DESIGN.md "MCP surface spec".
"""

from __future__ import annotations

import jsonschema
import pytest

from tests.mcp.helpers import mcp_client

TOOL_ARGS = {
    "list_traces": lambda trace_id: {},
    "get_step_context": lambda trace_id: {"trace_id": trace_id, "step_index": 0},
    "diff_step_contexts": lambda trace_id: {"trace_id": trace_id, "step_a": 0, "step_b": 1},
    "find_context_anomalies": lambda trace_id: {"trace_id": trace_id},
    "get_token_accounting": lambda trace_id: {"trace_id": trace_id},
}


@pytest.mark.parametrize("tool_name", list(TOOL_ARGS))
async def test_tool_declares_output_schema(mcp_server, tool_name):
    async with mcp_client(mcp_server) as session:
        tools = (await session.list_tools()).tools
        tool = next(t for t in tools if t.name == tool_name)
        assert tool.outputSchema is not None, f"{tool_name} has no outputSchema"


@pytest.mark.parametrize("tool_name", list(TOOL_ARGS))
async def test_tool_structured_content_conforms_to_output_schema(mcp_server, seeded_trace_id, tool_name):
    async with mcp_client(mcp_server) as session:
        tools = (await session.list_tools()).tools
        tool = next(t for t in tools if t.name == tool_name)

        result = await session.call_tool(tool_name, TOOL_ARGS[tool_name](seeded_trace_id))

        assert result.isError is False, f"{tool_name} call failed: {result.content}"
        assert result.structuredContent is not None, f"{tool_name} returned no structuredContent"
        jsonschema.validate(result.structuredContent, tool.outputSchema)


async def test_exactly_five_tools_registered(mcp_server):
    async with mcp_client(mcp_server) as session:
        tools = (await session.list_tools()).tools
        names = {t.name for t in tools}
        assert names == set(TOOL_ARGS), "tool set drifted from the designed 5 (see docs/DESIGN.md)"


async def test_trace_resources_registered(mcp_server):
    async with mcp_client(mcp_server) as session:
        templates = (await session.list_resource_templates()).resourceTemplates
        uri_templates = {t.uriTemplate for t in templates}
        assert uri_templates == {"trace://{trace_id}", "trace://{trace_id}/step/{step_index}"}
