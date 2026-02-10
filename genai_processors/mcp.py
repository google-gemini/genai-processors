# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MCP (Model Context Protocol) integration for function calling.

This module provides utilities to convert MCP server tools into Python callables
that can be used with the FunctionCalling processor.

Example usage:

```python
from mcp.client import ClientSession

# Connect to an MCP server
session: ClientSession = ...

# Use with FunctionCalling
fc = FunctionCalling(
    model=genai_model,
    # More than one session can be passed to FunctionCalling. Sessions can also
    # be mixed with regular Python callables. The same session should be passed
    # to the model as well.
    fns=[session],
)
```

See genai_processors/examples/chat.py for a complete example.
"""

import base64
from typing import Any, Callable

from genai_processors import content_api
from mcp import types as mcp_types
from mcp.client import session as mcp_session


class McpToolError(Exception):
  """Error raised when an MCP tool call fails."""

  def __init__(self, tool_name: str, content: list[mcp_types.ContentBlock]):
    self.tool_name = tool_name
    self.content = content
    message = f"MCP tool '{tool_name}' returned an error"
    if content:
      text_parts = [
          c.text for c in content if isinstance(c, mcp_types.TextContent)
      ]
      if text_parts:
        message = f"{message}: {' '.join(text_parts)}"
    super().__init__(message)


def _mcp_content_to_part(
    block: mcp_types.ContentBlock,
) -> content_api.ProcessorPart:
  """Convert an MCP content block to a ProcessorPart.

  Args:
    block: An MCP content block.

  Returns:
    A ProcessorPart representing the content.
  """
  if isinstance(block, mcp_types.TextContent):
    return content_api.ProcessorPart(block.text)
  elif isinstance(block, mcp_types.ImageContent):
    data = base64.b64decode(block.data)
    return content_api.ProcessorPart(data, mimetype=block.mimeType)
  elif isinstance(block, mcp_types.AudioContent):
    data = base64.b64decode(block.data)
    return content_api.ProcessorPart(data, mimetype=block.mimeType)
  elif isinstance(block, mcp_types.EmbeddedResource):
    resource = block.resource
    if isinstance(resource, mcp_types.TextResourceContents):
      return content_api.ProcessorPart(
          resource.text, mimetype=resource.mimeType or "text/plain"
      )
    else:
      data = base64.b64decode(resource.blob)
      return content_api.ProcessorPart(data, mimetype=resource.mimeType)
  elif isinstance(block, mcp_types.ResourceLink):
    return content_api.ProcessorPart(str(block.uri))
  else:
    raise ValueError(f"Unsupported MCP content type: {type(block)}")


def create_mcp_tool(
    session: mcp_session.ClientSession,
    tool: mcp_types.Tool,
) -> Callable[..., Any]:
  """Create a callable from an MCP tool for use with FunctionCalling.

  The returned callable has the same name as the MCP tool. Its docstring is the
  tool's description.

  Args:
    session: The MCP client session.
    tool: The MCP tool definition.

  Returns:
    An async callable that invokes the MCP tool. The callable returns a
    ProcessorPart with function_response containing the tool result, or raises
    an McpToolError if the tool call fails.
  """

  async def call_tool(**kwargs: Any) -> content_api.ProcessorPart:
    result = await session.call_tool(tool.name, kwargs)
    if result.isError:
      raise McpToolError(tool.name, result.content)
    return content_api.ProcessorPart.from_function_response(
        name=tool.name,
        response=[_mcp_content_to_part(block) for block in result.content],
    )

  call_tool.__name__ = tool.name
  call_tool.__doc__ = tool.description or f"Calls the {tool.name} MCP tool."

  return call_tool


async def mcp_tools_to_callables(
    session: mcp_session.ClientSession,
    tools: list[mcp_types.Tool] | None = None,
) -> list[Callable[..., Any]]:
  """Convert a list of MCP tools to callables for FunctionCalling.

  Args:
    session: The MCP client session.
    tools: List of MCP tool definitions from session.list_tools(). When not
      provided (i.e. set to None), all tools from the server will be used.

  Returns:
    List of async callables wrapping the MCP tools and returning ProcessorParts
    with function_response containing the tool result. Note that the tool
    results are returned as a list of ProcessorParts representing the
    content blocks returned by the tool.
  """
  if tools is None:
    tools = (await session.list_tools()).tools
  return [create_mcp_tool(session, tool) for tool in tools]
