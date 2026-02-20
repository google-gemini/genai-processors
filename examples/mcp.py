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
# ==============================================================================
"""MCP server for examples.

This modules provides several MCP servers that can be used in CLI or examples:

*  `get_demo_mcp_session`: a self-contained MCP server. Good for tests, but
   doesn't do anything useful.

*  `get_local_mcp_session`: launches a given binary with an MCP server locally.

*  `get_remote_mcp_session`: connects to a remote MCP server. Will likely
   require an API key.
"""

from collections.abc import AsyncGenerator
import contextlib
import shlex
import httpx
import mcp
from mcp.client import session as mcp_session
from mcp.client import stdio
from mcp.client import streamable_http
from mcp.server import fastmcp
from mcp.shared import memory as mcp_memory


def _create_demo_server() -> fastmcp.FastMCP:
  """Create a demo MCP server with simple example tools."""
  server = fastmcp.FastMCP(name='Demo MCP Server')

  @server.tool()
  def add(a: int, b: int) -> int:
    """Add two numbers together.

    Args:
      a: The first number to add.
      b: The second number to add.

    Returns:
      The sum of the two numbers.
    """
    return a + b

  @server.tool()
  def multiply(a: int, b: int) -> int:
    """Multiply two numbers together.

    Args:
      a: The first number to multiply.
      b: The second number to multiply.

    Returns:
      The product of the two numbers.
    """
    return a * b

  @server.tool()
  def greet(name: str) -> str:
    """Greet someone by name.

    Args:
      name: The name of the person to greet.

    Returns:
      A greeting message.
    """
    return f'Hello, {name}! Nice to meet you.'

  @server.tool()
  def get_weather(city: str) -> str:
    """Get the current weather for a city (demo data).

    Args:
      city: The name of the city to get the weather for.

    Returns:
      The current weather for the specified city.
    """
    weather_data = {
        'london': 'Cloudy, 15°C',
        'paris': 'Sunny, 22°C',
        'tokyo': 'Rainy, 18°C',
        'new york': 'Partly cloudy, 20°C',
    }
    return weather_data.get(
        city.lower(), f'Weather data not available for {city}'
    )

  return server


@contextlib.asynccontextmanager
async def get_demo_mcp_session() -> (
    AsyncGenerator[mcp_session.ClientSession, None]
):
  """Returns a demo MCP server session.

  Returns:
    A demo MCP server session with a few example tools.
  """
  server = _create_demo_server()
  async with mcp_memory.create_connected_server_and_client_session(
      server
  ) as session:
    yield session


@contextlib.asynccontextmanager
async def get_local_mcp_session(
    server_command: str = 'npx -y @modelcontextprotocol/server-filesystem .',
) -> AsyncGenerator[mcp_session.ClientSession, None]:
  """Returns an external MCP server session.

  Args:
    server_command: The command to run the local MCP server.

  Returns:
    A local MCP server session.
  """
  parts = shlex.split(server_command)
  server_params = stdio.StdioServerParameters(command=parts[0], args=parts[1:])
  print(f'Connecting to MCP server: {server_command}')
  async with stdio.stdio_client(server_params) as (read, write):
    async with mcp_session.ClientSession(read, write) as session:
      await session.initialize()
      yield session


@contextlib.asynccontextmanager
async def get_remote_mcp_session(
    server_address: str = 'https://mapstools.googleapis.com/mcp',
    api_key_header: dict[str, str] | None = None,
) -> AsyncGenerator[mcp_session.ClientSession, None]:
  """Returns a remote MCP server session.

  Args:
    server_address: The address of the remote MCP server.
    api_key_header: dictionary of headers containing the API key for the server.
      E.g. { 'X-Goog-Api-Key': <google_api_key> }.

  Returns:
    A remote MCP server session.
  """
  headers = {
      'Content-Type': 'application/json',
      'Accept': 'application/json, text/event-stream',
  }
  if api_key_header:
    headers.update(api_key_header)
  async with httpx.AsyncClient(
      headers=headers,
      timeout=httpx.Timeout(30, read=60),
  ) as client:
    async with streamable_http.streamable_http_client(
        server_address, http_client=client
    ) as (read, write, session_id):
      async with mcp.ClientSession(read, write) as session:
        print(f'Connecting to MCP server: {server_address}')
        await session.initialize()
        yield session
