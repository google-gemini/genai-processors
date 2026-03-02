# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
"""A simple turn by turn CLI chat.

Reads a line from the user, sends it to the model, streams the response. While
the chat itself is text-only, one can type a URL and it will be replaced with
its content. This allows sending images and other modalities to the model.
"""

import asyncio
import contextlib
import os
from typing import AsyncIterable, Sequence

from absl import app
from absl import flags
from genai_processors import content_api
from genai_processors import processor
from genai_processors.core import function_calling
from genai_processors.core import pdf
from genai_processors.core import realtime
from genai_processors.core import text
from genai_processors.dev import trace_file
from genai_processors.examples import mcp as mcp_examples
from genai_processors.examples import models
import httpx

_MCP_SERVER = flags.DEFINE_string(
    'mcp_server',
    'demo',
    'Address of the MCP server to use. Use "demo" to use a demo server. Use an'
    ' https address starting with "https://" to use a remote server. Use'
    ' "local:<command>" to use a local server, e.g. local:npx -y'
    ' @modelcontextprotocol/server-filesystem .',
)

_API_KEY_ENV = flags.DEFINE_string(
    'api_key_env',
    'API_KEY',
    'Name of the environment variable containing the API key for remote MCP'
    ' servers. Defined by `export API_KEY=...`.',
)

_API_KEY_HEADER = flags.DEFINE_string(
    'api_key_header',
    'X-Goog-Api-Key',
    'Name of the header containing the API key for remote MCP servers.',
)

_TRACE_DIR = flags.DEFINE_string(
    'trace_dir',
    None,
    'If set, enable tracing and write traces to this directory.',
)

SYSTEM_INSTRUCTIONS = [
    'You are an agent that interacts with the user in a conversation. Make'
    ' the conversation lively and interesting for the user. You can make jokes,'
    ' explain interesting facts, predict what could happen, etc. Respond to the'
    ' user in a few sentences maximum: keep it short and engaging.'
]

USER_PROMPT = '\n> '


def _get_mcp_session():
  """Returns a context manager for an MCP session."""
  if _MCP_SERVER.value == 'demo':
    return mcp_examples.get_demo_mcp_session()
  elif _MCP_SERVER.value.startswith('https://'):
    if _API_KEY_ENV.value:
      api_key = os.environ.get(_API_KEY_ENV.value)
      if not api_key:
        raise ValueError(
            f'API key not found in environment variable: {_API_KEY_ENV.value}'
        )
      api_key_header = {_API_KEY_HEADER.value: api_key}
    else:
      api_key_header = None

    return mcp_examples.get_remote_mcp_session(
        _MCP_SERVER.value, api_key_header
    )
  elif _MCP_SERVER.value.startswith('local:'):
    return mcp_examples.get_local_mcp_session(_MCP_SERVER.value[6:])
  else:
    raise ValueError(
        f'Unsupported MCP server: {_MCP_SERVER.value}. Use one of the'
        ' following:\n- demo\n- https://<address>\n- local:<command>'
    )


class _FetchUrl(processor.PartProcessor):
  """A PartProcessor that fetches the content for a given URL.

  DO NOT USE OUTSIDE OF THIS EXAMPLE: NOT PRODUCTION QUALITY.

  This is an oversimplified version of FetchUrl to allow testing multimodal
  content handling (images, PDFs). It will be replaced with a proper version
  from core.web once it is available.
  """

  def match(self, part: content_api.ProcessorPart) -> bool:
    """This processor matches on WebRequest parts."""
    return content_api.is_dataclass(part.mimetype, text.FetchRequest)

  @processor.yield_exceptions_as_parts
  async def call(
      self, part: content_api.ProcessorPart
  ) -> AsyncIterable[content_api.ProcessorPart]:
    """Gets the content for a given URL."""
    webrequest = part.get_dataclass(text.FetchRequest)
    async with httpx.AsyncClient(follow_redirects=True) as client:
      response = await client.get(webrequest.url)
      response.raise_for_status()

    yield content_api.ProcessorPart(
        response.content, mimetype=response.headers.get('content-type')
    )


async def run_chat() -> None:
  """Runs a simple turn by turn chat."""

  # The easiest way to track context between turns is to use Gemini Live API
  # genai_procesors.core.live_model.LiveProcessor. We then can send user
  # turns in and it will yield model responses.
  #
  # Here we take a more flexible but slightly more complex approach and use
  # genai_procesors.core.realtime.LiveProcessor - a client-side version of the
  # Live API. It wraps any turn-based model and provides a bidirectional
  # interface. It also supports customizable context compression.

  # See models.py for the list of supported models and flags used to select one.
  async with contextlib.AsyncExitStack() as stack:
    mcp_session = await stack.enter_async_context(_get_mcp_session())
    if _TRACE_DIR.value:
      await stack.enter_async_context(
          trace_file.SyncFileTrace(trace_dir=_TRACE_DIR.value, name='chat')
      )

    model = models.turn_based_model(
        system_instruction=SYSTEM_INSTRUCTIONS,
        disable_automatic_function_calling=True,
        tools=[mcp_session],
    )
    model = function_calling.FunctionCalling(
        model=realtime.LiveModelProcessor(model),
        fns=[mcp_session],
        is_bidi_model=True,
    )

    # Give the agent the ability to download multimodal content.
    chat_agent = text.UrlExtractor() + _FetchUrl() + pdf.PDFExtract() + model

    print('Welcome to the GenAI Processor Chat! Ask me anything.')
    print('You can also ask questions about images or PDFs by providing a URL.')
    print('For example:')
    print(
        ' - Describe the main points from the '
        ' https://storage.googleapis.com/gweb-developer-goog-blog-assets/images/gemini_2-5_ga_family_1-1__dark.original.png'
        ' diagram.'
    )
    print(' - Summarize https://arxiv.org/pdf/2312.11805')
    print('Press Ctrl + D to exit.')

    print(USER_PROMPT, end='')
    await text.terminal_output(
        chat_agent(text.terminal_input(prompt=USER_PROMPT)), prompt=USER_PROMPT
    )


def main(argv: Sequence[str]):
  del argv  # Unused.
  asyncio.run(run_chat())


if __name__ == '__main__':
  app.run(main)
