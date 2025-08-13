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
from typing import AsyncIterable, Sequence

from absl import app
from genai_processors import content_api
from genai_processors import context
from genai_processors import processor
from genai_processors.core import pdf
from genai_processors.core import realtime
from genai_processors.core import text
from genai_processors.examples import models
import httpx


SYSTEM_INSTRUCTIONS = [
    'You are an agent that interacts with the user in a conversation. Make'
    ' the conversation lively and interesting for the user. You can make jokes,'
    ' explain interesting facts, predict what could happen, etc. Respond to the'
    ' user in a few sentences maximum: keep it short and engaging.'
]

USER_PROMPT = '\n> '


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
  model = models.turn_based_model(system_instruction=SYSTEM_INSTRUCTIONS)
  chat_agent = realtime.LiveModelProcessor(model)

  # Give the agent the ability to download multimodal content.
  chat_agent = text.UrlExtractor() + _FetchUrl() + pdf.PDFExtract() + chat_agent

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

  async for part in chat_agent(text.terminal_input()):
    # Filter out status messages.
    if context.is_reserved_substream(part.substream_name):
      continue

    print(part.text, end='')

    if content_api.is_end_of_turn(part):
      print(USER_PROMPT, end='')


def main(argv: Sequence[str]):
  del argv  # Unused.
  asyncio.run(run_chat())


if __name__ == '__main__':
  app.run(main)
