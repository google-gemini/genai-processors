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
"""Utilities for fetching documents mentioned in the part stream.

Referencing URLs of documents (web pages, images, PDFs) in a prompt is a
convenient way to provide rich context for a model. While registering `http.get`
as a tool is a more flexible and robust approach, it requires an extra model
call and a round trip for each document loaded. This approach offers a more
hardwired but faster alternative.

We split the responsibility for fetching documents and deciding what needs
fetching. A special `genai_processors.core.text.FetchRequest` part must be used
to explicitly reference the document to be fetched. Then `UrlFetch` processor
would replace such FetchRequest Parts with the actual content.

While it is very convenient to just mention URL as text in the prompt, it
becomes easy to trigger the fetch unintentionally and can even be dangerous. So
it should be applied closer to the UI where user journeys are more well defined.
For example parsing URLs directly pasted into a chat interface is probably
fine but recursively following URLs from uncontrolled sources is dangerous. For
extra safety you may want to require the URL be on its own line.
`genai_processors.core.text.UrlExtractor` is a processor for the task.

This process can be refined further: e.g. one can use a fast model
(gemini-flash-lite or gemma-nano) to decide whether the URL should be fetched
before passing the prompt to a larger LLM. This way we can reduce latency by
making decisions fast and fetching multiple documents in parallel.

You can also consider using an alternative implementation from
https://github.com/mbeacom/genai-processors-url-fetch/ which has additional
security features and supports markdown.
"""
from collections.abc import AsyncIterable

from genai_processors import content_api
from genai_processors import mime_types
from genai_processors import processor
from genai_processors.core import text
import httpx


_HEADER_PART = 'Fetch result for URL: '


class UrlFetch(processor.PartProcessor):
  """A processor that fetches documents by URLs.

  This processor replaces genai_processors.core.text.FetchRequest with the
  referenced content. It ignores anything that is not a FetchRequest.

  It is recommended to chain genai_processors.core.text.HtmlCleaner after
  UrlFetch to simplify the HTML pages before sending them to a model.
  """

  def __init__(
      self,
      timeout_seconds: int = 10,
  ):
    """Initializes the UrlFetch processor.

    Args:
      timeout_seconds: The timeout in seconds for the HTTP request. Default set
        to 10 seconds.
    """

    self._client = httpx.AsyncClient(
        follow_redirects=True,
        timeout=timeout_seconds,
    )

  def match(self, part: content_api.ProcessorPart) -> bool:
    return mime_types.is_dataclass(part.mimetype, text.FetchRequest)

  @processor.yield_exceptions_as_parts
  async def call(
      self, part: content_api.ProcessorPart
  ) -> AsyncIterable[content_api.ProcessorPartTypes]:
    fetch_request = part.get_dataclass(text.FetchRequest)
    url = fetch_request.url
    yield _HEADER_PART + url + '\n'
    # Set stream=True to enable streaming the response body
    async with self._client.stream('GET', url, timeout=10) as response:
      # Raise an exception for bad status codes (4xx or 5xx)
      response.raise_for_status()

      # Iterate over the response chunks
      html_content = []
      async for chunk in response.aiter_text():
        html_content.append(chunk)
      yield content_api.ProcessorPart(
          ''.join(html_content), mimetype=mime_types.TEXT_HTML
      )
