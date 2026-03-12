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

"""Appends a preamble to a stream."""

from collections.abc import AsyncIterable, Callable

from genai_processors import content_api
from genai_processors import processor

PreambleFactory = Callable[[], content_api.ProcessorPartTypes]


class Preamble(processor.Processor):
  """Prepends a preamble to the full content."""

  def __init__(
      self,
      *,
      content: content_api.ProcessorContentTypes | None = None,
      content_factory: PreambleFactory | None = None,
  ):
    """Constructs a Preamble processor.

    Args:
      content: content to prepend.
      content_factory: function for returning a content given no input. This is
        helpful for when contents are not fully known on __init__, e.g. if they
        depend on the user or time of the request.

    Raises:
      ValueError if both `content` and `content_factory` are provided.
    """
    if content is not None and content_factory is not None:
      raise ValueError(
          'Only one of `content` and `content_factory` must be provided.'
      )

    self._content = (
        None if content is None else content_api.ProcessorContent(content)
    )
    self._content_factory = content_factory

  async def call(
      self, content: processor.ProcessorStream
  ) -> AsyncIterable[content_api.ProcessorPartTypes]:
    if self._content is not None:
      for content_part in self._content:
        yield content_part
    elif self._content_factory is not None:
      for content_part in content_api.ProcessorContent(self._content_factory()):
        yield content_part

    async for content_part in content:
      yield content_part


class Suffix(Preamble):
  """Appends a suffix to the full content."""

  async def call(
      self, content: AsyncIterable[content_api.ProcessorPart]
  ) -> AsyncIterable[content_api.ProcessorPart]:
    async for content_part in content:
      yield content_part

    if self._content is not None:
      for content_part in self._content:
        yield content_part
    elif self._content_factory is not None:
      for content_part in content_api.ProcessorContent(self._content_factory()):
        yield content_part
