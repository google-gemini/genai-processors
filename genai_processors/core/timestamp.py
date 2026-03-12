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
"""Processor adding timestamps to content."""

import functools
import time
from typing import AsyncIterable

from genai_processors import content_api
from genai_processors import processor


def to_timestamp(seconds: float | int, with_ms: bool = False) -> str:
  """Returns a timestamp as a string to indicate the time of an event."""
  if with_ms:
    seconds = round(seconds * 1000) / 1000
  else:
    seconds = round(seconds)
  minutes = seconds // 60
  seconds = seconds % 60
  if with_ms:
    return f'{minutes:02.0f}:{seconds:06.3f}'
  else:
    return f'{minutes:02.0f}:{seconds:02.0f}'


async def _add_timestamps(
    content: processor.ProcessorStream,
    with_ms: bool = False,
    substream_name: str | None = None,
) -> AsyncIterable[content_api.ProcessorPartTypes]:
  """Adds timestamps to image chunks (to be used by default when streaming)."""
  start = time.perf_counter()
  async for part in content:
    if content_api.is_image(part.mimetype):
      yield content_api.ProcessorPart(
          to_timestamp(time.perf_counter() - start, with_ms=with_ms),
          substream_name=substream_name,
          # Do not trigger a model generate call when the timestamp is added.
          metadata={'turn_complete': False},
      )
    yield part


def add_timestamps(
    with_ms: bool = False,
    substream_name: str | None = None,
) -> processor.Processor:
  """Adds timestamps to image chunks.

  By default the timestamps are added with the format `mm:ss` where
  `mm` is the number of minutes, `ss` is the number of seconds.

  Args:
    with_ms: Whether to add milliseconds to the timestamp. When `True`, the
      timestamp is added with the format `mm:ss.SSS` where `SSS` is the number
      of milliseconds.
    substream_name: The substream name to use for the timestamps.

  Returns:
    A processor that adds timestamps after each image chunk.
  """
  if substream_name is None:
    substream_name = ''
  return processor.processor_function(
      functools.partial(
          _add_timestamps,
          with_ms=with_ms,
          substream_name=substream_name,
      )
  )
