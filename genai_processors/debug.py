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
"""Utilities to add more debug information to content streams."""

import asyncio
from collections.abc import AsyncIterable
import time

from absl import logging
from genai_processors import content_api
from genai_processors import processor

ProcessorPart = content_api.ProcessorPart


class TTFTSingleStream(processor.Processor):
  """Wraps a processor to provide performance messaging."""

  def __init__(self, message: str, p: processor.Processor):
    """Wraps a processor to provide performance messaging.

      Should only be used for processors that consume their entire input before
      producing output (such as non-streaming or unidirectional/single streaming
      model calls). The TTFT is estimated by waiting first that the inputs
      stream is
      completely sent to the processor (`start` time is then set). When the
      processor outputs its first token, the duration from `start` is then
      reported.

    In a bidirectional streaming setup, the TTFT will not be reported at all.

    Args:
      message: header of the status chunk that will be returned. It is used to
        identify different calls to this function.
      p: processor for which we need to compute ttft. self._message = message
        self._p = p self._start = None self._ttft = None self._model_call_event
        = asyncio.Event() self._model_call_event.clear()
    """
    self._message = message
    self._p = p
    self._start = None
    self._ttft = None
    self._model_call_event = asyncio.Event()
    self._model_call_event.clear()

  def model_call_event(self) -> asyncio.Event:
    """Returns an event that is set when the wrapped processor has all parts.

    The event is set when the wrapped processor has all the input parts and
    is about to start generating the output.

    The event starts in a cleared state when the first part of the input
    stream is yielded. It is also cleared at the end of the wrappedprocessor,
    when all the output parts have been yielded.

    Its default value is unset and this event is set only for a short time
    during the call.

    Returns:
      An event that is set when the model call is started, that is when all the
      input parts have been sent to the wrapped processor.
    """
    return self._model_call_event

  def _ttft_processor(self) -> processor.Processor:

    @processor.processor_function
    async def log_on_close(
        content: AsyncIterable[ProcessorPart],
    ) -> AsyncIterable[ProcessorPart]:
      self._model_call_event.clear()
      async for part in content:
        yield part
      self._start = time.perf_counter()
      self._model_call_event.set()
      logging.info('ttft single stream start time: %s', self._start)

    @processor.processor_function
    async def log_on_first(
        content: AsyncIterable[ProcessorPart],
    ) -> AsyncIterable[ProcessorPart]:
      first_part = True
      async for part in content:
        if first_part:
          if self._start is not None:
            # We have finished going over all parts in the input content.
            # We can compute a proper TTFT.
            duration = time.perf_counter() - self._start
            self._ttft = duration
            self._message += f' TTFT={duration:.2f} seconds'
          else:
            # The input content is still being sent, typical of a bidi setup.
            # TTFT is 0 here by convention. We can't compute it yet.
            self._message += ' TTFT=0 - input stream still open.'
          yield processor.status(ProcessorPart(self._message))
        first_part = False
        yield part

    return log_on_close + self._p + log_on_first

  def ttft(self) -> float | None:
    """Returns the TTFT of the wrapped processor.

    Returns:
      the TTFT of the wrapped processor or None if the processor has not been
      called yet.
    """
    return self._ttft

  async def call(
      self, content: AsyncIterable[ProcessorPart]
  ) -> AsyncIterable[ProcessorPart]:
    async for chunk in self._ttft_processor()(content):
      yield chunk


def debug_string(part: ProcessorPart) -> str:
  return f'role {part.role} substream {part.substream_name} {part!r}'


def log_stream(message: str) -> processor.Processor:
  """Return a function that logs every part of a stream."""

  @processor.processor_function
  async def p(
      content: AsyncIterable[ProcessorPart],
  ) -> AsyncIterable[ProcessorPart]:
    async for part in content:
      logging.info('%s: %s', message, debug_string(part))
      yield part
    logging.info('%s: done', message)

  return p


def print_stream(message: str) -> processor.Processor:
  """Return a function that prints every part of a stream."""

  @processor.processor_function
  async def p(
      content: AsyncIterable[ProcessorPart],
  ) -> AsyncIterable[ProcessorPart]:
    async for part in content:
      print(f'{message}: {debug_string(part)}')
      yield part

  return p


def log_queue(
    message: str, queue: asyncio.Queue[ProcessorPart | None]
) -> asyncio.Queue[ProcessorPart | None]:
  """Return a function that logs every part of a queue."""
  output_queue = asyncio.Queue()

  async def log_and_output():
    while (part := await queue.get()) is not None:
      queue.task_done()
      logging.info('%s: %s', message, debug_string(part))
      output_queue.put_nowait(part)
    output_queue.put_nowait(None)

  processor.create_task(log_and_output())
  return output_queue
