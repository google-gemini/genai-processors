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
"""Rate limiter for audio output."""

import asyncio
import logging
import math
import time
from typing import AsyncIterable, Iterable, Optional
from absl import logging
from genai_processors import content_api
from genai_processors import context as context_lib
from genai_processors import processor

ProcessorPart = content_api.ProcessorPart
ProcessorPartTypes = content_api.ProcessorPartTypes

# Maximum audio chunk/part duration in seconds.
MAX_AUDIO_PART_SEC = 0.05

# Buffer in seconds to avoid the audio being cut off.
IN_FLIGHT_AUDIO_BUFFER_SEC = 0.05


def _audio_duration(audio_data: bytes, sample_rate: int) -> float:
  """Returns the duration of the audio data in seconds."""
  # 2 bytes per sample (16bits)
  return len(audio_data) / (2 * sample_rate)


def split_audio(
    audio_data: bytes,
    sample_rate: int,
    max_duration_sec: float = MAX_AUDIO_PART_SEC,
) -> Iterable[bytes]:
  """Splits audio data into chunks of max_duration_sec."""
  audio_data_length = len(audio_data)
  # 2 bytes per sample (16bits)
  chunk_target_bytes = int(max_duration_sec * sample_rate * 2)
  num_chunks = math.ceil(audio_data_length / chunk_target_bytes)
  for i in range(num_chunks):
    start = i * chunk_target_bytes
    end = min((i + 1) * chunk_target_bytes, audio_data_length)
    if start >= end:
      continue
    yield audio_data[start:end]


class RateLimitAudio(processor.Processor):
  """Splits and rate-limits the input audio parts for streaming audio output.

  Gemini API clients are expected to play streaming audio content to the user
  in its natural playback speed. As all audio parts are streamed at once, the
  client needs to stop playing back the audio when the user interrupts it.

  This processor does three things to address that:

    * Parts of potentially long streaming audio content are split into
      sub-parts of no more than 200 milliseconds. (Non-streaming audio is left
      alone, and count as "other parts" for the purposes of this processor.)
    * Parts are yielded from this processor at the rate of their natural
      playing speed, to put a reasonably tight limit on the amount of audio
      buffered beyond the agent's control.
    * Other parts are passed through unchanged. Debug/status parts are
      passed through as soon as possible, overtaking audio if needed.
  """

  def __init__(self, sample_rate: int, delay_other_parts: bool = True):
    """Initializes the rate limiter.

    Args:
      sample_rate: The sample rate of the audio. A typical value is 24000
        (24KHz)
      delay_other_parts: If true, other parts will be delayed until the audio is
        played out. If false, other parts will be passed through as soon as
        possible, overtaking audio if needed.
    """
    self._sample_rate = sample_rate
    self._delay_other_parts = delay_other_parts

  async def call(
      self, content: processor.ProcessorStream
  ) -> AsyncIterable[ProcessorPartTypes]:
    """Rate limits audio output."""
    # Most inputs queue here. When full, the fast-tracking of status/debug
    # chunks starts to block, so let's be generous with the queue size.
    audio_queue = asyncio.Queue[Optional[ProcessorPart]](10_000)
    # Delays in outputting from this queue distort the time estimations for
    # audio sub-chunks, so let's bound its size tightly.
    output_queue = asyncio.Queue[Optional[ProcessorPart]](3)

    async def consume_content():
      async for part in content:
        if content_api.is_audio(part.mimetype):
          # Split the audio into small parts so that when we interrupt between
          # them, we don't have to wait too long before interrupting.
          if (
              part.part.inline_data is not None
              and _audio_duration(part.bytes, self._sample_rate)
              > 2 * MAX_AUDIO_PART_SEC
          ):
            for sub_part in split_audio(
                part.part.inline_data.data, self._sample_rate
            ):
              audio_queue.put_nowait(
                  ProcessorPart(sub_part, mimetype=part.mimetype)
              )
          else:
            audio_queue.put_nowait(part)
        elif part.get_metadata('interrupted'):
          logging.debug(
              '%s - Interrupted - flush audio queue', time.perf_counter()
          )
          # Flush the audio queue - stop rate limiting audio asap.
          while not audio_queue.empty():
            audio_queue.get_nowait()
          self._audio_duration = 0.0
          audio_queue.put_nowait(part)
        elif (
            not self._delay_other_parts
            or part.substream_name in context_lib.get_reserved_substreams()
        ):
          await output_queue.put(part)
          await asyncio.sleep(0)  # Allow `yield` from output_queue to run
        else:
          await audio_queue.put(part)
      await audio_queue.put(None)

    async def consume_audio():
      start_playing_time = self._perf_counter() - 3600  # 1h back.
      while part := await audio_queue.get():
        if content_api.is_audio(part.mimetype):
          start_playing_time = max(
              self._perf_counter() - 0.05, start_playing_time
          )
          # Remove the 0.05 seconds delay to avoid the audio being cut off
          sleep_sec = max(0, start_playing_time - self._perf_counter())
          if sleep_sec > 1e-3:
            await self._asyncio_sleep(sleep_sec)
          await output_queue.put(part)
          await asyncio.sleep(0)  # Allow `yield` from output_queue to run
          start_playing_time += _audio_duration(
              part.part.inline_data.data, self._sample_rate
          )
        else:
          # Wait for the audio to be played out before passing on to the next
          # non-audio part.
          await self._asyncio_sleep(
              max(0, start_playing_time - self._perf_counter())
          )
          await output_queue.put(part)
      await output_queue.put(None)

    consume_audio_task = processor.create_task(consume_audio())
    consume_content_task = processor.create_task(consume_content())
    while part := await output_queue.get():
      yield part
    consume_content_task.cancel()
    consume_audio_task.cancel()

  # The following wrappers allow unit-tests to mock out walltime.
  def _perf_counter(self):
    return time.perf_counter()

  async def _asyncio_sleep(self, delay: float) -> None:
    await asyncio.sleep(delay)
