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
"""Windowing processors for working with long or realtime streams.

When dealing with long streams (e.g. video file) or doing contigous inference on
realtime streams one may need to slice them in-to prompts to be sent to an LLM.
This module contains utilities for that.

Also see genai_processors.core.realtime.LiveProcessor for bidirectional
"Live" chat implementation.
"""

import asyncio
import collections
from collections.abc import AsyncIterable, Awaitable
import contextlib
import time
from typing import Callable

from genai_processors import content_api
from genai_processors import processor
from genai_processors import streams


ProcessorPart = content_api.ProcessorPart
Processor = processor.Processor


class RollingPrompt:
  """Rolling prompt (aka iterator of prompts) for conversation processors.

  NOTE: This is a low level utility. Prefer using realtime.LiveProcessor or
  window.Window which provide higher level abstractions.

  This class acts as a buffer and organizer for a continuous, "infinite" stream
  of multimodal parts (e.g. from a real-time camera feed). It transforms this
  stream into finite, time-segmented prompts suitable for turn-based models.

  Adding Parts:
    add_part(part): Immediately appends a multimodal part to the current prompt.
    stash_part(part): Temporarily stores a part to be appended later. This is
      useful when it's not the client's "turn" to send parts (e.g. while a
      model is generating a response) and you want to delay its inclusion in the
      active prompt.
    apply_stash(): Appends all currently stashed parts to the active prompt.

  Accessing prompts:
    pending() -> `AsyncIterable[ProcessorPart]`: Returns an asynchronous
      iterator representing the current prompt state. This iterator first yields
      the compressed conversation history, then *continues* to stream any new
      parts added via `add_part` or `apply_stash`. This streaming behavior
      allows models to begin processing the current turn *before it is fully
      completed*, significantly minimizing Time To First Token (TTFT).
   finalize_pending(): This method should be called when it is time for the
     underlying model to make the turn.
     * The `AsyncIterable` previously returned by `pending()` will cease
       yielding parts after all current parts have been delivered, signaling the
       model to generate its response.
     * The history compression is applied to accumulated conversation (which
       does not include the ongoing model response, as it has not arrived yet).
     * A new `pending` iterator is then implicitly created for the *next* turn.
     * The compressed history is written to this new iterator, and any parts
       added subsequently will be directed to this new turn's pendig prompt.

  Consider a scenario where a model is actively generating a response, but the
  real-time stream (e.g. a video feed) continues to produce new parts (e.g.
  image frames). Directly using `add_part` for these new frames would
  incorrectly place them *within* or before the ongoing model response, falsely
  suggesting the model considered these new images during its computation for
  the *current* response.

  To address this, such parts coming from the application should be added using
  `stash_part` and appended to the prompt after the model turn is completed
  using `apply_stash`.
  """

  def __init__(
      self,
      *,
      duration_prompt_sec: float | None = None,
      compress_history: (
          Callable[[collections.deque[ProcessorPart]], Awaitable[None]] | None
      ) = None,
  ):
    """Initializes the rolling prompt.

    Args:
      duration_prompt_sec: the length of the prompt in terms of the time when
        the parts were added: we consider any part added after now -
        duration_prompt_sec. Set to None to keep all incoming parts. Set to 10
        minutes by default.
      compress_history: A callable that takes a deque of ProcessorPart and
        modifies it in place to compress the history.
    """
    # Current prompt as a queue.
    self._pending = asyncio.Queue[ProcessorPart | None]()
    # Main prompt content used to build the next prompt and the prompt queues.
    self._conversation_history = collections.deque(maxlen=10_000)
    # stashed parts to be added to the prompt later.
    self._stash: list[ProcessorPart] = []

    async def _noop_compress_history(
        _: collections.deque[ProcessorPart],
    ) -> None:
      pass

    self._compress_history = compress_history or _noop_compress_history
    if duration_prompt_sec:
      if compress_history:
        raise ValueError(
            'compress_history and duration_prompt_sec can not be set'
            ' simultaneously.'
        )

      self._compress_history = drop_old_parts(duration_prompt_sec)

  def add_part(
      self,
      part: ProcessorPart,
  ) -> None:
    """Adds a part to the current prompt."""
    part.metadata.setdefault('capture_time', time.perf_counter())
    self._conversation_history.append(part)
    self._pending.put_nowait(part)

  def stash_part(self, part: ProcessorPart):
    """Stashes the part to be appended to the prompt later using `apply_stash`.

    If it is not our turn to send parts (e.g. model is currently generating),
    we can stash them and append to the prompt when we get the turn (after the
    model generation is done).

    Args:
      part: The part to stash.
    """
    self._stash.append(part)

  def apply_stash(self):
    """Append all parts from the stash to the prompt."""
    for part in self._stash:
      self.add_part(part)
    self._stash = []

  def pending(self) -> content_api.ContentStream:
    """Returns the current pending prompt.

    Note that the same AsyncIterable is returned unless `finalize_pending` is
    called which creates a new pending prompt. So consuming Parts from it will
    affect all callers of `pending`.
    """
    return content_api.ContentStream(
        parts_generator=streams.dequeue(self._pending)
    )

  async def finalize_pending(self) -> None:
    """Close the current pending prompt and starts a new one.

    * The iterator previously returned from `pending` will stop after all Parts
      added so far.
    * Then history compression is applied.
    * At last, a new `pending` iterator is created and the compressed history
      is written to it.
    * Any parts added afterwards will go to the new iterator.
    """
    # Close the current queue.
    self._pending.put_nowait(None)
    # And create a new one.
    self._pending = asyncio.Queue[ProcessorPart | None]()
    await self._compress_history(self._conversation_history)
    for part in self._conversation_history:
      self._pending.put_nowait(part)


def drop_old_parts(
    age_sec: float,
) -> Callable[[collections.deque[ProcessorPart]], Awaitable[None]]:
  """A history compression policy that drops parts older than `age_sec`."""

  async def policy(history: collections.deque[ProcessorPart]):
    time_cut_point = time.perf_counter() - age_sec
    while history and history[0].metadata['capture_time'] < time_cut_point:
      history.popleft()

  return policy


def keep_last_n_turns(
    turns_to_keep: int,
) -> Callable[[collections.deque[ProcessorPart]], Awaitable[None]]:
  """Returns a history compression policy that keeps turns_to_keep last turns.

  Args:
    turns_to_keep: How many turns to keep, including the turn that comes after
      the policy is applied.
  """

  async def policy(history: collections.deque[ProcessorPart]):
    turns = 0
    for part in history:
      if content_api.is_end_of_turn(part):
        turns += 1

    turns_to_drop = turns + 1 - turns_to_keep

    while turns_to_drop > 0:
      if content_api.is_end_of_turn(history[0]):
        turns_to_drop -= 1
      history.popleft()

  return policy


class Window(Processor):
  """Invokes the given processor on a sliding window across incoming content.

  Accumulates incoming content. When `content_api.is_end_of_turn(part)` is
  encountered, `window_processor` is invoked with the accumulated content.

  NOTE: You can put a processor before the Window which will mark Parts with
  `part.metadata['turn_complete'] = true` or filter-out unwanted parts. For
  example when detecting events in a video stream you may leave only image parts
  and mark each as turn_complete.

  The output of `window_processor` is propagated to the Window output, but
  unlike `LiveProcessor` it is not kept in the prompt of `window_processor` and
  is not visible to the consecutive `window_processor` invocations.

  Many instances of `window_processor` may be run concurrently, but their output
  will be yielded in order.

  After `window_processor` is invoked, the accumulated prompt is compressed with
  `compress_history`. E.g. one can keep the last N turns or last M seconds or
  use an LLM to summarize the context. Note that new Parts added after the last
  turn won't be compressed. This allows handling each window in a streaming
  fashion.

  NOTE: If more sophisticated compression is needed (e.g. you want to leave
  image frames with increasing backoff: now, -1s, -5s, -15s, -30s) this can be
  done on per-window basis inside of `window_processor`. Similarly one can use
  Preamble/Postamble processors or set a system prompt to inject additional
  instructions in the `window_processor`.
  """

  def __init__(
      self,
      window_processor: Processor,
      compress_history: (
          Callable[[collections.deque[ProcessorPart]], Awaitable[None]] | None
      ) = None,
      max_concurrency: int = 0,
      stride: int = 1,
  ):
    """Initializes the window processor.

    Args:
      window_processor: The processor to invoke on the window.
      compress_history: A callable that takes a deque of ProcessorPart and
        modifies it in place to compress the history.
      max_concurrency: The maximum number of concurrent window_processor
        invocations. If 0 or less, concurrency is unlimited.
      stride: Only process every `stride` window, skipping the rest. Must be
        >= 1.
    """
    if stride < 1:
      raise ValueError('stride must be >= 1')
    self._window_processor = window_processor
    self._compress_history = compress_history
    self._stride = stride
    if max_concurrency > 0:
      self._semaphore = asyncio.Semaphore(max_concurrency)
    else:
      self._semaphore = contextlib.nullcontext()

  async def _consume_content(
      self,
      content: AsyncIterable[ProcessorPart],
      window_results_queue: asyncio.Queue[
          asyncio.Queue[ProcessorPart | None] | None
      ],
      rolling_prompt: RollingPrompt,
  ) -> None:
    """The main loop that creates and manages window processing tasks."""

    window_index = 0

    async def _create_window_task():
      prompt_for_window = rolling_prompt.pending()
      single_window_output_queue = asyncio.Queue[ProcessorPart | None]()
      window_results_queue.put_nowait(single_window_output_queue)

      async def run_window_processor():
        async with self._semaphore:
          try:
            async for part in self._window_processor(prompt_for_window):
              await single_window_output_queue.put(part)
          finally:
            await single_window_output_queue.put(
                content_api.ProcessorPart.end_of_turn()
            )
            await single_window_output_queue.put(None)

      processor.create_task(run_window_processor())

    await _create_window_task()
    window_index += 1

    async for part in content:
      rolling_prompt.add_part(part)
      if content_api.is_end_of_turn(part):
        await rolling_prompt.finalize_pending()
        if window_index % self._stride == 0:
          await _create_window_task()
        window_index += 1

    await rolling_prompt.finalize_pending()

    await window_results_queue.put(None)  # Signal end of all windows.

  async def call(
      self, content: AsyncIterable[ProcessorPart]
  ) -> AsyncIterable[ProcessorPart]:
    """The main entry point for the processor."""
    rolling_prompt = RollingPrompt(compress_history=self._compress_history)
    window_results_queue = asyncio.Queue[
        asyncio.Queue[ProcessorPart | None] | None
    ]()

    consume_content_task = processor.create_task(
        self._consume_content(
            content,
            window_results_queue,
            rolling_prompt,
        )
    )
    while single_window_output_queue := await window_results_queue.get():
      while part := await single_window_output_queue.get():
        yield part
    await consume_content_task
