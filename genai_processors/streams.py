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
"""Utilities for managing part streams."""

import asyncio
from collections.abc import AsyncIterable, AsyncIterator, Iterable
import copy
from typing import Any, TypeVar

from genai_processors import context

_T = TypeVar('_T')


def split(
    content: AsyncIterable[_T],
    *,
    n: int = 2,
    with_copy: bool = False,
) -> tuple[AsyncIterable[_T], ...]:
  """Split a stream into `n` identical streams.

  Recommended to be used with processor.context to ensure error propagation.

  Args:
    content: content to be split
    n: number of streams to return
    with_copy: whether to copy the items of the streams or not. It is
      recommended to copy the items when side effects between streams can
      happen. This is the case when one processor changes a part in place (e.g.
      update its metadata). As this can be expensive if the items are large and
      the number of streams is high, the default is to not copy. Consider
      setting this to True if there is a chance that a part can be modified in
      place.

  Returns:
    n streams of content.

  Raises:
    ValueError if n=0
  """
  if n == 0:
    raise ValueError('Cannot split a stream in n=0 streams.')
  if n == 1:
    return (content,)
  queues = [asyncio.Queue() for _ in range(n)]

  async def enqueue_parts() -> None:
    async for part in content:
      for queue in queues:
        if with_copy:
          queue.put_nowait(copy.deepcopy(part))
        else:
          queue.put_nowait(part)
    for queue in queues:
      queue.put_nowait(None)

  async def dequeue_parts(
      queue: asyncio.Queue[_T],
  ) -> AsyncIterable[_T]:
    while (part := await queue.get()) is not None:
      yield part

  context.create_task(enqueue_parts())

  return tuple(dequeue_parts(queue) for queue in queues)


async def concat(*contents: AsyncIterable[_T]) -> AsyncIterable[_T]:
  """Concatenate multiple streams into one.

  The streams are looped over concurrently before being assembled into a single
  output stream.

  Args:
    *contents: each stream to concat as a separate argument.

  Yields:
    The concatenation of all streams.
  """
  output_queues = [asyncio.Queue() for _ in contents]

  async def _stream_outputs(
      idx: int,
  ):
    async for c in contents[idx]:
      output_queues[idx].put_nowait(c)
    # Adds None to indicate end of output.
    output_queues[idx].put_nowait(None)

  tasks = []
  for idx, _ in enumerate(contents):
    tasks.append(context.create_task(_stream_outputs(idx)))

  for q in output_queues:
    while (part := await q.get()) is not None:
      q.task_done()
      yield part


async def merge(
    streams: Iterable[AsyncIterable[_T]],
    *,
    queue_maxsize: int = 0,
    stop_on_first: bool = False,
) -> AsyncIterable[_T]:
  """Merges multiple streams into one.

  The order is defined by the asyncio loop and will likely be determined by the
  time when the items are available.

  If a stream is cancelled, the overall merge will be cancelled and all other
  streams will be cancelled as well.

  Args:
    streams: The input streams to merge. These streams cannot be iterated over
      outside of this call. If you need to consume them outside of this call,
      use `streams.split()` to copy the stream first.
    queue_maxsize: The maximum number of items to buffer in an internal queue
      for each input stream. Set to 0 to use an unbounded queue.
    stop_on_first: If True, stop merging streams as soon as one of them is
      empty.

  Yields:
    a item from one of the input streams. The order in which the items are
    yielded is random and interleaved (likely order of generation). This means
    the order of the items within one stream is preserved but not across
    streams.
  """
  out = asyncio.Queue(maxsize=queue_maxsize)
  active_streams = 0

  async with asyncio.TaskGroup() as tg:
    running_tasks = []
    for s in streams:
      active_streams += 1
      running_tasks.append(tg.create_task(enqueue(s, out)))

    while active_streams:
      part = await out.get()
      out.task_done()
      if part is None:
        active_streams -= 1
        if stop_on_first:
          break
        continue
      yield part
    for t in running_tasks:
      t.cancel()


async def enqueue(
    content: AsyncIterable[_T], queue: asyncio.Queue[_T | None]
) -> None:
  """Enqueues all content into a queue.

  When the queue is unbounded, this function will not block. When the queue is
  bounded, this function will block until the queue has space.

  Args:
    content: The content to enqueue.
    queue: The queue to enqueue to.
  """
  try:
    async for part in content:
      await queue.put(part)
  finally:
    await queue.put(None)


async def dequeue(queue: asyncio.Queue[_T | None]) -> AsyncIterator[_T]:
  """Dequeues content from a queue.

  Args:
    queue: The queue to dequeue from. The queue must end with a None item.

  Yields:
    The content from the queue as an AsyncIterator. The items are yielded in
    the order they were enqueued.
  """
  while (part := await queue.get()) is not None:
    queue.task_done()
    yield part
  queue.task_done()


async def stream_content(
    content: Iterable[_T],
    with_delay_sec: float | None = None,
    delay_first: bool = False,
    delay_end: bool = True,
) -> AsyncIterable[_T]:
  """Converts non-async content into an AsyncIterable.

  Args:
    content: the items to yield. The state of the iterator provided by `content`
      is undefined post-invocation and should not be relied on.
    with_delay_sec: If set, asyncio.sleep() is called with this value between
      yielding parts, and optionally also before and after.
    delay_first: If set to True, a delay is added before the first part.
    delay_end: Unless set to False, a delay is added between the last part and
      stopping the returned AsyncIterator.

  Yields:
    an item in `content` (unchanged).
  """
  # Getting the next value from content must not block. For this reason
  # this function doesn't work for generators.
  delay_next = with_delay_sec is not None and delay_first
  for c in content:
    if delay_next:
      await asyncio.sleep(with_delay_sec)
    else:
      delay_next = with_delay_sec is not None  # From the 2nd iteration onwards.
    yield c
  if with_delay_sec is not None and delay_end:
    await asyncio.sleep(with_delay_sec)


async def gather_stream(content: AsyncIterable[_T]) -> list[_T]:
  """Gathers an AsyncIterable into a list of items."""
  return [c async for c in content]


async def aenumerate(
    aiterable: AsyncIterable[_T],
) -> AsyncIterable[tuple[int, _T]]:
  """Enumerate an async iterable."""
  i = 0
  async for x in aiterable:
    yield (i, x)
    i += 1


async def endless_stream() -> AsyncIterable[Any]:
  """Empty input stream for the live agents.

  The Live API and VideoIn / PyAudioIn processors lifetime is bound to the
  incoming Part stream. More complex setups would use the stream to send
  additional data such as the initial context or text entered in the terminal.

  For simpler use cases this is an empty never-ending stream to keep such
  processors alive.

  Yields:
    Nothing. The stream keeps waiting and will never end unless cancelled.
  """
  while True:
    await asyncio.sleep(1)
  # Unreachable. Needed to make the function a generator.
  yield None
