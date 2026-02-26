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
"""Switch processors to route parts to different processors."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, Callable
from typing import Generic, Self, TypeAlias, TypeVar

from genai_processors import content_api
from genai_processors import processor
from genai_processors import streams

_T = TypeVar('_T')

ProcessorPart: TypeAlias = content_api.ProcessorPart
PartProcessor: TypeAlias = processor.PartProcessor
Processor: TypeAlias = processor.Processor


class Switch(Processor, Generic[_T]):
  """Switch between processors.

  Convenient way to create a processor that routes the parts of the input stream
  to different processors based on a condition (aka a case). The condition can
  be:

  1. a function that takes a `ProcessorPart` and returns a boolean. Example:
    ```python
    switch_processor = (
        switch.Switch()
        .case(content_api.is_audio, audio_processor)
        .case(content_api.is_video, video_processor)
        .default(processor.passthrough())
    )
    ```
  2. a function that takes any value returned by the match_fn passed in the
     constructor and returns a boolean. We have a shortcut for boolean functions
     that tests for equality, e.g. `lambda x: x == "a"`. They can be replaced
     with the value itself, e.g. `"a"`. Example:
    ```python
    # The match_fn is applied to the input part and the result is compared to
    # the value passed in the case() method.
    switch_processor = (
        switch.Switch(content_api.get_substream_name)
        .case("a", p)  # equivalent to .case(lambda x: x == "a")
        .case("b", q)  # equivalent to .case(lambda x: x == "b")
        .default(processor.passthrough())
    )
    ```

  The order of the parts in the output and input streams is only kept for parts
  returned by the same processor, i.e. two parts matching two different cases
  are not guaranteed to be in the same order in the input and output stream.

  PartProcessors can be used instead of Processor, they will be converted to
  Processor automatically. The `processor.passthrough()` (a PartProcessor) in
  the examples above is equivalent to passing
  `processor.passthrough().to_processor()`.

  If the cases involve `PartProcessor`s only, it is best to use the
  `PartSwitch` class, which is optimized for concurrent processing of parts.
  """

  def __init__(
      self,
      match_fn: Callable[[ProcessorPart], _T] | None = None,
  ):
    self._cases: list[tuple[Callable[[_T], bool], Processor]] = []
    self._match = match_fn
    self._default_set = False

  async def call(
      self, content: AsyncIterable[ProcessorPart]
  ) -> AsyncIterable[content_api.ProcessorPartTypes]:
    input_queues = [asyncio.Queue() for _ in range(len(self._cases))]

    async def _triage():
      """Triage the input parts to the correct input queue."""
      async for part in content:
        for i, (filter_fn, _) in enumerate(self._cases):
          if filter_fn(part):
            await input_queues[i].put(part)
            break
      for q in input_queues:
        await q.put(None)

    triage_task = processor.create_task(_triage())

    # Process the parts in the input queues and merge all results.
    output_streams = [
        self._cases[i][1](streams.dequeue(queue))
        for i, queue in enumerate(input_queues)
    ]
    async for part in streams.merge(output_streams):
      yield part

    await triage_task

  def case(
      self,
      v: _T | Callable[[_T], bool],
      p: Processor | PartProcessor,
  ) -> Self:
    if self._default_set:
      raise ValueError(
          f'This case is added after the default processor is set: {v}'
      )
    if self._match is None:
      self._match = lambda x: x
    if isinstance(p, PartProcessor):
      case_processor = p.to_processor()
    else:
      case_processor = p
    if isinstance(v, Callable):
      self._cases.append((lambda x: v(self._match(x)), case_processor))
    else:
      self._cases.append((lambda x: v == self._match(x), case_processor))
    return self

  def default(self, p: Processor | PartProcessor) -> Self:
    if self._default_set:
      raise ValueError('The default processor is already set.')
    if isinstance(p, PartProcessor):
      self._cases.append((lambda x: True, p.to_processor()))
    else:
      self._cases.append((lambda x: True, p))
    self._default_set = True
    return self


class PartSwitch(PartProcessor, Generic[_T]):
  """Switch between part processors.

  Convenient way to create a switch processor that runs the first case that
  matches. A case is defined by a condition and a PartProcessor. The
  condition can be a function that takes a `ProcessorPart` and returns a boolean
  or a value. In the latter case, the condition is compared to the result of the
  match_fn passed to the constructor.

  By default, the switch processor does not return any part when no case
  matches. To return a part in that case, the default() method should be
  called after all cases have been added.

  Example usage:

  Simple match conditions based on equality:

  ```python
  # Applies content_api.as_text to the input part and checks equality with
  # the string "a" or "b".
  switch_processor = (
      switch.Switch(content_api.as_text)
      .case("a", p)
      .case("b", q)
      .default(processor.passthrough())
  ```

  Note that you can also compare parts directly by using the `lambda x: x`
  function in `Switch` but it is not recommended. For more complex match
  conditions, you can use cases operating on `part` directly as follows:

  ```python
  # Applies the lambda function defined in `case` on the input part to check
  # which case is valid.
  switch_processor = (
      switch.Switch()
      .case(lambda x: x.text.startswith("a"), p)
      .case(lambda x: x.text.startswith("b"), q)
      .default()
  )
  ```
  """

  def __init__(
      self,
      match_fn: Callable[[ProcessorPart], _T] | None = None,
  ):
    self._cases: list[tuple[Callable[[_T], bool], PartProcessor]] = []
    self._match = match_fn
    self._default_set = False

  async def call(
      self, part: ProcessorPart
  ) -> AsyncIterable[content_api.ProcessorPartTypes]:
    for filter_fn, p in self._cases:
      if filter_fn(part):
        async for c in p(part):
          yield c
        break

  def case(
      self,
      v: _T | Callable[[_T], bool],
      p: PartProcessor,
  ) -> Self:
    if self._default_set:
      raise ValueError(
          f'This case is added after the default processor is set: {v}'
      )
    if self._match is None:
      self._match = lambda x: x
    if isinstance(v, Callable):
      self._cases.append((lambda x: v(self._match(x)), p))
    else:
      self._cases.append((lambda x: v == self._match(x), p))
    return self

  def default(self, p: PartProcessor) -> Self:
    if self._default_set:
      raise ValueError('The default processor is already set.')
    self._cases.append((lambda x: True, p))
    self._default_set = True
    return self
