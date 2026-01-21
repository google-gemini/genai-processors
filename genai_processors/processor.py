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
"""Core processors library."""

from __future__ import annotations

import abc
import asyncio
from collections.abc import AsyncIterable, AsyncIterator, Callable, Sequence
import contextvars
import functools
import inspect
import types
import typing
from typing import Any, ParamSpec, Protocol, Self, TypeAlias, overload

from genai_processors import cache_base
from genai_processors import content_api
from genai_processors import context as context_lib
from genai_processors import map_processor
from genai_processors import mime_types
from genai_processors import streams
from genai_processors.dev import trace

# Aliases
context = context_lib.context
create_task = context_lib.create_task
PROMPT_STREAM = context_lib.PROMPT_STREAM
DEBUG_STREAM = context_lib.DEBUG_STREAM
STATUS_STREAM = context_lib.STATUS_STREAM

ProcessorPart = content_api.ProcessorPart
ProcessorPartTypes = content_api.ProcessorPartTypes
ProcessorContent = content_api.ProcessorContent
ProcessorContentTypes = content_api.ProcessorContentTypes
MatchFn: TypeAlias = Callable[[ProcessorPart], bool]

stream_content = streams.stream_content
gather_stream = streams.gather_stream


# Part queue size. It should be a big number to avoid blocking the processor.
_MAX_QUEUE_SIZE = 10_000

# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# Interface Definition & util functions


# This is needed because in some placess we pass async instead of ProcessorFn.
def _key_prefix(
    p: Processor | PartProcessor | ProcessorFn | PartProcessorFn,
) -> str:
  if isinstance(p, Processor):
    return p.key_prefix
  elif isinstance(p, PartProcessor):
    return p.key_prefix
  elif isinstance(p, functools.partial):
    return p.func.__qualname__  # pylint: disable=attribute-error
  return p.__qualname__  # pylint: disable=attribute-error


def _combined_key_prefix(
    processor_list: Sequence[
        Processor | PartProcessor | ProcessorFn | PartProcessorFn
    ],
) -> str:
  return ','.join(map(_key_prefix, processor_list))


async def _normalize_part_stream(
    content: AsyncIterable[ProcessorPartTypes],
    producer: Any = None,
) -> AsyncIterable[ProcessorPart]:
  """Yields ProcessorParts given a stream of content convertible to them."""
  async for part in content:
    match part:
      case ProcessorPart():
        yield part
      case _:
        try:
          yield ProcessorPart(part)
        except ValueError as e:
          raise ValueError(f'{e} produced by {producer}') from e


@typing.runtime_checkable
class ProcessorFn(Protocol):
  """A Processor function.

  The number of parts in and out of the processor, and their mimetypes, can be
  different. There is little constraint about what comes in or out.
  """

  def __call__(
      self, content: AsyncIterable[ProcessorPart]
  ) -> AsyncIterable[ProcessorPartTypes]:
    ...


class Processor(abc.ABC):
  """Any class implementing a processor should inherit from this."""

  @typing.final
  async def __call__(
      self, content: AsyncIterable[ProcessorPartTypes]
  ) -> AsyncIterable[ProcessorPart]:
    """Processes the given content.

    Descendants should override `call` method instead of this one:
      .__call__() is the convenient way to invoke a processor.
      .call() is the convenient way to implement a processor.

    Args:
      content: the input stream of content to process.

    Yields:
      the result of processing the input content.
    """
    normalized_content = _normalize_part_stream(content, producer=self)

    # Ensures that the same taskgroup is always added to the context and
    # includes the proper way of handling generators, i.e. use a queue inside
    # the task group instead of a generator.
    #
    # When an async generator yields control (e.g., via `yield` or `await`), it
    # temporarily exits any active `async with` context managers. If a task
    # *within* the TaskGroup` fails, the `TaskGroup` cancels all other tasks,
    # including any task consuming the generator. This cancellation appears as a
    # `CancelledError` at the `await` or `yield` point *in the consumer*, not
    # within the generator. The generator's internal `try...except` blocks
    # cannot catch this CancelledError` because it's raised *outside* the
    # generator's execution. Using a queue ensures that the `yield` statement is
    # always executed within the task group and that the `CancelledError` is
    # handled correctly.
    tg = context_lib.task_group()

    async with trace.call_scope(self.key_prefix) as current_trace:

      if current_trace:

        async def stream_input() -> AsyncIterable[ProcessorPart]:
          async for part in normalized_content:
            await current_trace.add_input(part)
            yield part

      else:
        stream_input = lambda: normalized_content

      if tg is None:
        output_queue = asyncio.Queue[ProcessorPart | None]()

        async def _with_context():
          async with context():
            try:
              async for p in _normalize_part_stream(
                  self.call(stream_input()), producer=self.call
              ):
                if current_trace:
                  await current_trace.add_output(p)
                output_queue.put_nowait(p)
            finally:
              output_queue.put_nowait(None)

        task = asyncio.create_task(_with_context())
        try:
          async for p in streams.dequeue(output_queue):
            yield p
        finally:
          await task
      else:
        async for p in _normalize_part_stream(
            self.call(stream_input()), producer=self.call
        ):
          if current_trace:
            await current_trace.add_output(p)
          yield p

  @abc.abstractmethod
  async def call(
      self, content: AsyncIterable[ProcessorPart]
  ) -> AsyncIterable[ProcessorPartTypes]:
    """Implements the Processor logic.

    Do not invoke this method directly:
      .__call__() is the convenient way to invoke a processor.
      .call() is the convenient way to implement a processor.

    It must be implemented by the processor and is responsible for processing
    the input content and yielding the output content.

    As with any async function it is highly recommended not to block inside the
    `call` method as this will prevent other coroutines to make progress.

    Args:
      content: the input stream of content to process.

    Yields:
      the result of processing the input content.
    """
    async for part in content:
      yield part

  @functools.cached_property
  def key_prefix(self) -> str:
    """Prefix for key to avoid collisions from different Processors.

    Defaults to classname. Processor() should override this if, for example, it
    accepts arguments that change output of __call__.

    Returns:
      Prefix that will be added to key.
    """
    return self.__class__.__qualname__

  def to_processor(self) -> Processor:
    return self

  def __add__(self, other: Self | PartProcessor) -> _ChainProcessor:
    """Adds `other` to this processor: self + other.

    Args:
      other: a processor to add to `self`.

    Returns:
      The chain of this process with `other`.
    """
    if isinstance(other, _ChainProcessor):
      return _ChainProcessor([self] + other._processors)
    return _ChainProcessor([self, other])


@typing.runtime_checkable
class PartProcessorFn(Protocol):
  """A PartProcessor function.

  PartProcessors are similar to Processors but they take a single part as
  input and can be executed concurrently over an async iterable. They allow
  higher level of concurrency than regular processors, especially if chained one
  after another.
  """

  def __call__(self, part: ProcessorPart) -> AsyncIterable[ProcessorPartTypes]:
    ...


@typing.runtime_checkable
class PartProcessorWithMatchFn(PartProcessorFn, Protocol):
  """A PartProcessor function with a match function.

  A match function indicates which parts should be processed by the part
  processor.
  """

  def match(self, part: ProcessorPart) -> bool:
    """Returns True if `part` should be processed by this part processor.

    Returns False if it sure that the part processor will not process the input
    part and that the part processor should pass the part as is.

    NOTE: the part processor `__call__` implementation should always skip the
    part (i.e. return the part as is) when `match` returns False.

    A typical example are part processors that are type-dependent, e.g. a part
    processor that parses a specific proto from the part or that only parses
    text.

    Args:
      part: the part to check.

    Returns:
      False if the part has no chance of being processed by this part
      processor. True otherwise.
    """
    ...


class PartProcessor(abc.ABC):
  """Any class implementing a part processor should inherit from this."""

  @typing.final
  async def __call__(self, part: ProcessorPart) -> AsyncIterable[ProcessorPart]:
    """Processes the given part.

    Descendants should override `call` method instead of this one:
      .__call__() is the convenient way to invoke a processor.
      .call() is the convenient way to implement a processor.

    Args:
      part: the Part to process.

    Yields:
      the result of processing the input Part.
    """
    if not self.match(part):
      yield part
      return
    async for result in _normalize_part_stream(
        self.call(part), producer=self.call
    ):
      yield result

  @abc.abstractmethod
  async def call(
      self, part: ProcessorPart
  ) -> AsyncIterable[ProcessorPartTypes]:
    """Implements the Processor logic.

    Do not invoke this method directly:
      .__call__() is the convenient way to invoke processor.
      .call() is the convenient way to implement processor.

    Args:
      part: the Part to process.

    Yields:
      the result of processing the input Part.
    """
    yield part

  def match(self, part: ProcessorPart) -> bool:
    del part
    return True

  @functools.cached_property
  def key_prefix(self) -> str:
    """Prefix for key to avoid collisions from different Processors."""
    return self.__class__.__qualname__

  @overload
  def __add__(self, other: PartProcessor) -> PartProcessor:
    ...

  @overload
  def __add__(self, other: Processor) -> Processor:
    ...

  def __add__(self, other: Self | Processor) -> PartProcessor | Processor:
    """Adds `other` to this processor.

    Args:
      other: a processor to add to `self`.

    Returns:
      The chain of this process with `other`.
    """
    if isinstance(other, PartProcessor):
      if isinstance(other, _ChainPartProcessor):
        return _ChainPartProcessor([self] + other._processor_list)
      return _ChainPartProcessor([self, other])

    return _ChainProcessor([self, other])

  def __floordiv__(self, other: Self | Processor) -> PartProcessor | Processor:
    """Make `other` be computed in parallel to this processor.

    Args:
      other: a processor to compute in parallel to `self`.

    Returns:
      The parallel computation of this process with `other`.
    """
    if isinstance(other, _ParallelPartProcessor):
      return _ParallelPartProcessor([self] + other._processor_list)
    elif isinstance(other, PartProcessor):
      return _ParallelPartProcessor([self, other])
    else:
      raise ValueError(
          'Parallel operator not valid between a PartProcessor and'
          f' {type(other)}.'
      )

  def to_processor(self) -> Processor:
    """Converts this PartProcessor to a Processor.

    Adds status and debug stream to the output streams.

    Returns:
      a processor that will run the part processor for each part concurrently
      in the input stream.
    """
    return _ProcessorWrapper(
        map_processor.map_part_function(
            _chain_part_processors(
                [self],
                task_name=self.key_prefix,
            )
        )
    )


def debug(content: ProcessorPartTypes, **kwargs) -> ProcessorPart:
  """Returns a ProcessorPart with the debug substream."""
  return ProcessorPart(content, substream_name=DEBUG_STREAM, **kwargs)


def status(content: ProcessorPartTypes, **kwargs) -> ProcessorPart:
  """Returns a ProcessorPart with the status substream."""
  return ProcessorPart(content, substream_name=STATUS_STREAM, **kwargs)


async def apply_async(
    processor: Processor | PartProcessor, content: ProcessorContentTypes
) -> list[ProcessorPart]:
  """Applies a Processor asynchronously.

  When a part processor is given as input, this method will first turn it into
  a processor and then will process the content asynchronously.

  Args:
    processor: the Processor to apply to the content.
    content: a collection of ProcessorParts on which to apply the Processor.

  Returns:
    the content, with the Processor applied to each content part.
  """
  async with context():
    content_processor = processor.to_processor()
    as_async = stream_content(ProcessorContent(content).all_parts)
    return await gather_stream(content_processor(as_async))


def apply_sync(
    processor: Processor | PartProcessor,
    content: ProcessorContentTypes,
) -> list[ProcessorPart]:
  """Applies a Processor synchronously.

  When a part processor is given as input, this method will first turn it into
  a processor and then will process the content concurrently.

  Args:
    processor: the Processor to apply to the content.
    content: a collection of ProcessorParts on which to apply the Processor.

  Returns:
    the content, with the Processor applied to each content part.
  """

  return asyncio.run(apply_async(processor, content))


def processor_function(
    func: ProcessorFn,
) -> Processor:
  """Decorator to transform a function into a processor."""
  # Wraps it into a processor class.
  assert inspect.isasyncgenfunction(
      func
  ), f'{func} is not an async function - define {func} with `async def`.'
  assert not _is_part_processor_protocol(func), f'{func} is not a ProcessorFn.'
  proc_func = typing.cast(ProcessorFn, func)
  return _ProcessorWrapper(proc_func)


@overload
def part_processor_function(
    *,
    match_fn: MatchFn | None = None,
) -> Callable[[PartProcessorFn], PartProcessor]:
  ...


@overload
def part_processor_function(
    func: PartProcessorFn,
    *,
    match_fn: MatchFn | None = None,
) -> PartProcessor:
  ...


def part_processor_function(
    func: PartProcessorFn | None = None,
    *,
    match_fn: MatchFn | None = None,
) -> PartProcessor | Callable[[PartProcessorFn], PartProcessor]:
  """Decorator to transform a function into a part processor."""
  if func:
    assert inspect.isasyncgenfunction(
        func
    ), f'{func} is not an async function - define {func} with `async def`.'
    proc_func = typing.cast(PartProcessorFn, func)
    if not match_fn:
      return _PartProcessorWrapper(proc_func)
    else:
      return _PartProcessorWrapper(proc_func, match_fn=match_fn)
  elif match_fn:
    return functools.partial(part_processor_function, match_fn=match_fn)
  else:
    raise ValueError('Either func or match_fn must be provided.')


def chain(
    processor_list: Sequence[Processor | PartProcessor],
) -> Processor:
  """Chain a sequence of processors.

  Args:
    processor_list: list of part processors or generic processors.

  Returns:
    A processor consisting of the chain of all the processors in the list. The
    execution is sequential from the first processor to the last but parts are
    processed concurrently overall.
  """
  if not processor_list:
    raise ValueError('processor_list is empty')
  chain_processor = processor_list[0]
  for p in processor_list[1:]:
    chain_processor = chain_processor + p
  if isinstance(chain_processor, PartProcessor):
    chain_processor = chain_processor.to_processor()
  return chain_processor


def parallel(processor_list: Sequence[PartProcessor]) -> PartProcessor:
  """Create a sequence of part processors to be run in parallel.

  Args:
    processor_list: list of part processors.

  Returns:
    A processor consisting of the parallel run of all the processors in the
    list. The execution is sequential from the first processor to the last but
    parts are processed concurrently overall.
  """
  if not processor_list:
    raise ValueError('processor_list is empty')
  return _ParallelPartProcessor(processor_list)


def parallel_concat(processor_list: Sequence[Processor]) -> Processor:
  """Create a sequence of processors to be run in parallel.

  The output is the concatenation of all processors, i.e.:

  parallel_concat([p1, p2])(stream) -> [p1(stream), p2(stream)]

  Args:
    processor_list: list of processors.

  Returns:
    A processor consisting of the parallel run of all the processors in the
    list. The execution is sequential from the first processor to the last and
    the result of each processor is concatenated
  """
  if not processor_list:
    raise ValueError('processor_list is empty')
  return _ParallelProcessor(processor_list)


def create_filter(condition: Callable[[ProcessorPart], bool]) -> PartProcessor:
  """Creates a processor that filters parts based on `condition`.

  Args:
    condition: a part is returned by this processor iff `condition(part)=True`

  Returns:
    a processor filtering the input stream
  """

  async def filter_with_condition(
      part: ProcessorPart,
  ) -> AsyncIterable[ProcessorPart]:
    if condition(part):
      yield part

  return _PartProcessorWrapper(filter_with_condition)


# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# Internal chaining mechanism for processors including concurrent execution.
# These methods should stay internal to this file. Use '+' or processor.chain
# to combine processors together.


def _is_part_processor_protocol(obj: Any) -> bool:
  """Returns True if `obj` implements PartProcessorFn.

  This function is needed as Processors and PartProcessors are Protocols and do
  not have proper runtime type checking.

  Args:
    obj: any object or function
  """

  def _full_name(obj: Any) -> str:
    """Returns the full qualified name of the object `obj`."""
    return obj.__module__ + '.' + getattr(obj, '__qualname__', '')

  if not callable(obj):
    return False
  # Extract callable argument hints
  if isinstance(obj, types.FunctionType):
    type_hint = typing.get_type_hints(obj)
  else:
    type_hint = typing.get_type_hints(obj.__call__)  # pylint: disable=attribute-error
  # Return type needs to be defined
  if 'return' not in type_hint:
    return False
  return_type = type_hint.pop('return')
  # Only one input parameter is defined
  if len(type_hint) != 1:
    return False
  # Only one generic type in the output type
  if len(typing.get_args(return_type)) != 1:
    return False
  # class names on output types must match.
  # We only test on AsyncIterable __qualname__ as AsyncIterable can be declared
  # in typing or collections.abc, and both should be recognized.
  if return_type.__qualname__ != 'AsyncIterable' or _full_name(
      typing.get_args(return_type)[0]
  ) != _full_name(ProcessorPart):
    return False
  # Type hints contains the input type only.
  if _full_name(next(iter(type_hint.values()))) != _full_name(ProcessorPart):
    return False
  return True


class _PartProcessorWrapper(PartProcessor):
  """A PartProcessorFn wrapped in a class."""

  def __init__(
      self,
      fn: PartProcessorFn,
      match_fn: MatchFn | None = None,
  ):
    assert inspect.isasyncgenfunction(fn), (
        f'{fn} is not an async function - define your PartProcessor with'
        ' `async def`.'
    )
    self._fn = fn
    self._match_fn = match_fn or (lambda _: True)

  async def call(self, part: ProcessorPart) -> AsyncIterable[ProcessorPart]:
    if not self._match_fn(part):
      yield part
      return
    async for result in self._fn(part):
      yield result

  def match(self, part: ProcessorPart) -> bool:
    return self._match_fn(part)

  @functools.cached_property
  def key_prefix(self) -> str:
    return '_PartProcessorWrapper:' + _key_prefix(self._fn)

  def __repr__(self):
    return f'{self.__class__.__name__}({self._fn})'


class _ChainPartProcessor(PartProcessor):
  """Chain of part processors."""

  def __init__(
      self,
      processor_list: Sequence[PartProcessorWithMatchFn],
  ):
    self._processor_list = list(processor_list)

  @functools.cached_property
  def key_prefix(self) -> str:
    return '_ChainPartProcessor:' + _combined_key_prefix(self._processor_list)

  def __add__(
      self, other: Processor | PartProcessor
  ) -> Processor | PartProcessor:
    if isinstance(other, _ChainPartProcessor):
      return _ChainPartProcessor(self._processor_list + other._processor_list)
    elif isinstance(other, PartProcessor):
      return _ChainPartProcessor([
          *self._processor_list,
          other,
      ])
    return _ChainProcessor([self, other])

  def match(self, part: ProcessorPart) -> bool:
    return any(p.match(part) for p in self._processor_list)

  async def call(
      self, part: ProcessorPart
  ) -> AsyncIterable[ProcessorPartTypes]:
    if not self._processor_list:
      # Empty chain = passthrough processor
      yield part
      return
    task_name = (
        f'_ChainPartProcessor({_key_prefix(self._processor_list[0])}...)'
    )
    async for result in _chain_part_processors(self._processor_list, task_name)(
        part
    ):
      yield result

  def to_processor(self) -> Processor:
    if not self._processor_list:
      return _ChainProcessor([])
    return super().to_processor()


class _ProcessorWrapper(Processor):
  """A ProcessorFn wrapped in a class."""

  def __init__(self, fn: ProcessorFn):
    self.call = fn

  def __repr__(self):
    return f'_ProcessorWrapper({repr(self.call)})'

  async def call(
      self, content: AsyncIterable[ProcessorPart]
  ) -> AsyncIterable[ProcessorPart]:
    # This method is overridden in the __init__.
    yield ProcessorPart('')

  @functools.cached_property
  def key_prefix(self) -> str:
    return _key_prefix(self.call)


class _ChainProcessor(Processor):
  """Chain of processors that fuses consecutive PartProcessors."""

  def __init__(
      self,
      processor_list: Sequence[Processor | PartProcessor],
  ):
    fused_processors: list[Processor | PartProcessor] = []
    for p in processor_list:
      if (
          isinstance(p, PartProcessor)
          and fused_processors
          and isinstance(fused_processors[-1], PartProcessor)
      ):
        fused_processors[-1] = fused_processors[-1] + p
      else:
        fused_processors.append(p)
    self._processors = fused_processors

    self._processor_list: list[ProcessorFn] = []
    for p in self._processors:
      if isinstance(p, PartProcessor):
        self._processor_list.append(p.to_processor().call)
      else:
        self._processor_list.append(p.call)

    self._task_name = (
        f'_ChainProcessor({_key_prefix(self._processors[0])}...)'
        if self._processors
        else '_ChainProcessor(empty)'
    )

  def __add__(self, other: Processor | PartProcessor) -> _ChainProcessor:
    other_list = (
        other._processors if isinstance(other, _ChainProcessor) else [other]
    )
    return _ChainProcessor(self._processors + other_list)

  async def call(
      self, content: AsyncIterable[ProcessorPart]
  ) -> AsyncIterable[ProcessorPartTypes]:
    if not self._processor_list:
      # Empty chain = passthrough processor
      async for part in content:
        yield part
      return
    async for result in _chain_processors(
        self._processor_list, self._task_name
    )(content):
      yield result

  @functools.cached_property
  def key_prefix(self) -> str:
    return '_ChainProcessor:' + _combined_key_prefix(self._processor_list)


async def _capture_reserved_substreams(
    content: AsyncIterable[ProcessorPart], queue: asyncio.Queue
) -> AsyncIterable[ProcessorPart]:
  reserved_substreams = context_lib.get_reserved_substreams()
  async for part in content:
    if any(
        part.substream_name.startswith(prefix) for prefix in reserved_substreams
    ):
      await queue.put(part)
    else:
      yield part


async def _enqueue_content(
    content: AsyncIterable[ProcessorPart], queue: asyncio.Queue
) -> None:
  async for part in content:
    queue.put_nowait(part)
  queue.put_nowait(None)


def _chain_processors(
    processors: Sequence[ProcessorFn], task_name: str
) -> ProcessorFn:
  """Combine processors in sequence.

  NOTE: Substreams debug and status are yielded immediately instead of passing
        them to the next processor.

  Args:
    processors: sequence of Processor functions to chain
    task_name: Name that will be assigned to the asyncio task running the
      processor.

  Returns:
    Processor that is a chain of the provided Sequence of Processors.
  """
  if len(processors) == 1:
    return processors[0]

  async def processor(
      content: AsyncIterable[ProcessorPart],
  ) -> AsyncIterable[ProcessorPart]:
    # Create a queue to put output parts
    output_queue = asyncio.Queue()
    # Chain all processors together
    for processor in processors:
      content = _capture_reserved_substreams(content, output_queue)
      content = _normalize_part_stream(processor(content), producer=processor)
    # Place output processed output parts on the queue.
    create_task(_enqueue_content(content, output_queue), name=task_name)
    while (part := await output_queue.get()) is not None:
      yield part
      output_queue.task_done()

  return processor


class _CaptureReservedSubstreams(PartProcessor):
  """ProcessorPart version of `_capture_reserved_substream`.

  __NOTE__: this class does not need a `match` method. It should always be used
  with map_processor methods that handle the `match` logic separately.
  """

  def __init__(
      self,
      queue: asyncio.Queue,
      p: PartProcessorWithMatchFn | PartProcessorFn,
  ):
    self._queue = queue
    self._part_processor_fn = p

  async def call(self, part: ProcessorPart) -> AsyncIterable[ProcessorPart]:
    if context_lib.is_reserved_substream(part.substream_name):
      await self._queue.put(part)
      return
    processed_stream: AsyncIterable[ProcessorPart] = _normalize_part_stream(
        self._part_processor_fn(part), producer=self._part_processor_fn
    )
    async for part in processed_stream:
      if context_lib.is_reserved_substream(part.substream_name):
        await self._queue.put(part)
      else:
        yield part

  def match(self, part: ProcessorPart) -> bool:
    return (
        context_lib.is_reserved_substream(part.substream_name)
        or not hasattr(self._part_processor_fn, 'match')
        or self._part_processor_fn.match(part)
    )


def _chain_part_processors(
    part_processors: Sequence[PartProcessorWithMatchFn],
    task_name: str,
) -> PartProcessorFn:
  """Combine **part processors** in sequence.

  Adds debug and status streams to the output.

  NOTE: Substreams debug and status are yielded immediately instead of passing
        them to the next processor.

  Args:
    part_processors: sequence of part processors to chain.
    task_name: Name that will be assigned to the asyncio task running the
      processor.

  Returns:
    Part processor that is a chain of the provided Sequence of part
    processors.
  """

  async def processor(
      part: ProcessorPart,
  ) -> AsyncIterable[ProcessorPart]:
    # Create a queue to put output parts
    output_queue = asyncio.Queue()
    processors = []
    match_fns = []
    for p in part_processors:
      processors.append(_CaptureReservedSubstreams(output_queue, p))
      match_fns.append(p.match)
    chained_processor = map_processor.chain_part_functions(
        processors, match_fns
    )
    # Processed content
    content = chained_processor(part)
    create_task(
        # Place output processed output parts on the queue.
        _enqueue_content(content, output_queue),
        name=task_name,
    )
    while (part := await output_queue.get()) is not None:
      yield part
      output_queue.task_done()

  return processor


# ---------------- Parallel operator ---------------


class _ParallelProcessor(Processor):
  """Parallel processors.

  A parallel processor is a function that takes a stream of content parts
  as input, runs multiple processors in parallel and concatenate their output
  into a single stream.
  """

  def __init__(
      self,
      processor_list: Sequence[ProcessorFn],
  ):
    self._processor_list = list(processor_list)

  def __repr__(self):
    list_repr = ','.join(map(repr, self._processor_list))
    return f'ParallelProcessor[{list_repr}]'

  async def call(
      self, content: AsyncIterable[ProcessorPart]
  ) -> AsyncIterable[ProcessorPart]:
    # Create a queue to put output parts
    output_queue = asyncio.Queue(maxsize=_MAX_QUEUE_SIZE)
    stream_inputs = streams.split(content, n=len(self._processor_list))
    output_streams = [
        _normalize_part_stream(
            processor(
                _capture_reserved_substreams(stream_inputs[idx], output_queue)
            ),
            producer=processor,
        )
        for idx, processor in enumerate(self._processor_list)
    ]
    # Place processed output parts on the queue.
    create_task(_enqueue_content(streams.concat(*output_streams), output_queue))
    while (part := await output_queue.get()) is not None:
      yield part
      output_queue.task_done()

  @functools.cached_property
  def key_prefix(self) -> str:
    return '_ParallelProcessor:' + _combined_key_prefix(self._processor_list)


class _ParallelPartProcessor(PartProcessor):
  """Parallel part processors.

  A parallel part processor is a function that takes a single content part
  as input, runs multiple part processors in parallel and merge their output
  into a single stream.
  """

  def __init__(
      self,
      processor_list: Sequence[PartProcessorWithMatchFn],
  ):
    self._processor_list = list(processor_list)
    self._is_passthrough = any(
        p is PASSTHROUGH_FALLBACK or p is PASSTHROUGH_ALWAYS
        for p in processor_list
    )

  def __floordiv__(self, processor: PartProcessor) -> PartProcessor:
    if isinstance(processor, _ParallelPartProcessor):
      return _ParallelPartProcessor(
          self._processor_list + processor._processor_list
      )
    else:
      return _ParallelPartProcessor(self._processor_list + [processor])

  async def call(
      self, part: ProcessorPart
  ) -> AsyncIterable[ProcessorPartTypes]:
    async for result in _parallel_part_processors(self._processor_list)(part):
      yield result

  def match(self, part: ProcessorPart) -> bool:
    if any(p.match(part) for p in self._processor_list):
      return True
    elif self._is_passthrough:
      # no processor will process the part but we still need to pass it
      # through. Return False means the part does not enter the parallel
      # processor and is passed through as is.
      return False
    else:
      # no processor will process the part and we want to drop it. Return
      # True will ensure the part enters the parallel processor that will drop
      # it.
      return True

  @functools.cached_property
  def key_prefix(self) -> str:
    return '_ParallelPartProcessor:' + _combined_key_prefix(
        self._processor_list
    )


def _parallel_part_processors(
    part_processors: Sequence[PartProcessorWithMatchFn],
) -> PartProcessorFn:
  """Combine **part processors** in parallel.

  Adds debug and status streams to the output.

  NOTE: Substreams debug and status are yielded immediately instead of passing
        them to the next processor.

  Args:
    part_processors: sequence of part processors to compute concurrently.

  Returns:
    Part processor that computes the output of the provided sequence of part
    processors concurrently.
  """

  async def part_processor(
      content: ProcessorPart,
  ) -> AsyncIterable[ProcessorPart]:
    # Create a queue to put output parts
    output_queue = asyncio.Queue()
    # Put the reserved stream capture before the parallel processor and after
    # each processor.
    processors = []
    match_fns = []
    passthrough_fallback = False
    passthrough_always = False
    for p in part_processors:
      if p is PASSTHROUGH_FALLBACK:
        passthrough_fallback = True
        continue
      if p is PASSTHROUGH_ALWAYS:
        passthrough_always = True
        continue
      processors.append(_CaptureReservedSubstreams(output_queue, p))
      match_fns.append(p.match)
    parallel_processor = _CaptureReservedSubstreams(
        output_queue,
        map_processor.parallel_part_functions(
            processors,
            match_fns,
            with_default_output=passthrough_fallback,
            with_always_output=passthrough_always,
        ),
    )
    # Processed content
    content = parallel_processor(content)
    create_task(
        # Place output processed output parts on the queue.
        _enqueue_content(content, output_queue)
    )
    while (part := await output_queue.get()) is not None:
      yield part
      output_queue.task_done()

  return part_processor


@part_processor_function(match_fn=lambda _: False)
async def _passthrough_fallback(
    content: ProcessorPart,
) -> AsyncIterable[ProcessorPart]:
  """Passthrough fallback used for the // operations."""
  yield content


@part_processor_function(match_fn=lambda _: False)
async def _passthrough_always(
    content: ProcessorPart,
) -> AsyncIterable[ProcessorPart]:
  """Passthrough always used for the // operations."""
  yield content


# Parallel processor to add to return the input part whenever the other
# processors in the // operations do not return anything.
PASSTHROUGH_FALLBACK = _passthrough_fallback

# Parallel processor to add to return the input part in any case.
PASSTHROUGH_ALWAYS = _passthrough_always

# Part processor yielding part unchanged.
# Useful as an initialization of a chain.
passthrough = lambda: _ChainPartProcessor([])


async def process_streams_parallel(
    processor: ProcessorFn,
    content_streams: Sequence[AsyncIterable[ProcessorPart]],
) -> AsyncIterable[ProcessorPart]:
  """Processes a sequence of content streams using the specified processor."""
  async for c in streams.concat(*[processor(s) for s in content_streams]):
    yield ProcessorPart(c)


_ProcessorParamSpec = ParamSpec('_ProcessorParamSpec')


class Source(Processor):
  """A Processor that produces ProcessorParts from some external source.

  Use @processor.source decorator to construct this class. Please see its
  docstring below for details.
  """

  @abc.abstractmethod
  def __aiter__(self) -> AsyncIterator[ProcessorPart]:
    """Maintains the original signature of the wrapped source function."""


class _SourceDecorator(Protocol):
  """Forward type definition to explain pytype how @source() propagates arguments."""

  def __call__(
      self,
      source_fn: Callable[
          _ProcessorParamSpec, AsyncIterable[ProcessorPartTypes]
      ],
  ) -> Callable[_ProcessorParamSpec, Source]:
    ...


def source(stop_on_first: bool = True) -> _SourceDecorator:
  """A Processor that produces ProcessorParts from some external source.

  Writing a source is as easy as writing a generator that yields the Parts.
  For example here is one reading input from stdin:

  ```py
  @processor.source()
  async def TerminalInput(prompt: str) -> AsyncIterable[ProcessorPartTypes]:
    # We rely on asyncio task cancellation to exit the loop.
    while True:
      yield await asyncio.to_thread(input, prompt)
  ```

  The wrapped source implements the Processor interface: it accepts an input
  stream and merges it with the generated parts. So multiple sources can be
  chained:

  ```py
  p = TerminalInput('>') + audio_io.AudioIn(...) + live_model.LiveModel(...)
  async for part in p(streams.endless_stream())
  ```

  Here the input stream of the first processor source is usually the
  `streams.endless_stream()` stream, that is, an open-ended stream that never
  ends. But Source can still be used as AsyncIterator directly:

  ```py
  p = live_model.LiveModel(...)
  async for part in p(TerminalInput('>'))
  ```

  Args:
    stop_on_first: Whether to interrupt the source if the incoming iterator
      ends. For realtime "endless" sources like "stream from a webcam" it should
      be True. Then the lifetime of the processor chain is controlled by the
      lifetime of incoming stream. For finite sources like "stream files from
      the given folder" it should be False.

  Returns:
    The decorator to wrap the source function.
  """

  def source_impl(
      source_fn: Callable[
          _ProcessorParamSpec, AsyncIterable[ProcessorPartTypes]
      ],
  ) -> Callable[_ProcessorParamSpec, Source]:

    class SourceImpl(Source):
      """Adapter from the source function to a Processor."""

      def __init__(self, *args, **kwargs):
        self._source = _normalize_part_stream(
            source_fn(*args, **kwargs), producer=source_fn
        )

      async def call(
          self, content: AsyncIterable[ProcessorPart]
      ) -> AsyncIterable[ProcessorPart]:
        async for part in streams.merge(
            [content, self._source], stop_on_first=stop_on_first
        ):
          yield part

      def __aiter__(self) -> AsyncIterator[ProcessorPart]:
        # This maintains the original signature of the wrapped function.
        return self._source

    return SourceImpl

  return source_impl


def yield_exceptions_as_parts(
    func: Callable[_ProcessorParamSpec, AsyncIterable[ProcessorPartTypes]],
) -> Callable[_ProcessorParamSpec, AsyncIterable[ProcessorPartTypes]]:
  """Decorates a `call` method to yield exceptions instead of raising them.

  This decorator can be applied to the `call` method of both `Processor` and
  `PartProcessor` classes.

    For the Processor pipeline to succeed each processor in it needs to succeed,
    and if there are many of them the probability of failure increases
    exponentially.

    To mitigate that we may want to let the model work with the partial results
    by interpreting exceptions as valid results.

    This decorator wraps the `PartProcessor.call` or `Processor.call` method in
    a try...except block.

    If the method raises an exception it is yielded as a special `status`
    `ProcessorPart` that the next processors in the pipeline can interpret.

    To be model-friendly, we format the exception as a text/x-exception part
    with a machine-readable representation in the part metadata.

    Example usage:

    ```py
    class FetchWebPageProcessor(processor.PartProcessor):

      @processor.yield_exceptions_as_parts
      async def call(
          self, part: processor.ProcessorPart
      ) -> AsyncIterable[processor.ProcessorPart]:
      ...
    ```

  Args:
    func: The `call` method to wrap.

  Returns:
    The wrapped `call` method.
  """

  @functools.wraps(func)
  async def wrapper(*args, **kwargs) -> AsyncIterable[ProcessorPartTypes]:
    try:
      async for item in func(*args, **kwargs):
        yield item
    except Exception as e:  # pylint: disable=broad-except
      yield ProcessorPart(
          # NOTE: The exact formatting might change, please use
          # mime_types.is_exception to detect exception parts.
          f'An unexpected error occurred: {e!r}',
          mimetype=mime_types.TEXT_EXCEPTION,
          substream_name=STATUS_STREAM,
          metadata={
              'original_exception': str(e),
              'exception_type': type(e).__name__,
          },
      )

  return wrapper


_PROCESSOR_PART_CACHE: contextvars.ContextVar[cache_base.CacheBase | None] = (
    contextvars.ContextVar('processor_part_cache', default=None)
)


class CachedPartProcessor(PartProcessor):
  """A PartProcessor that wraps another PartProcessor with a cache.

  For each incoming part it will write the output to the cache. If the same
  part is encountered again, the cached result will be used. As PartProcessors
  handle each part independently, we can cache each part independently too while
  preserving correct order, streaming behavior and correctly propagating errors
  on a cache miss.

  The cache to use is bound to a contextvars context and can be set via
  CachedPartProcessor.set_cache classmethod. This way servers can instantiate
  processor chain in a constructor but still have separate caches for each
  request, avoiding cross-talk between users.
  """

  def __init__(
      self,
      part_processor: PartProcessor,
      *,
      key_prefix: str | None = None,
      default_cache: cache_base.CacheBase | None = None,
  ):
    """Initializes the caching wrapper.

    Args:
      part_processor: The PartProcessor instance to wrap.
      key_prefix: Optional custom prefix for the cache key. If None, defaults to
        the key_prefix of the `part_processor_to_cache`.
      default_cache: The cache to use if one is not set in the context with
        .set_cache.
    """
    self._wrapped_processor = part_processor
    self.key_prefix = key_prefix or part_processor.key_prefix
    self._default_cache = default_cache

  @classmethod
  def set_cache(cls, cache: cache_base.CacheBase) -> None:
    """Update thread-local cache to be used.

    All CachedProcessor and CachedPartProcessor instance within the current
    contextvars context will use this cache to store and retrieve results.

    Args:
      cache: Cache to use.
    """
    _PROCESSOR_PART_CACHE.set(cache)

  def match(self, part: ProcessorPart) -> bool:
    """Matches if the underlying processor matches."""
    return self._wrapped_processor.match(part)

  async def call(
      self, part: ProcessorPart
  ) -> AsyncIterable[ProcessorPartTypes]:
    part_cache = _PROCESSOR_PART_CACHE.get(self._default_cache)

    if part_cache is not None:
      part_cache = part_cache.with_key_prefix(self.key_prefix)
      key = part_cache.hash_fn(part)
      cached_result = await part_cache.lookup(part, key=key)

      if isinstance(cached_result, ProcessorContent):
        for p in cached_result.all_parts:
          yield p
        return

      results_for_caching = []
      async for p in self._wrapped_processor(part):
        results_for_caching.append(p)
        yield p

      create_task(part_cache.put(key=key, value=results_for_caching))
    else:
      async for p in self._wrapped_processor(part):
        yield p


class CachedProcessor(Processor):
  """A Processor that wraps another Processor with a cache.

  Incoming content will be buffered (breaking the input streaming), hashed and
  checked whether it has been seen before. If not, wrapped processor would be
  invoked and its output would be written to the cache.

  The cache to use is bound to a contextvars context and can be set via
  CachedProcessor.set_cache classmethod.

  This way, servers can instantiate a processor chain in a constructor, but
  still have separate caches for each request - avoiding cross-talk between
  users.
  """

  def __init__(
      self,
      processor: Processor,
      *,
      key_prefix: str | None = None,
      default_cache: cache_base.CacheBase | None = None,
  ):
    """Initializes the caching wrapper.

    Args:
      processor: The Processor instance to wrap.
      key_prefix: Optional custom prefix for the cache key. If None, defaults to
        the key_prefix of the `part_processor_to_cache`.
      default_cache: The cache to use if one is not set in the context with
        .set_cache.
    """
    self._wrapped_processor = processor
    self.key_prefix = key_prefix or processor.key_prefix
    self._default_cache = default_cache

  @classmethod
  def set_cache(cls, cache: cache_base.CacheBase) -> None:
    """Update thread-local cache to be used.

    All CachedProcessor and CachedPartProcessor instance within the current
    contextvars context will use this cache to store and retrieve results.

    Args:
      cache: Cache to use.
    """
    _PROCESSOR_PART_CACHE.set(cache)

  async def call(
      self, content: AsyncIterable[ProcessorPartTypes]
  ) -> AsyncIterable[ProcessorPartTypes]:
    part_cache = _PROCESSOR_PART_CACHE.get(self._default_cache)

    if part_cache is not None:
      content = await gather_stream(content)

      part_cache = part_cache.with_key_prefix(self.key_prefix)
      key = part_cache.hash_fn(content)
      cached_result = await part_cache.lookup(content, key=key)

      if isinstance(cached_result, ProcessorContent):
        for p in cached_result.all_parts:
          yield p
        return

      results_for_caching = []
      async for p in self._wrapped_processor(stream_content(content)):
        results_for_caching.append(p)
        yield p

      if results_for_caching:
        create_task(part_cache.put(key=key, value=results_for_caching))
    else:
      async for p in self._wrapped_processor(content):
        yield p
