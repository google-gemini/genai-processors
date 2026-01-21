"""Abstract class of a trace to collect, work with and display processor traces.

A GenAIprocessor trace is a timeline of input and output events that
were used in a GenAI processor. It includes the user input and potentially the
audio and/or video stream in case of a realtime processor. The trace also
includes the function calls and responses made by the processor. Finally, it
includes the model output parts and any other arbitrary parts produced by the
processor. An event can also be a trace itself if a processor calls another one
internally.

A trace corresponds to a single processor call. If the processor is called
multiple times, multiple traces will be produced, each containing the input
used to call the processor and the output produced by the call.

__WARNING__: This is an incubating feature. The trace format is subject to
changes and we do not guarantee backward compatibility at this stage.
"""

from __future__ import annotations

import abc
import contextlib
import contextvars
import datetime
from typing import Any

from absl import logging
from genai_processors import content_api
import pydantic
import shortuuid


pydantic_converter = pydantic.TypeAdapter(Any)


class Trace(pydantic.BaseModel, abc.ABC):
  """A trace of a processor call.

  A trace contains some information about when the processor was called and
  includes methods to log input, output and sub-traces to the trace.

  The finalize method must be called to finalize the trace and release any
  resources.

  This is up to the implementer to decide how to store the trace.

  The add_sub_trace method should be used to create a new trace.
  """

  model_config = {'arbitrary_types_allowed': True}

  # Name of the trace.
  name: str | None = None

  # A description of the processor that produced this trace, i.e. arguments used
  # to construct the processor.
  processor_description: str | None = None

  # A unique ID for the trace.
  trace_id: str = pydantic.Field(default_factory=lambda: str(shortuuid.uuid()))

  # Boolean indicating whether the trace has just been created. This is used to
  # determine whether to create a subtrace when a processor is called or using
  # the existing trace when it's just been created.
  is_new: bool = False

  # The timestamp when the trace was started (the processor was called).
  start_time: datetime.datetime = pydantic.Field(
      default_factory=datetime.datetime.now
  )
  # The timestamp when the trace was ended (the processor returned).
  end_time: datetime.datetime | None = None

  _token: contextvars.Token[Trace | None] | None = pydantic.PrivateAttr(
      default=None
  )

  async def __aenter__(self) -> Trace:
    parent_trace = CURRENT_TRACE.get()

    if parent_trace:
      logging.warning(
          'Cannot enter a trace while another trace is already in scope: %s is'
          ' ignored in favor of %s',
          self,
          parent_trace,
      )

    self.is_new = True
    self._token = CURRENT_TRACE.set(self)
    return self

  async def __aexit__(
      self,
      exc_type: type[BaseException] | None,
      exc_val: BaseException | None,
      exc_tb: Any,
  ) -> None:
    if self._token is None:
      return

    self.end_time = datetime.datetime.now()
    CURRENT_TRACE.reset(self._token)
    await self._finalize()

  @abc.abstractmethod
  async def add_input(self, part: content_api.ProcessorPart) -> None:
    """Adds an input part to the trace events."""
    raise NotImplementedError()

  @abc.abstractmethod
  async def add_output(self, part: content_api.ProcessorPart) -> None:
    """Adds an output part to the trace events."""
    raise NotImplementedError()

  @abc.abstractmethod
  async def add_sub_trace(self, name: str) -> Trace:
    """Adds a sub-trace from a nested processor call to the trace events.

    Args:
      name: The name of the sub-trace.

    Returns:
      The trace that was added to the trace events.
    """
    # TODO(elisseeff, kibergus): consider adding a more generic relationship
    # between traces, e.g. traces generated one after another (wiht the + ops)
    # or traces generated in parallel (with the // ops).
    raise NotImplementedError()

  @abc.abstractmethod
  async def _finalize(self) -> None:
    """Finalize the trace.

    At this stage, the trace is ready to be stored and/or displayed. It is up
    to the implementer to decide how to store the trace. When this function
    returns all traces should be considered finalized and stored.
    """
    raise NotImplementedError()


CURRENT_TRACE: contextvars.ContextVar[Trace | None] = contextvars.ContextVar(
    'current_trace', default=None
)


@contextlib.asynccontextmanager
async def call_scope(processor_name: str):
  """Context manager for tracing a processor call."""
  parent_trace = CURRENT_TRACE.get()

  if parent_trace is None:
    # No tracing in scope - keep things as is.
    yield None
  elif parent_trace.is_new:
    # First call to a processor - re-use the root trace. It has been created
    # when the trace_scope was entered.
    parent_trace.name = processor_name
    parent_trace.is_new = False
    yield parent_trace
  else:
    # Parent is not None and corresponds to an existing trace: adds a new trace.
    async with await parent_trace.add_sub_trace(
        name=processor_name
    ) as new_trace:
      yield new_trace
