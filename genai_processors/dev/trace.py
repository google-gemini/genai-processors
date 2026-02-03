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
import asyncio
import contextvars
import datetime
from typing import Any

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

  # The timestamp when the trace was started (the processor was called).
  start_time: datetime.datetime = pydantic.Field(
      default_factory=datetime.datetime.now
  )
  # The timestamp when the trace was ended (the processor returned).
  end_time: datetime.datetime | None = None

  _token: contextvars.Token[Trace | None] | None = pydantic.PrivateAttr(
      default=None
  )
  # True if this trace is a sub-trace (not the main/root trace).
  is_sub_trace: bool = False

  async def __aenter__(self) -> Trace:
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

    # Skip finalizing sub-traces when the task is cancelled and no output part
    # has been produced yet. This avoids recording incomplete traces from
    # cancelled processors that have not run yet.
    try:
      current_task = asyncio.current_task()
      if (
          self.is_sub_trace
          and current_task is not None
          and current_task.cancelled()
      ):
        self.cancel()
        CURRENT_TRACE.reset(self._token)
        return
    except RuntimeError:
      pass

    self.end_time = datetime.datetime.now()
    CURRENT_TRACE.reset(self._token)
    # Shield the finalize call to avoid cancellation. This is to ensure that
    # the trace is always finalized (i.e. traces are saved), even if the context
    # is cancelled or the task is cancelled.
    await asyncio.shield(self._finalize())

  @abc.abstractmethod
  async def add_input(self, part: content_api.ProcessorPart) -> None:
    """Adds an input part to the trace events."""
    raise NotImplementedError()

  @abc.abstractmethod
  async def add_output(self, part: content_api.ProcessorPart) -> None:
    """Adds an output part to the trace events."""
    raise NotImplementedError()

  @abc.abstractmethod
  def add_sub_trace(self, name: str, relation: str) -> Trace:
    """Adds a sub-trace from a nested processor call to the trace events.

    Args:
      name: The name of the sub-trace.
      relation: The relation between this trace and sub-trace.

    Returns:
      The trace that was added to the trace events.
    """
    # TODO(elisseeff, kibergus): consider adding a more generic relationship
    # between traces, e.g. traces generated one after another (wiht the + ops)
    # or traces generated in parallel (with the // ops).
    raise NotImplementedError()

  @abc.abstractmethod
  def cancel(self) -> None:
    """Cancels the trace.

    This method is called when the task producing the trace is cancelled. The
    implementation should remove all unnecessary data from the trace to avoid
    recording incomplete traces from cancelled processors that have not produced
    anything yet.

    If the processor has output some parts, it is still considered successful
    and the trace should be stored, up to the cancellation point.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  async def _finalize(self) -> None:
    """Finalize the trace.

    At this stage, the trace is ready to be stored and/or displayed. It is up
    to the implementer to decide how to store the trace. When this function
    returns all traces should be considered finalized and stored.
    """
    raise NotImplementedError()


# TODO(kibergus): Context managers don't work well with Generators. Correct
# nested call tracking will be implemented later.
CURRENT_TRACE: contextvars.ContextVar[Trace | None] = contextvars.ContextVar(
    'current_trace', default=None
)


# Modules that should be excluded from tracing.
# Processors from these modules are internal/framework processors that add noise
# to traces. Sub-processors of these processors will still be traced.
# Use add_excluded_trace_module() to register additional modules.
EXCLUDED_TRACE_MODULES: set[str] = {
    'genai_processors.debug',
    'genai_processors.map_processor',
}


def add_excluded_trace_module(module_name: str) -> None:
  """Adds a module to the list of excluded trace modules.

  Processors from excluded modules will not create their own traces, but
  sub-processors they call will still be traced.

  Args:
    module_name: The module name to exclude (e.g. 'genai_processors.debug').
  """
  EXCLUDED_TRACE_MODULES.add(module_name)


def is_module_excluded(module_name: str | None) -> bool:
  """Returns True if the module should be excluded from tracing."""
  if module_name is None:
    return False
  return any(
      module_name == excluded or module_name.startswith(excluded + '.')
      for excluded in EXCLUDED_TRACE_MODULES
  )


def create_sub_trace(
    processor_name: str,
    parent_trace: Trace | None,
    *,
    module_name: str | None = None,
) -> Trace | None:
  """Context manager for tracing a processor call.

  Args:
    processor_name: The name/key_prefix of the processor.
    parent_trace: The parent trace (from the calling processor's stream).
    module_name: The module name of the processor. If provided, processors from
      excluded modules will be skipped.

  Returns:
    A new trace if tracing is enabled, or the parent trace if the processor
    should be skipped. Returns None if no tracing is in scope.
  """
  # NOTE: This interface will change when we add nested call tracking.

  # Skip tracing for excluded modules. Return None so the processor uses
  # nullcontext(). Sub-processors will still be traced because they get the
  # parent trace from CURRENT_TRACE.get() context variable.
  if is_module_excluded(module_name):
    return None

  relation = 'chain' if parent_trace else 'call'
  parent_trace = parent_trace or CURRENT_TRACE.get()

  if parent_trace is None:
    # No tracing in scope - keep things as is.
    return None
  else:
    # Parent is not None and corresponds to an existing trace: adds a new trace.
    new_trace = parent_trace.add_sub_trace(
        name=processor_name, relation=relation
    )
    new_trace.is_sub_trace = True
    return new_trace
