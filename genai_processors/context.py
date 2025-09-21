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
"""Context vars for managing task groups."""

import asyncio
from collections.abc import Coroutine, Iterable
import contextvars
from typing import Any, TypeVar

from absl import logging


_PROCESSOR_TASK_GROUP: contextvars.ContextVar['CancellableContextTaskGroup'] = (
    contextvars.ContextVar('processor_task_group')
)

PROMPT_STREAM = 'prompt'
DEBUG_STREAM = 'debug'
STATUS_STREAM = 'status'
_PROCESSOR_RESERVED_SUBSTREAMS: contextvars.ContextVar[frozenset[str]] = (
    contextvars.ContextVar(
        'processor_reserved_substreams',
        default=frozenset({DEBUG_STREAM, STATUS_STREAM}),
    )
)


def raise_flattened_exception_group(exception: Exception):
  e = exception
  while isinstance(e, ExceptionGroup):
    e = e.exceptions[0]
  if e is exception:
    raise exception
  else:
    raise e from exception


class CancellableContextTaskGroup(asyncio.TaskGroup):
  """TaskGroup that adds itself to a contextvar to be accessed by create_task.

  Includes a method for cancelling all tasks in the group.
  """

  def __init__(
      self, *args, reserved_substreams: Iterable[str] | None = None, **kwargs
  ):
    super().__init__(*args, **kwargs)
    self._cancel_tasks = set()
    self._reserved_substreams = reserved_substreams

  def create_task(self, *args, **kwargs) -> asyncio.Task:
    t = super().create_task(*args, **kwargs)
    self._cancel_tasks.add(t)
    t.add_done_callback(self._cancel_tasks.discard)
    return t

  async def __aenter__(self) -> 'CancellableContextTaskGroup':
    self._current_taskgroup_token = _PROCESSOR_TASK_GROUP.set(self)
    if self._reserved_substreams is not None:
      self._reserved_substreams_token = _PROCESSOR_RESERVED_SUBSTREAMS.set(
          frozenset(self._reserved_substreams)
      )
    else:
      self._reserved_substreams_token = None
    return await super().__aenter__()

  async def __aexit__(self, et, exc, tb):
    try:
      return await super().__aexit__(et, exc, tb)
    except BaseExceptionGroup as e:
      raise_flattened_exception_group(e)
    finally:
      try:
        _PROCESSOR_TASK_GROUP.reset(self._current_taskgroup_token)
        if self._reserved_substreams_token is not None:
          _PROCESSOR_RESERVED_SUBSTREAMS.reset(self._reserved_substreams_token)
      except ValueError:
        # ValueError is raised when the Context self._current_taskgroup_token
        # was created in doesn't match the current Context.
        # This can happen when an asyncgenerator is garbage collected.
        # The task loop will call loop.call_soon(loop.create_task, agen.aclose).
        # aclose raises GeneratorExit in the asyncgenerator which is seen here
        # but from a different context.
        if et is GeneratorExit:
          logging.log_first_n(
              logging.WARNING,
              'GeneratorExit was seen in processors.context. This'
              ' indicates that the asyncgenerator that opened the context has'
              ' been closed from a different context. For example, it has been'
              ' garbage collected. This is usually means a task executing the'
              ' generator was also garbage collected. Consider turning on'
              ' asyncio debug mode to investigate further.',
              1,
          )
          pass

  def cancel(self):
    for task in self._cancel_tasks:
      task.cancel()


def context(
    reserved_substreams: Iterable[str] | None = None,
) -> CancellableContextTaskGroup:
  return CancellableContextTaskGroup(reserved_substreams=reserved_substreams)


def task_group() -> CancellableContextTaskGroup | None:
  return _PROCESSOR_TASK_GROUP.get(None)


def get_reserved_substreams() -> frozenset[str]:
  return _PROCESSOR_RESERVED_SUBSTREAMS.get()


def is_reserved_substream(substream_name: str) -> bool:
  return any(
      substream_name.startswith(prefix) for prefix in get_reserved_substreams()
  )


# If a task is created without a task group then a reference to it must be kept.
_without_context_background_tasks = set()


def create_task(*args, **kwargs) -> asyncio.Task:
  """Creates a task that uses the context TaskGroup.

  If no context is available then `asyncio.create_task` will be used.

  Args:
    *args: Positional arguments to pass to `asyncio.create_task`.
    **kwargs: Keyword arguments to pass to `asyncio.create_task`.

  Returns:
    An asyncio task.
  """
  tg = task_group()

  if tg is None:
    task = asyncio.create_task(*args, **kwargs)
    _without_context_background_tasks.add(task)
    task.add_done_callback(_without_context_background_tasks.discard)
    return task

  return tg.create_task(*args, **kwargs)


_T = TypeVar('_T')


async def context_cancel_coro(
    f: Coroutine[Any, Any, _T],
) -> _T:
  """Wrapper that cancels all tasks in context if the wrapper is cancelled."""
  async with context() as ctx:
    try:
      return await f
    except asyncio.CancelledError:
      ctx.cancel()
      raise
