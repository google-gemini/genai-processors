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
r"""Chaining of a sequence of 1:many functions across a stream concurrently.

To execute all functions concurrently we build up a tree of results.
Given:
- A content stream of c={c0, c1, .., cN}
- A sequence of functions f={f0, f1, .., fM}

We have the following tree with streaming content on the x-axis and functions
on the y-axis.

    -> Content Stream ->
     c0   c1  c2  .. cN

f0   o    o   p   ..
     |\   |   |
f1   p o  o   o
       |  |
f2     o  o
       |  |
..
       |  |
fM     r  r

   o = complete node (all children fetched)
   p = pending node (potentially more children)
   r = result node (has a result to be yielded)

For the tree above the following has happened:
- f0(c0) has yielded 2 children
- f1(f0(c0)[0]) is executing and hasn't yielded any children yet
- fM(... f0(c0)[1]) has executed all functions and has a result to be returned
- fM(... f0(c1)[0]) has executed all functions and has a result to be returned
- f0(c2) has yielded 1 child and is pending (may yield more children)
- f1(f0(c2)[0]) has finished and returned no children


Currently no results have been returned to the user as the left-most node is
still pending. Once the 'f1(f0(c0)[0])' sub-tree completes, its results
(along with the two that are buffered) will be returned in order.
"""

import asyncio
from collections.abc import AsyncIterable, Callable, Iterable, Sequence
import functools
from typing import TypeAlias, TypeVar

from genai_processors import context
from genai_processors import streams


_T = TypeVar('_T')

# Models a part function (i.e. part processors): T -> stream[T]
PartFn: TypeAlias = Callable[[_T], AsyncIterable[_T]]
# Models a function that returns True if the part should be processed. When a
# part should not be processed, the part processor will not be called and
# the part will be passed as is for chains, and apply, or will be dropped for
# parallel.
MatchFn: TypeAlias = Callable[[_T], bool]
PartWithMatchFn: TypeAlias = tuple[PartFn, MatchFn]
# Models a stream function (i.e. processors): stream[T] -> stream[T]
StreamFn: TypeAlias = Callable[[AsyncIterable[_T]], AsyncIterable[_T]]


def apply_sync(fn: StreamFn, content: Iterable[_T]) -> list[_T]:
  """Applies a part function synchronously.

  Args:
    fn: the part function to apply to the content.
    content: a collection of inputs/parts on which to apply the function.

  Returns:
    the content, with the function `fn` applied to each input/part.
  """

  async def run_with_context():
    async with context.context():
      as_async = streams.stream_content(content)
      return await streams.gather_stream(fn(as_async))

  return asyncio.run(run_with_context())


def map_part_function(
    fn: PartFn,
    match_fn: MatchFn | None = None,
) -> StreamFn:
  """Converts a part function to a function taking a stream of parts.

  Adds a context if missing to ensure error propagation.

  Args:
    fn: a function that can be applied on a single part.
    match_fn: a function that returns True if the part should be processed by
      the part function. When the part should not be processed, the part
      processor will not be called and the part will be passed as is.

  Returns:
    A function that is applied concurrently across the parts of the input
    stream.
  """
  match_fn = match_fn or (lambda _: True)
  return functools.partial(_apply_part_function, (fn, match_fn))


def _to_tuple_fns(
    fns: Sequence[PartFn], match_fns: Sequence[MatchFn] | None
) -> Sequence[PartWithMatchFn]:
  """Converts a sequence of functions to a sequence of function tuples."""
  match_fns = match_fns or [lambda _: True for _ in fns]
  if len(fns) != len(match_fns):
    raise ValueError(
        'fns and match_fns must be the same length. Got'
        f' {len(fns)} != {len(match_fns)}.'
    )
  return list(zip(fns, match_fns))


def chain_part_functions(
    fns: Sequence[PartFn],
    match_fns: Sequence[MatchFn] | None = None,
) -> PartFn:
  """Chain the `fns` and execute them concurrently.

  See file comment.

  Args:
    fns: sequence of part functions to chain.
    match_fns: sequence of functions that return True if the part should be
      processed by the part function. When the part should not be processed, the
      part function will not be called and the part will be passed as is. When
      match_fns is not provided, all parts are processed by default.

  Returns:
    Part function that is a chain of the provided Sequence of functions.

  Raises:
    ValueError: if the length of fns and match_fns is not the same (when
    match_fns is provided).
  """
  return functools.partial(_chain_part_functions, _to_tuple_fns(fns, match_fns))


def parallel_part_functions(
    fns: Sequence[PartFn],
    match_fns: Sequence[MatchFn] | None = None,
    with_default_output: bool = False,
    with_always_output: bool = False,
) -> PartFn:
  """Combine `fns` to execute on _T in parallel across the `fns`.

  Args:
    fns: sequence of part functions to chain.
    match_fns: sequence of functions that return True if the part should be
      processed by the part function. When the part should not be processed, the
      part function will not be called and nothing will be yielded by the part
      function. When match_fns is not provided, all parts are processed by
      default.
    with_default_output: True when the parallel execution should fallback to
      return the input part as is when fns do not return any output part.
    with_always_output: True when the parallel execution should always return
      the input part as is independent of the output of the fns. This is a
      stronger condition than `with_default_output`. When `with_always_output`
      is True, `with_default_output` is basically ignored.

  Returns:
    Part function that runs all functions 'fns' in parallel. The output stream
    will keep the order of the input parts:

    f_0(c) = c00, c01
    f_1(c) = c10, c11, c12, c14

    f_0(c) // f_1(c) = c00, c01, c10, c11, c12
  """
  return functools.partial(
      _parallel_part_functions,
      _to_tuple_fns(fns, match_fns),
      with_default_output=with_default_output,
      with_always_output=with_always_output,
  )


# -------- Part Function Methods ----------


class _Finished:
  """A constant that represents that a generator end has been reached."""


_FinishedT = type[_Finished]


def _eager_run_fn(
    fn: PartFn,
    part: _T,
) -> AsyncIterable[_T]:
  """Executes fn on part in an asyncio.task.

  Must be called in an async context. It eagerly schedules a task on
  the event loop to execute the whole of `fn` on the part. Results from the
  AsyncIterable returned by `fn` can be retrieved via the AsyncIterable returned
  by this method.

  Args:
    fn: the part function to execute on the part.
    part: the part to execute the function on.

  Returns:
    An AsyncIterable that can be used to retrieve the results of `fn` on `part`
    in order.

  NOTE: this method is non-blocking.
  """
  q = asyncio.Queue[_T | _FinishedT]()

  async def call_fn():
    async for c in fn(part):
      q.put_nowait(c)
    q.put_nowait(_Finished)

  # Adds execution to the event loop
  context.create_task(call_fn())

  # AsyncIterable to retrieve results from q
  async def result_iter():
    while (c := await q.get()) is not _Finished:
      yield c

  return result_iter()


async def _passthrough(part: _T) -> AsyncIterable[_T]:
  yield part


async def _result_aiter(
    q: asyncio.Queue[AsyncIterable[_T] | _FinishedT],
) -> AsyncIterable[_T]:
  """Flattens the queue of aiters into a single aiter."""
  while (c_iter := await q.get()) is not _Finished:
    async for c in c_iter:
      yield c


def _chain_part_functions(
    fns: Sequence[PartWithMatchFn],
    part: _T,
) -> AsyncIterable[_T]:
  """Executes a sequence of functions (fn) on a part.

  This executes a tree of work as described in the module level comment.

  Consider composing 2 PartFns `(f, g)` on a part `c`. We create a new
  PartFn:
  ```
  c -> flatten(g(r) for r in f(c))
  ```
  This is a tree of work with depth 2. `_chain_part_functions` supports
  arbitrary amounts of functions.

  This must be called in an async context. It immediately schedules tasks
  on the event loop to execute the tree of work on the part. Results from the
  AsyncIterable returned by composing `fns` can be retrieved via the
  AsyncIterable returned by this method.

  Args:
    fns: the function tuples to execute on the part. The first element of the
      tuple is the part function, the second element is a function that returns
      True if the part should be processed by the part function. When the part
      should not be processed, the part processor will not be called and the
      part will be passed as is saving the creation of a new task.
    part: the part to execute the function on.

  Returns:
    An AsyncIterable that can be used to retrieve the results of running the
    composition of `fns` on `part` in order.

  NOTE: this method is non-blocking.
  """
  (fn, match_fn), *fns = fns

  if not fns:
    # Base case - do not spawn a new task if part is just passed through.
    if match_fn(part):
      return _eager_run_fn(fn, part)
    else:
      return _passthrough(part)
  else:
    # Recursive case
    q = asyncio.Queue[AsyncIterable[_T] | _FinishedT]()

    if not match_fn(part):
      return _chain_part_functions(fns, part)

    async def f():
      async for c in fn(part):
        q.put_nowait(_chain_part_functions(fns, c))
      q.put_nowait(_Finished)

    context.create_task(f())

    return _result_aiter(q)


def _apply_part_function(
    fn: PartWithMatchFn, content: AsyncIterable[_T]
) -> AsyncIterable[_T]:
  """Applies a part function to a stream of parts."""
  q = asyncio.Queue[AsyncIterable[_T] | _FinishedT]()
  fn, match_fn = fn

  async def f():
    async for c in content:
      if match_fn(c):
        q.put_nowait(_eager_run_fn(fn, c))
      else:
        q.put_nowait(_passthrough(c))
    q.put_nowait(_Finished)

  # Adds execution to the event loop
  context.create_task(f())

  return _result_aiter(q)


def _parallel_part_functions(
    fns: Sequence[PartWithMatchFn],
    part: _T,
    with_default_output: bool = False,
    with_always_output: bool = False,
) -> AsyncIterable[_T]:
  """Executes each part function in a sequence of part functions concurrently.

  This method is similar to `_chain_part_functions` except that all of the
  PartFns are executed on exactly `part` instead of being chained together.
  The resulting AsyncIterables returned by call each fn are concatenated
  together in the provided fns order.

  This must be called in an async context. It immediately schedules tasks
  on the event loop to execute each fn in fns on the part.

  Args:
    fns: the part functions to execute on the part.
    part: the part to execute the function on.
    with_default_output: When True if the resulting Iterable is empty `part`
      will be yielded.
    with_always_output: When True the input part will be yielded regardless of
      the output of the fns. This is a stronger condition than
      `with_default_output`. When `with_always_output` is True,
      `with_default_output` is basically ignored.

  Returns:
    An AsyncIterable that can be used to retrieve the results.

  NOTE: this method is non-blocking.
  """
  c_iters = [_eager_run_fn(fn, part) for fn, match_fn in fns if match_fn(part)]

  async def result_iter():
    has_output = False
    for c_iter in c_iters:
      async for c in c_iter:
        has_output = True
        yield c

    if with_always_output or (not has_output and with_default_output):
      yield part

  return result_iter()
