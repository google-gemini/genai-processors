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
"""Function calling processor.

The FunctionCalling processor is a loop over a model call that intercepts any
function call, runs the function, injects the result back into the model prompt
and goes to the next iteration until no function call is made.

The term function here is a bit of a misnomer as not all functions can be
called: its parameters must be JSON-serializable, see
https://ai.google.dev/gemini-api/docs/function-calling?example=meeting#schema.
Its responses should also either be JSON-serializable values or should return a
genai_types.FunctionResponse. We will therefore refer to such functions as tools
from now on to avoid confusion with functions defined in Python.

We define two types of tools: sync and async. Both can be easily defined from
sync of async Python functions.

For sync tools, the function calling processor will wait until the function
returns the result. This is the default behavior when the model is turn-based.
When the model is bidi (aka realtime), the function will be run in a separate
thread and any result will trigger an interruption of the model output and a new
model call.

If you provide an async tool, the function calling processor will return a 'OK,
running in the background.' response to the model immediately and will send back
the result later on when it is available. Such behavior happens for turn-based
models as well as for realtime models. For turn-based models, the model will
wait for the async tools to return the result before returning to the user.
If the model outputs more than one async function/tool call, these tools will
be run in parallel and the results can be interleaved. The function responses
contain the name of the tool that returned the response. This allows you to
detect which response corresponds to which tool call.

We therefore have the following 4 cases:

- Turn-based model + sync tool: tool is blocking future model calls. It is run
in a separate thread. Many can be run concurrently if the model output more than
one tool call at once. The model will wait for all tools to finish before
returning.
- Turn-based model + async tool: tool is non-blocking, allowing the model to
continue returning tool calls in subsequent model calls. The model will wait for
all tools to finish before returning control to the user.
- Realtime model + sync tool: tool is automatically converted to an async tool.
to let the realtime model process new incoming input without being blocked.
- Realtime model + async tool: tool is non-blocking, many can be run in
parallel and async tools can stream responses back to the model as they become
available.

For async tools and more generally for any tool used in with a realtime model,
you will get a first and immediate 'OK, running in the background.' response
containing a tool call ID, which is needed to associate future function
responses to the  correct calls. For sync tools, you will get the result as a
single function response when the tool returns. It is scheduled just after the
tool call.

We recommend using sync tools for turn-based models as they have
simpler formatting: they don't need the 'OK, running in the background.'
response with a tool call ID as the function response is returned after the tool
call.

When using a realtime model, sync tools are wrapped into a thread and de facto
becomes async to avoid blocking the realtime model. This means the model will
not wait for the tool to finish before returning and can process new input while
the tool is running. This means any tool used in a realtime model will have
a first 'OK, running in the background.' immediate response containing a tool
call ID.

Function calling processor provides a function to cancel any async tool. This
cancellation is implemented by the function calling processor directly. You
still need to provide the function declaration to the model in its prompt to
make it aware that this function is available. See below how to add the
cancel function to the model prompt. If you do not provide this cancel function
to the model, function calling will still work but the user will not be able to
stop an async tool.

You can define an async tool from a Python async function or from a Python async
generator. Async generators will stream responses back to the model whenever it
can. This can be used for proactive applications, e.g. geofence tool can yield a
response every time a user enters or exits their home, thus waking an assistant
and giving it a chance to react.

The response of async tools are automatically wrapped into
`genai_types.FunctionResponse`. This is what is returned by the function calling
processor. To control how the function response is added to the model prompt,
you can return `FunctionResponse` directly from your async tool function.

The `FunctionResponse` object includes a scheduling option that lets you control
how the function response is added to the model prompt:

- SILENT: the response is added to the model prompt but no model output is
generated. The model will see it next time something else triggers the
generation.
- INTERRUPT: the response is added to the model prompt, and the model output is
generated interrupting any ongoing model output.
- WHEN_IDLE: we wait for the model to output its current result (if there is
an ongoing one) then the response is appended to the prompt and the model is
requested to generate a response. This is the default option.

Note that you can return FunctionResponse in an async generator function.

Any async tool (or sync tool in a bidi model) called by the model will trigger
the following `FunctionResponse`. This is done automatically by this function
calling processor:

```python
FunctionResponse(
    name='<tool_name>',  # name of the tool or of the python function
    function_call_id='<id provided by the function call>',
    response={'result': 'Running in background.'},
    role='user',
    substream_name=<function_call_substream_name>, # provided to the ctor.
    scheduling='SILENT', # does not trigger any model output. just for info.
)
```

The function calling processor will populate any function ouptut with the proper
tool call ID and substream name, even when the function returns a
`FunctionResponse` directly.

This function calling processor focuses on the function calling loop and on the
execution of tools only. The model is responsible for formatting/parsing these
parts into its internal representation.

The overall logic can be depicted as:

```
    input -> pre_processor -> model -----+----> output
                  ^                      |
                  |                      v
            function run ----<---- function call
```

where:
-  `input` is the input stream.
-  `pre_processor` is a processor called once on `input` and on all results
obtained by tool invocation. This could for instance be an image tokenizer to
preprocess images and avoid re-tokenizing them on every model invocation.
-  `model` is a model processor that generates content. For the model to know
which tools are available, the samefunction/tool set should be given both to the
FunctionCalling and the model constructor. This model can be turn-based or
bidi but this should be specificed in the FunctionCalling ctor.
-  `function_call` executes the function calls returned by the model. This is
a private processor in this file. The function response is then fed back to the
model for another iteration. If there is no function call to execute and the
the model is turn-based, the loop is stopped and the current tool_use output is
sent back to the output stream.

The output is the model output including the function call and the responses.
They are all in the same substream as identified by `substream_name` in the
FunctionCalling ctor.

When used with GenAI models (i.e. Gemini API), the model should be defined as
follows:

```python
genai_processor = genai_model.GenaiModel(
    api_key=API_KEY,
    model_name='gemini-2.5-flash',
    generate_content_config=genai_types.GenerateContentConfig(
        tools=[fns],
        automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(
            disable=True
        ),
    ),
)
```

where `fns` are the python functions to be called. Note that we disable the
automatic function calling feature here to avoid duplicate function calls with
the GenAI automatic function calling feature.

If you want to allow cancelling ongoing async functions add `cancel_fc` and
`list_fc` tool defined in this file:

```python
generate_content_config=genai_types.GenerateContentConfig(
    tools=[fns, function_calling.cancel_fc, function_calling.list_fc],
    automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(
        disable=True
    ),
)
```

You define the function `fns` as regular python functions. Arguments and
outputs need to be JSON-serializable for models to be able to generate and
ingest them:

```python
def get_weather(location: str, period: Tuple[str, str] | None = None) -> str:
  '''Returns the weather for the provided period of time.

  Args:
    location: The location to get the weather for.
    period: A (start_time, end_time) tuple.

  Returns:
    A string describing the weather for the provided periods of time. If no
    period is provided, the current weather is returned.
  '''
  return 'Sun is shining'
```

We provide an extensive docstring here as this is usually used by the model to
add function declarations to its prompt. Supplying async Python functions
directly can be done as follows:

```python
async get_weather(location: str) -> str:
  '''Returns the current weather at the provided location.

  Args:
    location: The location to get the weather for.

  Returns:
    A string describing the weather for the provided location at the current
    time.
  '''
  response = await httpx.AsyncClient(...).get()
  return response.json()['temperature']
```

This creates an async tool automatically: the function calling processor will
return a 'OK, running in the background.' response to the model immediately and
will send back the result later on when it is available.

When an async function returns more than one response (over a longer period of
time), it is recommended to define an async generator function:

```python
async set_timer(countdown_seconds: int) -> AsyncIterable[str]:
  '''Sets a timer for the given number of seconds.

  Args:
    countdown_seconds: The number of seconds to set the timer for.

  Yields:
    A string describing the progress of the timer every second.
  '''
  await asyncio.sleep(countdown_seconds)
  yield f'{countdown_seconds} seconds have passed.'
```
  count_tick = 0
  while (count_tick < countdown_seconds)
    await asyncio.sleep(1)
    count_tick += 1
    yield f'{count_tick} seconds have passed.'

The function calling processor will detect async generators and will stream
responses back to the model whenever it can. This can be used for proactive
applications when an async generator function for instance sends signals to the
model every time it detects a change in a video feed.

When the end of the async generator is reached, the async tool will return an
empty `FunctionResponse` with the `will_continue` field set to `False` and the
scheduling set to `SILENT` (for info to the model). All other function responses
in the async generator are returned with the `will_continue` field set to
`True` to indicate that the tool is continuing to run and might produce more
responses.

As discussed above, you can return FunctionResponse directly from your async
tool function or async generator function to control how the function response
is added to the model prompt.
"""

import asyncio
import collections
from collections.abc import AsyncIterable
import dataclasses
import inspect
import math
from typing import Any, Callable

from genai_processors import content_api
from genai_processors import context as context_lib
from genai_processors import processor
from genai_processors import streams
from google.genai import _extra_utils
from google.genai import types as genai_types

# All function call parts (calls and responses) are emitted in a substream
# with this name. This is to help downstream processors to identify function
# calls that were executed from function calls returned directly by the model.
FUNCTION_CALL_SUBSTREAM_NAME = 'function_call'

# Internal metadata key to carry the scheduling of a function call in an
# EndOfTurn part.
_SCHEDULING_METADATA_KEY = '__scheduling__'


@dataclasses.dataclass
class _FunctionCallState:
  """State of the function calling processor."""

  # Number of async function calls that are currently running.
  running_fc_count: int = 0

  # Whether the model is currently outputting.
  model_outputting: bool = True
  # Number of tool calls that have been made.
  fn_call_count: int = 0
  # Maximum number of tool calls to make.
  fn_call_count_limit: float = math.inf
  # Whether a model call should be scheduled after an async function sent back
  # a part to the model prompt.
  schedule_model_call: bool = False


def _to_bidi(p: processor.Processor) -> processor.Processor:
  """Converts a unary processor to a bidi processor.

  A bidi (aka realtime) processor has actually more features than what we return
  here: eager processing of input, to handle start/end of speech signals, etc.
  Here, we need a thin adapter from turn-based to bidi using EoT to trigger a
  model call. This is to keep the same function calling loop between bidi and
  non-bidi so the logic stays the same as much as possible.

  Args:
    p: The unary processor to convert.

  Returns:
    A processor that can take an infinite input stream and will trigger model
    calls whenever an EndOfTurn is received.
  """

  @processor.processor_function
  async def call(
      content: AsyncIterable[content_api.ProcessorPart],
  ) -> AsyncIterable[content_api.ProcessorPart]:
    prompt = []
    dirty_prompt = False
    async for part in content:
      if content_api.is_end_of_turn(part):
        dirty_prompt = False
        async for part in p(streams.stream_content(prompt)):
          yield part
        yield content_api.END_OF_TURN
      else:
        prompt.append(part)
        if part.role == 'user':
          dirty_prompt = True
    if dirty_prompt:
      async for part in p(streams.stream_content(prompt)):
        yield part
      yield content_api.END_OF_TURN

  return call


async def _add_end_of_turn(
    content: AsyncIterable[content_api.ProcessorPart],
) -> AsyncIterable[content_api.ProcessorPart]:
  """Adds an end of turn to the content if it is not already present."""
  last_part = None
  async for part in content:
    yield part
    last_part = part
  if last_part is None or not content_api.is_end_of_turn(last_part):
    yield content_api.END_OF_TURN


class FunctionCalling(processor.Processor):
  """Tool use with Function Calling.

  See file level docstring for more details.
  """

  def __init__(
      self,
      model: processor.Processor,
      *,
      is_bidi_model: bool = False,
      substream_name: str = FUNCTION_CALL_SUBSTREAM_NAME,
      pre_processor: (
          processor.Processor | processor.PartProcessor | None
      ) = None,
      fns: list[Callable[..., Any]] | None = None,
      max_function_calls: int | None = None,
  ):
    """Initializes the FunctionCalling processor.

    Args:
      model: The processor to use for generation. The model can be a bidi or
        unary model. If it is a unary model, the function calling processor will
        convert it to a bidi processor. If this is a realtime model like the
        Gemini Live API or like the `realtime.LiveProcessor`, `is_bidi_model`
        should be set to True.
      is_bidi_model: Whether the model is a bidi model. Most Gemini API models -
        except the Live API - are unary streaming, i.e. `is_bidi_model` is
        False. If `is_bidi_model` is False, the function calling processor will
        convert the unary processor to a bidi processor that can be called with
        async function calls. This means the model will wait for the end of any
        async function called before returning, the TTFT is therefore bounded by
        the longest async function called. With true bidi models, the model
        stops whenever the input stream ends.
      substream_name: The substream name to use for function calls and function
        responses. This is to help downstream processors to identify function
        calls and responses that were executed from this function calling
        processor.
      pre_processor: An optional pre-processor to pass the model input (prompt,
        function responses, model output from previous iterations) through.
      fns: The functions to register for function calling. Those functions must
        be known to `model`, and will be called only if `model` returns a
        function call with the matching name. For Gemini API, this means the
        same functions should be passed in the `GenerationConfig(tools=[...])`
        to the `model` constructor. If the function name is not found in the
        `fns` list, the function calling processor will return the unknown
        function call part and will raise a `ValueError`. If the execution of
        the function fails, the function calling processor will return a
        function response with the error message.
      max_function_calls: maximum number of function calls to make (default set
        to 5 of turn-based models and infinity for bidi aka realtime models).
        When this limit is reached, the function calling loop will wait for the
        last function calls to return. If a function call is async and runs for
        a long time, the function calling loop will wait for it to finish.
    """
    self._model = model
    self._substream_name = substream_name
    self._is_bidi_model = is_bidi_model
    self._fns = fns
    self._pre_processor = (
        pre_processor.to_processor()
        if pre_processor
        else processor.passthrough().to_processor()
    )
    default_max_function_calls = 5 if not is_bidi_model else math.inf
    self._max_function_calls = max_function_calls or default_max_function_calls

  async def call(
      self, content: AsyncIterable[content_api.ProcessorPart]
  ) -> AsyncIterable[content_api.ProcessorPart]:
    state = _FunctionCallState(fn_call_count_limit=self._max_function_calls)
    # To support both bidi and unary models we reduce both cases to bidi.
    model = _to_bidi(self._model) if not self._is_bidi_model else self._model
    # We use input_queue to feed function responses to consecutive turns.
    input_queue = asyncio.Queue[content_api.ProcessorPart | None]()
    input_stream = streams.merge(
        streams=[
            _add_end_of_turn(content),
            streams.dequeue(input_queue),
        ],
        stop_on_first=self._is_bidi_model,
    )

    output_queue = asyncio.Queue[content_api.ProcessorPart | None]()
    execute_function_call = _ExecuteFunctionCall(
        fns=self._fns,
        output_queue=output_queue,
        fn_state=state,
        is_bidi_model=self._is_bidi_model,
        substream_name=self._substream_name,
    )

    async def processor_pipeline():
      async def preprocess():
        async for part in self._pre_processor(input_stream):
          if content_api.is_end_of_turn(part):
            state.model_outputting = True
          yield part

      try:
        async for part in model(preprocess()):
          # Models don't always set the role and content_api.END_OF_TURN is
          # explicitly set to role='user'.
          if not part.role or content_api.is_end_of_turn(part):
            part.role = 'model'
          if part.function_call:
            await execute_function_call(part)
          else:
            await output_queue.put(part)
      finally:
        await output_queue.put(None)

    pipeline_task = processor.create_task(processor_pipeline())

    async for part in streams.dequeue(output_queue):
      if content_api.is_end_of_turn(part) and part.role == 'model':
        state.model_outputting = False
        # EoT issued by the model. We need to check if we stop or if there is
        # a function call to schedule.
        if not self._is_bidi_model:
          if state.schedule_model_call:
            await input_queue.put(content_api.END_OF_TURN)
            state.schedule_model_call = False
          elif state.running_fc_count == 0:
            # Stop when there are no more function calls to execute and no more
            # async function calls running, or when the max function calls is
            # reached.
            await input_queue.put(None)
            await output_queue.put(None)
        elif state.schedule_model_call and not state.model_outputting:
          input_queue.put_nowait(content_api.END_OF_TURN)
          state.schedule_model_call = False
      else:
        if context_lib.is_reserved_substream(part.substream_name):
          yield part
        else:
          # Reinject output parts into the model when it's not bidi.
          if not self._is_bidi_model or part.function_response:
            await input_queue.put(part)
          yield part

          if part.function_call:
            state.running_fc_count += 1

          if part.function_response:
            if not part.function_response.will_continue:
              state.running_fc_count -= 1

            if part.function_response.scheduling:
              scheduling = part.function_response.scheduling
            else:
              scheduling = part.get_metadata(
                  _SCHEDULING_METADATA_KEY,
                  genai_types.FunctionResponseScheduling.WHEN_IDLE,
              )
            match scheduling:
              case genai_types.FunctionResponseScheduling.SILENT:
                pass
              case genai_types.FunctionResponseScheduling.WHEN_IDLE:
                if state.model_outputting:
                  state.schedule_model_call = True
                else:
                  await input_queue.put(content_api.END_OF_TURN)
              case _:
                await input_queue.put(content_api.END_OF_TURN)
            # If we reached max count, we should allow the model to react to
            # this response, so we make one extra iteration and stop after it.
            if state.fn_call_count >= state.fn_call_count_limit + 1:
              await input_queue.put(None)
              await output_queue.put(None)
              pipeline_task.cancel()


async def cancel_fc(function_ids: list[str]) -> content_api.ProcessorPart:
  """Cancels function calls whose ids are in the provided argument.

  Args:
    function_ids: The list of ids of the function calls to cancel.

  Returns:
    A text message stating whether the functions have been successufully
    cancelled or if there were some errors e.g. function with such id does not
    exist.
  """
  del function_ids
  raise NotImplementedError(
      'This cancel_fc is an interface without implementation that can be used'
      ' as a tool definition for models. Actual implementation is in'
      ' FunctionCalling.cancel_fc.'
  )


def list_fc() -> str:
  """Returns the list of function calls that are currently running.

  A function call is represented as a string describing the function name, its
  arguments, and its id.

  Returns:
    A text paragraph with one item per function call, each item describing the
    function call that were made and are currently running.
  """
  raise NotImplementedError(
      'This list_fc is an interface without implementation that can be used'
      ' as a tool definition for models. Actual implementation is in'
      ' FunctionCalling.list_fc.'
  )


class _ExecuteFunctionCall:
  """Executes a function call and returns the whole input with the result."""

  def __init__(
      self,
      fns: list[Callable[..., Any]],
      output_queue: asyncio.Queue[content_api.ProcessorPart | None],
      fn_state: _FunctionCallState,
      is_bidi_model: bool,
      substream_name: str,
  ):
    self._fns = {fn.__name__: fn for fn in fns}
    if is_bidi_model:
      # Cancel and list functions make only sense for bidi models.
      self._fns['cancel_fc'] = self.cancel_fc
      self._fns['list_fc'] = self.list_fc
    self._output_queue = output_queue
    # Map from function call id to the task running the function call.
    self._task_map: dict[str, asyncio.Task] = {}
    self._call_map: dict[str, genai_types.FunctionCall] = {}
    self._task_counter = collections.defaultdict(int)
    self._fn_state = fn_state
    self._is_bidi_model = is_bidi_model
    self._substream_name = substream_name

  def _create_task_id(self, fn_name: str) -> str:
    """Creates a function call id."""
    # Semantic task IDs help models to keep track of running tools and reduce
    # halucinations.
    task_id = f'{fn_name}_{self._task_counter[fn_name]}'
    self._task_counter[fn_name] += 1
    return task_id

  def list_fc(self) -> content_api.ProcessorPart:
    """Returns a list of function calls that are currently running."""
    f_calls = ['Background Functions currently running:']
    for k, v in self._task_map.items():
      if not v.done() and 'list_fc' not in k:
        f_calls.append(
            f'Function {self._call_map[k].name} is running with args'
            f' {self._call_map[k].args} and id: {k}'
        )
    return content_api.ProcessorPart.from_function_response(
        name='list_fc',
        response='\n'.join(f_calls) + '\n',
        role='user',
        substream_name=FUNCTION_CALL_SUBSTREAM_NAME,
    )

  async def cancel_fc(
      self, function_ids: list[str]
  ) -> content_api.ProcessorPart:
    """Cancels function calls whose ids are in the provided argument."""
    cancelled_tasks = []
    for function_id in function_ids:
      if function_id in self._task_map:
        if self._task_map[function_id].cancel():
          try:
            await self._task_map[function_id]
          except asyncio.CancelledError:
            cancelled_tasks.append(function_id)
        else:
          # When the task is already done, we assume cancellation was
          # successful.
          cancelled_tasks.append(function_id)
    diff = set(function_ids) - set(cancelled_tasks)
    if not diff:
      response = 'OK, cancelled.'
      is_error = False
    elif cancelled_tasks:
      response = (
          f'Cancelled the following function calls: {cancelled_tasks}.'
          f' The following function calls were not found: {diff}.'
      )
      is_error = True
    else:
      response = (
          f'Could not find any of the following function calls: {function_ids}.'
      )
      is_error = True
    return content_api.ProcessorPart.from_function_response(
        name='cancel_fc',
        response=response,
        role='user',
        substream_name=self._substream_name,
        is_error=is_error,
        scheduling='SILENT',
    )

  def _add_task(
      self, call: genai_types.FunctionCall, fn: Callable[..., Any]
  ) -> None:
    """Adds the Task running a function and returns task_id."""
    task_id = call.id
    task = processor.create_task(
        self._put_function_response_to_output_queue(call, fn)
    )
    self._task_map = {k: v for k, v in self._task_map.items() if not v.done()}
    self._task_map[task_id] = task
    self._call_map[task_id] = call
    self._call_map = {k: self._call_map[k] for k in self._task_map.keys()}

  def _to_function_response(
      self,
      parts: content_api.ProcessorContentTypes | Any,
      call: genai_types.FunctionCall,
      will_continue: bool | None = None,
      scheduling: genai_types.FunctionResponseScheduling | None = None,
  ) -> content_api.ProcessorPart:
    """Wraps the part in a FunctionResponse if it is not already one.

    Tools in function calling do not always return a FunctionResponse. This
    method wraps the part in a FunctionResponse if it is not already one. If it
    is already a FunctionResponse, we update the id to match the function call
    id.

    The response is structured as {'result': str(part)}. Part must therefore be
    convertible to a string (e.g. json serializable).

    Args:
      parts: The parts to wrap. You can pass more than one part. It will be
        wrapped in a single function response. When providing something
        different than a `ProcessorContentTypes`, `parts` needs to be JSON
        serializable. When providing a function response, the id will be set to
        the function call id and the substream name will be set to the function
        calling processor substream name (changed in place).
      call: The function call that was made.
      will_continue: should only be used for async generator functions. If set,
        will override any provided value. Otherwise, the existing value is
        preserved.
      scheduling: The scheduling of the function response. If set, will override
        any provided value. Otherwise, the existing value is preserved.

    Returns:
      A function response wrapping the content in the part.
    """
    substream_name = self._substream_name
    if isinstance(parts, content_api.ProcessorPart) and parts.substream_name:
      substream_name = parts.substream_name

    if isinstance(parts, content_api.ProcessorPart) and parts.function_response:
      parts.function_response.id = call.id
      if not parts.function_response.name:
        parts.function_response.name = call.name
      if not parts.substream_name:
        parts.substream_name = self._substream_name
      if not parts.role:
        parts.role = 'user'
      if will_continue is not None:
        parts.function_response.will_continue = will_continue
      if scheduling is not None:
        parts.function_response.scheduling = scheduling
      return parts
    else:
      return content_api.ProcessorPart.from_function_response(
          name=call.name,
          response=parts,
          function_call_id=call.id,
          substream_name=substream_name,
          scheduling=scheduling,
          will_continue=will_continue,
          role='user',
      )

  async def _put_function_response_to_output_queue(
      self, call: genai_types.FunctionCall, fn: Callable[..., Any]
  ) -> None:
    """Puts the function call results to the output queue.

    Wraps all function call parts in a function response and puts them to the
    output queue. An EoT is also put to the output queue to request a model call
    in function calling loop. This function should only be called for async
    functions.

    Args:
      call: The function call to execute.
      fn: The function to execute.
    """
    async for fc_part in self._execute_function(call, fn):
      await self._output_queue.put(fc_part)

  async def _execute_function(
      self,
      call: genai_types.FunctionCall,
      fn: Callable[..., Any],
  ) -> AsyncIterable[content_api.ProcessorPart]:
    """Runs the function into an async Task."""
    args = _extra_utils.convert_number_values_for_dict_function_call_args(
        call.args
    )
    converted_args = _extra_utils.convert_argument_from_function(args, fn)
    try:
      if inspect.isasyncgenfunction(fn):
        # When the generator exits we automatically add a will_continue=False
        # Part. We send one to the downstream processor and also a separate
        # one for each of the reserved substreams as they have distinct
        # consumers.
        substreams_to_close = {self._substream_name}
        async for part_type in fn(**converted_args):
          response = self._to_function_response(part_type, call)
          substream = (
              response.substream_name
              if context_lib.is_reserved_substream(response.substream_name)
              else self._substream_name
          )
          substreams_to_close.add(substream)

          if response.function_response.will_continue is False:  # pylint: disable=g-bool-id-comparison
            substreams_to_close.discard(substream)
          else:
            response.function_response.will_continue = True

          yield response

        # Send a function response to indicate that the function call is
        # done (will_continue=False).
        for substream in substreams_to_close:
          yield content_api.ProcessorPart.from_function_response(
              name=call.name,
              function_call_id=call.id,
              response='',
              substream_name=substream,
              role='user',
          )
      elif inspect.iscoroutinefunction(fn):
        yield self._to_function_response(
            await fn(**converted_args), call, will_continue=None
        )
      else:
        yield self._to_function_response(
            await asyncio.to_thread(fn, **converted_args),
            call,
            will_continue=None,
        )
    except Exception as e:  # pylint: disable=broad-except
      yield content_api.ProcessorPart.from_function_response(
          name=call.name,
          function_call_id=call.id,
          response=(
              f'Failed to invoke function {fn.__name__}({converted_args}): {e}'
          ),
          role='user',
          substream_name=self._substream_name,
          is_error=True,
      )

  async def __call__(self, part: content_api.ProcessorPart) -> None:
    """Executes a function call and returns the result."""
    # The match() method ensures that the part is a function call.
    self._fn_state.fn_call_count += 1
    if self._fn_state.fn_call_count > self._fn_state.fn_call_count_limit:
      return

    part.substream_name = FUNCTION_CALL_SUBSTREAM_NAME
    call = part.function_call

    await self._output_queue.put(part)

    try:
      fn = self._fns[call.name]
    except KeyError:
      function_names = ', '.join(repr(fn) for fn in sorted(self._fns.keys()))
      await self._output_queue.put(
          content_api.ProcessorPart.from_function_response(
              name=call.name,
              function_call_id=call.id,
              response=(
                  f'Function {call.name} not found. Available functions:'
                  f' {function_names}.'
              ),
              role='user',
              substream_name=self._substream_name,
              will_continue=False if self._is_bidi_model else None,
              is_error=True,
          )
      )
      return

    if (
        inspect.iscoroutinefunction(fn)
        or inspect.isasyncgenfunction(fn)
        or self._is_bidi_model
    ):
      if call.id is None:
        call.id = self._create_task_id(call.name)
      # Immediately returns a function response with a std result. The
      # scheduling is silent.
      await self._output_queue.put(
          self._to_function_response(
              'Running in background.',
              call,
              scheduling=genai_types.FunctionResponseScheduling.SILENT,
              will_continue=True,
          )
      )
      # Run function in a separate task and inject the results back into the
      # output queue.
      self._add_task(
          call,
          fn,
      )
    else:
      # This is a sync function invoked by a non-bidi model. So we just run it
      # synchronously.
      async for fc_part in self._execute_function(call, fn):
        await self._output_queue.put(fc_part)
