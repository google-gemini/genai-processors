"""Function calling processor.

The FunctionCalling processor is a loop over a model call that intercepts any
function call, runs the function, injects the result back into the model prompt
and goes to the next iteration until no function call is made.

It assumes that model uses Part.function_call / Part.function_response and
focuses on the function calling loop and the execution of tools. Model itself
is responsible for formatting/parsing these parts in-to its internal
representation.

The overall logic can be depicted as:

```
    input -> pre_processor -> model -> function_call -> output
                  |                    |
                  -<---function run---<-
```

where:
-  `input` is the input stream.
-  `pre_processor` is a processor called once on `input` and on all results
obtained by tool invocation.
-  `model` is an unary (not bidirectional) model processor that generates
   content. For the model to know which tools are available, the same
   function/tool set should be given both to the FunctionCalling and the model
   constructor.
-  `function_call` executes the function calls returned by the model. The
function response is then fed back to the model for another iteration. If there
is no function call to execute, the loop is stopped and the current tool_use
output is sent back to the output stream.

The output is the model output including the function call and the responses.

When used with GenAI models (i.e. Gemini API), the model should be defined as
follows:

```python
genai_processor = genai_model.GenaiModel(
    api_key=API_KEY,
    model_name="gemini-2.5-flash",
    generate_content_config=genai_types.GenerateContentConfig(
        tools=[fns],
        automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True),
    ),
)
```

where `fns` are the python functions to be called. Note that we disable the
automatic function calling feature here to avoid duplicate function calls with
the GenAI automatic function calling feature.
"""

from collections.abc import AsyncIterable
from typing import Any, Callable

from genai_processors import content_api
from genai_processors import processor
from genai_processors import streams
from google.genai import _extra_utils


# All function call parts (calls and responses) are emitted in a substream
# with this name. This is to help downstream processors to identify function
# calls that were executed from function calls returned directly by the model.
FUNCTION_CALL_SUBTREAM_NAME = "function_call"


class FunctionCalling(processor.Processor):
  """Tool use with Function Calling.

  See class docstring for more details.
  """

  def __init__(
      self,
      model: processor.Processor,
      *,
      pre_processor: (
          processor.Processor | processor.PartProcessor | None
      ) = None,
      fns: list[Callable[..., Any]] | None = None,
      max_function_calls: int = 5,
  ):
    """Initializes the FunctionCalling processor.

    Args:
      model: The processor to use for generation. This processor should not be a
        bidi model like the one obtained from `realtime.py`. It should be a
        unary-streaming model, waiting on the full input stream when called.
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
      max_function_calls: The maximum number of function calls to make (default
        set to 5). When this limit is reached, the function calling loop will
        stop even if the model is still invoking function calls.
    """
    self._model = model
    self._fns = fns
    self._pre_processor = (
        pre_processor.to_processor()
        if pre_processor
        else processor.passthrough().to_processor()
    )
    assert max_function_calls >= 0
    self._max_function_calls = max_function_calls

  async def call(
      self, content: AsyncIterable[content_api.ProcessorPart]
  ) -> AsyncIterable[content_api.ProcessorPart]:

    execute_function_call = ExecuteFunctionCall(fns=self._fns)

    if self._pre_processor:
      content = self._pre_processor(content)

    # Main loop - ensure we have at least one iteration and one model call
    # independent of the max_function_calls value.
    for _ in range(self._max_function_calls + 1):
      content, prev_content = streams.split(content, with_copy=True)
      next_content = (self._model + execute_function_call)(content)

      # Keep a copy of the model output
      output_content, next_content = streams.split(next_content, with_copy=True)
      tool_calls = False
      async for part in output_content:
        if part.function_response:
          tool_calls = True
        yield part

      if not tool_calls:
        break

      content = streams.concat(prev_content, self._pre_processor(next_content))


class ExecuteFunctionCall(processor.PartProcessor):
  """Executes a function call and returns the whole input with the result."""

  def __init__(self, fns: list[Callable[..., Any]]):
    self._fns = {fn.__name__: fn for fn in fns}

  def match(self, part: content_api.ProcessorPart) -> bool:
    return bool(part.function_call)

  async def call(
      self, part: content_api.ProcessorPart
  ) -> AsyncIterable[content_api.ProcessorPart]:
    """Executes a function call and returns the result."""
    # The match() method ensures that the part is a function call.
    part.substream_name = FUNCTION_CALL_SUBTREAM_NAME
    yield part
    call = part.function_call
    try:
      fn = self._fns[call.name]
    except KeyError as e:
      raise ValueError(
          f"Function {call.name} not found. Available functions:"
          f" {self._fns.keys()}."
      ) from e
    try:
      args = _extra_utils.convert_number_values_for_dict_function_call_args(
          call.args
      )
      function_response = {
          "result": _extra_utils.invoke_function_from_dict_args(args, fn)
      }
    except Exception as e:  # pylint: disable=broad-except
      function_response = {"error": str(e)}
    yield content_api.ProcessorPart.from_function_response(
        name=call.name,
        response=function_response,
        role="user",
        substream_name=FUNCTION_CALL_SUBTREAM_NAME,
    )
