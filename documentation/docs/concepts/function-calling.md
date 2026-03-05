# Function Calling

GenAI Processors provides a unified interface for tool use (function calling)
across different model backends, with specific support for asynchronous
execution making it a natural fit for low-latency, real-time agents.

## Core Concept: The Function Calling Loop

The `FunctionCalling` processor wraps a model processor (like `GenaiModel` or
`realtime.LiveProcessor`) to automate tool use. When the wrapped model returns a
`FunctionCall`, the processor intercepts it, executes the corresponding Python
function, and feeds the `FunctionResponse` back to the model as input for the
next iteration. This loop continues until the model responds with content other
than a function call, or `max_function_calls` is reached.

All function calls and responses are emitted on the `function_call` substream by
default.

## Defining Tools

Tools are defined as standard Python functions. The model uses the function
signature and docstring to understand when and how to call the tool. Arguments
and return types must be JSON-serializable. Providing a detailed docstring with
argument descriptions is crucial, as this is included in the prompt for the
model.

**Sync Tool:**

```python
def get_weather(city: str) -> str:
    """Returns the current weather for a given city.

    Args:
        city: The name of the city for which to get the weather.

    Returns:
        A string describing the current weather conditions in the city.
    """
    # This is a blocking call, the model waits until the function returns.
    return f"The weather in {city} is sunny."
```

**Async Tool:**

Sometimes tools take a while to finish, and you’ll want the model to keep
responding while they run in the background. This approach is especially useful
for launching multiple tools at once. For instance, a kitchen assistant might
set a timer and then immediately continue giving recipe instructions:

```python
import asyncio

async def set_alarm(seconds: float, message: str) -> str:
    """Sets an alarm that will trigger after some seconds.

    Args:
        seconds: The number of seconds to wait before the alarm triggers.
        message: The message to return when the alarm triggers.

    Returns:
        The message provided by the user.
    """
    await asyncio.sleep(seconds)
    return message
```

**Streaming Tool (Async Generator):**

If a tool needs to provide continuous updates over time—like progress reports or
live sensor data—you can use an `async` generator.

```python
async def smart_home_state() -> AsyncIterable[str]:
    """Yields smart home status updates as they occur.

    This includes events such as motion detection or lights being
    toggled on and off.

    Yields:
        A string describing the detected event.
    """
    while (home_status := await sensor.detect()):
        yield home_status
```

## Configuring the Processor

To enable function calling, you must:

1.  Let the model know which tools are available by providing the tool list in
    the model config.

2.  **Disable** any built-in automatic function calling in the model (e.g., for
    `GenaiModel`, set `automatic_function_calling=...disable=True`).

3.  Wrap the model processor with `FunctionCalling`, providing the same list of
    functions for execution.

```python
from genai_processors.core import function_calling, genai_model
from google.genai import types as genai_types

tools = [get_weather, set_alarm]

# 1 & 2: Configure model and disable its internal AFC.
model = genai_model.GenaiModel(
    model_name="gemini-2.0-flash",
    generate_content_config=genai_types.GenerateContentConfig(
        tools=tools,
        automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(
            disable=True
        ),
    ),
)

# 3. Wrap with FunctionCalling.
agent = function_calling.FunctionCalling(model=model, fns=tools)
```

## Async Tools and Real-Time Interaction

`FunctionCalling` is designed to work with both turn-based and real-time
(bidirectional) processors. Its behavior changes depending on whether the tool
is sync/async and whether `is_bidi_model=True` is set.

**If `is_bidi_model=True` (e.g., when wrapping `realtime.LiveProcessor`) or if a
tool is defined with `async def`, it is treated as an asynchronous tool to avoid
blocking the agent.**

When an async tool is called:

1.  `FunctionCalling` assigns a unique `function_call_id` to the call.

2.  It immediately returns a `FunctionResponse` to the model with the content
    `'Running in background.'`, and `scheduling='SILENT'`. This informs the
    model that the tool has started.

3.  The tool is executed in a background task.

4.  When the tool completes or yields a result, a new `FunctionResponse`
    containing the result and the unique `function_call_id` is sent to the
    model. This may trigger a new model turn, depending on the scheduling
    option.

### Response Scheduling

When an async tool result is ready, `FunctionCalling` needs to know if, and how,
it should prompt the model to respond to this new information. This is
controlled by the `scheduling` parameter of a `FunctionResponse`, which can be
one of:

-   **`SILENT`**: The response is added to the model's history but does not
    trigger a new model generation. The model will only see it in the prompt on
    its next natural turn.
-   **`WHEN_IDLE`** (Default): If the model is currently generating, wait for it
    to finish, then add the response to the prompt and trigger a new model
    generation. If the model is idle, trigger generation immediately.
-   **`INTERRUPT`**: Add the response to the prompt and trigger a new model
    generation immediately, cancelling any ongoing generation. This is useful if
    a tool result requires the agent to stop what it's saying and react to the
    tool result.

You can control scheduling by having your tool return a
`genai_types.FunctionResponse` object directly:

```python
from google.genai import types as genai_types

async def critical_alert(message: str) -> genai_types.FunctionResponse:
    """Sends a critical alert that should interrupt the agent's current speech
    and trigger an immediate response.

    Args:
        message: The alert message to process.

    Returns:
        A FunctionResponse with INTERRUPT scheduling to notify the model.
    """
    return genai_types.FunctionResponse(
        name='critical_alert',
        response={'status': 'ok'},
        scheduling='INTERRUPT',
    )
```

### Streaming Tool Responses (Async Generators)

If you use an `async` generator for a tool, each `yield` produces a
`FunctionResponse`. The processor automatically sets `will_continue=True` on
these responses. When the generator finishes, a final response with
`will_continue=False` is sent, signalling that the tool has finished executing.

## Managing Long-Running Tools

For bidirectional models (`is_bidi_model=True`), `FunctionCalling`
exposes two additional tools you can add to your model's tool list:

-   **`list_fc()`**: Returns a list of currently running async tools and their
    `function_call_id`s.
-   **`cancel_fc(function_ids: list[str])`**: Attempts to cancel running async
    tools by their IDs.

```python
from genai_processors.core import function_calling

# We explicitly add the list_fc and cancel_fc functions from function calling
# to let the model cancel async function calls. If tools contains only synced
# functions, list_fc and cancel_fc can be omitted.
tools = [my_async_tool, function_calling.list_fc, function_calling.cancel_fc]
model = genai_model.GenaiModel(..., tools=tools, ...)

agent = function_calling.FunctionCalling(model=model, fns=tools, is_bidi_model=True)
```

## Using MCP (Model Context Protocol)

GenAI Processors has first-class support for MCP. You can pass an
`McpClientSession` object directly in the `fns` list, and all tools exposed by
the MCP server will be made available to the model.

```python
from genai_processors import mcp

# Connect to an MCP server
session = await mcp.create_session(...)

# The tools from the session are automatically exposed
agent = function_calling.FunctionCalling(
    model=model,
    fns=[session] # Pass session. You could add functions as well here.
)
```

When used with a real-time model (setting `is_bidi_model=True`), all MCP
functions will be run in the background. This lets you build real-time agents
with MCP capabilities without adapting your MCP implementation.

## Tutorial

For a step-by-step guide on function calling, see the
[Function Calling with Processors Notebook](https://colab.research.google.com/github/google-gemini/genai-processors/blob/main/notebooks/function_calling.ipynb).
