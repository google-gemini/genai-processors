# Processor

The Processor is the fundamental unit of work in the GenAI Processors library.
It encapsulates logic, AI models, and tools into a composable, asynchronous
interface.

For a runnable introduction to the concepts described here, check out the
[Processor Introduction Notebook](https://github.com/google-gemini/genai-processors/blob/main/notebooks/processor_intro.ipynb).

## The Core Concept

At its heart, a Processor is a transformation pipeline: it takes an input stream
of
[ProcessorParts](https://github.com/google-gemini/genai-processors/blob/main/genai_processors/content_api.py#:~:text=class%20ProcessorPart\)
data and produces an output stream of parts. These parts represent different
modalities —such as text, images, or files— allowing the pipeline to handle
complex, multi-modal data.

To make development easier, the library handles the heavy lifting of unifying
data types for you. When receiving a stream, you'll work with a
`ProcessorStream` class (inheriting from `ContentStream`); this acts as an
`AsyncIterable`, but also provides convenient accessors like `.text()` for quick
extraction. When it’s time to send data back, you can simply yield whatever you
have —whether that’s a raw string or a part— and the library will automatically
normalize it into processor parts, filling in default values for any arguments
you didn't provide. This reliance on a standard `AsyncIterable` interface is
what makes it so simple to chain multiple Processors into a single, continuous
pipeline.

## The Processor Interface

Processors provide the following interface:

```python
  def __call__(
      self, content: AsyncIterable[ProcessorPartTypes] | ProcessorContentTypes
  ) -> ProcessorStream:
    ...
```

and there are two ways to define them:

### Functional Definition

For simple logic that does not maintain a state between processor calls, you can
use the `@processor.processor_function` decorator.

```python
from genai_processors import processor, content_api
from typing import AsyncIterable

@processor.processor_function
async def simple_filter(
    content: content_api.ProcessorStream,
    ) -> AsyncIterable[content_api.ProcessorPartTypes]:
    """Passes through only text parts buffering them first."""
    text_buffer = []
    async for part in content:
        if content_api.is_text(part.mimetype):
            text_buffer.append(part.text)
    yield "".join(text_buffer)
```

You’ll notice that a Processor’s implementation signature differs slightly from
its usage signature. This is intentional. When implementing a processor, you get
the full power of the `ProcessorStream` input and the flexibility to return any
valid `ProcessorPartTypes`. When calling a processor, the library allows you to
pass in various data types as input while ensuring you always receive a
`ProcessorStream` back, giving you a consistent, feature-rich output to work
with.

### Class-Based Definition

For logic requiring internal state (like API clients), you need to subclass
`processor.Processor` to keep the state as a class field.

```python
class MyStatefulProcessor(processor.Processor):
    def __init__(self, api_client: Any):
        self._api_client = api_client
        self._api_client.init()

    async def call(
        self, content: content_api.ProcessorStream
    ) -> AsyncIterable[content_api.ProcessorPartTypes]:
        request_str = await content_api.ContentStream(parts=content).text()
        response = await self._api_client.request(request_str)
        yield response
```

## The `PartProcessor` Interface

Parts in a stream can often be processed independently —for example, image
preprocessing or "verbalizing" PDFs into text and images. In these cases, we can
apply a critical optimization: not only can these parts be processed in
parallel, but we can also avoid head-of-line blocking when multiple processors
are chained into a pipeline.

To simplify this, we provide a specialized `PartProcessor` class designed to
handle a single part. The library automatically parallelizes the incoming stream
across a stack of these processors and ensures the results are reassembled in
the correct order for the next stage of the pipeline.

### Why use PartProcessor?

*   **Simplicity:** You write logic for one part, not a loop over a stream.

*   **Performance:** The library automatically runs `PartProcessor` logic
    concurrently on incoming parts.

```python
@processor.part_processor_function
async def shouter(
    part: content_api.ProcessorPart
    ) -> AsyncIterable[content_api.ProcessorPartTypes]:
    if content_api.is_text(part.mimetype):
        yield part.text.upper()
    else:
        yield part
```

### Filtering with `match_fn`

You can attach a match function to explicitly define which parts this processor
handles. This improves readability and performance and is recommended.

```python
def is_text_part(part: content_api.ProcessorPart) -> bool:
    return content_api.is_text(part.mimetype)

@processor.part_processor_function(match_fn=is_text_part)
async def shouter(
    part: content_api.ProcessorPart
    ) -> AsyncIterable[content_api.ProcessorPartTypes]:
    # We are guaranteed that 'part' is text because of match_fn
    yield part.text.upper()
```

### Comparison: Processor vs. PartProcessor

| Feature         | Processor                   | PartProcessor               |
| --------------- | --------------------------- | --------------------------- |
| **Input**       | Stream (AsyncIterable)      | Single Item (ProcessorPart) |
| **State**       | Can maintain state across   | Stateless per-part          |
|                 | the stream                  |                             |
| **Concurrency** | Sequential (unless manually | Automatic (Parallelized DFS |
|                 | managed)                    | map over a chain of         |
|                 |                             | PartProcessors              |
| **Use Case**    | Buffers, Accumulators, Full | Filters, Formatters,        |
|                 | Context Models              | Independent Transformations |

## Composition

The power of this library lies in composing simple processors into complex
pipelines.

### Sequential Chaining (`+`)

The `+` operator pipes the output of one processor into the input of the next.
It takes care of concurrency under the hood and is the recommended way of
chaining Processors or `PartProcessors` together.

```python
# Pipeline: Input -> Filter -> Model -> Output
pipeline = simple_filter + model_processor
```

### Parallel Execution

You can branch execution to run multiple processors simultaneously.

#### Parallel Parts (`//`)

Use // to run multiple `PartProcessors` on the same input part concurrently. The
results are concatenated (preserving input order).

```python
# For each input part, run a classifier AND a logger for that part.
classifier_and_logger = classifier_part_processor // logger_part_processor
```

#### Parallel Streams (`parallel_concat`)

For full `Processors` operating on streams, use `parallel_concat`. This
broadcasts the stream to multiple agents and merges their outputs.

```python
# Both agents receive the full stream context
mixture = processor.parallel_concat([agent_a, agent_b])
```

When iterating on `mixture`, both agents are processed in parallel but the
iterator will progress first on the result of `agent_a`. If `agent_b` comes
first, you will not be able to iterate on it before `agent_a` is done.

To merge the output of processors as they come, use the `merge` method from the
`streams` module on the output of each processor. See
[Async & Streaming](async-streaming.md) section for more details.

## Orchestration (Routing)

For advanced flows, you may need to route data dynamically based on its content.

### Switch (Stream Routing)

`Switch` dispatches parts of the input stream to different processors based on
conditions. The condition in each case statement defines the Parts that will be
sent to the attached processor. The order of the parts in the output and input
streams is only kept for parts returned by the same processor, i.e. two parts
matching two different cases are not guaranteed to be in the same order in the
input and output stream.

```python
from genai_processors import switch

# Route audio to audio_model, everything else to text_model
router = (
    switch.Switch(content_api.get_mimetype)
    .case(content_api.is_audio, audio_model)
    .default(text_model)
)
```

### PartSwitch (Part Routing)

`PartSwitch` provides the same part-level dispatching logic as a standard
`Switch`, but it is implemented as a `PartProcessor`. It directs individual
parts to different handlers and executes them concurrently. This is particularly
useful in a `PartProcessor` stack, ensuring that different modalities —like a
mix of heavy images and light text— are processed simultaneously so that one
slow part doesn't bottleneck the rest of the stream.

```python
# Process text tokens with one logic, images with another
part_router = (
    switch.PartSwitch()
    .case(content_api.is_text, text_handler)
    .case(content_api.is_image, image_handler)
    .default(processor.passthrough())
)
```

## Sources

Processors can ingest data from any AsyncIterator, but for convenience, we
provide built-in sources for common inputs like microphones, cameras, screen
captures, and terminals.

Because mixing these inputs is a frequent requirement, we’ve designed Sources to
implement the Processor interface. This allows you to combine them using the `+`
operator —for example: `TerminalInput('>') + audio_io.AudioIn(...) +
live_model.LiveModel(...)`. Most sources, especially those tailored for
real-time use, will stay active as long as their input stream is open. If you
need a pipeline to run indefinitely, you can use `streams.endless_stream()` to
keep the sources alive until they are explicitly cancelled or ended.

If you do not find what you need with built-in Source, you can define your own
source (e.g., from a microphone, a queue, or a file) using the
`@processor.source` decorator:

```python
@processor.source
async def my_source(filepath: str) -> AsyncIterable[content_api.ProcessorPart]:
    data = await read_file(filepath)
    yield content_api.ProcessorPart(data)

# Usage:
# p = file_source("data.txt") + simple_filter + model
# async for part in p(streams.endless_stream()):
#   ...
```

## Context & Error Propagation

When a processor in a chain fails, it is vital to terminate all concurrently
running tasks and notify the source that data processing has stopped. To achieve
this, pipelines should always be executed within an `async with
processor.context():` block. This context defines a TaskGroup, ensuring that if
one part of the pipeline fails, the entire chain is cleaned up and no background
tasks are left "hanging" as zombies.

```python
async with processor.context():
    # Run the pipeline
    async for part in my_processor_pipeline(input_stream):
        process_result(part)
```

### Yielding exceptions as parts

For complex pipelines, terminating them when something goes wrong might not be
an option: there are too many cogs to expect all of them working flawlessly.
Instead, we may want to let an LLM deal with the problem, autocorrect, or try a
different approach. In this case, wrap a processor with the
`@processor.yield_exceptions_as_parts` decorator. This will catch any exception
inside the processor and yield it as a `ProcessorPart` to the output stream.

```python
import random
from typing import AsyncIterable

from genai_processors import content_api
from genai_processors import mime_types
from genai_processors import processor


@processor.processor_function
@processor.yield_exceptions_as_parts
async def faulty_processor(
    content: content_api.ProcessorStream,
    ) -> AsyncIterable[content_api.ProcessorPartTypes]:
    """This processor will sometimes fail."""
    if random.random() < 0.5:
        raise ValueError("Something went wrong")
    yield "Hello"

@processor.processor_function
async def recover_from_error(
    content: content_api.ProcessorStream,
) -> AsyncIterable[content_api.ProcessorPartTypes]:
    """This processor will recover from an error."""
    async for part in content:
        if part.mimetype == mime_types.TEXT_EXCEPTION:
            yield f"Processor failed with: {part.text}, trying something else."
        else:
            yield part
```

With `@yield_exceptions_as_parts`, `faulty_processor` will not terminate the
pipeline when it fails. Instead, it will output a `ProcessorPart` with mimetype
`text/x-exception` that `recover_from_error` processor can use to react to the
failure.

### Background Tasks

If you need to perform a "fire-and-forget" operation (like saving to a DB)
without blocking the main generation flow, use `processor.create_task()`.

```python
async def process_with_background():
    async with processor.context():
        # Schedule the background task (fire and forget)
        # This is tracked by the context and cleaned up automatically
        processor.create_task(save_to_db(results))

        # Yield generation results immediately
        async for part in model(input_stream):
            yield part

        # When exiting the context, the task is awaited by the context.
```

### How the Lifecycle is Managed

To keep your pipelines robust and memory-safe, the `processor.context()` follows
a few key principles:

*   **Automatic Cleanup:** You don't have to worry about manual teardown. When
    the context exits, all background tasks—including buffers and parallel
    branches —are gracefully cleaned up for you.

*   **Smart Error Handling:** If a processor encounters an exception, the
    context acts as a safety net. It immediately cancels any remaining tasks in
    the chain and propagates the error, preventing "zombie" processes from
    lingering in the background.

*   **Integrated Task Management:** To ensure your background work stays synced
    with the pipeline’s lifecycle, we recommend using `processor.create_task()`.
    Unlike a standard `asyncio.create_task()`, this keeps your work attached to
    the processor’s context so it can be managed and cleaned up automatically.
