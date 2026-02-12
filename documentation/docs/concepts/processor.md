# Processor

For a runnable introduction to the concepts describe here, check out the
[Processor Introduction Notebook](../notebooks/processor_intro.ipynb).

The Processor is the fundamental unit of work in the GenAI Processors library.
It encapsulates logic, AI models, and tools into a composable, asynchronous
interface.

This library solves the problem of managing complex generative AI streams.
Instead of writing messy while loops and manual buffer management, you define
small, reusable units (Processors) and chain them together.

## The Core Concept

At its heart, a Processor is a transformation pipeline. It takes an input stream
of data and produces an output stream.

*. Input: `AsyncIterable[ProcessorPartTypes]`

*. Output: `AsyncIterable[ProcessorPart]`

This stream-to-stream design enables three key capabilities:

1.  Real-time Streaming: Process content the moment it arrives (e.g.,
    token-by-token generation).

2.  Context Awareness: Maintain state across a long conversation or interaction.

3.  Complex Flows: A single input can trigger zero, one, or multiple outputs
    (e.g., a user question yielding a "Thinking..." status followed by the final
    answer).

## The Processor Interface

Use the standard Processor when you need to handle the entire stream as a whole,
or when you need to maintain state between different parts of the stream.

### Functional Definition

For stateless or simple logic, use the @processor.processor_function decorator.

```python
from genai_processors import processor, content_api
from typing import AsyncIterable

@processor.processor_function
async def simple_filter(
    content: AsyncIterable[content_api.ProcessorPart]
    ) -> AsyncIterable[content_api.ProcessorPart]:
    """Passes through only text parts."""
    async for part in content:
        if content_api.is_text(part.mimetype):
            yield part
```

### Class-Based Definition

For logic requiring internal state (like buffering tokens, history, or API
clients), subclass `processor.Processor`.

```python
class MyStatefulProcessor(processor.Processor):
    def __init__(self, prefix: str):
        self.prefix = prefix

    async def call(
        self,
        content: AsyncIterable[content_api.ProcessorPart]
        ) -> AsyncIterable[content_api.ProcessorPart]:
        async for part in content:
            if content_api.is_text(part.mimetype):
                yield content_api.ProcessorPart(f"{self.prefix}: {part.text}")
            else:
                yield part
```

## The `PartProcessor` Interface

Often, you don't need the full stream context. If you want to process distinct
items (like individual user messages) independently, use `PartProcessor`.

This is a specialized version of a Processor that accepts a **single**
`ProcessorPart` as input and yields a stream of results.

### Why use PartProcessor?

*. **Simplicity:** You write logic for one item, not a loop over a stream.

*. **Performance:** The library automatically runs `PartProcessor` logic
concurrently on incoming parts.

```python
@processor.part_processor_function
async def shouter(
    part: content_api.ProcessorPart
    ) -> AsyncIterable[content_api.ProcessorPart]:
    if content_api.is_text(part.mimetype):
        yield content_api.ProcessorPart(part.text.upper())
    else:
        yield part
```

### Filtering with `match_fn`

You can attach a match function to explicitly define which parts this processor
handles. This improves readability and performance.

```python
def is_text_part(part: content_api.ProcessorPart) -> bool:
    return content_api.is_text(part.mimetype)

@processor.part_processor_functio(match_fn=is_text_part)
async def shouter(
    part: content_api.ProcessorPart
    ) -> AsyncIterable[content_api.ProcessorPart]:
    # We are guaranteed that 'part' is text because of match_fn
    yield content_api.ProcessorPart(part.text.upper())
```

### Comparison: Processor vs. PartProcessor

Feature Processor PartProcessor Input Stream (AsyncIterable) Single Item
(ProcessorPart) State Can maintain state across the stream Stateless per-part
Concurrency Sequential (unless manually managed) Automatic (Parallel Map) Use
Case Buffers, Accumulators, Full Context Models Filters, Formatters, Independent
Transformations

## Composition

The power of this library lies in chaining simple processors into complex
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

Use // to run multiple `PartProcessors` on the same input part at the *same*
time. The results are concatenated (preserving input order).

```python
# For each input part, run a classifier AND a logger for that part.
classifier_and_logger = classifier_part_processor // logger_part_processor
```

#### Parallel Streams (`parallel_concat`)

For full `Processors` operating on streams, use `parallel_concat`. This
broadcasts the stream to multiple agents and merges their outputs as they
arrive.

```python
# Both agents receive the full stream context
# Outputs are merged as they become available
mixture = processor.parallel_concat([agent_a, agent_b])
```

When iterating on mixture, both agents are processed in parallel but the
iterator will progress first on the result of `agent_a`. If `agent_b` comes
first, you will not be able to iterate on it before `agent_a` is done. To merge
the output of processors at they come, use the `merge` method from the `streams`
module on the output of each processor. See
[Async & Streaming](async-streaming.md) section for more details.

## Orchestration (Routing)

For advanced flows, you may need to route data dynamically based on its content.

### Switch (Stream Routing)

`Switch` routes the **entire stream** (or subsequences of it) to different
processors based on conditions. The condition in each case statement defines the
parts that will be parts of the stream sent to the attached processor.

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

`PartSwitch` is the `PartProcessor` equivalent. It routes individual parts to
different handlers, which are executed concurrently.

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

A `Source` is the entry point of a pipeline. It is technically a`Processor` that
ignores its input stream and generates data from an external origin (microphone,
file, queue, etc.).

To define a stream (e.g., from a microphone, a queue, or a file), use
`@processor.source`.

```python
@processor.source
async def my_source(filepath: str) -> AsyncIterable[content_api.ProcessorPart]:
    data = await read_file(filepath)
    yield content_api.ProcessorPart(data)

# Usage:
# file_source("data.txt") + simple_filter + model
```

Sources are just Processors that don't need an input stream to produce data, but
they can still be chained (e.g., `my_source + processor`).

## Context & Error Propagation

GenAI Processors uses a strict context manager system to ensure no background
task is left "hanging" (zombie processes).

You must run your pipeline within a `processor.context()`. This acts like a
Python TaskGroup.

```python
async with processor.context():
    # Run the pipeline
    async for part in my_processor_pipeline(input_stream):
        process_result(part)
```

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

        # When exiting the context, the task is awaited by the context. Ensure
        # it is finished or the program will wait here for a long time.
```

### Critical Rules

1.  **Automatic Cleanup:** When the context exits, all background tasks
    (buffers, parallel branches) are cleaned up.

2.  **Error Propagation:** If any processor raises an exception, the context
    cancels all other tasks immediately and propagates the error.

3.  **Task Creation:** Always use `processor.create_task()` instead of
    `asyncio.create_task()` to ensure your background work is attached to the
    lifecycle of the processor.
