# Tracing

GenAI Processors includes a powerful hierarchical tracing mechanism for
monitoring and debugging processor pipelines. Traces capture a time-stamped log
of processor executions, including input/output parts, nested processor calls,
errors, and cancellations. This is invaluable for understanding complex data
flows, diagnosing issues, analyzing timing, and inspecting multimodal data at
each stage of a pipeline.

## Overview

When tracing is enabled, every call to a `Processor` or `PartProcessor` can be
recorded. A trace logs:

-   **Input/Output**: Each `ProcessorPart` that enters or leaves a processor.
-   **Timing**: Start and end times for each processor execution.
-   **Hierarchy**: Nested calls are captured as sub-traces, showing how
    processors call each other (e.g., in a `chain` or `parallel` operation).
-   **Content**: Multimodal content like text, images, and audio can be
    inspected. Metadata, function calls, tool responses, and code execution are
    also captured.
-   **Errors**: Exceptions raised by a processor are logged in the trace, along
    with stack traces.
-   **Cancellations**: If a processor's task is cancelled, this is marked in the
    trace.

## How Tracing Works

The tracing mechanism is built into the `Processor` and `PartProcessor` base
classes and integrates with Python's `asyncio` and `contextvars`.

### Trace Context

Tracing is activated when a processor is executed within an active `trace.Trace`
asynchronous context. The root `Trace` object manages trace collection for its
scope.

When a `Processor` is called, it checks for an active trace: - If tracing is
active, it creates a **sub-trace** representing its own execution and attaches
it to the parent trace. - If no trace is active, the processor runs without
tracing.

Certain internal or debugging processor modules (e.g., `genai_processors.debug`)
are excluded by default to reduce noise in traces. You can view or extend this
list via `trace.EXCLUDED_TRACE_MODULES`.

### Event Logging

Within a trace, each input part consumed and output part produced by a processor
is logged as a `TraceEvent` with a timestamp. If a processor calls another
processor, this call is also logged as an event containing a nested sub-trace.

## Enabling Tracing

Tracing is enabled by wrapping the processor pipeline execution in a `Trace`
context manager. **You do not need to modify your processor implementations.**
The library provides `SyncFileTrace` for file-based trace logging.

```python
import asyncio
from collections.abc import AsyncIterable

from genai_processors import content_api
from genai_processors import processor
from genai_processors.dev import trace_file


@processor.processor_function
async def my_pipeline(
    content: content_api.ProcessorStream,
) -> AsyncIterable[content_api.ProcessorPartTypes]:
    async for part in content:
        yield part.text.upper()


async def main():
    # When my_pipeline() is called, its execution will be traced
    # because it's inside the SyncFileTrace context.
    async with trace_file.SyncFileTrace(
        trace_dir='/tmp/traces', name='my_pipeline_trace'
    ):
        result = await my_pipeline('my input').text()
        print(result)

asyncio.run(main())
```

This will run `my_pipeline`, record its execution trace, and save it to the
`/tmp/traces` directory.

## File-Based Tracing: `SyncFileTrace`

`SyncFileTrace` is the default backend for tracing, which saves traces to disk
when its context exits. For each traced execution, it generates two files in the
specified `trace_dir`:

*   **`.json`**: A JSON file containing all trace events, parts, and metadata.
    This file can be loaded for programmatic analysis using
    `trace_file.SyncFileTrace.load(path)`.
*   **`.html`**: An interactive HTML trace viewer that can be opened in a
    browser for visual inspection of the processor execution timeline, nested
    calls, and multimodal data.

### Trace Viewer

The HTML trace viewer provides an interactive interface for exploring traces:

-   A hierarchical view of nested processor calls on the left panel.
-   A detailed, time-ordered log of inputs and outputs for the selected
    processor on the right panel.
-   Inline rendering for text, images, and audio parts.
-   Formatted display for function calls, function responses, executable code,
    and file data.
-   Playback controls for audio streams.
-   Metadata inspection for each part.

### Configuration

`SyncFileTrace` can be configured with options like:

-   `trace_dir`: Directory to save trace files.
-   `name`: A name for the trace, used in filenames and the viewer title.
-   `image_size`: A `(width, height)` tuple to resize images to for saving space
    in traces, e.g., `(200, 200)`. Set to `None` to disable resizing.
-   `max_size_bytes`: If set, trace part content will be omitted if the total
    trace size exceeds this limit, to prevent excessive memory usage or huge
    trace files. Metadata is still kept. This is handy for real-time agents that
    take video input and can therefore generate very large traces quickly.

## Custom Trace Backends

To send traces to a different backend (e.g., a database, or a streaming
service), you can implement a custom trace class by inheriting from
`trace.Trace` and implementing its abstract methods for handling inputs,
outputs, sub-traces, errors, and finalization.

## Guidelines

-   **Development**: Tracing is invaluable during development and debugging.
    Enable `SyncFileTrace` to understand how data flows through your pipeline
    and to diagnose issues.
-   **Production**: Tracing adds some overhead due to data collection and
    serialization. For production environments, consider disabling tracing or
    implementing a sampling mechanism or a more lightweight trace backend if
    needed.
-   **Multimodal Data**: The trace viewer is especially useful for pipelines
    that handle images and audio, allowing you to see or hear the data at each
    stage.
-   **Errors and Cancellations**: If a processor raises an exception or is
    cancelled, the trace will record this state, which is useful for debugging
    failures in complex asynchronous pipelines.
