# Development Tools for Processors

## Processor Trace

Processor traces allow you to record the inputs, outputs, and internal steps of
a processor during its execution. This is useful for debugging, analysis, and
understanding processor behavior.

A trace is a timeline of events, where each event represents an input part, an
output part, or a call to a sub-processor. Events are time stamped and ordered
chronologically. If a processor calls other processors, sub-traces are created
and nested within the main trace, providing a hierarchical view of execution.

### Enabling Tracing

To enable trace collection for a processor, use a `Trace` context manager. We
provide examples here with the `SyncFileTrace` context manager implementation.
Other approaches could be implemented in the future (e.g. stored in a DB or
streaming into a file instead for writing when the trace is done).

```python
import asyncio
from genai_processors import processor
from genai_processors.dev import trace_file

@processor.processor_function
async def my_processor_fn(content):
  ...

async def main():
  trace_dir = '/path/to/your/trace/directory'
  # Any processor call within this context will be traced.
  # Change `trace_file.SyncFileTrace` with other tracing implementation if
  # needed.
  async with trace_file.SyncFileTrace(trace_dir):
      await processor.apply_async(my_processor_fn, parts)
```

### Default implementation: write to files

The default implementation of tracing is done with `trace_file.SyncFileTrace`.
When a processor is called within a `SyncFileTrace`, it records its execution
and saves it into two files under `trace_dir` provided to the trace scope:

-   `{processor_name}_{trace_id}.json` containing a json dictionary that can be
loaded for further programmatic analysis using `SyncFileTrace.load`:

    ```python
    import os
    from genai_processors.dev import trace_file

    trace_dir = '/path/to/your/trace/directory'
    traces = []
    for f in os.listdir(trace_dir):
        if f.endswith('.json'):
            traces.append(trace_file.SyncFileTrace.load(
              os.path.join(trace_dir, f)
              )
            )
    ```

-   `{processor_name}_{trace_id}.html` containing an HTML representation of the
trace that can easily be viewed on a web browser. This is the same content as
the json dictionary.

### Implementing a new tracing

To implement a custom trace sink (e.g., save to a database, stream to a network
 location), you need to extend the abstract base class `trace.Trace` from
`genai_processors.dev.trace` and implement its abstract methods. Your new class
can then be used in place of `SyncFileTrace`.

You must implement the following methods:

*   `async def add_input(self, part: content_api.ProcessorPart) -> None`:
Handles input parts received by the processor.
*   `async def add_output(self, part: content_api.ProcessorPart) -> None`:
Handles output parts produced by the processor.
*   `async def add_sub_trace(self) -> Trace`:
Handles the start of a nested processor call. The returned `trace` should be an
instance of your custom trace implementation.
*   `async def _finalize(self) -> None:`: Called when the trace context is
exited. Use this to perform final actions like flushing buffers, closing
connections, or writing data to disk.

**Asynchronous Design**

All event-handling methods (`add_input`, `add_output`, `add_sub_trace`) and
`_finalize` are `async`. This design prevents tracing from blocking the
processor's execution thread, which is critical in an asynchronous framework.
If your tracing implementation needs to perform I/O (e.g., writing to a remote
database or file system), you can use `await` for these operations without
blocking the processor.
