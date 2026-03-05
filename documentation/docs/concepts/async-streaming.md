# Async & Streaming

Content streams are the primary mechanism through which processors ingest,
transform, and output data. At their core, streams are
`AsyncIterable[ProcessorPart]`. This means the most direct way to interact with
them is via an asynchronous loop:

```python
async for part in my_processor(input_stream):
    print(part.text)
```

The
[ContentStream](https://github.com/google-gemini/genai-processors/blob/main/genai_processors/content_api.py#:~:text=class%20ContentStream)
class is an extension of `AsyncIterable[ProcessorPart]` that simplifies data
handling with built-in convenience methods:

*   `await stream.text()`: Concatenates all parts and returns the full text.

*   `content = await stream.gather()`: Aggregates the entire stream into a
    single content object.

For more advanced workflows, the
[streams](https://github.com/google-gemini/genai-processors/blob/main/genai_processors/streams.py)
module provides additional utilities for direct stream manipulation.

## Splitting, forking or teeing Streams

If the same input needs to be sent to multiple destinations at once, the `split`
method comes in handy. It creates many streams identical to the original one
which can be iterated over independently.

```python
# 'split' creates two independent iterators from one source. Note that the
# parts are not copied. If copies are required, use `with_copy=True` as
# described below.
stream1, stream2 = streams.split(content, n=2)
output_stream1 = processor1(stream1)
output_stream2 = processor2(stream2)

# Use 'with_copy=True' if downstream processors might mutate the objects
s1, s2 = streams.split(content, n=2, with_copy=True)
```

A typical usage of splitting is when you want to "fan out" the user request to
many independent agents that will run separate searches or analysis. You would
create as many streams as processors and apply them to these streams. The next
section describes the operations to recombine their outputs.

## Combining Streams

When you have multiple streams, you can combine them in two distinct ways:

### A. Concatenation (`concat`)

*   **Behavior:** Concurrent. Compute the parts in Stream A and in Stream B
    concurrently (e.g. fetch content from two websites at the same time).
*   **Order:** Preserved strictly. Even if Stream B has content available early,
    Stream A content is streamed back first.

```python
# stream1 is returned before stream2
async for part in streams.concat(output_stream1, output_stream2):
    print(part.text)
```

This concatenation operation would typically be used in the "fan out" example,
collecting the results of all sub-processors in contiguous paragraphs. You can
then apply another "synthesis" step with a processor that takes this newly
concatenated stream. Doing so ensures that concurrency is managed optimally: you
do not have to worry about parallelism as it is handled by the framework behind
the scene.

### B. Merging (`merge`)

*   **Behavior:** Interleaved. Yields items from whichever stream has data ready
    first (e.g. receive audio and images from two streaming sources of the same
    live event).
*   **Order:** Based on availability (time).

```python
# Output: Mixed parts from stream1 and stream2 as they arrive
async for part in streams.merge([output_stream1, output_stream2]):
    print(part.text)
```

`merge` will end when all streams end. If it needs to end when any stream ends,
set its `stop_on_first` argument to `True`.

A typical usage of such a merge operation is when the streams come from
real-time sources that are all sync'ed with time. You can imagine images coming
from different cameras, with sound tracks, or a flow a coordinates provided by
different sensors.

## Streams and Queues

Processors return streams. If you have `AsyncIterable[ProcessorPart]` or some
other form of content,
[ContentStream](https://github.com/google-gemini/genai-processors/blob/main/genai_processors/content_api.py#:~:text=class%20ContentStream\)
can be constructed directly from it. Another common pattern is turning
`asyncio.Queue[content_api.ProcessorPart | None]` into a stream. It is commonly
used to merge multiple sources or to feed parts from non asyncio code. `None` is
used then as the end of the queue or of the stream. This can be done easily with
the `enqueue` and `dequeue` methods:

```python
import asyncio
from genai_processors import content_api
from genai_processors import streams

# Queue of parts. None is added to indicate when the queue is done.
q = asyncio.Queue[content_api.ProcessorPart | None]()

# Builds an iterable over the parts of q until None is met.
stream = streams.dequeue(q)

# Adds the content of stream into the queue q, adds None when done.
# p has the same content as q.
p = asyncio.Queue[content_api.ProcessorPart | None]()
streams.enqueue(stream, p)
```

Queues provide a flexible way to create streams collecting parts from different
processors and places: you can easily pass a queue to many processors in their
constructors and dequeue the queue into a single stream to pass it to another
processor. When used with the `merge` operation, this provides a powerful tool
to create complex logic between processors as described in the next section.

## Creating Graphs of Processors

You can easily build complex processing graphs—like a large fan-out and merge
agent—using simple stream splitting and concatenation. The framework handles all
the heavy lifting for concurrency and schedules operations to ensure you get
your first token as fast as possible.

Since streams can be created from queues anywhere in your code, you can also
build loops and custom behaviors. The following example shows this in action,
starting with stream1 fed by an external input processor.

```python
from genai_processors import context
from genai_processors import streams

# We introduce a queue that will receive the main content.
queue1 = asyncio.Queue[content_api.ProcessorPart | None]()

# We create a concurrent task that adds the input_stream to the queue. This is
# equivalent to a merge operation inside queue1 where parts are ordered by
# arrival time.
context.create_task(streams.enqueue(input_stream, queue1))

# We transform queue1 into an async iterable. This async iterable will end as
# soon as input_stream ends or queue1 is fed with None.
stream1 = streams.dequeue(queue1)

# Now we can inject the content of stream1 anywhere in the program where
# queue1 is available. This enables us here to loop back the output of
# processor1 into its input.
async for part in processor1(stream1):
    if content_api.is_text(part.mimetype):
        if "error" in part.text:
            # We signal to the processor1 that it needs to try again.
            queue1.put_nowait(content_api.ProcessorPart("error, try again"))
```

This example is obviously a bit constructed but it shows how easy it is to
transfer parts between streams. We used the `enqueue` and `dequeue` operation
here but you could also use the `merge` or `concat` functions as well and get
different types of merges (e.g. only end `stream1` when both `queue1` and
`input_stream` are done).
