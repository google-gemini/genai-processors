# Async & Streaming

The genai_processors library is built entirely on Python asyncio. Because
Generative AI models often involve network latency and long generation times,
using asynchronous processing allows your application to remain
responsive—streaming tokens to the user the instant they are generated, rather
than waiting for the full response.

## The Fundamentals

At its core, a `Processor` is an asynchronous generator. It takes an input
stream and yields `ProcessorPart` objects.

### The Basic Loop

To use a processor, you iterate over its output using async for.

```python
import asyncio
from genai_processors import processor

async def main():
    # 1. Initialize your processor
    my_proc = MyProcessor()

    # 2. Process the input stream
    # The processor yields 'parts' (chunks of data/text) as they are ready.
    async for part in my_proc(input_stream):
        print(part.text)

asyncio.run(main())
```

### Streaming vs. Buffering

You have two main patterns for handling model output:

#### A. Token-by-Token Streaming (Low Latency)

This is ideal for chat interfaces. You display tokens immediately as the model
generates them.

```python
async for part in model(input_stream):
    # 'part.text' contains the latest token or chunk
    print(part.text, end="", flush=True)
```

#### B. Buffering (Atomic Processing)

If you need the full response before taking action (e.g., to parse JSON), use
`gather_stream`.

```python
from genai_processors import streams

# Collect all stream parts into a single list
parts = await streams.gather_stream(model(input_stream))

# Combine them into a single string
full_text = content_api.as_text(parts)
```

## Manipulating Streams

The streams module provides powerful tools to manage how data flows through your
application.

### Creating Streams

You can turn standard synchronous data into an async stream compatible with
processors.

```python
from genai_processors import streams
from genai_processors import content_api

# From a list
stream = content_api.ContentStream(content=["Hello", "World"])

# From a list with delays (useful for testing) between iterations
stream = streams.stream_content(content=["Hi", "there"], with_delay_sec=0.5)

# Create an infinite stream (keeps agents alive waiting for input)
stream = streams.endless_stream():
    pass  # Never ends unless cancelled
```

Streams and queues are interchangeable with the `enqueue` and `dequeu` method:

```python
import asyncio
from genai_processors import content_api
from genai_processors import streams

# Queue of parts. None is added to indicate when the queue is done.
q = asyncio.Queue[content_api.ProcessorPart | None]()

# Builds an iterable over the parts of q until None is met.
stream = streams.dequeue(q)

# Adds the content of stream into the queue q, adds None when done.
# p has the same content than q.
p = asyncio.Queue[content_api.ProcessorPart | None]()
streams.enqeue(stream, p)
```

### Splitting Streams

Sometimes you need to send the same input to multiple destinations
simultaneously (e.g., logging inputs to a database while sending them to an
LLM).

```python
# 'split' creates two independent iterators from one source
stream1, stream2 = streams.split(content, n=2)

# Use 'with_copy=True' if downstream processors might mutate the objects
s1, s2 = streams.split(content, n=2, with_copy=True)
```

### Combining Streams

When you have multiple streams, you can combine them in two distinct ways:

#### A. Concatenation (`concat`)

*   **Behavior:** Concurrent. Compute the parts in Stream A and in Stream B
    concurrently (e.g. fetch content from two websites at the same time).
*   **Order:** Preserved strictly. Even if Stream B has content available early,
    Stream A content is streamed back first.

```python
# stream1 is returned before stream2
async for part in streams.concat(stream1, stream2):
    print(part.text)
```

#### B. Merging (`merge`)

*   **Behavior:** Interleaved. Yields items from whichever stream has data ready
first (e.g. receive audio and images from two streaming sources of the same live
event).
*   **Order:** Based on availability (time), not source sequence.

```python
# Output: Mixed parts from stream1 and stream2 as they arrive
async for part in streams.merge([stream1, stream2]):
    print(part.text)
```

`merge` will end when all streams end. If it needs to end when any stream ends,
set its `stop_on_first` argument to `True`.
