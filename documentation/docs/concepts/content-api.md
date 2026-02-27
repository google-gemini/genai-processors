# Content API

The Content API is the foundation of GenAI Processors. It defines how data
—whether text, images, audio, or tool calls— is represented and passed between
processors.

This page introduces the core concepts around Content in GenAI Processors. For a
hands-on introduction, check out the
[Content API Tutorial](../notebooks/content_api_intro.ipynb).

## ProcessorPart: The Atomic Unit

At the core of the GenAI processor library is `ProcessorPart`, a unified wrapper
around `google.genai.types.Part`.

### What is a Part?

A `ProcessorPart` is a **container for multi-modal data**. Unlike simple
strings, a part can hold:

*   **Text**: User prompts or model responses.
*   **Binary Data**: Images, Audio, Video, PDFs.
*   **Function Calls**: Requests from the model to execute tools.
*   **Function Responses**: Results from tools to be sent back to the model.

In an agentic pipeline, we need to carry **extra bits** of information along
with that content without altering the authentic payload. `ProcessorPart` adds:

*   **Substreams**: To route data through logical channels.
*   **Metadata**: To attach arbitrary context (timestamps, trace IDs).
*   **Role**: To explicitly track who produced the content (`user`, `model`,
    `tool`, `system`).

**NOTE**: `ProcessorPart` is built on top of `google.genai.types.Part`. You can
easily wrap a GenAI part into a `ProcessorPart`, or access the underlying GenAI
part via `.part`.

### Creating Parts

You can create `ProcessorPart`s from various raw types or existing GenAI
objects.

#### From Simple Types

```python
from genai_processors import content_api

# Text (defaults to role='')
part = content_api.ProcessorPart("Hello world")

# Text with explicit role
part = content_api.ProcessorPart("Why is the sky blue?", role="user")

# Binary Data (Images, Audio, PDF)
# You MUST specify the mimetype for binary data
image_part = content_api.ProcessorPart(
    image_bytes,
    mimetype="image/png",
    role="user"
)

# Pydantic/Dataclasses (Serialized as JSON)
# Mimetype becomes 'application/json; type=MyClass'
item = MyDataClass(value=123)
data_part = content_api.ProcessorPart.from_dataclass(dataclass=item)
```

#### From GenAI SDK Types

If you already have `google.genai.types` objects:

```python
from google.genai import types

# Existing Part
genai_part = types.Part(text="foo")
part = content_api.ProcessorPart(genai_part)

# Function Call (Model Output)
part = content_api.ProcessorPart.from_function_call(
    name="get_weather",
    args={"city": "Paris"},
    )

# Function Response (Tool Output)
part = content_api.ProcessorPart.from_function_response(
    name="get_weather",
    response={"temp": 20},
    )
```

### Common Properties and Methods

Once you have a `ProcessorPart`, you can inspect and transform it.

*   `part`: Access the underlying `google.genai.types.Part`.
*   `text`: Returns the text content. Raises `ValueError` if the part is not
    text (e.g., an image).
*   `bytes`: Returns the raw bytes (for images/audio) or encoded text.
*   `mimetype`: The media type string.
*   `metadata`: A dictionary of extra information.

**Serialization to Dictionary** You can also send a part over the wire or save
it, using `to_dict()`. It returns a JSON-compatible dictionary.

```python
data = part.to_dict()
# {'part': {...}, 'role': 'user', 'metadata': {...}, ...}

# Reconstruct
new_part = content_api.ProcessorPart.from_dict(data)
```

**Copying Parts** When you copy a `ProcessorPart`, the underlying GenAI `Part`
is **shared** (not deep-copied), but the **metadata** is deep-copied.

```python
# 'copy' shares the underlying heavy data (image bytes, etc.)
# but gives you a fresh metadata dict to modify safely.

derived_part = original_part.copy()
derived_part.metadata['processed'] = True
```

### Helper Methods

You often need to check the type of content a part holds. The library provides
helper functions for this:

```python
if content_api.is_text(part.mimetype):
    print(part.text)

if content_api.is_image(part.mimetype):
    show_image(part.pil_image)

if content_api.is_json(part.mimetype):
    data = json.loads(part.text)
```

## Special Concepts

### Substreams and Routing

A key feature of GenAI Processors is **Substreams**. A single stream of parts
can carry multiple logical streams interleaved together.

Some substreams have **special behavior** in the processor framework:

*   **`debug`** and **`status`**: Parts on these substreams are considered
    "out-of-band". They are **returned directly to the user** and are **NOT**
    passed to downstream processors in a chain. This allows any processor in a
    deep pipeline to emit debug logs or progress updates that bubble up
    immediately without interfering with the main processing flow.
*   **`default`** (empty string): The main conversation content.

```python
# Emitting a debug log that bypasses downstream processing
debug_part = content_api.ProcessorPart(
    "Starting complex calculation...",
    substream_name="debug"
)
```

### End of Turn

In conversational systems, it is crucial to know when a "turn" (a complete
message from user or model) is finished.

`ProcessorPart.end_of_turn()` creates a special sentinel part. This is used by
processors to signal boundaries, for example, to know when to stop waiting for
more chunks and trigger a model call.

```python
# Create an end-of-turn market
eot = content_api.ProcessorPart.end_of_turn()

# Check for it
if content_api.is_end_of_turn(part):
    flush_buffer()
```

## ProcessorContent: A Collection of Parts

`ProcessorContent` is a container for multiple `ProcessorPart`s. It represents a
user prompt or a whole conversation history with different roles and subtreams,
potentially containing text, audio and a images.

It acts like a list:

```python
content = content_api.ProcessorContent([
    content_api.ProcessorPart("Look at this:", role="user"),
    content_api.ProcessorPart(
        image_bytes,
        mimetype="image/png",
        role="user"
        )
    ])

# Iterate
for part in content:
    print(part.mimetype)
```

### Utility: `as_text`

To quickly extract all text from a content object (ignoring non-text parts):
`full_text = content_api.as_text(content)`

## Flexible Inputs: `ProcessorPartTypes`

One of the strengths of the usage of `ProcessorPart` is that it allows for a
flexible input API. Most methods in the library that accept content do not
strictly demand a `ProcessorPart` object. Instead, they accept
`ProcessorPartTypes`, a union of compatible types that are automatically
promoted to `ProcessorPart`.

Accepted types include: * `str`: Converted to a text part. * `bytes`: Converted
to a blob (mimetype must be known contextually or specified as an extra arg). *
`PIL.Image.Image`: Converted to an image part. * `google.genai.types.Part`:
Wrapped directly. * `google.genai.types.File`: Wrapped as a file references.

This means you can often pass a string or an image directly to a processor
without manually wrapping it.

## ContentStream: Unified Streaming

In an agentic pipeline, data flows as a stream. However, sources of data vary:
sometimes you have a static string, sometimes a list of images, and sometimes a
live async iterator from a model.

`ContentStream` is an **adaptor** that normalizes these sources into a unified
`AsyncIterable[ProcessorPart]`.

### Usage

```python
# 1. From a simple string
stream = content_api.ContentStream(content="Hello world")

# 2. From a list of mixed types (promoted via ProcessorPartTypes)
stream = content_api.ContentStream(content=["Text", image_obj])

# 3. From an existing AsyncIterator (passthrough)
stream = content_api.ContentStream(parts_generator=my_async_gen())

# Consumption (async - returns a part as soon as it is generated)
async for part in stream:
    print(part.text)

# Consumption (sync - returns once all parts are generated)
print(await stream.text())
```

## Processors and Streams

Finally, it is important to understand how `Processor`s interact with these
concepts.

A `Processor` typically accepts an `AsyncIterable[ProcessorPartTypes]`. This
means:

1.  You do **not** need to manually convert everything to `ProcessorPart` before
    calling a processor.
2.  You do **not** need to manually wrap your data in a `ContentStream`.

The processor will automatically:

1.  Normalize the input into a `ProcessorStream` (which is a `ContentStream`).
2.  Convert all elements to `ProcessorPart`s.
3.  Process them asynchronously.

```python
# Valid processor call with mixed types
await my_processor(content_api.ContentStream(["Hello", img_bytes]))

# Valid processor call with a generator
await my_processor(my_async_gen())
```
