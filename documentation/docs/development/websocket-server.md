# Wrapping a Processor into a WebSocket Server

The
[`live_server`](https://github.com/google-gemini/genai-processors/blob/main/dev/live_server.py)
module provides a simple way to serve a GenAI Processor over a WebSocket
connection. This allows you to bridge a local agent pipeline to an AI Studio
Applet or a custom web interface.

## Quick Start

To initialize the server, define a `processor_factory` function to creates an
instance of your processor and pass it to `live_server.run_server`:

```python
import asyncio
from typing import Any
from genai_processors import processor
from genai_processors.dev import live_server

def create_my_processor(config: dict[str, Any]) -> processor.Processor:
  # 'config' is sent by the client at start.
  my_processor = ... # Your processor initialization
  return my_processor

async def main():
  # Defaults to localhost.
  await live_server.run_server(create_my_processor, port=8765)

if __name__ == '__main__':
  asyncio.run(main())
```

## Communication Protocol

The client and server exchange JSON-stringified messages. Each message follows a
consistent envelope representing a `ProcessorPart`.

**1. Initialization (Client → Server)**

Before sending data, the client should send a configuration message. This
dictionary is passed directly to your `processor_factory` as the `config`
argument.

```json
{
  "mimetype": "application/x-config",
  "metadata": {
    "my_setting": "my_value"
  }
}
```


**2. Sending Data & Control (Client → Server)**

Input (Text, Audio, Images) is wrapped in a `ProcessorPart` object. Binary data
must be Base64-encoded.

-   **Text**: `json { "part": { "text": "Hello World" }, "role": "user" }`

-   **Media**:

> ```json
> {
>  "part": {
>    "inline_data": {
>      "data": "SGVsbG8gV29ybGQ=",
>      "mime_type": "audio/l16;rate=24000"
>    }
>  },
>  "role": "user",
>  "substream_name": "realtime"
> }
> ```

-   **Control Signals:** The client can send empty `ProcessorPart` objects with
    `metadata` to manage the session:

    -   `{"metadata": {"reset": true}}`: Resets the processor state on the
        server.

    -   `{"metadata": {"mic": "off"}}`: Notifies the server the client has muted
        their microphone. This sends a `part` to the underlying processor on the
        `realtime` substream with the metadata entry for `audio_stream_end` set
        to `True`.

**3. Receiving Data & State (Server → Client)**

The server streams responses back using the same structure. Common mimetypes to
handle include `text/plain`, `audio/*`, and the internal `application/x-state`.

**State Metadata to Monitor:**

-   `generation_complete: True`: The processor has finished generating a
    response.

-   `interrupted: True`: The processor was interrupted (e.g., by a "start of
    speech" signal).

-   `health_check: True`: Sent periodically to ensure the connection is alive.

**Example Response:**

```json
{
  "part": { "text": "Hello! How can I help?" },
  "role": "model",
  "mimetype": "text/plain"
}
```

**Implementation Examples (TypeScript):**

A typical way to handle these messages in your Applet frontend:

```ts
this.agentWebSocket.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    const { mimetype, part, metadata } = msg;

    if (mimetype?.startsWith('audio/')) {
        // Decode Base64 and play
        this.playBuffer(base64.decode(part.inline_data.data));
    }
    else if (mimetype === 'application/x-state') {
        if (metadata.generation_complete) console.log("Done!");
    }
};
```

You can see real examples of client apps using the websocket server here:

*   [Live Commentator UI](https://github.com/google-gemini/genai-processors/blob/main/examples/live_commentator/ais_app/index.tsx)
*   [Live Ilustrator UI](https://github.com/google-gemini/genai-processors/blob/main/examples/live_illustrator/ais_app/index.tsx)
*   [Widgets Agent UI](https://github.com/google-gemini/genai-processors/blob/main/examples/widgets/ais_app/index.tsx)
