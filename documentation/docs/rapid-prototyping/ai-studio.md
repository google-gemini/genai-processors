# Building AI Studio Applets for Live Agents

Google's [AI Studio](https://aistudio.google.com/) allows you to build
Applets—custom web-based frontends for rapidly prototyping interactive AI
agents. Applets are ideal for demos requiring microphone/camera access or
specialized UI components that the standard chat interface doesn't support.

## Applet Structure

An Applet is a standard web bundle consisting of three core files:

-   `index.html`: The UI skeleton.

-   `index.js / .tsx`: The client-side logic (WebSocket handling, audio
    processing, UI state).

-   `metadata.json`: Defines the app's identity and required browser
    permissions.

**Example** `metadata.json`

```json
{
  "name": "Live Voice Assistant",
  "description": "Real-time voice agent using WebSockets",
  "requestFramePermissions": [
    "microphone",
    "camera"
  ]
}
```

## Development Workflow

The typical Applet logic follows this lifecycle:

1.  **Establish Connection:** Open a WebSocket to your backend (e.g.,
    `ws://localhost:8765`).

2.  **Handshake:** Send an initial `application/x-config` message to initialize
    the remote Processor.

3.  **Input Streaming:** Capture microphone/text and send Base64-encoded
    `ProcessorPart` objects.

4.  **Output Rendering:** Parse incoming JSON messages and update the DOM or
    play audio buffers.

## Generating Applets with AI

AI Studio is highly effective at generating the boilerplate for these Applets.
To get a functional starting point, use a prompt that specifies the
communication requirements:

**Example of Prompt snippet:**

> "Create an AI Studio Applet with a clean UI for a voice commentator. The
> applet must:
>
> 1.  Connect to a WebSocket backend at ws://localhost:8765. The server is not
>     hosted on the applet nor in typescript. It is already built elsewhere, do
>     not build it.
>
> 2.  Send/receive JSON serialized representations of ProcessorParts (from the
>     GenAI Processor library).

## Reference Implementation

For a complete example including audio buffering and state management, see the
[`Live Commentator Applet Source`](https://github.com/google-gemini/genai-processors/blob/main/examples/live_commentator/ais_app/).
