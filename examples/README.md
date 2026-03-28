# Example of Usage 📝

This directory contains examples of CLIs and Colabs built from a few processors
each. Go over the examples to see how processors can be used to build various
agents.

We recommend checking the following CLI examples first:

*   The
    [Real-Time Simple CLI](https://github.com/google-gemini/genai-processors/blob/main/examples/realtime_simple_cli.py)
    is an Audio-in Audio-out Live processor with google search as a tool. It is
    a full client-side implementation of a Live processor that demonstrates the
    streaming and orchestration capabilities of GenAI Processors. It uses
    [realtime.py](https://github.com/google-gemini/genai-processors/blob/main/genai_processors/core/realtime.py)
    to transform any text-based LLM (or processor) into a Live agent.

*   The
    [Live CLI](https://github.com/google-gemini/genai-processors/blob/main/examples/live_simple_cli.py)
    is a full multimodal Live processor using the Google Live API. In contrast
    to the Real-Time Simple CLI above, it also handles images at a 1 FPS rate.

*   The
    [Trip Request CLI](https://github.com/google-gemini/genai-processors/blob/main/examples/trip_request_cli.py)
    is a simple trip planner that returns a high level plan for a trip defined
    by a destination, start and an end date. It is an example of concurrency and
    processor usage in a turn-based context.

*   The
    [Chat CLI](https://github.com/google-gemini/genai-processors/blob/main/examples/chat.py)
    is a versatile chat agent using async function calling with MCP to use tools
    during a conversation. It demonstrates how to use MCP (Model Context
    Protocol) tools with GenAI Processors. It is meant to be re-used and
    extended.

Sub-directories include more complex agents like
[Research](https://github.com/google-gemini/genai-processors/blob/main/examples/research/README.md)
(deep research agent),
[Commentator](https://github.com/google-gemini/genai-processors/blob/main/examples/live_commentator/README.md)
(live commentator on a video feed including an interruption mechanism), or
[Live Illustrator](https://github.com/google-gemini/genai-processors/blob/main/examples/live_illustrator/README.md)
(continuously listens to audio and generates accompanying images triggered by speech).
Check the README files in these subdirectories to get an in-depth
description of how they work and how they were built.

Other CLIs like
[speech_to_text_cli](https://github.com/google-gemini/genai-processors/blob/main/examples/speech_to_text_cli.py),
[text_to_speech_cli](https://github.com/google-gemini/genai-processors/blob/main/examples/text_to_speech_cli.py),
or
[vad_cli](https://github.com/google-gemini/genai-processors/blob/main/examples/vad_cli.py)
are simple wrappers around an existing processor and can be used to check that your
environment is set up correctly, e.g. to use the Google Speech API or local VAD.
