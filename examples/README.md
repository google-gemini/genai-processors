# Example of Usage 📝

This directory contains examples of CLIs and Colabs built from a few processors
each. Go over the examples to see how processors can be used to build various
agents.

We recommend checking the following CLI examples first:

*   The [Real-Time Simple CLI](realtime_simple_cli.py) is an Audio-in Audio-out
    Live processor with google search as a tool. It is a full client-side
    implementation of a Live processor that demonstrates the streaming and
    orchestration capabilities of GenAI Processors. It uses
    [realtime.py](https://github.com/google-gemini/genai-processors/blob/main/genai_processors/core/realtime.py)
    to transform any text-based LLM (or processor) into a Live agent.

*   The [Live CLI](live_simple_cli.py) is a full multimodal Live processor
    using the Google Live API. In contrast to the Real-Time Simple CLI above, it
    also handles images at a 1 FPS rate.

*   The [Trip Request CLI](trip_request_cli.py) is a simple trip planner that
    returns a high level plan for a trip defined by a destination, start and an
    end date. It is an example of concurrency and processor usage in a
    turn-based context.

Sub-directories include more complex agents like [Research](research/README.md)
(deep research agent) or [Commentator](live/README.md) (live commentator on a
video feed including an interruption mechanism). Check the README files in these
subdirectories to get an in-depth description of how they work and how they were
built.

Other CLIs like [speech_to_text_cli](speech_to_text_cli.py) or
[text_to_speech_cli](text_to_speech_cli.py) are simple wrappers around existing
processor and can be used to check that your environment is set up correctly,
e.g. to use the Google Speech API.
