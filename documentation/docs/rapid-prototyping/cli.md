# Building CLI Examples

Building a Command Line Interface (CLI) is the fastest way to prototype and test
your agents. This guide outlines the high-level principles for creating robust
CLI examples using GenAI Processors.

## The Pipeline Pattern

The core philosophy of the library is **composition**. Most CLI examples follow
a simple pipeline pattern:

```python
agent = input_processor + logic_processor + output_processor
```

By using the `+` operator, you can chain inputs (mic, camera, terminal) with
logic (Gemini, STT/TTS) and outputs (speakers, terminal output) into a single
executable agent.

## Core Guidelines

When building a new CLI example, follow these guidelines:

### Standard Input/Output Utilities

Avoid writing custom loop logic for terminal interaction. Use the optimized
utilities in `genai_processors.core`:

-   **Text**: Use
    [`text.terminal_input()`](https://github.com/google-gemini/genai-processors/blob/main/genai_processors/core/text.py)
    and `text.terminal_output()`.
-   **Audio**: Use
    [`audio_io.PyAudioIn()`](https://github.com/google-gemini/genai-processors/blob/main/genai_processors/core/audio_io.py)
    and `audio_io.PyAudioOut()`.
-   **Video**: Use
    [`video.VideoIn(video_mode=...)`](https://github.com/google-gemini/genai-processors/blob/main/genai_processors/core/video.py)
    for camera or screen sharing.

### Asynchronous Execution

Processors are designed to be used from async Python, so it is often convenient
to make the `main` function async and run the whole program using
`asyncio.run()`. For instance, most examples are wrapped in an `async def run()`
function and executed via `asyncio.run()`.

## Template Structure

A typical CLI example looks like this:

```python
import asyncio
import os

from genai_processors.core import genai_model
from genai_processors.core import text

# Load configuration from environment
API_KEY = os.environ['GOOGLE_API_KEY']

async def run_agent():
    model = genai_model.GenaiModel(
        api_key=API_KEY,
        model_name='gemini-3.0-flash'
    )

    # Compose the pipeline (Input + Model + Output)
    # text.terminal_output handles the loop and formatting automatically
    prompt = '> '
    await text.terminal_output(
        model(text.terminal_input(prompt=prompt)),
        prompt=prompt
    )

if __name__ == '__main__':
    asyncio.run(run_agent())
```

See the
[`examples/`](https://github.com/google-gemini/genai-processors/tree/main/examples)
directory for full implementations of these patterns.
