# GenAI Processors Library 📚

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/genai-processors.svg)](https://pypi.org/project/genai-processors/)
[![Documentation](https://img.shields.io/badge/View-Documentation-blue?&logo=read-the-docs)](https://google-gemini.github.io/genai-processors/)

**Build Modular, Asynchronous, and Composable AI Pipelines for Generative AI.**

GenAI Processors is a lightweight Python library that enables efficient,
parallel content processing. It addresses the fragmentation of LLM APIs through
three core pillars:

1.  **Unified Content Model**: A single, consistent representation for inputs
    and outputs across models, agents, and tools.
2.  **Processors**: Simple, composable Python classes that transform content
    streams using native `asyncio`.
3.  **Streaming**: Asynchronous streaming capabilities built-in by default,
    without added plumbing complexity.

At the ecosystem's core lies the `Processor`, which encapsulates a unit of work.
Through a "dual-interface" pattern, it handles the complexity of asynchronous,
multimodal data streaming while exposing a simple API to developers:

```python
from typing import AsyncIterable
from genai_processors import content_api
from genai_processors import processor

class EchoProcessor(processor.Processor):
  # The PRODUCER interface (for the processor author):
  # Takes a robust ProcessorStream as input, and yields part types.
  async def call(
      self, content: content_api.ProcessorStream
  ) -> AsyncIterable[content_api.ProcessorPartTypes]:
      # Process content as it streams in!
      async for part in content:
          yield part
```

Applying a `Processor` is just as straightforward. The CONSUMER interface
accepts wide, forgiving input types and returns a powerful stream that can be
awaited entirely or streamed chunk-by-chunk:

```python
# The CONSUMER interface (for the caller):
# Provide input effortlessly. Strings are automatically cast into Parts.
input_content = ["Hello ", content_api.ProcessorPart("World")]

# 1. Gather all outputs easily into one object:
result: content_api.ProcessorContent = await simple_text_processor(input_content).gather()

# 2. Or for text-only agents, get the text directly:
print(await simple_text_processor(input_content).text())

# 3. And for streaming use cases, iterate over the parts as they arrive:
async for part in simple_text_processor(input_content):
  print(part.text, end="")
```

The concept of `Processor` provides a common abstraction for Gemini model calls
and increasingly complex behaviors built around them, accommodating both
turn-based interactions and live streaming.

## ✨ Key Features

*   **Modular**: Breaks down complex tasks into reusable `Processor` and
    `PartProcessor` units, which are easily chained (`+`) or parallelized (`//`)
    to create sophisticated data flows and agentic behaviors.
*   **Integrated with GenAI API**: Includes ready-to-use processors like
    `GenaiModel` for turn-based API calls and `LiveProcessor` for real-time
    streaming interactions.
*   **Extensible**: Lets you create custom processors by inheriting from base
    classes or using simple function decorators.
*   **Rich Content Handling**:
    *   `ProcessorPart`: A wrapper around `genai.types.Part` enriched with
        metadata like MIME type, role, and custom attributes.
    *   Supports various content types (text, images, audio, custom JSON).
*   **Asynchronous & Concurrent**: Built on Python's familiar `asyncio`
    framework to orchestrate concurrent tasks (including network I/O and
    communication with compute-heavy subthreads).
*   **Stream Management**: Has utilities for splitting, concatenating, and
    merging asynchronous streams of `ProcessorPart`s.

## 📦 Installation

The GenAI Processors library requires Python 3.10+.

Install it with:

```bash
pip install genai-processors
```

## Code generation

Generative models are often unaware of recent API and SDK updates and may
suggest outdated or legacy code.

We recommend using our [Code Generation instructions](llms.txt) when generating
code that uses GenAI Processors to guide your model towards using the more
recent SDK features. Copy and paste the instructions into your development
environment to provide the model with the necessary context.

## 🚀 Getting Started

We recommend to start with the
[documentation microsite](https://google-gemini.github.io/genai-processors/)
which covers the core concepts, development guides, and architecture.

You can also check the following colabs to get familiar with GenAI processors
(we recommend following them in order):

*   [Content API Colab](https://colab.research.google.com/github/google-gemini/genai-processors/blob/main/notebooks/content_api_intro.ipynb) -
    explains the basics of `ProcessorPart`, `ProcessorContent`, and how to
    create them.
*   [Processor Intro Colab](https://colab.research.google.com/github/google-gemini/genai-processors/blob/main/notebooks/processor_intro.ipynb) -
    an introduction to the core concepts of GenAI Processors.
*   [Create Your Own Processor](https://colab.research.google.com/github/google-gemini/genai-processors/blob/main/notebooks/create_your_own_processor.ipynb) -
    a walkthrough of the typical steps to create a `Processor` or a
    `PartProcessor`.
*   [Work with the Live API](https://colab.research.google.com/github/google-gemini/genai-processors/blob/main/notebooks/live_processor_intro.ipynb) -
    a couple of examples of real-time processors built from the Gemini Live API
    using the `LiveProcessor` class.

## 📖 Examples

Explore the [examples/](examples/) directory for practical demonstrations:

*   [Real-Time Live Example](examples/realtime_simple_cli.py) - an Audio-in
    Audio-out Live agent with google search as a tool. It is a client-side
    implementation of a Live processor (built with text-based
    [Gemini API](https://ai.google.dev/gemini-api/docs) models) that
    demonstrates the streaming and orchestration capabilities of GenAI
    Processors.
*   [Research Agent Example](examples/research/README.md) - a research agent
    built with Processors, comprising 3 sub-processors, chaining, creating
    `ProcessorPart`s, etc.
*   [Live Commentary Example](examples/live/README.md) - a description of a live
    commentary agent built with the
    [Gemini Live API](https://ai.google.dev/gemini-api/docs/live), composed of
    two agents: one for event detection and one for managing the conversation.

## 🧩 Built-in Processors

The [core/](genai_processors/core/) directory contains a set of basic processors
that you can leverage in your own applications. It includes the generic building
blocks needed for most real-time applications and will evolve over time to
include more core components.

Community contributions expanding the set of built-in processors are located
under [contrib/](genai_processors/contrib/) - see the section below on how to
add code to the GenAI Processor library.

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for
guidelines on how to contribute to this project.

## 📜 License

This project is licensed under the Apache License, Version 2.0. See the
[LICENSE](LICENSE) file for details.

## Gemini Terms of Services

If you make use of Gemini via the Genai Processors framework, please ensure you
review the [Terms of Service](https://ai.google.dev/gemini-api/terms).
