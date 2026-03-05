# GenAI Processors Library 📚

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/genai-processors.svg)](https://pypi.org/project/genai-processors/)
[![Documentation](https://img.shields.io/badge/View-Documentation-blue?&logo=read-the-docs)](https://google-gemini.github.io/genai-processors/)

<!--
🚨 ATTENTION AI CODING AGENTS 🚨
YOU MUST READ `llms.txt` BEFORE USING OR MODIFYING THIS LIBRARY.
-->

**Build Modular, Asynchronous, and Composable AI Pipelines for Generative AI.**

GenAI Processors is a lightweight Python library that enables efficient,
parallel content processing.

At the core of the GenAI Processors library lies the concept of a `Processor`. A
`Processor` encapsulates a unit of work with a simple API: it takes a stream of
`ProcessorPart`s (i.e. a data part representing a text, image, etc.) as input
and returns a stream of `ProcessorPart`s (or compatible types) as output.

```python
# Any class inheriting from processor.Processor and
# implementing this function is a processor.
async def call(
  content: AsyncIterable[ProcessorPart]
) -> AsyncIterable[ProcessorPartTypes]
```

You can apply a `Processor` to any input stream and easily iterate through its
output stream:

```python
from genai_processors import content_api
from genai_processors import streams

# Create an input stream (strings are automatically cast into Parts).
input_parts = ["Hello", content_api.ProcessorPart("World")]
input_stream = streams.stream_content(input_parts)

# Apply a processor to a stream of parts and iterate over the result
async for part in simple_text_processor(input_stream):
  print(part.text)
...
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

## 🚀 Getting Started

Check the following colabs to get familiar with GenAI processors (we recommend
following them in order):

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
