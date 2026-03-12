# GenAI Processors

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/google-gemini/genai-processors/blob/main/LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/genai-processors.svg)](https://pypi.org/project/genai-processors/)
[![Source Code](https://img.shields.io/badge/View-Source%20Code-green?&logo=github)](https://github.com/google-gemini/genai-processors/tree/main)

**Composable Building Blocks for Generative AI Pipelines**

GenAI Processors is a lightweight Python library designed for building modular,
asynchronous, and composable AI applications and agents.

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } __Getting Started__

    ---

    Install the library and build your first pipeline in minutes.

    [:octicons-arrow-right-24: Quick Start](getting-started.md)

    [:octicons-arrow-right-24: Examples](examples/cli.md)

-   :material-lightbulb-outline:{ .lg .middle } __Design Principles__

    ---

    The philosophy behind GenAI Processors: unified content, streaming, and composability.

    [:octicons-arrow-right-24: Principles](principles.md)

-   :material-book-open-variant:{ .lg .middle } __Core Concepts__

    ---

    Deep dive into Processors, Parts, Streams, and Orchestration.

    [:octicons-arrow-right-24: Concepts](concepts/processor.md)


-   :material-chip:{ .lg .middle } __Built-in Processors__

    ---

    Documentation for core processors like GenaiModel and LiveProcessor.

    [:octicons-arrow-right-24: Processors](development/built-in-processors.md)

</div>

## ✨ Key Features

Genai Processors are built for rapid prototyping where low latency and
responsive behavior are the priority, whether you're building for real-time
interaction or traditional text. The goal is to make developing streaming,
multimodal agents as straightforward as using text-only bots; by simplifying the
way you mix streaming and non-streaming code, we ensure that achieving a fast
time-to-first-token is obtained out-of-the-box.

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
