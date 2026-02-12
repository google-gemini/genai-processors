# GenAI Processors

**Composable Building Blocks for Generative AI Pipelines**

GenAI Processors is a lightweight Python library designed for building modular,
asynchronous, and composable AI applications and agents.

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } __Getting Started__

    ---

    Install the library and build your first pipeline in minutes.

    [:octicons-arrow-right-24: Quick Start](getting-started.md)

-   :material-book-open-variant:{ .lg .middle } __Core Concepts__

    ---

    Deep dive into Processors, Parts, Streams, and Orchestration.

    [:octicons-arrow-right-24: Concepts](concepts/content-api.md)

-   :material-code-tags:{ .lg .middle } __Examples__

    ---

    Explore real-world applications, agents, and CLI tools.

    [:octicons-arrow-right-24: Examples](examples/realtime-cli.md)

-   :material-chip:{ .lg .middle } __Built-in Processors__

    ---

    Documentation for core processors like GenaiModel and LiveProcessor.

    [:octicons-arrow-right-24: Processors](built-in-processors/index.md)

</div>

## ✨ Key Features

Genai Processors is particularly well suited to experiment with realtime agents
or more generally for any prototyping where time to first token or responsive
behavior is key.

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
