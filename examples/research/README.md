# Research Agent Example 🧠

**Automate Information Gathering and Synthesis with GenAI Processors.**

This example demonstrates how to build a multi-step "Research Agent" using the
`genai-processors` library. The agent takes a user query, breaks it down into
researchable topics, gathers information using AI tools, and then synthesizes a
comprehensive answer.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](../../LICENSE)

## ✨ Key Features

This example highlights several key capabilities of the `genai-processors`
library:

*   **Modular Agent Design**: Decomposes a complex research task into distinct,
    reusable processors (`TopicGenerator`, `TopicResearcher`,
    `TopicVerbalizer`).
*   **Structured Data Flow**: Utilizes custom dataclasses (`interfaces.Topic`)
    embedded within `ProcessorPart`s to pass structured information through the
    pipeline.
*   **Search Tool Integration**: Leverages the GenAI API's Google Search tool
    use capabilities within the `TopicResearcher` for dynamic information
    retrieval.
*   **Dynamic Content Generation**: Generates research topics and synthesized
    reports based on runtime user input and intermediate findings.
*   **Pipeline Composition**: Chains custom-built processors with core library
    processors (like `GenaiModel` and `Preamble`) to create a complete,
    end-to-end workflow.
*   **Configuration Management**: Uses a dedicated `Config` object
    (`interfaces.Config`) to manage model names, prompt parameters, and tool
    configurations.
*   **Asynchronous Processing**: Inherits the asynchronous and concurrent nature
    of the `genai-processors` framework for efficient operation.

## ⚙️ How it Works

The Research Agent follows a structured pipeline:

1.  **Input**: Receives a user's research query as a stream of `ProcessorPart`s.
2.  **Topic Generation (`TopicGenerator`)**:
    *   A `GenaiModel` is prompted (using `prompts.TOPIC_GENERATION_PREAMBLE`)
        to analyze the user's query and generate a list of distinct research
        topics.
    *   Outputs these topics as `ProcessorPart`s, each containing a `Topic`
        dataclass (initially without research text).
3.  **Topic Research (`TopicResearcher`)**:
    *   For each `Topic` part, another `GenaiModel` (configured with tools like
        Google Search) is prompted (using `prompts.TOPIC_RESEARCH_PREAMBLE`) to
        find relevant information. Using PartProcessors, the calls are all made
        concurrently, minimizing Time to First Token (TTFT).
    *   The research findings are added to the `research_text` field of the
        `Topic` dataclass.
4.  **Verbalization (`TopicVerbalizer`)**:
    *   Each researched `Topic` part is transformed into a human-readable
        Markdown string, summarizing the topic, its relation to the original
        query, and the research findings. The verbalization done with a
        [Jinja2 template processor](https://github.com/google-gemini/genai-processors/blob/main/genai_processors/core/jinja_template.py).
5.  **Synthesis (within `ResearchAgent`)**:
    *   All verbalized research texts are collected.
    *   A final `GenaiModel` is prompted (using `prompts.SYNTHESIS_PREAMBLE`) to
        synthesize these individual research pieces into a single, coherent
        response that addresses the user's original query.
6.  **Output**: Streams the final synthesized research report.

## 🧩 Key Components

*   **`agent.py`**:
    *   `ResearchAgent`: The main processor that orchestrates the entire
        pipeline by chaining the sub-processors.
*   **`interfaces.py`**:
    *   `Topic`: Dataclass defining the structure for a research topic (topic
        string, relationship, research text).
    *   `Config`: Dataclass for configuring the agent (model names, number of
        topics, enabled tools).
*   **`prompts.py`**:
    *   Contains the string preambles used to instruct the GenAI models at each
        stage (topic generation, research, synthesis).
*   **`processors/`**:
    *   `topic_generator.py`: Implements `TopicGenerator` for identifying
        research sub-topics.
    *   `topic_researcher.py`: Implements `TopicResearcher` for gathering
        information on each topic using tools.

## 🛠️ Configuration

The behavior of the `ResearchAgent` can be customized through the
`interfaces.Config` object, allowing you to specify:

*   `topic_generator_model_name`
*   `topic_researcher_model_name`
*   `research_synthesizer_model_name`
*   `num_topics` (number of topics to generate)
*   `excluded_topics` (list of topics to avoid)
*   `enabled_research_tools` (list of GenAI tools for the researcher, e.g.,
    Google Search)

## 📚 Example Notebook

An example notebook can be found
[here](https://colab.research.google.com/github/google-gemini/genai-processors/blob/main/notebooks/research_example.ipynb).

## 📜 License

This example is licensed under the Apache License, Version 2.0.
