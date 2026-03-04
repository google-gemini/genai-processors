# Getting Started

This guide will walk you through setting up GenAI Processors and building your
first AI pipeline.

## Prerequisites

-   **Python 3.11+**
-   **Google AI Studio API Key** as the tutorials and the demos use Gemini. You
    can get the key for free
    [here](https://ai.google.dev/gemini-api/docs/api-key). Once you know more
    about the library, you will be able to use Gemma and other models available
    via Ollama or Transformers.

## Installation

Install the core library:

```bash
pip install genai-processors
```

## Google API Configuration

We will use Gemini API extensively in our examples. Set your API key as an
environment variable for ease of use:

```bash
export GOOGLE_API_KEY="your_api_key_here"
```

## Your First Processor

Let's build a simple processor that accepts text input and generates a response
using Gemini. For this example, we use a Gemini API model.

```python
import asyncio
import os

from genai_processors import content_api
from genai_processors.core import genai_model

async def main():
    # Get your key from the environment variable
    api_key = os.environ.get("GOOGLE_API_KEY")

    # Initialize the model.
    model = genai_model.GenaiModel(
        model_name="gemini-3.0-flash",
        api_key=api_key,
    )

    # Process a single prompt.
    response = model(content_api.ProcessorContent("Hello, GenAI!"))
    # Convenient method to collect all textual parts.
    print(await response.text())

if __name__ == "__main__":
    asyncio.run(main())
```

## Composition

The true power of GenAI Processors lies in composition. You can chain processors
using the `+` operator. More generally, you can combine processors in different
ways knowing they always follow the same simple signature: streams of parts
coming in and out. We recommend to use the `+` operator whenever possible to
benefit from the framework optimizations done under the hood: the processing of
parts is indeed implemented to minimize TTFT and concurrency is used extensively
whenever possible.

```python
from genai_processors import content_api
from genai_processors.core import preamble

# Create a processor that adds a prefix to any input.
system_prompt = preamble.Preamble(
    "You are a pirate styling assistant. Answer everything in pirate speak."
)

# Chain it with the model
pirate_bot = system_prompt + model

# Now the model will always act like a pirate
response = pirate_bot(
    content_api.ContentStream(content=["What color matches blue?"])
    )
print(await response.text())
```

We used textual response but could work with multi-model response as well using
a direct iteration over the output of the processor.

```python
async for part in response:
  # Print all objects returned by the processor.
  print(f"Response part: {part}")
```

## Next Steps

Now that you have the basics, dive deeper into the core concepts:

-   **[Tutorials](https://github.com/google-gemini/genai-processors/blob/main/notebooks/content_api_intro.ipynb)**:
    Learn about the different components of the library step-by-step with
    examples [external colabs].

-   **[Core Concepts](concepts/processor.md)**: get an overview of the core
    concepts of the library from processor to content stream and more.
