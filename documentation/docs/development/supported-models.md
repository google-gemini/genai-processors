# Supported Models

GenAI Processors supports multiple model backends, enabling flexible deployments
from cloud to local. All model backends are processors and inputs/outputs are
handled like any other processor in the library: they accept various content
types as input and return an async iterable of `ProcessorPart` as output. See
[Processor](../concepts/processor.md) for more on handling processor inputs and
outputs.

## Overview

Backend      | Class                                                                                                                          | Use Case
------------ | ------------------------------------------------------------------------------------------------------------------------------ | --------
Gemini API   | [`GenaiModel`](https://github.com/google-gemini/genai-processors/blob/main/genai_processors/core/genai_model.py)               | Cloud-based, full-featured
Ollama       | [`OllamaModel`](https://github.com/google-gemini/genai-processors/blob/main/genai_processors/core/ollama_model.py)             | Local inference, many models
Transformers | [`TransformersModel`](https://github.com/google-gemini/genai-processors/blob/main/genai_processors/core/transformers_model.py) | HuggingFace models

## Gemini API (`GenaiModel`)

The primary backend using Google's Gemini models. `GenaiModel` provides a
unified processor interface for the Gemini API, handling text, images, and video
inputs.

See
[`genai_model.py`](https://github.com/google-gemini/genai-processors/blob/main/genai_processors/core/genai_model.py)
for more details.

All `model_name`s below should be taken from the list provided in
https://ai.google.dev/gemini-api/docs/models.

### Basic Usage

```python
from genai_processors.core import genai_model

model = genai_model.GenaiModel(model_name="gemini-...", api_key=...)

print asyncio.run(model("Hello, world!").text())
```

### Configuration

You can configure generation parameters, system instructions, and tools:

```python
model = genai_model.GenaiModel(
    model_name="gemini-...",
    api_key=...,
    generate_content_config={
        "temperature": 0.7,
        "top_p": 0.9,
        "max_output_tokens": 1024,
        "tools": [my_tool1, my_tool2],  # Example function calling
        "system_instruction": "You are a helpful assistant.",
    },
)
```

### Multimodal Input

You can send images, audio, and video:

```python
from genai_processors import content_api

# Image input
image_part = content_api.ProcessorPart(image_bytes, mimetype="image/png")
content=[image_part, "What's in this image?"]

response = model(content)
```

## Ollama (`OllamaModel`)

Run models locally with [Ollama](https://ollama.com/). This backend supports
many open models, including Google's Gemma family.

See
[`ollama_model.py`](https://github.com/google-gemini/genai-processors/blob/main/genai_processors/core/ollama_model.py)
for more details.

### Setup

1.  Install Ollama: https://ollama.com
2.  Pull a model, e.g., `ollama pull gemma3` or `ollama pull llama3`
3.  Start Ollama: `ollama serve`

### Basic Usage

```python
from genai_processors.core import ollama_model

model = ollama_model.OllamaModel(model_name="gemma3")
response = model("Hello, Gemma!")
```

### Configuration

```python
model = ollama_model.OllamaModel(
    model_name="gemma3",
    host="http://localhost:11434",  # Ollama server
    generate_content_config={
        "temperature": 0.7,
        "seed": 42,
        "system_instruction": "You are a helpful assistant.",
    },
)
```

## Transformers (`TransformersModel`)

Run models locally using HuggingFace
[Transformers](https://huggingface.co/docs/transformers/index). This allows
access to a vast range of models from the HuggingFace Hub, including Gemma.

See
[`transformers_model.py`](https://github.com/google-gemini/genai-processors/blob/main/genai_processors/core/transformers_model.py)
for more details.

`model_name`s for Gemma should be extracted from the
[HuggingFace site](https://huggingface.co/models?search=gemma).

### Basic Usage

```python
from genai_processors.core import transformers_model

# Example with Gemma 2B
model = transformers_model.TransformersModel(
    model_name="google/gemma-..."
)
response = model("Hello from Transformers!")
```

### Configuration

```python
model = transformers_model.TransformersModel(
    model_name="google/gemma-...",
    generate_content_config={
        "temperature": 0.8,
        "max_output_tokens": 512,
        "system_instruction": "You are a helpful assistant.",
    }
)
```
