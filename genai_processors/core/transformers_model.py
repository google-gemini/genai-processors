# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Wraps a Hugging Face Transformers model into a Processor.

This module allows running Processor pipelines with locally-run LLMs, such as
Gemma. Also before working with Gemma models, make sure you have requested
access via Kaggle [https://ai.google.dev/gemma/docs/setup#get-access] and
reviewed the Gemma terms of use [https://ai.google.dev/gemma/terms].
"""

import asyncio
from collections.abc import AsyncIterable
import json
from typing import Any, Callable

from genai_processors import content_api
from genai_processors import processor
from genai_processors import tool_utils
from google.genai import types as genai_types
import transformers
from typing_extensions import TypedDict


class GenerateContentConfig(TypedDict, total=False):
  """Optional model configuration parameters."""

  system_instruction: content_api.ProcessorContentTypes
  """Instructions for the model to steer it toward better performance.

  For example, "Answer as concisely as possible" or "Don't use technical
  terms in your response".
  """

  seed: int | None
  """Seed."""

  stop_sequences: list[str]
  """Stop sequences."""

  temperature: float | None
  """Controls the randomness of predictions."""

  top_k: float | None
  """If specified, top-k sampling will be used."""

  top_p: float | None
  """If specified, nucleus sampling will be used."""

  tools: list[genai_types.Tool | Callable[..., Any]] | None
  """Tools the model may call."""

  max_output_tokens: int | None
  """Maximum number of tokens that can be generated in the response."""


class TransformersModel(processor.Processor):
  """`Processor` that calls the Hugging Face Transformers model.

  Note: All content is buffered prior to calling the model.
  """

  def __init__(
      self,
      *,
      model_name: str = '',
      generate_content_config: GenerateContentConfig | None = None,
  ):
    """Initializes the Transformers model.

    Args:
      model_name: Pretrained model name or path.
      generate_content_config: Inference settings.

    Returns:
      A `Processor` that calls Hugging Face Transformers model in turn-based
      fashion.
    """  # fmt: skip
    self._generate_content_config = generate_content_config or {}

    self._hf_processor = transformers.AutoProcessor.from_pretrained(model_name)
    self._model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, device_map='auto'
    )

    self._tools = []
    if tools_config := self._generate_content_config.get('tools'):
      for fdecl in tool_utils.to_function_declarations(tools_config):
        self._tools.append(tool_utils.function_declaration_to_json(fdecl))

    self._system_instruction = []
    for part in content_api.ProcessorContent(
        self._generate_content_config.get('system_instruction', ())
    ):
      self._system_instruction.append(
          _to_hf_message(part, default_role='system')
      )
    self._generation_kwargs = {}
    for arg in ['temperature', 'top_k', 'top_p']:
      if self._generate_content_config.get(arg) is not None:
        self._generation_kwargs[arg] = self._generate_content_config[arg]

    self._generation_kwargs['max_new_tokens'] = (
        self._generate_content_config.get(
            'max_output_tokens', self._model.config.max_position_embeddings
        )
    )

    if seed := self._generate_content_config.get('seed'):
      transformers.set_seed(seed)

  async def call(
      self, content: AsyncIterable[content_api.ProcessorPartTypes]
  ) -> AsyncIterable[content_api.ProcessorPartTypes]:
    """Internal method to call the Ollama API and stream results."""
    messages = list(self._system_instruction)
    async for part in content:
      messages.append(_to_hf_message(part, default_role='user'))
    if not messages:
      return

    inputs = self._hf_processor.apply_chat_template(
        messages,
        tools=self._tools,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors='pt',
    )

    input_len = len(inputs['input_ids'][0])
    output_queue = asyncio.Queue[content_api.ProcessorPart | None]()

    generate_task = processor.create_task(
        asyncio.to_thread(
            self._model.generate,
            **inputs.to(self._model.device),
            streamer=_Streamer(
                input_len,
                asyncio.get_running_loop(),
                self._hf_processor,
                output_queue,
            ),
            pad_token_id=self._hf_processor.eos_token_id,
            **self._generation_kwargs,
        )
    )

    try:
      while (part := await output_queue.get()) is not None:
        yield part
    finally:
      await generate_task


def _to_hf_message(
    part: content_api.ProcessorPart, default_role: str = ''
) -> dict[str, Any]:
  """Returns HF message JSON."""
  # Gemini API uses upper case for roles, while transformers uses lower case.
  role = part.role.lower() or default_role
  if role == 'model':
    role = 'assistant'

  message: dict[str, Any] = {'role': role}

  if part.function_call:
    message['role'] = 'assistant'
    message['tool_calls'] = [{
        'type': 'function',
        'function': {
            'name': part.function_call.name,
            'arguments': json.dumps(part.function_call.args),
        },
    }]
    return message
  elif part.function_response:
    message['role'] = 'tool'
    message['content'] = json.dumps(part.function_response.response)
    message['name'] = part.function_response.name
    return message
  elif content_api.is_text(part.mimetype):
    message['content'] = [{'type': 'text', 'text': part.text}]
  elif content_api.is_image(part.mimetype):
    raise ValueError('Images are not supported yet.')
    # TODO(kibergus): Add image support. Can they be passed as data and not URL?
    # message['content'] = [{
    #     "type": "image",
    #     "image": [{"type": "image", "url": part.text},
    # ]
  else:
    raise ValueError(f'Unsupported Part type: {part.mimetype}')

  return message


class _Streamer(transformers.generation.BaseStreamer):
  """A utility class for streaming tokens out of a transformer models.

  As the model generation is run in a separate thread we use asyncio.Queue and
  asyncio.Loop.call_soon_threadsafe to push the output back to the event loop.
  """

  def __init__(
      self,
      skip_tokens: int,
      loop: asyncio.AbstractEventLoop,
      hf_processor: transformers.AutoProcessor,
      output_queue: asyncio.Queue[content_api.ProcessorPartTypes | None],
  ):
    self._skip_tokens = skip_tokens
    self._loop = loop
    self._hf_processor = hf_processor
    self._queue = output_queue

  def put(self, value):
    """Invoked to push new tokens."""
    tokens = value.flatten()
    tokens_to_skip = min(len(tokens), self._skip_tokens)
    tokens = tokens[tokens_to_skip:]
    self._skip_tokens -= tokens_to_skip

    part = self._hf_processor.decode(tokens, skip_special_tokens=True)
    if part:
      self._loop.call_soon_threadsafe(self._queue.put_nowait, part)

  def end(self):
    """Invoked to signal the end of generation."""
    self._loop.call_soon_threadsafe(self._queue.put_nowait, None)
