# %%
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
import dataclasses
import json
import re
from typing import Any, Callable, Literal

from absl import logging
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
      log_chat_template: bool = False,
      tool_response_format: Literal['dict', 'string'] = 'string',
  ):
    """Initializes the Transformers model.

    Args:
      model_name: Pretrained model name or path.
      generate_content_config: Inference settings.
      log_chat_template: Whether to log the HF chat template. This is useful for
        debugging, but can be verbose.
      tool_response_format: The format of the tool response. By default, the tool response is returned as a
        string.

    Returns:
      A `Processor` that calls Hugging Face Transformers model in turn-based
      fashion.
    """  # fmt: skip
    self._log_chat_template = log_chat_template
    self._tool_response_format = tool_response_format
    self._generate_content_config = generate_content_config or {}

    self._hf_processor = transformers.AutoProcessor.from_pretrained(model_name)
    self._model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, device_map='auto'
    )
    self._tools = []
    self._parse_function_calls = False
    if tools_config := self._generate_content_config.get('tools'):
      self._parse_function_calls = True
      for fdecl in tool_utils.to_function_declarations(tools_config):
        self._tools.append(tool_utils.function_declaration_to_json(fdecl))

    self._system_instruction = []
    for part in content_api.ProcessorContent(
        self._generate_content_config.get('system_instruction', ())
    ):
      self._system_instruction.append(
          _to_hf_message(
              part,
              tool_response_format=self._tool_response_format,
              default_role='system',
          )
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
      messages.append(
          _to_hf_message(
              part,
              tool_response_format=self._tool_response_format,
              default_role='user',
          )
      )
    if not messages:
      return

    inputs = self._hf_processor.apply_chat_template(
        messages,
        tools=self._tools,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors='pt',
    )

    if self._log_chat_template:
      inputs_str = self._hf_processor.apply_chat_template(
          messages,
          tools=self._tools,
          add_generation_prompt=True,
          return_tensors='pt',
          tokenize=False,
      )
      logging.info('HF CHAT TEMPLATE: %s', inputs_str)

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
                parse_function_calls=self._parse_function_calls,
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
    part: content_api.ProcessorPart,
    tool_response_format: str,
    default_role: str = '',
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
            'arguments': part.function_call.args,
        },
    }]
    return message
  elif part.function_response:
    message['role'] = 'tool'
    message['name'] = part.function_response.name
    response = part.function_response.response
    match tool_response_format:
      case 'string':
        if 'result' in response:
          response = response['result']
        message['content'] = (
            json.dumps(response) if not isinstance(response, str) else response
        )
      case 'dict':
        if 'result' in response and isinstance(response['result'], dict):
          response = response['result']
        message['content'] = {
            'name': part.function_response.name,
            'response': response,
        }
      case _:
        raise ValueError(
            f'Unsupported tool response format: {tool_response_format}'
        )
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


@dataclasses.dataclass(frozen=True)
class _Tokens:
  segment: Literal['function_call', 'text']
  ids: list[int]


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
      parse_function_calls: bool = True,
  ):
    self._skip_tokens = skip_tokens
    self._loop = loop
    self._hf_processor = hf_processor
    self._queue = output_queue
    self._parse_function_calls = parse_function_calls
    # To mitigate prompt injection attacks we parse function calls directly from
    # tokens and not from text representation. This assumes that the model is
    # trained to use dedicated tokens for the start and end of the call and
    # prompt tokenization correctly escapes attempts to pass fake function
    # calls.
    self._current_function_call_tokens = None
    # Collect the token ids for the start and end of the function call. They
    # will be used to detect the tokens corresponding to a full function call.
    function_call_token_ids = hf_processor.encode(
        text='<start_function_call><end_function_call><escape>',
        add_special_tokens=False,
    )
    assert len(function_call_token_ids) == 3
    self._start_function_id = function_call_token_ids[0]
    self._end_function_id = function_call_token_ids[1]
    self._escape_id = function_call_token_ids[2]
    # Pattern to capture the function name and arguments as groups.
    self._function_call_pattern = re.compile(
        r'^<start_function_call>call:([^{]+)(\{.*\})<end_function_call>$'
    )

  def _process_function_call_tokens(self, tokens: list[int]) -> list[_Tokens]:
    """Extract function calls from the tokens.

    The tokens list can include many function calls in a row (potentially). This
    method will extract the function calls and return the tokens before and
    after each function call. The returned list contains _Tokens segments
    that are either text or a complete function call.

    When _Tokens.segment == 'function_call', the tokens include the function
    call segment corresponding to:
    <start_function_call>call:fn_name...<end_function_call>.

    Args:
      tokens: The token ids list to extract function calls from.

    Returns:
      A list of _Tokens segments.
    """
    if not tokens:
      return []
    if not self._parse_function_calls:
      return [_Tokens(segment='text', ids=tokens)]
    if self._current_function_call_tokens is None:
      # We are not in a function call.
      try:
        lind = tokens.index(self._start_function_id)
      except ValueError:
        # No function call start token id found, return the tokens as is.
        return [_Tokens(segment='text', ids=tokens)]
      # We have found a function call start token id, start collecting the
      # tokens that are part of the function call.
      self._current_function_call_tokens = []
      remaining_tokens = self._process_function_call_tokens(tokens[lind:])
      return [
          _Tokens(segment='text', ids=tokens[:lind]),
          *remaining_tokens,
      ]
    else:
      try:
        rind = tokens.index(self._end_function_id)
      except ValueError:
        # No function call end token id found, collect the tokens for the
        # current function call.
        self._current_function_call_tokens.extend(tokens)
        return []
      # We have found the function call end token id, yield the tokens
      # collected so far and process the remaining tokens.
      fc_tokens = self._current_function_call_tokens + tokens[: rind + 1]
      self._current_function_call_tokens = None
      remaining_tokens = self._process_function_call_tokens(tokens[rind + 1 :])
      return [
          _Tokens(segment='function_call', ids=fc_tokens),
          *remaining_tokens,
      ]

  def _extract_function_call_part(
      self, part: str
  ) -> content_api.ProcessorPartTypes:
    """Extracts a structured function call from its string representation."""
    fc_match = self._function_call_pattern.match(part)
    if not fc_match:
      return part
    fc_name = fc_match.group(1)
    try:
      fc_args = json.loads(fc_match.group(2))
    except json.JSONDecodeError as e:
      return content_api.ProcessorPart(
          f'Could not parse function call arguments: {fc_match.group(2)} for'
          f' function call {fc_name}: {e}',
          substream_name=processor.DEBUG_STREAM,
          role='model',
      )

    return content_api.ProcessorPart.from_function_call(
        name=fc_name,
        args=fc_args,
        role='model',
    )

  def _add_quotes_on_properties(self, token_str: str) -> str:
    """Adds quotes around the property name in a function call."""
    return re.sub(r'([{,]\s*)(\w+)\b:', r'\1"\2":', token_str)

  def put(self, value):
    """Invoked to push new tokens."""
    tokens = value.flatten()
    tokens_to_skip = min(len(tokens), self._skip_tokens)
    tokens = tokens[tokens_to_skip:]
    self._skip_tokens -= tokens_to_skip
    # tokens are torch tensors, we convert them to list here.
    for tokens in self._process_function_call_tokens(tokens.cpu().tolist()):
      if tokens.segment == 'function_call':
        # We have found a function call, process the escape tokens.
        text_tokens = []
        token_ids = []
        inside_escape = False
        for token in tokens.ids:
          # Function arguments may contain special symbols like " and \. While
          # models can escape them, sometimes they get confused, especially if
          # the tool parameter is a code and contains escaping on its own.
          # By using a special <escape> token to separate parameters we can
          # achieve a more robust operation: we escape all the text in between
          # <escape> tokens and parse the result as JSON.
          if token == self._escape_id:
            token_str = self._hf_processor.decode(
                token_ids, skip_special_tokens=True
            )
            if inside_escape:
              # We are inside a string argument, keep things as is but escape
              # all quotes inside to be parsed correctly as JSON.
              text_tokens.append(json.dumps(token_str))
            else:
              # We are not inside a string argument, the property name should be
              # quoted to be parsed correctly as JSON.
              text_tokens.append(self._add_quotes_on_properties(token_str))
            token_ids = []
            inside_escape = not inside_escape
          else:
            token_ids.append(token)
        if token_ids:
          text_tokens.append(
              self._add_quotes_on_properties(
                  self._hf_processor.decode(token_ids, skip_special_tokens=True)
              )
          )
        part = ''.join(text_tokens)
        # Parse the function call arguments as plain json object.
        part = self._extract_function_call_part(part)
      else:
        part = self._hf_processor.decode(tokens.ids, skip_special_tokens=True)
      if part:
        self._loop.call_soon_threadsafe(self._queue.put_nowait, part)

  def end(self):
    """Invoked to signal the end of generation."""
    self._loop.call_soon_threadsafe(self._queue.put_nowait, None)
