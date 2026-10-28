# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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

"""Turn-based MiniMax text generation through Chat Completions."""

from collections.abc import AsyncIterable
import copy
from typing import Literal

from genai_processors import content_api
from genai_processors import processor
import httpx


RECOMMENDED_MODELS = ('MiniMax-M3.1-Flash', 'MiniMax-M3')
DEFAULT_MODEL = RECOMMENDED_MODELS[0]
_BASE_URLS = {
    'global': 'https://api.minimax.io/v1',
    'china': 'https://api.minimaxi.com/v1',
}
Effort = Literal['low', 'medium', 'high', 'xhigh', 'max']


class MiniMaxModel(processor.Processor):
  """Generates text, preserving response messages for client-managed history.

  This processor buffers input and returns one text part per turn. Pass the
  complete conversation, including prior output parts, on every call. Tools
  and media inputs are not supported by this text processor.
  """

  def __init__(
      self,
      *,
      api_key: str,
      model_name: str = DEFAULT_MODEL,
      region: Literal['global', 'china'] = 'global',
      base_url: str | None = None,
      thinking: Literal['adaptive', 'disabled'] | None = None,
      reasoning_effort: Effort | None = None,
      max_completion_tokens: int | None = None,
      client: httpx.AsyncClient | None = None,
  ):
    """Initializes the processor.

    Args:
      api_key: API key for the selected endpoint.
      model_name: Model ID. Explicit custom and legacy IDs are sent unchanged.
      region: Direct API region; ignored when base_url is supplied.
      base_url: Optional compatible API root, including its version path.
      thinking: Optional thinking mode. Flash requires adaptive thinking.
      reasoning_effort: Flash thinking depth; omitted uses the API default.
      max_completion_tokens: Optional generation limit, including reasoning.
      client: Optional caller-owned HTTP client for custom transports.
    """
    if not model_name:
      raise ValueError('model_name must not be empty')
    if base_url is None and region not in _BASE_URLS:
      raise ValueError(f'Unknown region: {region}')
    if thinking not in (None, 'adaptive', 'disabled'):
      raise ValueError('thinking must be adaptive or disabled')
    if model_name == DEFAULT_MODEL and thinking == 'disabled':
      raise ValueError('MiniMax-M3.1-Flash requires adaptive thinking')
    if reasoning_effort not in (None, 'low', 'medium', 'high', 'xhigh', 'max'):
      raise ValueError('Unsupported reasoning_effort')
    if max_completion_tokens is not None and max_completion_tokens <= 0:
      raise ValueError('max_completion_tokens must be positive')

    self._model_name = model_name
    self._url = (base_url or _BASE_URLS[region]).rstrip(
        '/'
    ) + '/chat/completions'
    self._headers = {'Authorization': f'Bearer {api_key}'}
    self._options = {}
    if thinking is not None:
      self._options['thinking'] = {'type': thinking}
    if reasoning_effort is not None:
      self._options['reasoning_effort'] = reasoning_effort
    if max_completion_tokens is not None:
      self._options['max_completion_tokens'] = max_completion_tokens
    self._owns_client = client is None
    self._client = (
        client if client is not None else httpx.AsyncClient(timeout=300)
    )

  async def call(
      self, content: processor.ProcessorStream
  ) -> AsyncIterable[content_api.ProcessorPartTypes]:
    messages = []
    async for part in content:
      if part.function_call or part.function_response:
        raise ValueError('MiniMaxModel does not support tool parts')
      if not content_api.is_text(part.mimetype):
        raise ValueError('MiniMaxModel accepts text parts only')
      role = 'assistant' if part.role == 'model' else (part.role or 'user')
      if role not in ('system', 'user', 'assistant'):
        raise ValueError(f'Unsupported text role: {role}')
      # Preserve reasoning and any additional response fields across turns.
      previous = part.metadata.get('minimax_message')
      if role == 'assistant' and previous is not None:
        message = copy.deepcopy(previous)
        message['content'] = part.text
      else:
        message = {'role': role, 'content': part.text}
      messages.append(message)
    if not messages:
      return

    response = await self._client.post(
        self._url,
        headers=self._headers,
        json={
            'model': self._model_name,
            'messages': messages,
            'stream': False,
            **self._options,
        },
    )
    response.raise_for_status()
    data = response.json()
    if 'error' in data:
      raise ValueError(f'MiniMax API error: {data["error"]}')
    choice = data['choices'][0]
    message = choice['message']
    if message.get('tool_calls') or message.get('function_call'):
      raise ValueError('MiniMaxModel does not support tool responses')
    yield content_api.ProcessorPart(
        message.get('content') or '',
        role='model',
        metadata={
            'minimax_message': copy.deepcopy(message),
            'usage': data.get('usage'),
            'model': data.get('model'),
            'finish_reason': choice.get('finish_reason'),
            'turn_complete': True,
        },
    )

  async def aclose(self):
    """Closes the internally created HTTP client, if any."""
    if self._owns_client:
      await self._client.aclose()

  async def __aenter__(self):
    return self

  async def __aexit__(self, exc_type, exc_val, exc_tb):
    await self.aclose()
