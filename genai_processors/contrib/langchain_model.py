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

"""A lightweight LangChain processor that wraps any BaseChatModel.

Provides:
 * Turn based, single prompt inference
 * Multimodal input support (text + images)
 * Flexible prompt templating

Tools are currently not supported on GenAI processors level: tool calls and
responses are not translated beween ProcessorParts and LangChain AIMessages.
There is no translation between genai.Tool definition and LangChain.

There is no special support for structured / constrained decoding either. You
can use "json_mode" provided by LangChain but produced ProcessorParts will be
just streamed text, not structured ProcessorParts like shown in
notebooks/constrained_decoding.ipynb.
"""

import base64
from collections.abc import AsyncIterable
from typing import Any, Iterable, Optional, Union

from genai_processors import content_api
from genai_processors import processor
from langchain_core import messages as langchain_messages
from langchain_core.language_models import chat_models
from langchain_core.prompts import ChatPromptTemplate


class LangChainModel(processor.Processor):
  """A simple turn based wrapper around any LangChain BaseChatModel.

  Buffers one user turn, then streams the LLM response.
  """

  def __init__(
      self,
      model: chat_models.BaseChatModel,
      *,
      system_instruction: content_api.ProcessorContentTypes = (),
      prompt_template: Optional[ChatPromptTemplate] = None,
  ):
    """Initializes the LangChain model.

    Args:
      model: LangChain model to use.
      system_instruction: Instructions for the model to steer it toward better
        performance. If provided, the system instruction (SI) is prepended to
        the prompt, with a `system` role.
      prompt_template: A pre-built LangChain ChatPromptTemplate to format
        messages before passing them to the LLM
    """
    super().__init__()
    self._model = model
    self._model_name = getattr(self._model, 'model', type(self._model).__name__)

    self._system_instruction = content_api.ProcessorContent(system_instruction)
    # Force parts in the system_instruction to have system role.
    for part in self._system_instruction:
      part.role = 'system'

    self._prompt_template = prompt_template

  async def call(
      self, content_stream: AsyncIterable[content_api.ProcessorPart]
  ) -> AsyncIterable[content_api.ProcessorPart]:
    parts: list[content_api.ProcessorPart] = []
    async for part in content_stream:
      parts.append(part)
    content = self._system_instruction + parts

    msgs = self._convert_to_langchain_messages(content.all_parts)

    payload = (
        {'input': self._prompt_template.format(messages=msgs)}
        if self._prompt_template
        else msgs
    )

    async for chunk in self._model.astream(payload):
      if not isinstance(chunk.content, str):
        raise NotImplementedError(
            'Multimodal content output is not implemented yet.'
        )

      yield content_api.ProcessorPart(
          chunk.content,
          mimetype='text/plain',
          role='model',
          metadata={'model': self._model_name},
      )

  def _convert_to_langchain_messages(
      self, parts: Iterable[content_api.ProcessorPart]
  ) -> list[
      Union[
          langchain_messages.HumanMessage,
          langchain_messages.SystemMessage,
          langchain_messages.AIMessage,
      ]
  ]:
    messages: list[
        Union[
            langchain_messages.HumanMessage,
            langchain_messages.SystemMessage,
            langchain_messages.AIMessage,
        ]
    ] = []
    content_parts: list[dict[str, Any]] = []
    last_role: Optional[str] = None
    last_part: Optional[content_api.ProcessorPart] = None

    def flush():
      nonlocal content_parts
      if content_parts:
        cls = {
            'system': langchain_messages.SystemMessage,
            'model': langchain_messages.AIMessage,
        }.get(last_role, langchain_messages.HumanMessage)
        if len(content_parts) == 1 and content_parts[0].get('type') == 'text':
          content = content_parts[0]['text']
        else:
          content = content_parts

        messages.append(
            cls(
                content=content,
                additional_kwargs={
                    'metadata': last_part.metadata if last_part else {}
                },
            )
        )
        content_parts = []

    for part in parts:
      if content_api.is_text(part.mimetype):
        part_content = {'type': 'text', 'text': part.text}
      elif content_api.is_image(part.mimetype) and part.bytes:
        b64 = base64.b64encode(part.bytes).decode('utf-8')
        part_content = {
            'type': 'image_url',
            'image_url': {'url': f'data:{part.mimetype};base64,{b64}'},
        }
      else:
        raise ValueError(f'Unsupported mimetype: {part.mimetype}')

      if part.role != last_role and last_role is not None:
        flush()
      content_parts.append(part_content)
      last_role = part.role
      last_part = part

    flush()
    return messages
