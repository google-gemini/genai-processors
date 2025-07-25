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
 * Flexible Jinja-based prompt templating
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
      llm: chat_models.BaseChatModel,
      system_instruction: Optional[str] = None,
      prompt_template: Optional[str] = None,
  ):
    super().__init__()
    self.llm = llm
    self.system_instruction = system_instruction or ''
    self.prompt_template = (
        ChatPromptTemplate.from_template(prompt_template)
        if prompt_template
        else None
    )

  async def call(
      self, content_stream: AsyncIterable[content_api.ProcessorPart]
  ) -> AsyncIterable[content_api.ProcessorPart]:
    parts: list[content_api.ProcessorPart] = []
    async for part in content_stream:
      parts.append(part)
    content = content_api.ProcessorContent(parts)

    msgs = self._convert_to_langchain_messages(content.all_parts)
    if self.system_instruction:
      msgs.insert(
          0, langchain_messages.SystemMessage(content=self.system_instruction)
      )

    payload = (
        {'input': self.prompt_template.format(messages=msgs)}
        if self.prompt_template
        else msgs
    )

    async for chunk in self.llm.astream(payload):
      model_name = getattr(self.llm, 'model', type(self.llm).__name__)

      yield content_api.ProcessorPart(
          chunk.content,
          mimetype='text/plain',
          role='model',
          metadata={'model': model_name},
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
      nonlocal content_parts, last_role, last_part
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
