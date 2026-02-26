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
"""Processor which researches a `Topic`, using tools."""

from typing import AsyncIterable

from genai_processors import content_api
from genai_processors import processor
from genai_processors.core import genai_model
from genai_processors.core import jinja_template
from genai_processors.core import preamble
from google.genai import types

from .. import interfaces
from .. import prompts

ProcessorPart = processor.ProcessorPart


class TopicResearcher(processor.PartProcessor):
  """Processor which researches a given `Topic`.

  This processor uses the GenAI API's tool feature to research a given
  topic, and yields `TopicResearch` ProcessorParts with the research results.
  """

  def __init__(
      self,
      api_key: str,
      config: interfaces.Config | None = None,
  ):
    """Initializes the TopicResearcher.

    Args:
      api_key: The API key to use for the GenAI API.
      config: The agent configuration.
    """
    self._config = config or interfaces.Config()
    self._genai_processor = genai_model.GenaiModel(
        api_key=api_key,
        model_name=self._config.topic_researcher_model_name,
        generate_content_config=types.GenerateContentConfig(
            tools=self._config.enabled_research_tools,
        ),
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(
                attempts=100,
            ),
        ),
    )
    p_preamble = preamble.Preamble(
        content=[
            ProcessorPart(prompts.TOPIC_RESEARCH_PREAMBLE),
            ProcessorPart('Topic to research: '),
        ]
    )
    p_verbalizer = jinja_template.RenderDataClass(
        template_str=(
            '## {{ data.topic }}\n'
            '*{{ data.relationship_to_user_content }}*'
            '{% if data.research_text|trim != "" %}'
            '\n\n### Research\n\n{{ data.research_text }}'
            '{% endif %}'
        ),
        data_class=interfaces.Topic,
    )
    p_suffix = preamble.Suffix(content=[ProcessorPart('Your research: ')])
    self._pipeline = (
        p_verbalizer + p_preamble + p_suffix + self._genai_processor
    )

  def match(self, part: ProcessorPart) -> bool:
    return content_api.is_dataclass(part.mimetype, interfaces.Topic)

  async def call(
      self,
      part: ProcessorPart,
  ) -> AsyncIterable[ProcessorPart]:

    input_topic = part.get_dataclass(interfaces.Topic)
    response = self._pipeline(part)

    updated_topic = interfaces.Topic(
        topic=input_topic.topic,
        relationship_to_user_content=input_topic.relationship_to_user_content,
        research_text=await response.text(),
    )
    yield ProcessorPart.from_dataclass(updated_topic)
    yield processor.status(f"""Researched topic!

## {updated_topic.topic}

### Research

{updated_topic.research_text}""")
