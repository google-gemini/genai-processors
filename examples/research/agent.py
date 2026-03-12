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

"""Research agent example, using Processors library.

This agent takes user content as input, generates a set of topics based on
that content, researches each topic, and then synthesizes the research into a
final output.

The agent is composed of three main processors:
  - TopicGenerator: Generates `Topics` based on the user content.
  - TopicResearcher: Researches each `Topic`.
  - TopicVerbalizer: Converts a `Topic` ProcessorPart to a human-readable
  string.
  - The output of this flow is then passed to a GenAI model, which synthesizes
  the research into a final output.

The agent orchestrates these processors, passing the output of one to the
input of the next.
"""

from typing import AsyncIterable

from genai_processors import processor
from genai_processors.core import genai_model
from genai_processors.core import jinja_template
from genai_processors.core import preamble
from google.genai import types

from . import interfaces
from . import prompts
from .processors import topic_generator
from .processors import topic_researcher

ProcessorPart = processor.ProcessorPart
ProcessorPartTypes = processor.ProcessorPartTypes


class ResearchAgent(processor.Processor):
  """Research agent example, using Processors library.

  This agent takes user content as input, generates a set of topics based on
  that content, researches each topic, and then synthesizes the research into a
  final output.

  The agent is composed of three main processors:
    - TopicGenerator: Generates `Topics` based on the user content.
    - TopicResearcher: Researches each `Topic`.
    - TopicVerbalizer: Converts a `Topic` ProcessorPart to a human-readable
    string.
    - The output of this flow is then passed to a GenAI model, which synthesizes
    the research into a final output.

  The agent orchestrates these processors, passing the output of one to the
  input of the next.
  """

  def __init__(
      self, api_key: str, config: interfaces.Config = interfaces.Config()
  ) -> None:
    """Initializes the Research Agent.

    Args:
      api_key: The API key to use for the GenAI API.
      config: The configuration for the Research Agent.
    """
    self._config = config

    p_topic_generator = topic_generator.TopicGenerator(
        api_key=api_key,
        config=config,
    )
    p_topic_researcher = topic_researcher.TopicResearcher(
        api_key=api_key,
        config=config,
    )
    p_topic_verbalizer = jinja_template.RenderDataClass(
        template_str=(
            '## {{ data.topic }}\n'
            '*{{ data.relationship_to_user_content }}*'
            '{% if data.research_text|trim != "" %}'
            '\n\n### Research\n\n{{ data.research_text }}'
            '{% endif %}'
        ),
        data_class=interfaces.Topic,
    )
    p_genai_model = genai_model.GenaiModel(
        api_key=api_key,
        model_name=self._config.research_synthesizer_model_name,
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(
                attempts=100,
            ),
        ),
    )
    p_preamble = preamble.Preamble(
        content=[
            prompts.SYNTHESIS_PREAMBLE,
            'Research text: ',
        ]
    )
    p_suffix = preamble.Suffix(
        content='Your synthesized research: '
    )
    self._pipeline = (
        p_topic_generator
        + p_topic_researcher
        + p_topic_verbalizer
        + p_preamble
        + p_suffix
        + p_genai_model
    )

  async def call(
      self, content: processor.ProcessorStream
  ) -> AsyncIterable[ProcessorPartTypes]:
    async for content_part in self._pipeline(content):
      yield content_part
    yield processor.status('Produced research synthesis!')
