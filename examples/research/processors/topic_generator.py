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
"""Processor which generate topics for a given research query."""

import json
from typing import AsyncIterable

from genai_processors import content_api
from genai_processors import processor
from genai_processors.core import genai_model
from genai_processors.core import preamble
from google.genai import types

from .. import interfaces
from .. import prompts

ProcessorPart = processor.ProcessorPart


class TopicGenerator(processor.Processor):
  """Processor which generates research topics based on user content.

  This processor demonstrates how custom structured parts can be added to the
  Part stream.

  In this case each topic to research is represented as a self-sufficient JSON
  chunk.

  This allows further processors to easily transform them!
  """

  def __init__(
      self,
      api_key: str,
      config: interfaces.Config | None = None,
  ):
    """Initializes the TopicGenerator.

    Args:
      api_key: The API key to use for the GenAI API.
      config: The agent configuration.
    """
    self._config = config or interfaces.Config()
    self._p_genai_model = genai_model.GenaiModel(
        api_key=api_key,
        model_name=self._config.topic_generator_model_name,
        generate_content_config={
            'response_mime_type': 'application/json',
            'response_schema': list[interfaces.Topic],
        },
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(
                attempts=100,
            ),
        ),
    )
    self._num_topics = self._config.num_topics

    preamble_content = [
        ProcessorPart(prompts.TOPIC_GENERATION_PREAMBLE),
        ProcessorPart(
            f'Please provide exactly {self._config.num_topics} research'
            ' topics, along with each topic\'s "relationship" to the user'
            ' prompt.'
        ),
    ]
    if self._config.excluded_topics:
      preamble_content.append(
          ProcessorPart(
              'Here is a list of topics that should be excluded:'
              f' {self._config.excluded_topics}'
          )
      )
    preamble_content.append(
        ProcessorPart('You will now be provided with the user content.')
    )
    p_preamble = preamble.Preamble(content=preamble_content)
    p_suffix = preamble.Suffix(
        content=[
            ProcessorPart(
                f"""Return your response as Topics JSON in the format below.

You MUST return exactly {self._config.num_topics} topics.

Topic
  topic: str
  relationship_to_user_content: list[str]

Topics
  list[Topic]

Your JSON:
"""
            )
        ]
    )

    self._pipeline = p_preamble + p_suffix + self._p_genai_model

  async def call(
      self, content: AsyncIterable[ProcessorPart]
  ) -> AsyncIterable[ProcessorPart]:
    topics = []
    async for content_part in self._pipeline(content):
      topics.append(content_part.get_dataclass(interfaces.Topic))

    yield processor.status(f'Generated {len(topics)} topics to research!')
    for i, t in enumerate(topics):
      yield processor.status(
          f'Topic {i + 1}: "{t.topic}"\n\n*({t.relationship_to_user_content})*'
      )
      yield ProcessorPart.from_dataclass(dataclass=t)
