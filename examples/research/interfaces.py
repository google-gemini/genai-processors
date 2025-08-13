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

"""Research Agent interfaces."""

import dataclasses
import dataclasses_json
from genai_processors import content_api
from google.genai import types as genai_types

ProcessorPart = content_api.ProcessorPart


@dataclasses_json.dataclass_json
@dataclasses.dataclass(frozen=True)
class Topic:
  """A topic to be researched.

  Attributes:
    topic: The topic to be researched.
    relationship_to_user_content: A description of how the topic is related to
      the user content.
    research_text: The research text for the topic. This field is optional and
      is expected to be populated by the TopicResearcher processor.
  """

  topic: str
  relationship_to_user_content: str
  research_text: str | None = None


@dataclasses.dataclass
class Config:
  """Config used by the Research agent.

  Attributes:
    topic_generator_model_name: The model name to use for the topic generator.
    topic_researcher_model_name: The model name to use for the topic researcher.
    research_synthesizer_model_name: The model name to use for synthesizing the
      research results.
    num_topics: The number of topics to generate.
    excluded_topics: A list of topics to exclude from the research. E.g., if the
      user prompt is "Starting a vegetable garden in my small backyard in
      London", then the excluded topics could be "what vegetables can grow in
      the London climate?", "vegetables that can grow with limited space", etc.
    enabled_research_tools: A list of research tools to enable. By default, only
      the Google Search tool is enabled.
  """

  topic_generator_model_name: str = 'gemini-2.5-flash'
  topic_researcher_model_name: str = 'gemini-2.5-flash'
  research_synthesizer_model_name: str = 'gemini-2.5-flash'
  num_topics: int = 5
  excluded_topics: list[str] | None = None
  # Use default factory instead
  enabled_research_tools: list[genai_types.ToolConfigOrDict] = (
      dataclasses.field(
          default_factory=lambda: [
              genai_types.Tool(google_search=genai_types.GoogleSearch())
          ]
      )
  )
