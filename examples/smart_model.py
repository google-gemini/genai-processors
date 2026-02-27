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

"""A collection of agents that invoke underlying model in a way to improve quality."""

from typing import AsyncIterable

from genai_processors import content_api
from genai_processors import processor


class CriticReviser(processor.Processor):
  """Agent that uses a critic-reviser loop to improve responses."""

  def __init__(
      self,
      model: processor.Processor,
      max_iterations: int = 5,
  ):
    """Initializes the SmartModel.

    Args:
      model: The base generative model to use.
      max_iterations: Maximum number of critic-reviser loop iterations.
    """
    self._model = model
    self._max_iterations = max_iterations

  async def call(
      self, content: content_api.ContentStream
  ) -> AsyncIterable[content_api.ProcessorPartTypes]:
    # We gather content from the stream as we will need to reuse it multiple times.
    input_content = await content.gather()

    current_response = await self._model(input_content).gather()

    for _ in range(self._max_iterations):
      critic_response = await self._model([
          input_content,
          '\n\nDraft response:\n\n',
          current_response,
          (
              '\n\nYou are a harsh critic. Review the draft response to the'
              " user's prompt. If the draft fully answers the prompt and has no"
              " obvious flaws, simply output 'OK'. Otherwise, concisely list"
              ' the flaws or missing information. Do not rewrite the response.'
          ),
      ]).gather()

      critic_text = await critic_response.text(strict=False)
      if critic_text.strip().upper() == 'OK':
        break

      current_response = await self._model([
          input_content,
          '\n\nDraft response:\n\n',
          current_response,
          '\n\nCriticism:\n',
          critic_response,
          (
              '\n\nUpdate your previous draft response to address the'
              ' criticism. Keep the parts that are already good.'
          ),
      ]).gather()

    for part in current_response:
      yield part
