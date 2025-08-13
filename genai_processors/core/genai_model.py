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

"""Wraps the Gemini API into a Processor.

## Example usage

```py
p = GenaiModel(
    api_key=API_KEY,
    model_name="gemini-2.0-flash-exp-image-generation",
    generate_content_config=genai.types.GenerateContentConfig(
      response_modalities=['Text', 'Image']
    )
)
```

### Sync Execution

```py
INPUT_PROMPT = 'Create an image of two dalmatians & a cute poem'

genai_content = processors.apply_sync(p, [ProcessorPart(INPUT_PROMPT)])
for part in genai_content:
  if part.text:
    print(part.text)
  elif part.pil_image:
    display(part.pil_image)
```

### Async Execution

```py
input_stream = processor.stream_content([ProcessorPart(INPUT_PROMPT)])
async for part in p(input_stream):
  if part.text:
    print(part.text)
  elif part.pil_image:
    display(part.pil_image)
```
"""

from collections.abc import AsyncIterable
from typing import Any
from genai_processors import content_api
from genai_processors import processor
from genai_processors.core import constrained_decoding
from google.genai import client
from google.genai import types as genai_types


def genai_response_to_metadata(
    response: genai_types.GenerateContentResponse,
) -> dict[str, Any]:
  """Converts a Genai response to metadata, to be attached to a ProcessorPart."""
  return {
      'create_time': response.create_time,
      'response_id': response.response_id,
      'model_version': response.model_version,
      'prompt_feedback': response.prompt_feedback,
      'usage_metadata': response.usage_metadata,
      'automatic_function_calling_history': (
          response.automatic_function_calling_history
      ),
      'parsed': response.parsed,
  }


class GenaiModel(processor.Processor):
  """`Processor` that calls the Genai API in turn-based fashion.

  Note: All content is buffered prior to calling the Genai API.
  """

  def __init__(
      self,
      api_key: str,
      model_name: str,
      generate_content_config: (
          genai_types.GenerateContentConfigOrDict | None
      ) = None,
      debug_config: client.DebugConfig | None = None,
      http_options: (
          genai_types.HttpOptions | genai_types.HttpOptionsDict | None
      ) = None,
      stream_json: bool = False,
  ):
    """Initializes the GenaiModel.

    Args:
      api_key: The `API key <https://ai.google.dev/gemini-api/docs/api-key>`_ to
        use for authentication. Applies to the Gemini Developer API only.
      model_name: The name of the model to use.
      generate_content_config: The configuration for generating content.
      debug_config: Config settings that control network behavior of the client.
        This is typically used when running test code.
      http_options: Http options to use for the client. These options will be
        applied to all requests made by the client. Example usage: `client =
        genai.Client(http_options=types.HttpOptions(api_version='v1'))`.
      stream_json: By default, if a `response_schema` is present in the
        `generate_content_config`, this processor will buffer the model's JSON
        output and parse it into dataclass/enum instances. Set this to True to
        disable this behavior and stream the raw JSON text instead.

    Returns:
      A `Processor` that calls the Genai API in turn-based fashion.

    ## Model Name Usage

    Supported formats for **Vertex AI API** include:

      * The Gemini model ID, for example: 'gemini-2.0-flash'
      * The full resource name starts with 'projects/', for example:
      'projects/my-project-id/locations/us-central1/publishers/google/models/gemini-2.0-flash'
      * The partial resource name with 'publishers/', for example:
      'publishers/google/models/gemini-2.0-flash' or
      'publishers/meta/models/llama-3.1-405b-instruct-maas' / separated
      publisher and model name, for example: 'google/gemini-2.0-flash' or
      'meta/llama-3.1-405b-instruct-maas'

    Supported formats for **Gemini API** include:

      * The Gemini model ID, for example: 'gemini-2.0-flash'
      * The model name starts with 'models/', for example:
      'models/gemini-2.0-flash'
      * For tuned models, the model name starts with 'tunedModels/', for
      example: 'tunedModels/1234567890123456789'
    """
    self._client = client.Client(
        api_key=api_key,
        debug_config=debug_config,
        http_options=http_options,
    )
    self._model_name = model_name
    self._generate_content_config = generate_content_config
    self._parser = None

    schema = None
    if generate_content_config:
      if isinstance(generate_content_config, genai_types.GenerateContentConfig):
        if hasattr(generate_content_config, 'response_schema'):
          schema = generate_content_config.response_schema
      elif isinstance(generate_content_config, dict):
        schema = generate_content_config.get('response_schema')

    # If schema is present and the user has not opted for raw JSON streaming,
    # set up the JSON decoding processor.
    if schema and not stream_json:
      self._parser = constrained_decoding.StructuredOutputParser(schema)

  async def _generate_from_api(
      self, content: AsyncIterable[content_api.ProcessorPartTypes]
  ) -> AsyncIterable[content_api.ProcessorPartTypes]:
    """Internal method to call the GenAI API and stream results."""
    turn = genai_types.Content(parts=[])
    contents = []
    async for content_part in content:
      content_part = content_api.ProcessorPart(content_part)
      if turn.role and content_part.role != turn.role:
        contents.append(turn)
        turn = genai_types.Content(parts=[])

      turn.role = content_part.role or 'user'
      turn.parts.append(content_api.to_genai_part(content_part))  # pylint: disable=attribute-error

    if turn.role:
      contents.append(turn)

    if not contents:
      return

    async for res in await self._client.aio.models.generate_content_stream(
        model=self._model_name,
        contents=contents,
        config=self._generate_content_config,
    ):
      res: genai_types.GenerateContentResponse = res
      if res.candidates:
        content = res.candidates[0].content
        if content and content.parts:
          for part in content.parts:
            yield processor.ProcessorPart(
                part,
                metadata=genai_response_to_metadata(res),
                role=content.role or 'model',
            )

  async def call(
      self, content: AsyncIterable[content_api.ProcessorPartTypes]
  ) -> AsyncIterable[content_api.ProcessorPartTypes]:
    api_stream = self._generate_from_api(content)

    if self._parser:
      async for part in self._parser(api_stream):
        yield part
    else:
      async for part in api_stream:
        yield part
