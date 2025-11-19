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

"""Model selector for examples.

NOTE: THIS MODULE IS ONLY INTENDED TO BE USED BY CODE IN THE EXAMPLES DIRECTORY.

This is a helper for examples to allow selecting a model backend from command
line flags. It is not intended to be used as a generic model selector. It is
deliberately designed to be inflexible and only work with command line flags
to demonstrate that examples can run with various backends.
"""

import enum
import os
from absl import flags
from genai_processors import content_api
from genai_processors import processor
from genai_processors.contrib.langchain_model import LangChainModel
from genai_processors.core import genai_model
from genai_processors.core import ollama_model
from genai_processors.core import transformers_model
from google.genai import types as genai_types
import langchain_google_genai

FLAGS = flags.FLAGS


class _ModelType(enum.Enum):
  GEMINI = 'gemini'
  OLLAMA = 'ollama'
  LANGCHAIN = 'langchain'
  TRANSFORMERS = 'transformers'


_MODEL_TYPE = flags.DEFINE_enum(
    'model_type',
    _ModelType.GEMINI.value,
    [e.value for e in _ModelType],
    'Name of the model type to use.',
)
_MODEL_NAME = flags.DEFINE_string(
    'model_name',
    None,
    'Name of the generative model to use.',
)

# You need to define the Google API key in the environment variables to use
# Gemini models.
API_KEY = os.environ['GOOGLE_API_KEY']


def turn_based_model(
    system_instruction: content_api.ProcessorContentTypes,
) -> processor.Processor:
  """Returns a turn-based model based on command line flags.

  This is a helper to allow examples to run with as variety of model backends.
  It is not intended to be used as a generic model selector.

  Args:
    system_instruction: The system instruction to use for the model.

  Returns:
    A turn-based LLM model.
  """
  model_name = _MODEL_NAME.value
  if _MODEL_TYPE.value == _ModelType.GEMINI.value:
    if not API_KEY:
      raise ValueError(
          'Google API key is not set. Define a GOOGLE_API_KEY environment '
          'variable with a key obtained from AI Studio.'
      )
    if not model_name:
      model_name = 'gemini-2.0-flash-lite'

    return genai_model.GenaiModel(
        api_key=API_KEY,
        model_name=model_name,
        generate_content_config=genai_types.GenerateContentConfig(
            system_instruction=[
                part.text
                for part in content_api.ProcessorContent(system_instruction)
            ],
            response_modalities=['TEXT'],
            # Adds google search as a tool. This is not needed for the model to
            # work but it is useful to ask questions that can be answered by
            # google search.
            tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())],
        ),
        # Make the newest features available for the examples.
        http_options=genai_types.HttpOptions(api_version='v1alpha'),
    )

  if _MODEL_TYPE.value == _ModelType.OLLAMA.value:
    if not model_name:
      model_name = 'gemma3'
    return ollama_model.OllamaModel(
        model_name=model_name,
        generate_content_config=ollama_model.GenerateContentConfig(
            system_instruction=system_instruction
        ),
    )

  if _MODEL_TYPE.value == _ModelType.LANGCHAIN.value:
    if not model_name:
      model_name = 'gemini-2.0-flash-lite'
    llm = langchain_google_genai.ChatGoogleGenerativeAI(model=model_name)
    return LangChainModel(model=llm, system_instruction=system_instruction)

  if _MODEL_TYPE.value == _ModelType.TRANSFORMERS.value:
    if not model_name:
      model_name = 'google/gemma-2b'
    return transformers_model.TransformersModel(
        model_name=model_name,
        generate_content_config=transformers_model.GenerateContentConfig(
            system_instruction=system_instruction
        ),
    )

  raise ValueError(f'{_MODEL_TYPE.value!r} is not supported.')
