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

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from genai_processors import content_api
from genai_processors import processor
from genai_processors.core import transformers_model
import transformers


class MockInputs(dict):
  """Mock inputs returned by apply_chat_template."""

  def __init__(self):
    super().__init__(input_ids=[[1, 2]])

  def to(self, device):
    del device  # Unused.
    return self


class TransformersModelTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.mock_processor = mock.Mock()
    self.mock_processor.eos_token_id = 0
    self.mock_processor.apply_chat_template.return_value = MockInputs()

    def mock_decode(tokens, skip_special_tokens):
      del skip_special_tokens  # Unused.
      if tokens == [1, 2]:
        return 'Test prompt'
      elif tokens == [3, 4]:
        return 'Hello, world!'
      else:
        return None

    self.mock_processor.decode.side_effect = mock_decode

    self.mock_model = mock.Mock()
    self.mock_model.device = 'cpu'
    self.mock_model.config.max_position_embeddings = 1024
    def mock_generate_fn(*args, **kwargs):
      del args  # Unused.
      streamer = kwargs['streamer']
      streamer.put(mock.Mock(**{'flatten.return_value': [1, 2]}))
      streamer.put(mock.Mock(**{'flatten.return_value': [3, 4]}))
      streamer.end()

    self.mock_model.generate.side_effect = mock_generate_fn

    self.enter_context(
        mock.patch.object(
            transformers.AutoProcessor,
            'from_pretrained',
            return_value=self.mock_processor,
        )
    )
    self.enter_context(
        mock.patch.object(
            transformers.AutoModelForCausalLM,
            'from_pretrained',
            return_value=self.mock_model,
        )
    )

  def test_simple_inference(self):
    model = transformers_model.TransformersModel(model_name='unused')

    output = processor.apply_sync(model, ['Test prompt'])

    self.assertEqual(content_api.as_text(output), 'Hello, world!')
    self.mock_processor.apply_chat_template.assert_called_once()
    self.mock_model.generate.assert_called_once()

  def test_system_instruction(self):
    model = transformers_model.TransformersModel(
        model_name='unused',
        generate_content_config={'system_instruction': 'Be nice.'},
    )
    processor.apply_sync(model, ['Test prompt'])
    self.mock_processor.apply_chat_template.assert_called_once_with(
        [
            {
                'role': 'system',
                'content': [{'type': 'text', 'text': 'Be nice.'}],
            },
            {
                'role': 'user',
                'content': [{'type': 'text', 'text': 'Test prompt'}],
            },
        ],
        tools=[],
        add_generation_prompt=True,
        return_dict=True,
        return_tensors='pt',
    )

  def test_function_call_from_previous_turn(self):
    """Tests that function calls from previous turns are formatted correctly.

    In this case FunctionCall is passed as a part of the conversation history,
    rather than returned by the model. We check that when fed back to the model
    it is represented properly.
    """

    def emulate_vigorous_thinking(thinking_budget: float) -> int:
      """The "smart" function."""
      return '🧠' * int(thinking_budget)

    model = transformers_model.TransformersModel(
        model_name='unused',
        generate_content_config={'tools': [emulate_vigorous_thinking]},
    )
    processor.apply_sync(
        model,
        [
            content_api.ProcessorPart.from_function_call(
                name='emulate_vigorous_thinking', args={'thinking_budget': 1.5}
            )
        ],
    )
    self.mock_processor.apply_chat_template.assert_called_once_with(
        [{
            'role': 'assistant',
            'tool_calls': [{
                'type': 'function',
                'function': {
                    'name': 'emulate_vigorous_thinking',
                    'arguments': '{"thinking_budget": 1.5}',
                },
            }],
        }],
        tools=[{
            'type': 'function',
            'function': {
                'name': 'emulate_vigorous_thinking',
                'description': 'The "smart" function.',
                'parameters': {
                    'type': 'object',
                    'properties': {'thinking_budget': {'type': 'number'}},
                    'required': ['thinking_budget'],
                },
            },
        }],
        add_generation_prompt=True,
        return_dict=True,
        return_tensors='pt',
    )

  def test_function_response(self):
    """Tests that function responses are correctly formatted."""
    model = transformers_model.TransformersModel(model_name='unused')
    processor.apply_sync(
        model,
        [
            content_api.ProcessorPart.from_function_response(
                name='emulate_vigorous_thinking', response=42
            )
        ],
    )
    self.mock_processor.apply_chat_template.assert_called_once_with(
        [{
            'role': 'tool',
            'content': '{"result": 42}',
            'name': 'emulate_vigorous_thinking',
        }],
        tools=[],
        add_generation_prompt=True,
        return_dict=True,
        return_tensors='pt',
    )


if __name__ == '__main__':
  absltest.main()
