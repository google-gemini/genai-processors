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
import torch
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

    self.decode_map = {
        1: 'Test ',
        2: 'prompt',
        3: 'Hello, ',
        4: 'world!',
        5: '<start_function_call>',
        6: '<end_function_call>',
        7: '<escape>',
    }

    def mock_decode(tokens, skip_special_tokens):
      del skip_special_tokens  # Unused.
      output = []
      for token in tokens:
        output.append(self.decode_map[token])
      return ''.join(output)

    def mock_encode(text, add_special_tokens):
      del add_special_tokens  # Unused.
      del text  # Unused.
      # 5=start_function_call, 6=end_function_call
      return [5, 6, 7]

    self.mock_processor.decode.side_effect = mock_decode
    self.mock_processor.encode.side_effect = mock_encode

    self.mock_model = mock.Mock()
    self.mock_model.device = 'cpu'
    self.mock_model.config.max_position_embeddings = 1024

    self.output_tokens = []

    def mock_generate_fn(*args, **kwargs):
      del args  # Unused.
      streamer = kwargs['streamer']
      for token in self.output_tokens:
        streamer.put(mock.Mock(**{'flatten.return_value': torch.tensor(token)}))
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
    self.output_tokens = [[1, 2], [3, 4]]
    model = transformers_model.TransformersModel(model_name='unused')

    output = processor.apply_sync(model, ['Test prompt'])

    self.assertEqual(content_api.as_text(output), 'Hello, world!')
    self.mock_processor.apply_chat_template.assert_called_once()
    self.mock_model.generate.assert_called_once()

  def test_system_instruction(self):
    self.output_tokens = [[1, 2], [3, 4]]
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

    self.output_tokens = [[1, 2], [3, 4]]

    # pylint: disable=g-doc-return-or-yield
    def emulate_vigorous_thinking(thinking_budget: float) -> int:
      """The "smart" function.

      Args:
        thinking_budget: time covered by our budget.
      """
      return '🧠' * int(thinking_budget)

    # pylint: enable=g-doc-return-or-yield

    model = transformers_model.TransformersModel(
        model_name='unused',
        generate_content_config={'tools': [emulate_vigorous_thinking]},
    )
    model._parse_function_calls = True
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
                    'properties': {
                        'thinking_budget': {
                            'type': 'number',
                            'description': 'time covered by our budget.',
                        }
                    },
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
                name='emulate_vigorous_thinking',
                response=42,
            )
        ],
    )
    self.mock_processor.apply_chat_template.assert_called_once_with(
        [{
            'role': 'tool',
            'content': '42',
            'name': 'emulate_vigorous_thinking',
        }],
        tools=[],
        add_generation_prompt=True,
        return_dict=True,
        return_tensors='pt',
    )

  def test_function_call_decoding(self):
    self.output_tokens = [
        [1, 2],
        [3, 4],
        [5],  # start of function call
        [8, 7, 9, 7, 10],
        [6],  # end of function call
    ]
    self.decode_map.update({
        8: 'call:emulate_vigorous_thinking{thinking_budget:1.5,reason:',
        9: 'cheap',
        10: '}',
    })
    model = transformers_model.TransformersModel(model_name='unused')
    model._parse_function_calls = True
    output = processor.apply_sync(model, ['Test prompt'])
    expected_output = [
        content_api.ProcessorPart('Hello, world!'),
        content_api.ProcessorPart.from_function_call(
            name='emulate_vigorous_thinking',
            args={'reason': 'cheap', 'thinking_budget': 1.5},
            role='model',
        ),
    ]
    self.assertEqual(output, expected_output)
    self.mock_processor.apply_chat_template.assert_called_once()
    self.mock_model.generate.assert_called_once()

  @parameterized.named_parameters(
      ('split', [[1, 2], [3, 4], [5], [8, 6], [5, 8, 6]]),
      ('fine_split', [[1, 2], [3, 4], [5], [8], [6], [5], [8], [6]]),
      ('merged', [[1, 2], [3, 4], [5, 8, 6, 5, 8, 6]]),
      ('merged_with_text', [[1, 2], [3, 4, 5, 8], [6], [5, 8, 6]]),
      ('monolith', [[1, 2, 3, 4, 5, 8, 6, 5, 8, 6]]),
  )
  def test_double_function_call_decoding(self, output_tokens):
    self.output_tokens = output_tokens
    self.decode_map.update({
        8: 'call:emulate_vigorous_thinking{thinking_budget:1.5}',
    })
    model = transformers_model.TransformersModel(model_name='unused')
    model._parse_function_calls = True
    output = processor.apply_sync(model, ['Test prompt'])
    expected_output = [
        content_api.ProcessorPart('Hello, world!'),
        content_api.ProcessorPart.from_function_call(
            name='emulate_vigorous_thinking',
            args={'thinking_budget': 1.5},
            role='model',
        ),
        content_api.ProcessorPart.from_function_call(
            name='emulate_vigorous_thinking',
            args={'thinking_budget': 1.5},
            role='model',
        ),
    ]
    self.assertEqual(output, expected_output)
    self.mock_processor.apply_chat_template.assert_called_once()
    self.mock_model.generate.assert_called_once()

  def test_escape_function_arg_parsing(self):
    # 7 = escape token id
    self.output_tokens = [
        [1, 2],
        [3, 4],
        [5],  # start of function call
        [8, 7, 9],
        [7, 15, 10, 7, 11, 7, 15, 12, 7, 13, 7, 14],
        [6],  #  end of function call
    ]
    self.decode_map.update({
        8: 'call:emulate_vigorous_thinking{thinking_budget:1.5,reason:',
        9: 'cheap<escape>',
        10: 'complex_arg: { arg1:',
        11: 'val1',
        12: ' arg2: {arg21:',
        13: 'this is a "fun" function with , arg: "13"',
        14: '}}}',
        15: ',',
    })
    model = transformers_model.TransformersModel(model_name='unused')
    model._parse_function_calls = True
    output = processor.apply_sync(model, ['Test prompt'])
    expected_output = [
        content_api.ProcessorPart('Hello, world!'),
        content_api.ProcessorPart.from_function_call(
            name='emulate_vigorous_thinking',
            args={
                'reason': 'cheap<escape>',
                'thinking_budget': 1.5,
                'complex_arg': {
                    'arg1': 'val1',
                    'arg2': {
                        'arg21': 'this is a "fun" function with , arg: "13"'
                    },
                },
            },
            role='model',
        ),
    ]
    self.assertEqual(output, expected_output)
    self.mock_processor.apply_chat_template.assert_called_once()
    self.mock_model.generate.assert_called_once()

  def test_escape_another_function_arg_parsing(self):
    # 7 = escape token id
    self.output_tokens = [
        [1, 2],
        [3, 4],
        [5],  # start of function call
        [8, 7, 9],
        [7, 10, 7, 11, 7, 12],
        [6],  # end of function call
    ]

    self.decode_map.update({
        8: 'call:read_a_file{file_data:{display_name:',
        9: 'foo.txt',
        10: ',file_uri:',
        11: '/path/to/foo.txt',
        12: '}}',
    })
    model = transformers_model.TransformersModel(model_name='unused')
    model._parse_function_calls = True
    output = processor.apply_sync(model, ['Test prompt'])
    expected_output = [
        content_api.ProcessorPart('Hello, world!'),
        content_api.ProcessorPart.from_function_call(
            name='read_a_file',
            args={
                'file_data': {
                    'display_name': 'foo.txt',
                    'file_uri': '/path/to/foo.txt',
                }
            },
            role='model',
        ),
    ]
    self.assertEqual(output, expected_output)
    self.mock_processor.apply_chat_template.assert_called_once()
    self.mock_model.generate.assert_called_once()


if __name__ == '__main__':
  absltest.main()
