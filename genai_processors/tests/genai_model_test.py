import dataclasses
import enum
from typing import Any, Callable, Sequence
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import dataclasses_json
from genai_processors import content_api
from genai_processors import processor
from genai_processors.core import genai_model
from google.genai import types as genai_types


@dataclasses_json.dataclass_json
@dataclasses.dataclass(frozen=True)
class Actor:
  """A test dataclass representing an actor."""

  name: str
  birth_year: int


class MovieGenre(enum.StrEnum):
  """A test enum for movie genres."""

  SCI_FI = 'SCI_FI'
  ACTION = 'ACTION'
  COMEDY = 'COMEDY'


async def async_genai_response_stream(text_chunks: list[str]):
  for chunk in text_chunks:
    yield genai_types.GenerateContentResponse.model_validate({
        'candidates': [{
            'content': {'parts': [{'text': chunk}], 'role': 'model'},
        }]
    })


class GenaiModelTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'simple_inference',
          dict(),  # No specific generate_content_config
          ['Hello, world!'],
          lambda self, output: self.assertEqual(
              content_api.as_text(output), 'Hello, world!'
          ),
      ),
      (
          'streaming_inference',
          dict(),
          ['Hel', 'lo,', ' world', '!'],
          lambda self, output: self.assertEqual(
              content_api.as_text(output), 'Hello, world!'
          ),
      ),
      (
          'constrained_to_dataclass',
          dict(
              generate_content_config=genai_types.GenerateContentConfig(
                  response_mime_type='application/json',
                  response_schema=Actor,
              )
          ),
          ['{"name": "Keanu Reeves",', ' "birth_year": 1964}'],
          lambda self, output: self.assertEqual(
              output[0].get_dataclass(Actor),
              Actor(name='Keanu Reeves', birth_year=1964),
          ),
      ),
      (
          'constrained_to_list_of_dataclasses',
          dict(
              generate_content_config=genai_types.GenerateContentConfig(
                  response_mime_type='application/json',
                  response_schema=list[Actor],
              )
          ),
          ['[{"name": "A", "birth_year": 0}, {"name": "B", "birth_year": 1}]'],
          lambda self, output: (
              self.assertLen(output, 2),
              self.assertEqual(
                  output[0].get_dataclass(Actor),
                  Actor(name='A', birth_year=0),
              ),
              self.assertEqual(
                  output[1].get_dataclass(Actor),
                  Actor(name='B', birth_year=1),
              ),
          ),
      ),
      (
          'constrained_to_enum',
          dict(
              generate_content_config=genai_types.GenerateContentConfig(
                  response_mime_type='text/plain',
                  response_schema=MovieGenre,
              )
          ),
          ['"SCI_FI"'],
          lambda self, output: self.assertEqual(
              content_api.as_text(output), 'SCI_FI'
          ),
      ),
      (
          'stream_json_bypasses_parsing',
          dict(
              generate_content_config=genai_types.GenerateContentConfig(
                  response_mime_type='application/json',
                  response_schema=Actor,
              ),
              stream_json=True,
          ),
          ['{"name": "test", "birth_year": 1}'],
          lambda self, output: self.assertEqual(
              content_api.as_text(output),
              '{"name": "test", "birth_year": 1}',
          ),
      ),
  )
  @mock.patch('google.genai.client.AsyncModels.generate_content_stream')
  def test_model_processor(
      self,
      model_kwargs: dict[str, Any],
      mock_response_chunks: list[str],
      validate_output: Callable[
          ['GenaiModelTest', Sequence[content_api.ProcessorPart]], None
      ],
      mock_generate_stream,
  ):
    mock_generate_stream.return_value = async_genai_response_stream(
        mock_response_chunks
    )

    model = genai_model.GenaiModel(
        api_key='unused', model_name='gemini-1.5-pro', **model_kwargs
    )

    output = processor.apply_sync(model, ['Test prompt'])
    mock_generate_stream.assert_called_once()
    validate_output(self, output)

  @mock.patch('google.genai.client.AsyncModels.generate_content_stream')
  def test_multi_turn_chat_roles(self, mock_generate_stream):
    mock_generate_stream.return_value = async_genai_response_stream(
        ['response']
    )
    model = genai_model.GenaiModel(
        api_key='unused', model_name='gemini-1.5-pro'
    )

    chat_history = [
        content_api.ProcessorPart('Hello', role='user'),
        content_api.ProcessorPart('Hi there!', role='model'),
        content_api.ProcessorPart('How are you?', role='user'),
    ]
    processor.apply_sync(model, chat_history)

    mock_generate_stream.assert_called_once()
    passed_contents = mock_generate_stream.call_args.kwargs['contents']
    self.assertLen(passed_contents, 3)
    self.assertEqual(passed_contents[0].role, 'user')
    self.assertEqual(passed_contents[1].role, 'model')
    self.assertEqual(passed_contents[2].role, 'user')

  @mock.patch('google.genai.client.AsyncModels.generate_content_stream')
  def test_multi_part_turn(self, mock_generate_stream):
    mock_generate_stream.return_value = async_genai_response_stream(
        ['response']
    )
    model = genai_model.GenaiModel(
        api_key='unused', model_name='gemini-1.5-pro'
    )

    input_parts = [
        content_api.ProcessorPart(
            b'imagedata', mimetype='image/png', role='user'
        ),
        content_api.ProcessorPart('what is this image?', role='user'),
    ]
    processor.apply_sync(model, input_parts)

    mock_generate_stream.assert_called_once()
    passed_contents = mock_generate_stream.call_args.kwargs['contents']
    self.assertLen(passed_contents, 1)  # Only one turn.
    self.assertLen(passed_contents[0].parts, 2)  # Both parts should be there.
    self.assertEqual(passed_contents[0].parts[0].inline_data.data, b'imagedata')
    self.assertEqual(passed_contents[0].parts[1].text, 'what is this image?')


if __name__ == '__main__':
  absltest.main()
