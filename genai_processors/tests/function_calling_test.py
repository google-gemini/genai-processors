from collections.abc import AsyncIterable
import unittest

from absl.testing import absltest
from genai_processors import content_api
from genai_processors import processor
from genai_processors import streams
from genai_processors.core import function_calling


def get_weather(location: str) -> str:
  """Returns sunny weather for any location."""
  return f'Weather in {location} is sunny'


def get_time() -> str:
  """Returns 12:00."""
  return '12:00'


def failing_function() -> str:
  """This function always fails."""
  raise ValueError('<this function failed>')


class MockGenerateProcessor(processor.Processor):

  def __init__(self, responses: list[list[content_api.ProcessorPart]]):
    self._responses = responses
    self._requests = []
    self._call_count = 0

  async def call(
      self, content: AsyncIterable[content_api.ProcessorPart]
  ) -> AsyncIterable[content_api.ProcessorPart]:
    self._requests.append(await streams.gather_stream(content))
    if self._call_count < len(self._responses):
      response_parts = self._responses[self._call_count]
      self._call_count += 1
      for part in response_parts:
        yield part
    else:
      yield content_api.ProcessorPart('fallback response')


class FunctionCallingTest(unittest.IsolatedAsyncioTestCase):

  async def test_no_function_call(self):
    generate_processor = MockGenerateProcessor(
        [[content_api.ProcessorPart('Hello!')]]
    )
    fc_processor = function_calling.FunctionCalling(
        generate_processor,
        fns=[get_weather, get_time],
    )
    input_content = [content_api.ProcessorPart('Hi')]
    output = await streams.gather_stream(
        fc_processor(streams.stream_content(input_content))
    )
    self.assertEqual(output, [content_api.ProcessorPart('Hello!')])

  async def test_one_function_call(self):
    generate_processor = MockGenerateProcessor([
        [
            content_api.ProcessorPart.from_function_call(
                name='get_weather', args={'location': 'London'}
            )
        ],
        [content_api.ProcessorPart('Weather in London is sunny')],
    ])
    fc_processor = function_calling.FunctionCalling(
        generate_processor,
        fns=[get_weather],
    )
    input_content = [
        content_api.ProcessorPart('What is the weather in London?')
    ]
    output = await streams.gather_stream(
        fc_processor(streams.stream_content(input_content))
    )
    self.assertEqual(
        output,
        [
            content_api.ProcessorPart.from_function_call(
                name='get_weather',
                args={'location': 'London'},
                substream_name=function_calling.FUNCTION_CALL_SUBTREAM_NAME,
            ),
            content_api.ProcessorPart.from_function_response(
                name='get_weather',
                response={'result': 'Weather in London is sunny'},
                role='user',
                substream_name=function_calling.FUNCTION_CALL_SUBTREAM_NAME,
            ),
            content_api.ProcessorPart('Weather in London is sunny'),
        ],
    )

  async def test_two_function_calls(self):
    # Call get_time() and get_weather(). The model returns 'let me check'.
    # The expectation is that the functions get called at the right iteration
    # and the function responses are returned properly to the model and to the
    # final output.
    model_output_0 = [
        content_api.ProcessorPart.from_function_call(name='get_time', args={})
    ]
    model_output_1 = [
        content_api.ProcessorPart('let me check', role='model'),
        content_api.ProcessorPart.from_function_call(
            name='get_weather', args={'location': 'Paris'}
        ),
    ]
    model_output_2 = [
        content_api.ProcessorPart('Time is 12:00, Weather in Paris is sunny')
    ]
    input_content = [
        content_api.ProcessorPart('What is the time and weather in Paris?')
    ]
    request_0 = input_content
    fc_output_0 = model_output_0 + [
        content_api.ProcessorPart.from_function_response(
            name='get_time',
            response={'result': '12:00'},
            role='user',
            substream_name=function_calling.FUNCTION_CALL_SUBTREAM_NAME,
        ),
    ]
    request_1 = request_0 + fc_output_0
    fc_output_1 = model_output_1 + [
        content_api.ProcessorPart.from_function_response(
            name='get_weather',
            response={'result': 'Weather in Paris is sunny'},
            role='user',
            substream_name=function_calling.FUNCTION_CALL_SUBTREAM_NAME,
        ),
    ]
    request_2 = request_1 + fc_output_1

    generate_processor = MockGenerateProcessor([
        model_output_0,
        model_output_1,
        model_output_2,
    ])
    fc_processor = function_calling.FunctionCalling(
        generate_processor,
        fns=[get_weather, get_time],
    )
    output = await streams.gather_stream(
        fc_processor(streams.stream_content(input_content))
    )
    self.assertEqual(output, fc_output_0 + fc_output_1 + model_output_2)
    self.assertEqual(
        generate_processor._requests,
        [
            request_0,
            request_1,
            request_2,
        ],
    )

  async def test_max_function_calls(self):
    max_function_calls = 1
    generate_processor = MockGenerateProcessor(
        [[
            content_api.ProcessorPart.from_function_call(
                name='get_time', args={}
            )
        ]]
        * (max_function_calls + 1)
    )
    fc_processor = function_calling.FunctionCalling(
        generate_processor,
        fns=[get_weather, get_time],
        max_function_calls=max_function_calls,
    )
    input_content = [content_api.ProcessorPart('What is the time?')]
    output = await streams.gather_stream(
        fc_processor(streams.stream_content(input_content))
    )
    self.assertEqual(
        output,
        [
            content_api.ProcessorPart.from_function_call(
                name='get_time',
                args={},
                substream_name=function_calling.FUNCTION_CALL_SUBTREAM_NAME,
            ),
            content_api.ProcessorPart.from_function_response(
                name='get_time',
                response={'result': '12:00'},
                role='user',
                substream_name=function_calling.FUNCTION_CALL_SUBTREAM_NAME,
            ),
        ]
        * (max_function_calls + 1),
    )

  async def test_function_not_found(self):
    generate_processor = MockGenerateProcessor([[
        content_api.ProcessorPart.from_function_call(name='get_time', args={})
    ]])
    fc_processor = function_calling.FunctionCalling(
        generate_processor,
        fns=[get_weather],
    )
    input_content = [content_api.ProcessorPart('What is the time?')]
    with self.assertRaisesRegex(ValueError, 'Function get_time not found'):
      await streams.gather_stream(
          fc_processor(streams.stream_content(input_content))
      )

  async def test_failing_function(self):
    generate_processor = MockGenerateProcessor([
        [
            content_api.ProcessorPart.from_function_call(
                name='failing_function', args={}
            )
        ],
        [content_api.ProcessorPart('The function failed')],
    ])
    fc_processor = function_calling.FunctionCalling(
        generate_processor,
        fns=[failing_function],
    )
    input_content = [content_api.ProcessorPart('Call failing function')]
    output = await streams.gather_stream(
        fc_processor(streams.stream_content(input_content))
    )
    self.assertEqual(
        output,
        [
            content_api.ProcessorPart.from_function_call(
                name='failing_function',
                args={},
                substream_name=function_calling.FUNCTION_CALL_SUBTREAM_NAME,
            ),
            content_api.ProcessorPart.from_function_response(
                name='failing_function',
                response={
                    'error': (
                        'Failed to invoke function failing_function with'
                        ' converted arguments {} from model returned function'
                        ' call argument {} because of error <this function'
                        ' failed>'
                    )
                },
                role='user',
                substream_name=function_calling.FUNCTION_CALL_SUBTREAM_NAME,
            ),
            content_api.ProcessorPart('The function failed'),
        ],
    )


if __name__ == '__main__':
  absltest.main()
