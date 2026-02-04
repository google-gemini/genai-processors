import asyncio
from collections.abc import AsyncIterable
import time
import unittest

from absl.testing import absltest
from absl.testing import parameterized
from genai_processors import content_api
from genai_processors import processor
from genai_processors import streams
from genai_processors.core import function_calling
from google.genai import types as genai_types


def get_weather(location: str) -> str:
  """Returns sunny weather for any location."""
  return f'Weather in {location} is sunny'


def get_time() -> str:
  """Returns 12:00."""
  return '12:00'


def failing_function() -> str:
  """This function always fails."""
  raise ValueError('<this function failed>')


def sleep_sync(sleep_seconds: float) -> str:
  """Sleeps for a given number of seconds and returns how long it took."""
  time.sleep(sleep_seconds)
  return f'Slept for {sleep_seconds} seconds'


class MockGenerateProcessor(processor.Processor):
  """Mock a turn-based processor."""

  def __init__(self, responses: list[list[content_api.ProcessorPart]]):
    self._responses = responses
    self._requests = []
    self._call_count = 0

  async def call(
      self, content: content_api.ContentStream
  ) -> AsyncIterable[content_api.ProcessorPart]:
    self._requests.append(list(await content.gather()))
    if self._call_count < len(self._responses):
      response_parts = self._responses[self._call_count]
      self._call_count += 1
      for part in response_parts:
        yield part
    else:
      yield content_api.ProcessorPart('fallback response')


class MockBidiGenerateProcessor(processor.Processor):
  """Mock a bidi processor, aka realtime processor."""

  def __init__(self, responses: list[list[content_api.ProcessorPart]]):
    self._responses = responses
    self._requests = []
    self._call_count = 0

  async def call(
      self, content: AsyncIterable[content_api.ProcessorPart]
  ) -> AsyncIterable[content_api.ProcessorPart]:
    prompt = []
    async for part in content:
      if content_api.is_end_of_turn(part):
        self._requests.append(prompt.copy())
        if self._call_count < len(self._responses):
          response_parts = self._responses[self._call_count]
          self._call_count += 1
          for part in response_parts:
            yield part
        else:
          yield content_api.ProcessorPart('fallback response')
        yield content_api.END_OF_TURN
      else:
        prompt.append(part)


END_OF_STREAM = content_api.ProcessorPart('end of stream')


def create_model(
    model_output: list[list[content_api.ProcessorPart]], is_bidi: bool
) -> tuple[processor.Processor, int]:
  if is_bidi:
    generate_processor = MockBidiGenerateProcessor(
        model_output + [[END_OF_STREAM]]
    )
    delay_sec = 2
  else:
    generate_processor = MockGenerateProcessor(model_output)
    delay_sec = 0
  return generate_processor, delay_sec


class FunctionCallingSyncTest(unittest.IsolatedAsyncioTestCase):

  async def test_no_function_call(self):
    model_output = [content_api.ProcessorPart('Hello!', role='model')]
    generate_processor = MockGenerateProcessor([model_output])
    fc_processor = function_calling.FunctionCalling(
        generate_processor,
        fns=[get_weather, get_time],
    )
    self.assertSequenceEqual(
        await fc_processor(content_api.ProcessorContent('Hi')).gather(),
        model_output,
    )

  async def test_one_function_call(self):
    model_output_0 = [
        content_api.ProcessorPart.from_function_call(
            name='get_weather',
            args={'location': 'London'},
            role='model',
            substream_name=function_calling.FUNCTION_CALL_SUBSTREAM_NAME,
        )
    ]
    model_output_1 = [content_api.ProcessorPart('Sun will shine', role='model')]

    generate_processor = MockGenerateProcessor([
        model_output_0,
        model_output_1,
    ])
    fc_processor = function_calling.FunctionCalling(
        generate_processor,
        fns=[get_weather],
    )
    input_content = content_api.ProcessorContent(
        'What is the weather in London?'
    )
    self.assertSequenceEqual(
        await fc_processor(input_content).gather(),
        model_output_0
        + [
            content_api.ProcessorPart.from_function_response(
                name='get_weather',
                response='Weather in London is sunny',
                role='user',
                substream_name=function_calling.FUNCTION_CALL_SUBSTREAM_NAME,
            ),
        ]
        + model_output_1,
    )

  async def test_two_function_calls(self):
    # Call get_time() and get_weather(). The model returns 'let me check'.
    # The expectation is that the functions get called at the right iteration
    # and the function responses are returned properly to the model and to the
    # final output.
    model_output_0 = [
        content_api.ProcessorPart.from_function_call(
            name='get_time', args={}, role='model'
        )
    ]
    model_output_1 = [
        content_api.ProcessorPart('let me check', role='model'),
        content_api.ProcessorPart.from_function_call(
            name='get_weather',
            args={'location': 'Paris'},
            role='model',
        ),
    ]
    model_output_2 = [
        content_api.ProcessorPart(
            'Time is 12:00, Weather in Paris is sunny', role='model'
        )
    ]
    input_content = [
        content_api.ProcessorPart(
            'What is the time and weather in Paris?', role='user'
        )
    ]
    request_0 = input_content
    fc_output_0 = model_output_0 + [
        content_api.ProcessorPart.from_function_response(
            name='get_time',
            response='12:00',
            role='user',
            substream_name=function_calling.FUNCTION_CALL_SUBSTREAM_NAME,
        ),
    ]
    request_1 = request_0 + fc_output_0
    fc_output_1 = model_output_1 + [
        content_api.ProcessorPart.from_function_response(
            name='get_weather',
            response='Weather in Paris is sunny',
            role='user',
            substream_name=function_calling.FUNCTION_CALL_SUBSTREAM_NAME,
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
    output = await fc_processor(streams.stream_content(input_content)).gather()
    self.assertSequenceEqual(output, fc_output_0 + fc_output_1 + model_output_2)
    self.assertSequenceEqual(
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
                name='get_time',
                args={},
                role='model',
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
    output = await fc_processor(streams.stream_content(input_content)).gather()
    self.assertSequenceEqual(
        output,
        [
            content_api.ProcessorPart.from_function_call(
                name='get_time',
                args={},
                role='model',
                substream_name=function_calling.FUNCTION_CALL_SUBSTREAM_NAME,
            ),
            content_api.ProcessorPart.from_function_response(
                name='get_time',
                response='12:00',
                role='user',
                substream_name=function_calling.FUNCTION_CALL_SUBSTREAM_NAME,
            ),
        ]
        * max_function_calls,
    )

  async def test_function_not_found(self):
    model_output_0 = [
        content_api.ProcessorPart.from_function_call(
            name='get_nothing',
            args={},
            role='model',
        )
    ]
    model_output_1 = [
        content_api.ProcessorPart('ok, trying another function', role='model'),
    ]
    generate_processor = MockGenerateProcessor([model_output_0, model_output_1])
    fc_processor = function_calling.FunctionCalling(
        generate_processor,
        fns=[get_weather, get_time],
    )
    input_content = [content_api.ProcessorPart('What is the time?')]
    output = await fc_processor(streams.stream_content(input_content)).gather()
    self.assertSequenceEqual(
        output,
        model_output_0
        + [
            content_api.ProcessorPart.from_function_response(
                name='get_nothing',
                response=(
                    'Function get_nothing not found. Available functions:'
                    " 'get_time', 'get_weather'."
                ),
                role='user',
                substream_name=function_calling.FUNCTION_CALL_SUBSTREAM_NAME,
                is_error=True,
            ),
        ]
        + model_output_1,
    )

  async def test_failing_function(self):
    generate_processor = MockGenerateProcessor([
        [
            content_api.ProcessorPart.from_function_call(
                name='failing_function',
                args={},
                role='model',
            )
        ],
        [content_api.ProcessorPart('The function failed')],
    ])
    fc_processor = function_calling.FunctionCalling(
        generate_processor,
        fns=[failing_function],
    )
    input_content = [content_api.ProcessorPart('Call failing function')]
    output = await fc_processor(streams.stream_content(input_content)).gather()
    self.assertSequenceEqual(
        output,
        [
            content_api.ProcessorPart.from_function_call(
                name='failing_function',
                args={},
                role='model',
                substream_name=function_calling.FUNCTION_CALL_SUBSTREAM_NAME,
            ),
            content_api.ProcessorPart.from_function_response(
                name='failing_function',
                response=(
                    'Failed to invoke function failing_function({}): <this'
                    ' function failed>'
                ),
                role='user',
                substream_name=function_calling.FUNCTION_CALL_SUBSTREAM_NAME,
                is_error=True,
            ),
            content_api.ProcessorPart('The function failed', role='model'),
        ],
    )


async def sleep_async(sleep_seconds: int) -> str:
  """Sleeps for a given number of seconds and returns how long it took."""
  await asyncio.sleep(sleep_seconds)
  return f'Slept for {sleep_seconds} seconds'


async def sleep_async_generator(
    sleep_seconds: int,
) -> AsyncIterable[str]:
  """Yields a string every second for the given number of seconds."""
  for i in range(1, sleep_seconds + 1):
    await asyncio.sleep(1)
    yield f'Slept for {i} seconds'


async def failing_async_function() -> str:
  """This async function always fails."""
  await asyncio.sleep(0.01)
  raise ValueError('<this async function failed>')


class FunctionCallingAsyncTest(
    unittest.IsolatedAsyncioTestCase, parameterized.TestCase
):

  @parameterized.named_parameters(
      ('unary', False),
      ('bidi', True),
  )
  async def test_async_function(self, is_bidi):
    input_content = [content_api.ProcessorPart('Call sleep async function')] + (
        [content_api.END_OF_TURN] if is_bidi else []
    )
    model_output_0 = [
        content_api.ProcessorPart(
            genai_types.Part(
                function_call=genai_types.FunctionCall(
                    name='sleep_async',
                    id='sleep_async-1',
                    args={'sleep_seconds': 1},
                )
            ),
            role='model',
        )
    ]
    model_output_1 = [
        content_api.ProcessorPart(
            'yes, I slept for 1 second',
            role='model',
        )
    ]
    generate_processor, delay_sec = create_model(
        [
            model_output_0,
            model_output_1,
        ],
        is_bidi,
    )
    fc_processor = function_calling.FunctionCalling(
        generate_processor,
        fns=[sleep_async],
        is_bidi_model=is_bidi,
    )
    output = await fc_processor(
        streams.stream_content(
            input_content, with_delay_sec=delay_sec, delay_end=True
        )
    ).gather()
    # The function calling is similar to the sync version except that the model
    # waits for the function to finish before yielding the next part.
    self.assertSequenceEqual(
        output,
        model_output_0
        + [
            content_api.ProcessorPart.from_function_response(
                name='sleep_async',
                function_call_id='sleep_async-1',
                response='Running in background.',
                role='user',
                substream_name=function_calling.FUNCTION_CALL_SUBSTREAM_NAME,
                scheduling='SILENT',
                will_continue=True,
            ),
        ]
        + [
            content_api.ProcessorPart.from_function_response(
                name='sleep_async',
                response='Slept for 1 seconds',
                function_call_id='sleep_async-1',
                role='user',
                substream_name=function_calling.FUNCTION_CALL_SUBSTREAM_NAME,
            ),
        ]
        + model_output_1,
    )

  @parameterized.named_parameters(
      ('unary', False),
      ('bidi', True),
  )
  async def test_async_sync_function(self, is_bidi):
    input_content = [content_api.ProcessorPart('Call sleep async function')] + (
        [content_api.END_OF_TURN] if is_bidi else []
    )
    model_output_0 = [
        content_api.ProcessorPart(
            genai_types.Part(
                function_call=genai_types.FunctionCall(
                    name='sleep_async',
                    id='sleep_async-1',
                    args={'sleep_seconds': 1},
                )
            ),
            role='model',
        )
    ]
    model_output_1 = [
        content_api.ProcessorPart(
            'OK, slept for 1 second, checking weather in London',
            role='model',
        ),
        content_api.ProcessorPart.from_function_call(
            name='get_weather',
            args={'location': 'London'},
            role='model',
        ),
    ]
    model_output_2 = [
        content_api.ProcessorPart(
            'got the weather',
            role='model',
        )
    ]
    generate_processor, delay_sec = create_model(
        [
            model_output_0,
            model_output_1,
            model_output_2,
        ],
        is_bidi,
    )
    fc_processor = function_calling.FunctionCalling(
        generate_processor,
        fns=[sleep_async, get_weather],
        is_bidi_model=is_bidi,
    )
    output = await fc_processor(
        streams.stream_content(
            input_content, with_delay_sec=delay_sec, delay_end=True
        )
    ).gather()
    self.assertSequenceEqual(
        output,
        model_output_0
        + [
            content_api.ProcessorPart.from_function_response(
                name='sleep_async',
                function_call_id='sleep_async-1',
                response='Running in background.',
                role='user',
                substream_name=function_calling.FUNCTION_CALL_SUBSTREAM_NAME,
                scheduling='SILENT',
                will_continue=True,
            ),
        ]
        + [
            content_api.ProcessorPart.from_function_response(
                name='sleep_async',
                function_call_id='sleep_async-1',
                response='Slept for 1 seconds',
                role='user',
                substream_name=function_calling.FUNCTION_CALL_SUBSTREAM_NAME,
            ),
        ]
        + model_output_1
        # For bidi models, even sync functions are async as they are run in a
        # separate thread to avoid blocking the main input stream (remember bidi
        # is often equivalent to realtime with parts coming in every second).
        + (
            [
                content_api.ProcessorPart.from_function_response(
                    name='get_weather',
                    # No function call id provided in the function call part.
                    function_call_id='get_weather_0',
                    response='Running in background.',
                    role='user',
                    substream_name=function_calling.FUNCTION_CALL_SUBSTREAM_NAME,
                    scheduling='SILENT',
                    will_continue=True,
                ),
            ]
            if is_bidi
            else []
        )
        + [
            content_api.ProcessorPart.from_function_response(
                name='get_weather',
                # No function call id provided in the function call part.
                function_call_id='get_weather_0' if is_bidi else None,
                response='Weather in London is sunny',
                substream_name=function_calling.FUNCTION_CALL_SUBSTREAM_NAME,
                role='user',
            ),
        ]
        + model_output_2,
    )

  @parameterized.named_parameters(
      ('unary', False),
      ('bidi', True),
  )
  async def test_async_generator_function(self, is_bidi):
    input_content = [content_api.ProcessorPart('Call sleep async function')] + (
        [content_api.END_OF_TURN] if is_bidi else []
    )
    model_output_0 = [
        content_api.ProcessorPart.from_function_call(
            name='sleep_async_generator',
            args={'sleep_seconds': 2},
            role='model',
        )
    ]
    model_output_1 = [
        content_api.ProcessorPart(
            'got first part',
            role='model',
        )
    ]
    model_output_2 = [
        content_api.ProcessorPart(
            'got second part, all done',
            role='model',
        )
    ]
    model_output_3 = [
        content_api.ProcessorPart(
            'stream ended',
            role='model',
        )
    ]

    generate_processor, delay_sec = create_model(
        [
            model_output_0,
            model_output_1,
            model_output_2,
            model_output_3,
        ],
        is_bidi,
    )
    fc_processor = function_calling.FunctionCalling(
        generate_processor,
        fns=[sleep_async_generator],
        is_bidi_model=is_bidi,
    )
    output = await fc_processor(
        streams.stream_content(
            input_content, with_delay_sec=delay_sec * 2, delay_end=True
        )
    ).gather()
    self.assertSequenceEqual(
        output,
        model_output_0
        + [
            content_api.ProcessorPart.from_function_response(
                name='sleep_async_generator',
                function_call_id='sleep_async_generator_0',
                response='Running in background.',
                role='user',
                substream_name=function_calling.FUNCTION_CALL_SUBSTREAM_NAME,
                scheduling='SILENT',
                will_continue=True,
            ),
        ]
        + [
            content_api.ProcessorPart.from_function_response(
                name='sleep_async_generator',
                function_call_id='sleep_async_generator_0',
                response='Slept for 1 seconds',
                role='user',
                substream_name=function_calling.FUNCTION_CALL_SUBSTREAM_NAME,
                will_continue=True,
            ),
        ]
        + model_output_1
        + [
            content_api.ProcessorPart.from_function_response(
                name='sleep_async_generator',
                function_call_id='sleep_async_generator_0',
                response='Slept for 2 seconds',
                role='user',
                substream_name=function_calling.FUNCTION_CALL_SUBSTREAM_NAME,
                will_continue=True,
            ),
        ]
        + [
            content_api.ProcessorPart.from_function_response(
                name='sleep_async_generator',
                function_call_id='sleep_async_generator_0',
                response='',
                role='user',
                substream_name=function_calling.FUNCTION_CALL_SUBSTREAM_NAME,
            )
        ]
        + model_output_2
        + model_output_3,
    )

  async def test_async_max_tool_calls(self):
    # Bidi models do not limit tool calls, so we only test unary.
    input_content = [content_api.ProcessorPart('sleep for 1 second twice')]
    model_output_0 = [
        content_api.ProcessorPart.from_function_call(
            name='sleep_async',
            args={'sleep_seconds': 1},
            role='model',
        )
    ]
    model_output_1 = [
        content_api.ProcessorPart(
            'ok, we slept for 1 second once',
            role='model',
        ),
        content_api.ProcessorPart.from_function_call(
            name='sleep_async',
            args={'sleep_seconds': 1},
            role='model',
        ),
    ]
    generate_processor = MockGenerateProcessor([
        model_output_0,
        model_output_1,
    ])
    fc_processor = function_calling.FunctionCalling(
        generate_processor,
        fns=[sleep_async],
        max_function_calls=1,
    )
    output = await fc_processor(streams.stream_content(input_content)).gather()
    self.assertSequenceEqual(
        output,
        content_api.ProcessorContent([
            model_output_0,
            content_api.ProcessorPart.from_function_response(
                name='sleep_async',
                function_call_id='sleep_async_0',
                response='Running in background.',
                role='user',
                substream_name=function_calling.FUNCTION_CALL_SUBSTREAM_NAME,
                scheduling='SILENT',
                will_continue=True,
            ),
            content_api.ProcessorPart.from_function_response(
                name='sleep_async',
                function_call_id='sleep_async_0',
                response='Slept for 1 seconds',
                role='user',
                substream_name=function_calling.FUNCTION_CALL_SUBSTREAM_NAME,
            ),
            model_output_1[:1],
        ]),
    )

  @parameterized.named_parameters(
      ('unary', False),
      ('bidi', True),
  )
  async def test_failing_async_function(self, is_bidi):
    input_content = [
        content_api.ProcessorPart('Call failing async function')
    ] + ([content_api.END_OF_TURN] if is_bidi else [])
    model_output_0 = [
        content_api.ProcessorPart.from_function_call(
            name='failing_async_function',
            args={},
            role='model',
        )
    ]
    model_output_1 = [
        content_api.ProcessorPart(
            'The function failed as expected',
            role='model',
        )
    ]
    generate_processor, delay_sec = create_model(
        [
            model_output_0,
            model_output_1,
        ],
        is_bidi,
    )
    fc_processor = function_calling.FunctionCalling(
        generate_processor,
        fns=[failing_async_function],
        is_bidi_model=is_bidi,
    )
    output = await fc_processor(
        streams.stream_content(
            input_content, with_delay_sec=delay_sec, delay_end=True
        )
    ).gather()
    self.assertSequenceEqual(
        output,
        content_api.ProcessorContent([
            model_output_0,
            content_api.ProcessorPart.from_function_response(
                name='failing_async_function',
                function_call_id='failing_async_function_0',
                response='Running in background.',
                role='user',
                substream_name=function_calling.FUNCTION_CALL_SUBSTREAM_NAME,
                scheduling='SILENT',
                will_continue=True,
            ),
            content_api.ProcessorPart.from_function_response(
                name='failing_async_function',
                function_call_id='failing_async_function_0',
                response=(
                    'Failed to invoke function failing_async_function({}):'
                    ' <this async function failed>'
                ),
                role='user',
                substream_name=function_calling.FUNCTION_CALL_SUBSTREAM_NAME,
                is_error=True,
            ),
            model_output_1,
        ]),
    )

  async def test_bidi_streaming_with_sync_function(self):

    async def input_generator():
      yield content_api.ProcessorPart('call sleep sync for 1 second')
      yield content_api.END_OF_TURN
      await asyncio.sleep(0.1)  # ensure sync function starts
      yield content_api.ProcessorPart(
          'still streaming while sync function runs'
      )
      yield content_api.END_OF_TURN
      await asyncio.sleep(1.5)  # ensure we get the function response

    model_output_0 = [
        content_api.ProcessorPart.from_function_call(
            name='sleep_sync',
            args={'sleep_seconds': 1},
            role='model',
        )
    ]
    model_output_1 = [
        content_api.ProcessorPart(
            'model saw second input',
            role='model',
        )
    ]

    model_output_2 = [
        content_api.ProcessorPart(
            'slept for 1 second',
            role='model',
        )
    ]

    generate_processor, _ = create_model(
        [
            model_output_0,
            model_output_1,
            model_output_2,
        ],
        is_bidi=True,
    )

    fc_processor = function_calling.FunctionCalling(
        generate_processor,
        fns=[sleep_sync],
        is_bidi_model=True,
    )
    output = await fc_processor(input_generator()).gather()

    self.assertSequenceEqual(
        output,
        content_api.ProcessorContent([
            model_output_0,
            content_api.ProcessorPart.from_function_response(
                name='sleep_sync',
                function_call_id='sleep_sync_0',
                response='Running in background.',
                role='user',
                substream_name=function_calling.FUNCTION_CALL_SUBSTREAM_NAME,
                scheduling='SILENT',
                will_continue=True,
            ),
            model_output_1,
            content_api.ProcessorPart.from_function_response(
                name='sleep_sync',
                function_call_id='sleep_sync_0',
                response='Slept for 1 seconds',
                role='user',
                substream_name=function_calling.FUNCTION_CALL_SUBSTREAM_NAME,
            ),
            model_output_2,
        ]),
    )

  @parameterized.named_parameters(
      (
          'non_existent_id',
          ['sleep_async_0', 'non-existent-id'],
          (
              'Cancelled the following function calls:'
              " ['sleep_async_0']. The following function calls were not found:"
              " {'non-existent-id'}."
          ),
          True,
      ),
      ('ok', ['sleep_async_0'], 'OK, cancelled.', False),
      ('empty', [], 'OK, cancelled.', False),
  )
  async def test_cancel_async_function(self, function_ids, response, is_error):
    input_content = [
        content_api.ProcessorPart('Call sleep and cancel'),
        content_api.END_OF_TURN,
        content_api.END_OF_TURN,
    ]
    sleep_time_sec = 3
    model_output_0 = [
        content_api.ProcessorPart.from_function_call(
            name='sleep_async',
            args={'sleep_seconds': sleep_time_sec},
            role='model',
        ),
    ]
    model_output_1 = [
        content_api.ProcessorPart(
            'OK, running sleep async function, now cancel it',
            role='model',
        ),
        content_api.ProcessorPart.from_function_call(
            name='cancel_fc',
            args={'function_ids': function_ids},
            role='model',
        ),
    ]
    generate_processor, delay_sec = create_model(
        [
            model_output_0,
            model_output_1,
        ],
        is_bidi=True,
    )
    fc_processor = function_calling.FunctionCalling(
        generate_processor,
        fns=[sleep_async, function_calling.cancel_fc],
        is_bidi_model=True,
    )
    output = await fc_processor(
        streams.stream_content(
            input_content, with_delay_sec=delay_sec, delay_end=True
        )
    ).gather()
    # When no cancellation happens, the async sleep function should return
    # the result.
    async_sleep_output = (
        [
            content_api.ProcessorPart.from_function_response(
                name='sleep_async',
                function_call_id='sleep_async_0',
                response=f'Slept for {sleep_time_sec} seconds',
                role='user',
                substream_name=function_calling.FUNCTION_CALL_SUBSTREAM_NAME,
            ),
            END_OF_STREAM,
        ]
        if not function_ids
        else []
    )
    self.assertSequenceEqual(
        output,
        content_api.ProcessorContent([
            model_output_0,
            content_api.ProcessorPart.from_function_response(
                name='sleep_async',
                function_call_id='sleep_async_0',
                response='Running in background.',
                role='user',
                substream_name=function_calling.FUNCTION_CALL_SUBSTREAM_NAME,
                scheduling='SILENT',
                will_continue=True,
            ),
            model_output_1,
            content_api.ProcessorPart.from_function_response(
                name='cancel_fc',
                function_call_id='cancel_fc_0',
                response='Running in background.',
                substream_name=function_calling.FUNCTION_CALL_SUBSTREAM_NAME,
                role='user',
                scheduling='SILENT',
                will_continue=True,
            ),
            content_api.ProcessorPart.from_function_response(
                name='cancel_fc',
                function_call_id='cancel_fc_0',
                response=response,
                role='user',
                substream_name=function_calling.FUNCTION_CALL_SUBSTREAM_NAME,
                is_error=is_error,
                scheduling='SILENT',
            ),
            async_sleep_output,
        ]),
    )

  async def test_list_fc(self):
    input_content = [
        content_api.ProcessorPart('Call sleep and list_fc'),
        content_api.END_OF_TURN,
        content_api.END_OF_TURN,
    ]
    sleep_time_sec = 3
    model_output_0 = [
        content_api.ProcessorPart.from_function_call(
            name='sleep_async',
            args={'sleep_seconds': sleep_time_sec},
            role='model',
        ),
    ]
    model_output_1 = [
        content_api.ProcessorPart(
            'OK, running sleep async function, now list functions',
            role='model',
        ),
        content_api.ProcessorPart.from_function_call(
            name='list_fc',
            args={},
            role='model',
        ),
    ]
    model_output_2 = [
        content_api.ProcessorPart(
            'OK, obtained list of functions',
            role='model',
        ),
    ]
    model_output_3 = [
        content_api.ProcessorPart(
            'OK',
            role='model',
        ),
    ]
    generate_processor, delay_sec = create_model(
        [
            model_output_0,
            model_output_1,
            model_output_2,
            model_output_3,
        ],
        is_bidi=True,
    )
    fc_processor = function_calling.FunctionCalling(
        generate_processor,
        fns=[sleep_async, function_calling.list_fc],
        is_bidi_model=True,
    )
    output = await fc_processor(
        streams.stream_content(
            input_content, with_delay_sec=delay_sec, delay_end=True
        )
    ).gather()

    list_fc_response = (
        'Background Functions currently running:\nFunction sleep_async is'
        " running with args {'sleep_seconds': 3} and id: sleep_async_0\n"
    )

    self.assertSequenceEqual(
        output,
        content_api.ProcessorContent([
            model_output_0,
            content_api.ProcessorPart.from_function_response(
                name='sleep_async',
                function_call_id='sleep_async_0',
                response='Running in background.',
                role='user',
                substream_name=function_calling.FUNCTION_CALL_SUBSTREAM_NAME,
                scheduling='SILENT',
                will_continue=True,
            ),
            model_output_1,
            content_api.ProcessorPart.from_function_response(
                name='list_fc',
                function_call_id='list_fc_0',
                response='Running in background.',
                substream_name=function_calling.FUNCTION_CALL_SUBSTREAM_NAME,
                role='user',
                scheduling='SILENT',
                will_continue=True,
            ),
            content_api.ProcessorPart.from_function_response(
                name='list_fc',
                function_call_id='list_fc_0',
                response=list_fc_response,
                role='user',
                substream_name=function_calling.FUNCTION_CALL_SUBSTREAM_NAME,
            ),
            model_output_2,
            content_api.ProcessorPart.from_function_response(
                name='sleep_async',
                function_call_id='sleep_async_0',
                response='Slept for 3 seconds',
                role='user',
                substream_name=function_calling.FUNCTION_CALL_SUBSTREAM_NAME,
            ),
            model_output_3,
        ]),
    )


if __name__ == '__main__':
  absltest.main()
