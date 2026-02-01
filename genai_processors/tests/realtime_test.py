import asyncio
from collections.abc import AsyncIterable
import unittest

from absl.testing import absltest
from absl.testing import parameterized
from genai_processors import content_api
from genai_processors import processor
from genai_processors import streams
from genai_processors.core import realtime
from genai_processors.core import window
from PIL import Image


ProcessorPart = content_api.ProcessorPart
ProcessorContent = content_api.ProcessorContent


def create_image(width, height):
  return Image.new('RGB', (width, height))


# Fake realtime models - simply wraps the parts around a model() call.
@processor.processor_function
async def main_model_fake(
    content: AsyncIterable[ProcessorPart],
) -> AsyncIterable[ProcessorPart]:
  yield 'model('
  async for part in content:
    if content_api.is_text(part.mimetype):
      yield part
    else:
      yield f'[{part.mimetype}]'
  yield ')'


# Fake of a realtime model raising an error
@processor.part_processor_function
async def main_model_exception_fake(
    part: ProcessorPart,
) -> AsyncIterable[ProcessorPart]:
  yield part
  raise ValueError('model error')


class RealTimeConversationTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  @parameterized.parameters([
      dict(
          input_stream=[
              ProcessorPart('hello', role='user'),
              ProcessorPart(
                  b'\x01\x00\x01\x00',
                  mimetype='audio/wav',
                  role='user',
              ),
              ProcessorPart(
                  create_image(100, 100),
                  mimetype='image/png',
                  role='user',
              ),
          ],
          output_text='model(hello[audio/wav][image/png])',
      ),
      dict(
          input_stream=[
              ProcessorPart('hello', role='user'),
              ProcessorPart(
                  b'\x01\x00\x01\x00',
                  mimetype='audio/wav',
                  role='user',
              ),
              content_api.ProcessorPart.end_of_turn(),
              ProcessorPart('yo', role='user'),
              ProcessorPart(
                  create_image(100, 100),
                  mimetype='image/png',
                  role='user',
              ),
          ],
          # The first model call is cancelled, the second model call is made
          # with the full prompt.
          output_text='model(hello[audio/wav]yo[image/png])',
      ),
  ])
  async def test_realtime_single_ok(self, input_stream, output_text):
    input_stream = streams.stream_content(input_stream)
    output_parts = await streams.gather_stream(
        realtime.LiveProcessor(
            main_model_fake.to_processor(),
        )(input_stream)
    )
    actual = content_api.as_text(output_parts, substream_name='')
    self.assertEqual(actual, output_text)

  async def test_realtime_raise_exception(self):
    conversation_mgr = realtime.LiveProcessor(
        turn_processor=main_model_exception_fake.to_processor()
    )
    input_stream = streams.stream_content([
        ProcessorPart('hello', role='user'),
    ])
    with self.assertRaises(ValueError):
      await streams.gather_stream(conversation_mgr(input_stream))


@processor.processor_function
async def model_fake(
    content: AsyncIterable[ProcessorPart],
) -> AsyncIterable[ProcessorPart]:
  buffer = content_api.ProcessorContent()
  async for part in content:
    buffer += part
  # Assume a long model call.
  await asyncio.sleep(1)
  yield ProcessorPart(f'model({buffer.as_text()})', role='model')


class RealTimeConversationModelTest(unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    super().setUp()
    self.output_queue = asyncio.Queue()
    self.user_not_talking = asyncio.Event()
    self.user_not_talking.set()
    self.rolling_prompt = window.RollingPrompt()

  def end_conversation(self):
    # To be called within an asyncio loop.
    async def _end_conversation():
      await asyncio.sleep(5)
      self.output_queue.put_nowait(None)

    processor.create_task(_end_conversation())

  async def test_output_order_ok(self):
    model = realtime._RealTimeConversationModel(
        output_queue=self.output_queue,
        generation=model_fake,
        rolling_prompt=self.rolling_prompt,
        user_not_talking=self.user_not_talking,
    )
    model.user_input(ProcessorPart('hello'))
    model.user_input(ProcessorPart('world'))

    # A turn takes 1 sec (see model_fake).
    await model.turn()
    model.user_input(ProcessorPart('done', role='user'))

    self.end_conversation()
    output_parts = await streams.gather_stream(
        streams.dequeue(self.output_queue)
    )
    actual = content_api.as_text(output_parts, substream_name='')
    self.assertEqual(actual, 'model(helloworld)')

  async def test_prompt_order_ok(self):
    model = realtime._RealTimeConversationModel(
        output_queue=self.output_queue,
        generation=model_fake,
        rolling_prompt=self.rolling_prompt,
        user_not_talking=self.user_not_talking,
    )
    model.user_input(ProcessorPart('hello'))
    model.user_input(ProcessorPart('world'))
    # A turn takes 1 sec (see model_fake).
    await model.turn()

    # Wait for the conversation to end.
    self.end_conversation()
    _ = await streams.gather_stream(streams.dequeue(self.output_queue))

    # Check that the rolling prompt put all the parts in the correct order.
    await self.rolling_prompt.finalize_pending()
    prompt_pending = self.rolling_prompt.pending()
    await self.rolling_prompt.finalize_pending()
    prompt_actual = await streams.gather_stream(prompt_pending)
    self.assertEqual(
        content_api.as_text(prompt_actual, substream_name=''),
        'helloworldmodel(helloworld)',
    )


if __name__ == '__main__':
  absltest.main()
