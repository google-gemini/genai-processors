from collections.abc import AsyncIterable
import logging
import re
import time
import unittest

from absl.testing import absltest
from genai_processors import content_api
from genai_processors import debug
from genai_processors import processor
from genai_processors import streams

ProcessorContent = content_api.ProcessorContent
ProcessorPart = content_api.ProcessorPart


@processor.processor_function
async def model_fake(
    content: AsyncIterable[ProcessorPart],
) -> AsyncIterable[ProcessorPart]:
  buffer = []
  async for chunk in content:
    buffer.append(ProcessorPart(f'model({chunk.text})'))
  for chunk in buffer:
    yield chunk


class DebugTest(unittest.IsolatedAsyncioTestCase):

  async def test_ttft_status_ok(self):
    input_stream = ProcessorContent(['Hello', 'world', '!'])
    p_with_debug = debug.TTFTSingleStream('test', model_fake)
    actual = await p_with_debug(input_stream).gather()
    # Check that the processor is called with the input stream properly
    self.assertEqual(
        actual.as_text(substream_name=''), 'model(Hello)model(world)model(!)'
    )
    # Check that the TTFT is reported in the status stream and is consistent.
    match = re.search(
        r'test TTFT=(\d+\.\d+) seconds', actual.as_text(substream_name='status')
    )
    self.assertIsNotNone(match)
    self.assertEqual(match.group(1), f'{p_with_debug.ttft():2.2f}')

  async def test_ttft_with_async_for(self):
    p_with_debug = debug.TTFTSingleStream('test', model_fake)
    input_stream = ProcessorContent(['Hello', 'world', '!'])
    async for _ in p_with_debug(input_stream):
      self.assertIsNotNone(p_with_debug.ttft())
      break

  async def test_debug_stream_in_chain(self):
    input_stream = ProcessorContent(['Hello ', 'world', '!'])

    @processor.part_processor_function
    async def p(chunk: ProcessorPart) -> AsyncIterable[ProcessorPart]:
      yield ProcessorPart(f'p({chunk.text})')

    @processor.part_processor_function
    async def q(chunk: ProcessorPart) -> AsyncIterable[ProcessorPart]:
      logging.info('q(%s)', chunk)
      yield ProcessorPart(f'q({chunk.text})')

    chain = p + debug.log_stream('test') + q
    output_stream = chain(input_stream)
    with self.assertLogs(level='INFO') as log_output:
      self.assertEqual(
          await output_stream.text(), 'q(p(Hello ))q(p(world))q(p(!))'
      )
      self.assertIn(
          'p(Hello )',
          ','.join(log_output.output),
      )
      self.assertIn(
          'p(world)',
          ','.join(log_output.output),
      )
      self.assertIn(
          'p(!)',
          ','.join(log_output.output),
      )

  async def test_debug_with_model_call(self):
    waiting_time = 0.5
    input_stream = streams.stream_content(
        ProcessorContent('test'), with_delay_sec=waiting_time
    )
    now = time.perf_counter()
    model_fake_with_debug = debug.TTFTSingleStream('test', model_fake)

    async def model_call_event_wait(t: float):
      await model_fake_with_debug.model_call_event().wait()
      time.perf_counter()
      self.assertGreaterEqual(time.perf_counter() - t, waiting_time)

    async for _ in model_fake_with_debug(input_stream):
      # Wait for the model call and check that it is happening after the input
      # stream has been streamed.
      await processor.create_task(model_call_event_wait(now))
      break


if __name__ == '__main__':
  absltest.main()
