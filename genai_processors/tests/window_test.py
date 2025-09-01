import asyncio
import collections
from collections.abc import AsyncIterable
import time
import unittest

from absl.testing import absltest
from genai_processors import content_api
from genai_processors import processor
from genai_processors import streams
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


class RealTimePromptTest(unittest.IsolatedAsyncioTestCase):

  async def test_add_part(self):
    rolling_prompt = window.RollingPrompt()
    prompt_content = rolling_prompt.pending()
    part_list = [ProcessorPart(str(i)) for i in range(5)]
    for c in part_list:
      rolling_prompt.add_part(c)
    await rolling_prompt.finalize_pending()
    prompt_text = ProcessorContent(
        await streams.gather_stream(prompt_content)
    ).as_text()
    # prompt = part0-4.
    self.assertEqual(prompt_text, '01234')

  async def test_stashing(self):
    rolling_prompt = window.RollingPrompt()
    prompt_content = rolling_prompt.pending()
    part_list = [ProcessorPart(str(i)) for i in range(5)]
    for c in part_list:
      rolling_prompt.add_part(c)
    rolling_prompt.stash_part(ProcessorPart('while_outputting'))
    for c in part_list:
      rolling_prompt.add_part(c)
    rolling_prompt.apply_stash()
    await rolling_prompt.finalize_pending()
    prompt_text = ProcessorContent(
        await streams.gather_stream(prompt_content)
    ).as_text()
    # prompt = part0-4, part0-4, part while outputting, prompt_suffix
    # -> part while outputting should always be put at the end.
    self.assertEqual(prompt_text, '0123401234while_outputting')

  async def test_cut_history(self):
    rolling_prompt = window.RollingPrompt(duration_prompt_sec=0.1)
    prompt_content = rolling_prompt.pending()
    part_count = 2
    part_list = [ProcessorPart(str(i)) for i in range(part_count)]
    img_list = [ProcessorPart(create_image(3, 3))] * part_count
    for idx, part in enumerate(part_list):
      rolling_prompt.add_part(part)
      rolling_prompt.add_part(img_list[idx])
      await asyncio.sleep(0.01)
    await rolling_prompt.finalize_pending()
    prompt_text = [
        content_api.as_text(c) if content_api.is_text(c.mimetype) else 'img'
        for c in await streams.gather_stream(prompt_content)
    ]
    # First prompt gets the full history.
    self.assertEqual(prompt_text, ['0', 'img', '1', 'img'])
    await asyncio.sleep(0.08)
    await rolling_prompt.finalize_pending()
    prompt_content = rolling_prompt.pending()
    for part in part_list:
      rolling_prompt.add_part(part)
    await rolling_prompt.finalize_pending()
    prompt_text = [
        content_api.as_text(c) if content_api.is_text(c.mimetype) else 'img'
        for c in await streams.gather_stream(prompt_content)
    ]
    # Second prompt gets the cut history + what was fed now.
    self.assertEqual(prompt_text, ['1', 'img', '0', '1'])


class WindowProcessorTest(unittest.IsolatedAsyncioTestCase):

  async def test_window(self):
    input_stream = streams.stream_content([
        '1',
        content_api.ProcessorPart.end_of_turn(),
        '2',
    ])
    output_parts = await streams.gather_stream(
        window.Window(
            main_model_fake.to_processor(),
        )(input_stream)
    )
    actual = content_api.as_text(output_parts)
    self.assertEqual(actual, 'model(1)model(12)')

  async def test_window_with_stride(self):
    input_stream = streams.stream_content([
        '1',
        content_api.ProcessorPart.end_of_turn(),
        '2',
        content_api.ProcessorPart.end_of_turn(),
        '3',
        content_api.ProcessorPart.end_of_turn(),
        '4',
    ])
    output_parts = await streams.gather_stream(
        window.Window(
            main_model_fake.to_processor(),
            stride=2,
        )(input_stream)
    )
    actual = content_api.as_text(output_parts)
    self.assertEqual(actual, 'model(1)model(123)')

  async def test_history_compression(self):
    async def compress_history(history):
      history.clear()

    input_stream = streams.stream_content([
        '1',
        content_api.ProcessorPart.end_of_turn(),
        '2',
    ])
    output_parts = await streams.gather_stream(
        window.Window(
            main_model_fake.to_processor(),
            compress_history=compress_history,
        )(input_stream)
    )
    actual = content_api.as_text(output_parts)
    self.assertEqual(actual, 'model(1)model(2)')

  async def test_max_concurrency(self):
    max_concurrency = 2
    concurrent_executions = 0

    @processor.processor_function
    async def window_processor(
        content: AsyncIterable[ProcessorPart],
    ) -> AsyncIterable[ProcessorPart]:
      nonlocal concurrent_executions
      concurrent_executions += 1
      self.assertLessEqual(concurrent_executions, 1)

      # Simulate some work
      await asyncio.sleep(0.1)
      async for part in content:
        yield part

      concurrent_executions -= 1

    input_stream = streams.stream_content(
        ['1', content_api.ProcessorPart.end_of_turn()] * 10
    )
    output_parts = await streams.gather_stream(
        window.Window(window_processor, max_concurrency=1)(input_stream)
    )


class HistoryCompressionTest(absltest.TestCase):

  def test_drop_old_parts(self):
    now = time.perf_counter()
    policy = window.drop_old_parts(age_sec=10)
    history = collections.deque()
    history.append(ProcessorPart('a', metadata={'capture_time': now - 15}))
    history.append(ProcessorPart('b', metadata={'capture_time': now - 5}))
    history.append(ProcessorPart('c', metadata={'capture_time': now}))

    asyncio.run(policy(history))

    self.assertLen(history, 2)
    self.assertEqual(history[0].text, 'b')
    self.assertEqual(history[1].text, 'c')

  def test_keep_last_n_turns(self):
    policy = window.keep_last_n_turns(turns_to_keep=2)
    history = collections.deque()
    history.append(ProcessorPart('a'))
    history.append(content_api.ProcessorPart.end_of_turn())
    history.append(ProcessorPart('b'))
    history.append(content_api.ProcessorPart.end_of_turn())
    history.append(ProcessorPart('c'))

    asyncio.run(policy(history))

    self.assertLen(history, 3)
    self.assertEqual(history[0].text, 'b')
    self.assertEqual(history[1], content_api.ProcessorPart.end_of_turn())
    self.assertEqual(history[2].text, 'c')


if __name__ == '__main__':
  absltest.main()
