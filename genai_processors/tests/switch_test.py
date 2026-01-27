import asyncio
from collections.abc import AsyncIterable
import time
import unittest

from absl.testing import absltest
from absl.testing import parameterized
from genai_processors import content_api
from genai_processors import processor
from genai_processors import streams
from genai_processors import switch


class SwitchProcessorTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  def setUp(self):
    super().setUp()

    @processor.processor_function
    async def p(
        content: AsyncIterable[content_api.ProcessorPart],
    ) -> AsyncIterable[content_api.ProcessorPartTypes]:
      async for part in content:
        await asyncio.sleep(0.01)
        yield f'p({content_api.as_text(part)})'

    @processor.processor_function
    async def q(
        content: AsyncIterable[content_api.ProcessorPart],
    ) -> AsyncIterable[content_api.ProcessorPartTypes]:
      async for part in content:
        yield f'q({content_api.as_text(part)})'

    self.p = p
    self.q = q

  async def test_switch_simple_ok(self):
    match = (
        switch.Switch(content_api.as_text)
        .case('a', self.p)
        .case('b', self.q)
        .default(processor.passthrough())
    )
    result = match(streams.stream_content(['a', 'b', 'c']))
    self.assertEqual(await result.text(), 'q(b)cp(a)')

  async def test_switch_different_substreams(self):
    input_stream = [content_api.ProcessorPart('a', substream_name='a')] * 3
    input_stream.append(content_api.ProcessorPart('b', substream_name='b'))
    input_stream.append(content_api.ProcessorPart('c', substream_name='c'))

    match = (
        switch.Switch(content_api.get_substream_name)
        .case('a', self.p)
        .default(processor.passthrough())
    )
    result = match(streams.stream_content(input_stream))
    self.assertEqual(await result.text(), 'bcp(a)p(a)p(a)')

    match = switch.Switch(content_api.get_substream_name).case('b', self.q)
    start_sec = time.perf_counter()
    result = match(streams.stream_content(input_stream))
    # There is no default case, only b passed through.
    self.assertEqual(await result.text(), 'q(b)')
    end_sec = time.perf_counter()
    # Check that the await in p does not slow down the switch.
    self.assertLess(end_sec - start_sec, 0.009)

  async def test_switch_single_run_match_none_ok(self):

    match = (
        switch.Switch(content_api.as_text)
        .case(lambda x: x.startswith('a'), self.p)
        .case('b', self.q)
        .default(processor.passthrough())
    )
    result = match(streams.stream_content(['a1', 'b', 'c', 'a2']))
    # p(a) is the last one as p takes longer to compute: the order is not
    # guaranteed between cases but the order within a case is guaranteed.
    self.assertEqual(await result.text(), 'q(b)cp(a1)p(a2)')


class PartSwitchProcessorTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  def setUp(self):
    super().setUp()

    @processor.part_processor_function
    async def p(
        part: content_api.ProcessorPart,
    ) -> AsyncIterable[content_api.ProcessorPartTypes]:
      yield f'p({part.text})'

    @processor.part_processor_function
    async def q(
        part: content_api.ProcessorPart,
    ) -> AsyncIterable[content_api.ProcessorPartTypes]:
      yield f'q({content_api.as_text(part)})'

    self.p = p
    self.q = q

  async def test_switch_processor_simple_ok(self):

    match = (
        switch.PartSwitch(content_api.as_text)
        .case('a', self.p)
        .case('b', self.q)
        .default(processor.passthrough())
    )
    result = match.to_processor()(
        streams.stream_content(
            ['a', 'b', 'c'],
        )
    )
    self.assertEqual(await result.text(), 'p(a)q(b)c')

  async def test_switch_processor_single_run_ok(self):

    match = (
        switch.PartSwitch(lambda c: c.mimetype)
        .case('text/plain', self.p)
        .case('text/plain', self.q)
        .default(processor.passthrough())
    )
    result = match.to_processor()(
        streams.stream_content(
            ['a', 'b', 'c'],
        )
    )
    self.assertEqual(await result.text(), 'p(a)p(b)p(c)')

  async def test_switch_processor_single_run_match_none_ok(self):

    match = (
        switch.PartSwitch(content_api.as_text)
        .case('a', self.p)
        .case('b', self.q)
        .default(processor.passthrough())
    )
    result = match.to_processor()(
        streams.stream_content(
            ['a', 'b', 'c'],
        )
    )
    self.assertEqual(await result.text(), 'p(a)q(b)c')

  async def test_switch_processor_complex_cases_ok(self):

    match = (
        switch.PartSwitch(content_api.as_text)
        .case(lambda x: x.startswith('a'), self.p)
        .case(lambda x: x.startswith('b'), self.q)
        .default((processor.passthrough()))
    )
    result = match.to_processor()(
        streams.stream_content(
            ['ab', 'bc', 'cd'],
        )
    )
    self.assertEqual(await result.text(), 'p(ab)q(bc)cd')

  async def test_switch_processor_match_and_case_ok(self):

    match = (
        switch.PartSwitch(lambda x: x.mimetype)
        .case('text/plain', self.p)
        .case('audio/wav', self.q)
        .default(processor.passthrough())
    )
    c1 = content_api.ProcessorPart('text', mimetype='text/plain')
    c2 = content_api.ProcessorPart('audio', mimetype='audio/wav')
    result = match.to_processor()(
        streams.stream_content(
            [c1, c2],
        )
    )
    # empty param in `q` as `c2` is not text but audio.
    self.assertEqual(await result.text(), 'p(text)q()')

  async def test_switch_processor_many_processors_few_parts(self):
    max_active_tasks = []

    @processor.part_processor_function
    async def p(
        part: content_api.ProcessorPart,
    ) -> AsyncIterable[content_api.ProcessorPart]:
      max_active_tasks.append(
          len([
              task
              for task in asyncio.all_tasks()
              if not task.done() and task.get_name().startswith('eager_run_fn')
          ]),
      )
      yield content_api.ProcessorPart(f'p({part.text})')

    part_count = 10
    processor_count = 1000
    match = switch.PartSwitch(lambda x: x.text)
    for i in range(processor_count):
      match.case(f'{i} ', p)
    match.default(processor.passthrough())
    _ = await streams.gather_stream(
        match.to_processor()(
            streams.stream_content(
                [f'{i} ' for i in range(part_count)],
            )
        )
    )
    # The switch.Match processor create 3 tasks per part: 1 for the
    # to_processor() and 2 for the parallel processors.
    self.assertLessEqual(max(max_active_tasks), 3 * part_count)


if __name__ == '__main__':
  absltest.main()
