import asyncio
from collections.abc import AsyncIterable
import unittest

from absl.testing import absltest
from absl.testing import parameterized
from genai_processors import content_api
from genai_processors import streams
from google.genai import types as genai_types


async def _parts(*data: str) -> AsyncIterable[content_api.ProcessorPart]:
  for d in data:
    yield content_api.ProcessorPart(genai_types.Part(text=d))


async def text(content: AsyncIterable[content_api.ProcessorPart]) -> list[str]:
  return [part.text async for part in content]


class StreamsTest(parameterized.TestCase, unittest.IsolatedAsyncioTestCase):

  def test_split_distributes_parts(self):
    async def _distribute_parts():
      streams_ = streams.split(_parts('foo', 'bar'), n=3)
      return await asyncio.gather(*[text(stream_) for stream_ in streams_])

    output = asyncio.run(_distribute_parts())

    self.assertEqual(output, [['foo', 'bar'], ['foo', 'bar'], ['foo', 'bar']])

  def test_split_tolerates_empty_stream(self):
    async def _empty_stream():
      streams_ = streams.split(_parts(), n=2)
      return await asyncio.gather(*[text(stream_) for stream_ in streams_])

    output = asyncio.run(_empty_stream())

    self.assertEqual(output, [[], []])

  def test_concat_merges_streams(self):
    async def _concat_streams():
      streams_ = [_parts(f'foo {i}') for i in range(3)]
      concat_ = streams.concat(*streams_)
      return await text(concat_)

    output = asyncio.run(_concat_streams())

    self.assertEqual(output, ['foo 0', 'foo 1', 'foo 2'])

  def test_concat_returns_empty_stream_when_given_no_streams(self):
    self.assertEqual(asyncio.run(text(streams.concat())), [])

  def test_concat_tolerates_empty_streams(self):
    async def _concat_streams():
      streams_ = [_parts(f'foo {i}') for i in range(3)]
      concat_ = streams.concat(*streams_, _parts())
      return await text(concat_)

    output = asyncio.run(_concat_streams())

    self.assertEqual(output, ['foo 0', 'foo 1', 'foo 2'])

  async def test_merge_empty(self):
    merged = streams.merge([])
    self.assertEqual(await streams.gather_stream(merged), [])

  async def test_merge(self):
    input_streams = [
        streams.stream_content([1, 2, 3], with_delay_sec=0.1, delay_first=True),
        streams.stream_content(
            [4, 5, 6], with_delay_sec=0.12, delay_first=True
        ),
    ]
    merged = streams.merge(input_streams)
    self.assertEqual(await streams.gather_stream(merged), [1, 4, 2, 5, 3, 6])
    # The delays make sure the order of the merge is deterministic (time based).
    input_streams = [
        streams.stream_content([1, 2, 3], with_delay_sec=0.1, delay_first=True),
        streams.stream_content(
            [4, 5, 6], with_delay_sec=0.09, delay_first=True
        ),
    ]
    merged = streams.merge(input_streams)
    self.assertEqual(await streams.gather_stream(merged), [4, 1, 5, 2, 6, 3])

  async def test_merge_cancel(self):
    async def stream_1():
      async for c in streams.stream_content(
          [1, 2, 3], with_delay_sec=0.1, delay_first=True
      ):
        yield c
        raise asyncio.CancelledError()

    stream_2 = streams.stream_content(
        [4, 5, 6], with_delay_sec=0.05, delay_first=True
    )
    merged = streams.merge([stream_1(), stream_2])
    # The stream cancelled is stopped, does not prevent others from continuing.
    self.assertEqual(await streams.gather_stream(merged), [4, 1, 5, 6])

  async def test_merge_with_error(self):
    async def stream_1():
      async for c in streams.stream_content(
          [1, 2, 3], with_delay_sec=0.1, delay_first=True
      ):
        yield c
        raise ValueError()

    with self.assertRaises(ExceptionGroup) as value_error:
      await streams.gather_stream(
          streams.merge([stream_1(), streams.stream_content([4, 5, 6])])
      )
    self.assertIsInstance(value_error.exception.exceptions[0], ValueError)

  async def test_streams_and_enumerates_content(self):
    as_stream = streams.stream_content([1, 2])
    async for i, x in streams.aenumerate(as_stream):
      self.assertEqual(i + 1, x)

  async def test_finishes(self):

    async def slow_noop(
        inputs: AsyncIterable[int],
    ) -> AsyncIterable[int]:
      async for x in inputs:
        await asyncio.sleep(1)
        yield x

    inputs = streams.stream_content([0, 1, 2])
    response = await streams.gather_stream(slow_noop(inputs))
    self.assertEqual(response, [0, 1, 2])

  @parameterized.named_parameters(
      dict(testcase_name='bounded_queue', maxsize=2),
      dict(testcase_name='unbounded_queue', maxsize=0),
  )
  async def test_enqueue(self, maxsize: int):
    q = asyncio.Queue(maxsize=maxsize)
    t = asyncio.create_task(
        streams.enqueue(streams.stream_content([1, 2, 3]), q)
    )
    self.assertEqual(await q.get(), 1)
    self.assertEqual(await q.get(), 2)
    self.assertEqual(await q.get(), 3)
    self.assertIsNone(await q.get())
    await t


if __name__ == '__main__':
  absltest.main()
