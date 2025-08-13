import asyncio
from collections.abc import AsyncIterable
import gc
import time
import unittest

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
from genai_processors import map_processor
from genai_processors import streams


class MapProcessorTest(parameterized.TestCase):

  def test_part_function_execute_order(self):
    execute_order = []

    async def reverse_echo(content: int) -> AsyncIterable[int]:
      # Wait for duration inverse to content
      await asyncio.sleep(2 - content)

      execute_order.append(content)
      yield content

    debug_content = [0, 1, 2]
    transformed_content = map_processor.apply_sync(
        map_processor.map_part_function(reverse_echo), debug_content
    )

    self.assertEqual(transformed_content, debug_content)
    self.assertEqual(execute_order, [2, 1, 0])

  def test_part_function_yield_multiple(self):
    execute_order = []

    async def reverse_echo(content: int) -> AsyncIterable[int]:
      execute_order.append(content)
      yield content
      await asyncio.sleep(2 - content)
      execute_order.append(content)
      yield content

    debug_content = [0, 1, 2]
    transformed_content = map_processor.apply_sync(
        map_processor.map_part_function(reverse_echo), debug_content
    )

    self.assertEqual(transformed_content, [0, 0, 1, 1, 2, 2])
    self.assertEqual(execute_order, [0, 1, 2, 2, 1, 0])

  def test_part_function_apply_sync(self):
    async def plus_one(content: int) -> AsyncIterable[int]:
      await asyncio.sleep(0.1)
      yield content + 1

    output = map_processor.apply_sync(
        map_processor.map_part_function(plus_one), [1, 2, 3]
    )
    self.assertEqual(output, [2, 3, 4])

  def test_chain_part_functions(self):
    async def not_one(content: int) -> AsyncIterable[int]:
      if content != 1:
        yield content

    async def two_to_four(content: int) -> AsyncIterable[int]:
      if content == 2:
        yield 4
      else:
        yield content

    debug_content = [0, 1, 2]
    transformed_content = map_processor.apply_sync(
        map_processor.map_part_function(
            map_processor.chain_part_functions(
                [not_one, two_to_four],
            )
        ),
        debug_content,
    )

    self.assertEqual(transformed_content, [0, 4])

  def test_part_function_chain_execute_ahead(self):
    """Test part function chains can run ahead.

    This is useful behaviour for when a particular function requires to block
    for a long time.
    Should execute in order:
            c0  c1 ..> content
        f0  2  0
        f1  3  1
    fns ..
        v
    """
    event = asyncio.Event()
    execute_order = []

    async def f0(c: str) -> AsyncIterable[str]:
      if c == 'c0':
        await event.wait()
      execute_order.append(('f0', c))
      yield c

    async def f1(c: str) -> AsyncIterable[str]:
      if c == 'c1':
        event.set()
      execute_order.append(('f1', c))
      yield c

    ins = ['c0', 'c1']
    out = map_processor.apply_sync(
        map_processor.map_part_function(
            map_processor.chain_part_functions([f0, f1])
        ),
        ins,
    )
    # Check outputs are ordered correctly
    self.assertEqual(ins, out)
    # Check that `c1`` was fully processed before `c0` and that `f1` was always
    # executed before `f0` for each item.
    self.assertEqual(
        execute_order, [('f0', 'c1'), ('f1', 'c1'), ('f0', 'c0'), ('f1', 'c0')]
    )

  def test_parallel_processor_order_execution_ok(self):

    async def plusone(
        part: str,
    ) -> AsyncIterable[str]:
      yield part + 'a'
      yield part + 'b'

    async def plustwo(
        part: str,
    ) -> AsyncIterable[str]:
      yield part + '1'
      yield part + '2'
      yield part + '3'

    content = ['1', '2']
    output = map_processor.apply_sync(
        map_processor.map_part_function(
            map_processor.parallel_part_functions([plusone, plustwo])
        ),
        content,
    )
    expected = ['1a', '1b', '11', '12', '13', '2a', '2b', '21', '22', '23']
    self.assertEqual(output, expected)

  def test_parallel_processor_return_something(self):

    async def plusone(
        part: str,
    ) -> AsyncIterable[str]:
      if part == '1':
        yield part + 'a'

    async def plustwo(
        part: str,
    ) -> AsyncIterable[str]:
      if part == '1':
        yield part + '1'

    content = ['1', '2']
    output = map_processor.apply_sync(
        map_processor.map_part_function(
            map_processor.parallel_part_functions(
                [plusone, plustwo], with_default_output=True
            )
        ),
        content,
    )
    # We expect the processors to return 1 one value each for '1' but zero for
    # '2', the parallel execution should then return the input part '2' as is.
    expected = ['1a', '11', '2']
    self.assertEqual(output, expected)

  def test_parallel_processor_parallel_execution_ok(self):

    execution_order = []

    async def slow(
        part: str,
    ) -> AsyncIterable[str]:
      # We delay this function to check that it gets executed in second.
      await asyncio.sleep(1)
      execution_order.append(1)
      yield part + 'slow'

    async def fast(
        part: str,
    ) -> AsyncIterable[str]:
      execution_order.append(2)
      yield part + 'fast'

    content = ['1', '2', '3']
    output = map_processor.apply_sync(
        map_processor.map_part_function(
            map_processor.parallel_part_functions([slow, fast])
        ),
        content,
    )
    self.assertLen(output, len(content) * 2)
    self.assertEqual(execution_order, [2, 2, 2, 1, 1, 1])

  def test_parallel_with_part_processor(self):
    async def longone(part: str) -> AsyncIterable[str]:
      if part[-1] != '1':
        yield part
        return
      await asyncio.sleep(1)
      yield part + '_1'

    async def shortone(
        part: str,
    ) -> AsyncIterable[str]:
      if part[-1] != '1':
        yield part
        return
      yield part + '_1'

    async def shorttwo(
        part: str,
    ) -> AsyncIterable[str]:
      if part[-1] != '2':
        yield part
        return
      yield part + '_2'

    async def longtwo(
        part: str,
    ) -> AsyncIterable[str]:
      if part[-1] != '2':
        yield part
        return
      await asyncio.sleep(1)
      yield part + '_2'

    content = ['1', '2', '1', '2']
    # (long_1 // short_2) + (short_1 // long_2)
    p1 = map_processor.parallel_part_functions([longone, shorttwo])
    p2 = map_processor.parallel_part_functions([shortone, longtwo])
    p_start = time.perf_counter()
    p_output = map_processor.apply_sync(
        map_processor.map_part_function(
            map_processor.chain_part_functions([p1, p2])
        ),
        content,
    )
    p_end = time.perf_counter()
    self.assertLen(p_output, len(content) * 4)
    c_start = time.perf_counter()
    _ = map_processor.apply_sync(
        map_processor.map_part_function(
            map_processor.chain_part_functions(
                [longone, shorttwo, shortone, longtwo]
            )
        ),
        content,
    )
    c_end = time.perf_counter()
    # Check execution time is within 1sec (should be within 0.01 sec).
    # Both chain and // should have the same execution pattern.
    self.assertAlmostEqual(p_end - p_start, c_end - c_start, places=0)


class MapProcessorWithValidFnTest(
    unittest.IsolatedAsyncioTestCase, parameterized.TestCase
):

  async def test_match_fn_map_no_task(self):
    # Check that set match_fn to False does not create a task and the parts
    # are passed through.
    fn_run_count = 0

    async def f(c: str) -> AsyncIterable[str]:
      nonlocal fn_run_count
      fn_run_count += 1
      await asyncio.sleep(0.3)
      yield f'f({c})'

    input_str = ['0', '1', '2', '3'] * 10
    result = await streams.gather_stream(
        map_processor.map_part_function(f, lambda _: False)(
            streams.stream_content(input_str)
        )
    )
    self.assertEqual(fn_run_count, 0)
    self.assertEqual(result, input_str)

  async def test_match_fn_chain_ok(self):
    # Check that match_fn is respected and parts are passed through when
    # match_fn returns False (for a chain).
    async def f(c: str) -> AsyncIterable[str]:
      yield f'f({c})'

    async def g(c: str) -> AsyncIterable[str]:
      yield f'g({c})'

    chain = map_processor.chain_part_functions(
        [f, g],
        [lambda x: x == '0', lambda x: x == '1'],
    )
    result = await streams.gather_stream(
        map_processor.map_part_function(chain)(
            streams.stream_content(['0', '1', '2', '3'])
        )
    )
    self.assertEqual(result, ['f(0)', 'g(1)', '2', '3'])

  @parameterized.parameters(
      dict(
          pass_through=False,
          expected=['f(0)', 'g(1)'],
      ),
      dict(
          pass_through=True,
          expected=['f(0)', 'g(1)', '2', '3'],
      ),
  )
  async def test_match_fn_parallel_ok(self, pass_through, expected):
    # Check that match_fn is respected and parts are _not_ passed through when
    # match_fn returns False (for a parallel).
    async def f(c: str) -> AsyncIterable[str]:
      yield f'f({c})'

    async def g(c: str) -> AsyncIterable[str]:
      yield f'g({c})'

    parallel = map_processor.parallel_part_functions(
        [f, g],
        [lambda x: x == '0', lambda x: x == '1'],
        with_default_output=pass_through,
    )
    result = await streams.gather_stream(
        map_processor.map_part_function(parallel)(
            streams.stream_content(['0', '1', '2', '3'])
        )
    )
    self.assertEqual(result, expected)


class GCMapProcessorTest(parameterized.TestCase):

  def test_gc_time(self):
    gc.disable()
    use_map = True

    async def echo_p(c):
      async for x in c:
        yield x

    async def echo(c):
      yield c

    async def run_chain():
      s = streams.stream_content(['test' for _ in range(10)])

      if use_map:
        # Chain map processors
        map_p = map_processor.chain_part_functions([echo, echo, echo])
        p = map_processor.map_part_function(map_p)
        s = p(s)
      else:
        # Chain processors
        for _ in range(3):
          s = echo_p(s)

      async for _ in s:
        pass

    async def launch():
      # Clear GC
      gc.collect()
      start = time.perf_counter()
      ts = [asyncio.create_task(run_chain()) for _ in range(1000)]
      await asyncio.gather(*ts)
      duration = time.perf_counter() - start
      logging.info('Echos took %s', duration)

      # Run GC
      start = time.perf_counter()
      gc.collect()
      duration = time.perf_counter() - start
      logging.info('GC took %s', duration)
      assert duration < 0.5, duration

    asyncio.run(launch())

    gc.enable()


if __name__ == '__main__':
  absltest.main()
