import asyncio
from collections.abc import AsyncIterable
import io
import json
import os
import shutil
from typing import cast
import unittest
import wave

from absl.testing import absltest
from genai_processors import content_api
from genai_processors import context as context_lib
from genai_processors import debug
from genai_processors import mime_types
from genai_processors import processor
from genai_processors import streams
from genai_processors.core import function_calling
from genai_processors.core import realtime
from genai_processors.dev import trace_file
from google.genai import types as genai_types
import numpy as np
from PIL import Image


@processor.processor_function
async def to_upper_fn(
    content: AsyncIterable[content_api.ProcessorPart],
) -> AsyncIterable[content_api.ProcessorPartTypes]:
  async for part in content:
    await asyncio.sleep(0.01)  # to ensure timestamps are different
    if mime_types.is_text(part.mimetype):
      yield part.text.upper() + '_sub_trace'
    else:
      yield part


@processor.part_processor_function
async def add_one(
    part: content_api.ProcessorPart,
) -> AsyncIterable[content_api.ProcessorPartTypes]:
  await asyncio.sleep(0.01)  # to ensure timestamps are different
  if mime_types.is_text(part.mimetype):
    yield part.text + '_1'
  else:
    yield part


def create_image_part() -> content_api.ProcessorPart:
  # Create a small green image using PIL
  img = Image.new('RGB', (10, 10), color='green')
  img_bytes_io = io.BytesIO()
  img.save(img_bytes_io, format='PNG')
  return content_api.ProcessorPart.from_bytes(
      data=img_bytes_io.getvalue(),
      mimetype='image/png',
  )


def create_audio_part() -> content_api.ProcessorPart:
  # Generate a small random WAV audio part
  sample_rate = 16000  # samples per second
  duration = 0.1  # seconds
  num_samples = int(sample_rate * duration)
  # Generate random samples between -1 and 1
  random_samples = np.random.uniform(-1, 1, num_samples)
  # Scale to int16 range
  audio_data = (random_samples * 32767).astype(np.int16)

  audio_bytes_io = io.BytesIO()
  with wave.open(audio_bytes_io, 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(audio_data.dtype.itemsize)
    wf.setframerate(sample_rate)
    wf.writeframes(audio_data.tobytes())
  return content_api.ProcessorPart.from_bytes(
      data=audio_bytes_io.getvalue(),
      mimetype='audio/wav',
  )


def collect_processor_names(t: trace_file.SyncFileTrace) -> set[str]:
  """Collects processor names from a trace and its sub-traces."""
  names = set()
  if t.name:
    names.add(t.name)
  for event in t.events:
    if event.sub_trace:
      names.update(collect_processor_names(event.sub_trace))
  return names


class SubTraceProcessor(processor.Processor):

  def __init__(self):
    super().__init__()
    self.sub_processor = to_upper_fn + add_one
    self.sub_processor = debug.TTFTSingleStream(
        'TEST_SUB_PROCESSOR',
        self.sub_processor,
    )

  async def call(
      self, content: AsyncIterable[content_api.ProcessorPart]
  ) -> AsyncIterable[content_api.ProcessorPartTypes]:
    async for part in self.sub_processor(content):
      if (
          isinstance(part, content_api.ProcessorPart)
          and mime_types.is_text(part.mimetype)
          and not context_lib.is_reserved_substream(part.substream_name)
      ):
        yield part.text + '_outer'
      else:
        yield part


class TraceTest(unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    super().setUp()
    self.trace_dir = os.path.join(absltest.get_default_test_tmpdir(), 'traces')
    os.makedirs(self.trace_dir, exist_ok=True)

  def tearDown(self):
    super().tearDown()
    shutil.rmtree(self.trace_dir)

  async def test_trace_generation_and_timestamps(self):
    p = SubTraceProcessor()
    input_parts = content_api.ProcessorContent('hello')
    async with trace_file.SyncFileTrace(trace_dir=self.trace_dir):
      results = await p(input_parts).gather()
    # Check we return the status part with the debug information.
    self.assertIn('TEST_SUB_PROCESSOR', results[0].text)
    self.assertEqual(results[1].text, 'HELLO_sub_trace_1_outer')
    json_files = [f for f in os.listdir(self.trace_dir) if f.endswith('.json')]
    self.assertTrue(len(json_files), 1)
    trace_path = os.path.join(self.trace_dir, json_files[0])
    self.assertTrue(os.path.exists(trace_path.replace('.json', '.html')))

    root_trace = trace_file.SyncFileTrace.load(trace_path)
    self.assertEqual(root_trace.events[0].relation, 'call')

    # We have (with debug module exclusion, TTFTSingleStream and its internal
    # processors log_on_close/log_on_first are not traced):
    # root_trace:
    #  \__ SubTraceProcessor (call)
    #      \__ chain (from _ChainProcessor wrapping the user processors)
    #          \__ to_upper_fn
    #          \__ add_one

    # Get SubTraceProcessor
    trace = cast(trace_file.SyncFileTrace, root_trace.events[0].sub_trace)
    self.assertFalse(trace.events[0].is_input)
    self.assertEqual(trace.events[0].relation, 'chain')
    # Get chain (was TTFTSingleStream -> _ChainProcessor, now just chain)
    sub_trace = cast(trace_file.SyncFileTrace, trace.events[0].sub_trace)
    self.assertIsNotNone(sub_trace)
    self.assertIn('chain', sub_trace.name)
    # Get to_upper_fn (first sub-trace of chain)
    to_upper_trace = cast(
        trace_file.SyncFileTrace, sub_trace.events[0].sub_trace
    )
    self.assertIsNotNone(to_upper_trace)
    self.assertIn('to_upper_fn', to_upper_trace.name)
    # Check the output of to_upper_fn
    self.assertFalse(to_upper_trace.events[1].is_input)
    self.assertEqual(
        root_trace.parts_store[to_upper_trace.events[1].part_hash]['part'][
            'text'
        ],
        'HELLO_sub_trace',
    )
    self.assertIsNotNone(to_upper_trace.start_time)
    self.assertIsNotNone(to_upper_trace.end_time)
    self.assertLess(to_upper_trace.start_time, to_upper_trace.end_time)

    # Check events from SubTraceProcessor.
    self.assertTrue(trace.events[1].is_input)
    self.assertEqual(
        root_trace.parts_store[trace.events[1].part_hash]['part']['text'],
        'hello',
    )
    self.assertFalse(trace.events[2].is_input)
    self.assertIn(
        'TEST_SUB_PROCESSOR',
        root_trace.parts_store[trace.events[2].part_hash]['part']['text'],
    )
    self.assertFalse(trace.events[3].is_input)
    self.assertEqual(
        root_trace.parts_store[trace.events[3].part_hash]['part']['text'],
        'HELLO_sub_trace_1_outer',
    )

  async def test_trace_save_load(self):
    trace = trace_file.SyncFileTrace(name='test')
    async with trace:
      await trace.add_input(content_api.ProcessorPart('in'))
      await trace.add_input(
          content_api.ProcessorPart.from_bytes(
              data=b'bytes',
              mimetype='image/jpeg',
          )
      )
      sub_trace = trace.add_sub_trace(name='sub_test', relation='chain')
      await sub_trace.add_input(content_api.ProcessorPart('sub_in'))
      await sub_trace.add_output(content_api.ProcessorPart('sub_out'))
      await trace.add_output(content_api.ProcessorPart('out'))

    tmpdir = absltest.get_default_test_tmpdir()
    trace_path = os.path.join(tmpdir, 'trace.json')

    trace.save(trace_path)
    loaded_trace = trace_file.SyncFileTrace.load(trace_path)

    self.assertEqual(
        json.loads(trace.to_json_str()),
        json.loads(loaded_trace.to_json_str()),
    )

    sub_trace_event = next(
        event for event in loaded_trace.events if event.relation == 'chain'
    )
    sub_trace = cast(trace_file.SyncFileTrace, sub_trace_event.sub_trace)
    self.assertEqual(sub_trace.name, 'sub_test')
    # Verify _root_trace is properly set for sub-traces after loading
    self.assertIsNone(loaded_trace._root_trace)
    self.assertIs(sub_trace._root_trace, loaded_trace)

  async def test_save_html(self):
    p = SubTraceProcessor()
    trace_dir = self.trace_dir

    img_part = create_image_part()
    audio_part = create_audio_part()
    parts = [
        img_part,
        audio_part,
        content_api.ProcessorPart('hello', substream_name='input', role='user'),
        content_api.ProcessorPart(
            'how ',
            substream_name='input',
            role='user',
            metadata={'is_over': False},
        ),
        img_part,
        content_api.ProcessorPart('are ', substream_name='input', role='user'),
        content_api.ProcessorPart('you?', substream_name='input', role='user'),
        audio_part,
    ]
    async with trace_file.SyncFileTrace(trace_dir=trace_dir, name='Trace test'):
      await processor.apply_async(p, parts)

    html_files = [
        f
        for f in os.listdir(trace_dir)
        if f.endswith('.html') and 'Trace test' in f
    ]
    self.assertEqual(len(html_files), 1)
    trace_path = os.path.join(trace_dir, html_files[0])
    self.assertTrue(os.path.exists(trace_path))

  async def test_image_resizing(self):
    img_part = content_api.ProcessorPart(
        Image.new('RGB', (400, 300), color='green')
    )

    trace = trace_file.SyncFileTrace(name='test_image_resizing')
    async with trace:
      await trace.add_input(img_part)

    self.assertEqual(len(trace.events), 1)
    event = trace.events[0]
    self.assertIsNotNone(event.part_hash)

    part_dict = trace.parts_store[event.part_hash]
    part_image_bytes = part_dict['part']['inline_data']['data']
    part_image = Image.open(io.BytesIO(part_image_bytes))
    self.assertEqual(part_image.size, (200, 150))

  async def test_html_with_complex_parts(self):
    """Test that HTML is generated correctly for all ProcessorPart types."""
    trace_dir = self.trace_dir

    img_part = create_image_part()
    audio_part = create_audio_part()
    exec_code_part = genai_types.Part.from_executable_code(
        code=(
            'print("Hello from Python!")\nresult = 2 + 2\nprint(f"Result:'
            ' {result}")'
        ),
        language=genai_types.Language.PYTHON,
    )
    code_result_part = genai_types.Part.from_code_execution_result(
        outcome=genai_types.Outcome.OUTCOME_OK,
        output='Hello from Python!\nResult: 4',
    )
    root_trace = trace_file.SyncFileTrace(
        trace_dir=trace_dir,
        name='Complex Parts Test',
    )
    async with root_trace:
      trace = root_trace.add_sub_trace(name='sub_trace', relation='call')
      await trace.add_input(
          content_api.ProcessorPart('User query: what is the weather?')
      )
      await trace.add_input(img_part)
      await trace.add_input(audio_part)
      await trace.add_output(
          content_api.ProcessorPart.from_function_call(
              name='get_weather',
              args={'location': 'San Francisco', 'units': 'celsius'},
              role='model',
          )
      )
      await trace.add_input(
          content_api.ProcessorPart.from_function_response(
              name='get_weather',
              response={'temperature': 22, 'conditions': 'sunny'},
              function_call_id='call_12345',
              substream_name='tool_response',
              role='user',
          )
      )
      await trace.add_output(
          content_api.ProcessorPart(
              'The weather in San Francisco is 22°C and sunny.',
              role='model',
          )
      )
      await trace.add_output(
          content_api.ProcessorPart.from_function_call(
              name='generate_image',
              args={'prompt': 'A sunny day in San Francisco'},
              role='model',
              metadata={'generation_complete': True, 'turn_id': 123},
          )
      )
      await trace.add_output(
          content_api.ProcessorPart(exec_code_part, role='model')
      )
      await trace.add_output(
          content_api.ProcessorPart(code_result_part, role='model')
      )
      sub_trace = trace.add_sub_trace(name='tool_execution', relation='call')
      await sub_trace.add_input(
          content_api.ProcessorPart('Executing tool: generate_image')
      )
      await sub_trace.add_output(img_part)
      await trace.add_output(img_part)
      await trace.add_output(
          content_api.ProcessorPart(
              'Here is the generated image of a sunny day in San Francisco.',
              role='model',
              metadata={'generation_complete': True, 'turn_id': 123},
          )
      )

    # Find the files matching our trace name
    complex_html = [
        f
        for f in os.listdir(trace_dir)
        if 'Complex Parts Test' in f and f.endswith('.html')
    ]
    complex_json = [
        f
        for f in os.listdir(trace_dir)
        if 'Complex Parts Test' in f and f.endswith('.json')
    ]
    self.assertEqual(len(complex_html), 1)
    self.assertEqual(len(complex_json), 1)

    # Load and verify the JSON structure
    json_path = os.path.join(trace_dir, complex_json[0])
    loaded_root_trace = trace_file.SyncFileTrace.load(json_path)
    loaded_trace = loaded_root_trace.events[0].sub_trace
    self.assertIsNotNone(loaded_trace)

    # Verify we have the expected number of events
    self.assertGreater(len(loaded_trace.events), 5)

    # Verify function call is present
    function_call_found = False
    function_response_found = False
    executable_code_found = False
    code_result_found = False
    sub_trace_found = False

    for event in loaded_trace.events:
      if event.sub_trace:
        sub_trace_found = True
        continue
      part_dict = loaded_root_trace.parts_store.get(event.part_hash)
      if part_dict:
        part = part_dict.get('part', {})
        if part.get('function_call'):
          function_call_found = True
        if part.get('function_response'):
          function_response_found = True
        if part.get('executable_code'):
          executable_code_found = True
        if part.get('code_execution_result'):
          code_result_found = True

    self.assertTrue(function_call_found)
    self.assertTrue(function_response_found)
    self.assertTrue(executable_code_found)
    self.assertTrue(code_result_found)
    self.assertTrue(sub_trace_found)

    # Verify HTML file exists and has content
    html_path = os.path.join(trace_dir, complex_html[0])
    self.assertTrue(os.path.exists(html_path))
    with open(html_path, 'r') as f:
      html_content = f.read()
    self.assertIn('Complex Parts Test', html_content)
    self.assertIn('function_call', html_content)
    self.assertIn('function_response', html_content)
    self.assertIn('executable_code', html_content)
    self.assertIn('code_execution_result', html_content)

  async def test_debug_processors_excluded_from_trace(self):
    """Test that processors from genai_processors.debug are excluded from trace.

    This test verifies that when a processor is wrapped with debug utilities
    like TTFTSingleStream or log_stream, those debug processors do not appear
    in the trace. Only the user-defined processor chain should be traced.
    """
    trace_dir = self.trace_dir

    # Create a simple processor chain wrapped with debug utilities
    processor_chain = to_upper_fn + add_one
    # Wrap with TTFTSingleStream (from debug module - should be excluded)
    wrapped_processor = debug.TTFTSingleStream('Debug Test', processor_chain)
    # Also wrap with log_stream (from debug module - should be excluded)
    wrapped_processor = debug.log_stream('Test Log') + wrapped_processor

    # Create mixed input: audio, image, and text
    img_part = create_image_part()
    audio_part = create_audio_part()
    parts = [
        audio_part,
        img_part,
        content_api.ProcessorPart('hello', role='user'),
        content_api.ProcessorPart('world', role='user'),
    ]

    # First call
    async with trace_file.SyncFileTrace(
        trace_dir=trace_dir, name='Debug Exclusion Test'
    ):
      _ = await processor.apply_async(wrapped_processor, parts[:2])
      _ = await processor.apply_async(wrapped_processor, parts[2:])

    # Verify HTML file was created
    html_files = [
        f
        for f in os.listdir(trace_dir)
        if 'Debug Exclusion Test' in f and f.endswith('.html')
    ]
    json_files = [
        f
        for f in os.listdir(trace_dir)
        if 'Debug Exclusion Test' in f and f.endswith('.json')
    ]
    self.assertEqual(len(html_files), 1)
    self.assertEqual(len(json_files), 1)

    # Load the trace and verify no debug processors are present
    json_path = os.path.join(trace_dir, json_files[0])
    root_trace = trace_file.SyncFileTrace.load(json_path)

    all_names = collect_processor_names(root_trace)

    # Debug processors should NOT be in the trace
    debug_processor_names = [
        'TTFTSingleStream',
        'log_stream',
        'print_stream',
        'log_on_close',
        'log_on_first',
    ]
    for debug_name in debug_processor_names:
      for name in all_names:
        self.assertNotIn(
            debug_name,
            name,
            f'Debug processor "{debug_name}" should not appear in trace,'
            f' but found "{name}"',
        )

    # User-defined processors SHOULD be in the trace
    self.assertTrue(
        any('to_upper_fn' in name for name in all_names),
        f'to_upper_fn should be in trace, found: {all_names}',
    )
    self.assertTrue(
        any('add_one' in name for name in all_names),
        f'add_one should be in trace, found: {all_names}',
    )

  async def test_live_processor_trace(self):
    """Test that LiveProcessor generates correct traces."""
    trace_dir = os.getenv('TEST_UNDECLARED_OUTPUTS_DIR') or self.trace_dir

    call_counter = 0

    @processor.processor_function
    async def fake_turn_model(
        content: AsyncIterable[content_api.ProcessorPart],
    ) -> AsyncIterable[content_api.ProcessorPartTypes]:
      nonlocal call_counter
      call_counter += 1
      buffer = content_api.ProcessorContent()
      async for part in content:
        buffer += part
      await asyncio.sleep(0.001 * call_counter)
      yield content_api.ProcessorPart(
          f'model({buffer.as_text()})', role='model'
      )

    live_processor = realtime.LiveProcessor(
        turn_processor=fake_turn_model,
        duration_prompt_sec=60,
    )

    async with trace_file.SyncFileTrace(
        trace_dir=trace_dir, name='Live Processor Test'
    ):
      input_parts = []
      for i in range(10):
        input_parts += [
            content_api.ProcessorPart(f'hello_{i}', role='user'),
            content_api.ProcessorPart(f'world_{i}', role='user'),
            content_api.END_OF_TURN,
        ]
      input_stream = streams.stream_content(input_parts, with_delay_sec=0.01)
      _ = await streams.gather_stream(live_processor(input_stream))

    json_files = [
        f
        for f in os.listdir(trace_dir)
        if 'Live Processor Test' in f and f.endswith('.json')
    ]
    self.assertEqual(len(json_files), 1)

    json_path = os.path.join(trace_dir, json_files[0])
    root_trace = trace_file.SyncFileTrace.load(json_path)

    all_names = collect_processor_names(root_trace)

    self.assertTrue(
        any('LiveProcessor' in name for name in all_names),
        f'LiveProcessor should be in trace, found: {all_names}',
    )
    self.assertTrue(
        any('fake_turn_model' in name for name in all_names),
        f'fake_turn_model should be in trace, found: {all_names}',
    )

  async def test_parallel_processor_trace_names(self):
    """Test that parallel processors have human-readable names in traces."""
    trace_dir = os.getenv('TEST_UNDECLARED_OUTPUTS_DIR') or self.trace_dir

    @processor.processor_function
    async def processor_a(
        content: AsyncIterable[content_api.ProcessorPart],
    ) -> AsyncIterable[content_api.ProcessorPartTypes]:
      async for part in content:
        if mime_types.is_text(part.mimetype):
          yield part.text + '_a'
        else:
          yield part

    @processor.processor_function
    async def processor_b(
        content: AsyncIterable[content_api.ProcessorPart],
    ) -> AsyncIterable[content_api.ProcessorPartTypes]:
      async for part in content:
        if mime_types.is_text(part.mimetype):
          yield part.text + '_b'
        else:
          yield part

    parallel_processor = processor.parallel_concat([processor_a, processor_b])

    async with trace_file.SyncFileTrace(
        trace_dir=trace_dir, name='Parallel Processor Test'
    ):
      _ = await processor.apply_async(
          parallel_processor, [content_api.ProcessorPart('hello')]
      )

    json_files = [
        f
        for f in os.listdir(trace_dir)
        if 'Parallel Processor Test' in f and f.endswith('.json')
    ]
    self.assertEqual(len(json_files), 1)

    json_path = os.path.join(trace_dir, json_files[0])
    root_trace = trace_file.SyncFileTrace.load(json_path)

    all_names = collect_processor_names(root_trace)

    self.assertTrue(
        any('parallel' in name.lower() for name in all_names),
        f'parallel should be in trace, found: {all_names}',
    )
    self.assertTrue(
        any('processor_a' in name for name in all_names),
        f'processor_a should be in trace, found: {all_names}',
    )
    self.assertTrue(
        any('processor_b' in name for name in all_names),
        f'processor_b should be in trace, found: {all_names}',
    )
    for name in all_names:
      self.assertNotIn(
          '_ParallelProcessor:',
          name,
          f'Trace name "{name}" should not have internal class prefix',
      )

  async def test_part_processor_chain_trace_names(self):
    """Test that chained part processors have human-readable names in traces."""
    trace_dir = os.getenv('TEST_UNDECLARED_OUTPUTS_DIR') or self.trace_dir

    @processor.part_processor_function
    async def part_proc_x(
        part: content_api.ProcessorPart,
    ) -> AsyncIterable[content_api.ProcessorPartTypes]:
      if mime_types.is_text(part.mimetype):
        yield part.text + '_x'
      else:
        yield part

    @processor.part_processor_function
    async def part_proc_y(
        part: content_api.ProcessorPart,
    ) -> AsyncIterable[content_api.ProcessorPartTypes]:
      if mime_types.is_text(part.mimetype):
        yield part.text + '_y'
      else:
        yield part

    chained_part_processor = part_proc_x + part_proc_y

    async with trace_file.SyncFileTrace(
        trace_dir=trace_dir, name='Part Processor Chain Test'
    ):
      _ = await processor.apply_async(
          chained_part_processor, [content_api.ProcessorPart('hello')]
      )

    json_files = [
        f
        for f in os.listdir(trace_dir)
        if 'Part Processor Chain Test' in f and f.endswith('.json')
    ]
    self.assertEqual(len(json_files), 1)

    json_path = os.path.join(trace_dir, json_files[0])
    root_trace = trace_file.SyncFileTrace.load(json_path)

    all_names = collect_processor_names(root_trace)

    for name in all_names:
      self.assertNotIn(
          '_PartProcessorWrapper:',
          name,
          f'Trace name "{name}" should not have internal class prefix',
      )
      self.assertNotIn(
          '_ChainPartProcessor:',
          name,
          f'Trace name "{name}" should not have internal class prefix',
      )

  async def test_realtime_async_function_with_image(self):
    """Test that images returned by async functions are traced.

    This test verifies that when using FunctionCalling with an async function
    that returns images, the function calls, function responses with images,
    and image parts are all correctly captured in the trace.
    """
    trace_dir = os.getenv('TEST_UNDECLARED_OUTPUTS_DIR') or self.trace_dir

    # Create an image part that will be returned by the async function
    test_image = create_image_part()

    # Define an async function that generates an image
    async def generate_image(prompt: str) -> content_api.ProcessorPart:
      """Generates an image based on the prompt."""
      await asyncio.sleep(0.01)  # Simulate async work
      del prompt  # Unused in test
      return test_image

    # Mock model that issues a function call and then responds after
    call_count = 0

    @processor.processor_function
    async def mock_model(
        content: AsyncIterable[content_api.ProcessorPart],
    ) -> AsyncIterable[content_api.ProcessorPartTypes]:
      """Mock model that issues a generate_image function call."""
      nonlocal call_count
      async for part in content:
        del part  # consume input
      call_count += 1
      if call_count == 1:
        # First call: issue a function call
        yield content_api.ProcessorPart.from_function_call(
            name='generate_image',
            args={'prompt': 'A cat'},
            role='model',
        )
      else:
        # Second call (after function response): respond with text
        yield content_api.ProcessorPart(
            'Here is the generated image.',
            role='model',
        )

    # Wrap model with FunctionCalling
    fc_processor = function_calling.FunctionCalling(
        model=realtime.LiveProcessor(mock_model),
        fns=[generate_image],
        is_bidi_model=True,
    )

    async with trace_file.SyncFileTrace(
        trace_dir=trace_dir, name='Async Function Image Test'
    ):
      input_parts = [
          content_api.ProcessorPart('Generate an image of a cat', role='user'),
          content_api.END_OF_TURN,
      ]
      input_stream = streams.stream_content(
          input_parts, with_delay_sec=0.1, delay_end=True
      )
      _ = await streams.gather_stream(fc_processor(input_stream))

    json_files = [
        f
        for f in os.listdir(trace_dir)
        if 'Async Function Image Test' in f and f.endswith('.json')
    ]
    self.assertEqual(len(json_files), 1)

    json_path = os.path.join(trace_dir, json_files[0])
    root_trace = trace_file.SyncFileTrace.load(json_path)

    all_names = collect_processor_names(root_trace)

    # Verify FunctionCalling is in trace
    self.assertTrue(
        any('FunctionCalling' in name for name in all_names),
        f'FunctionCalling should be in trace, found: {all_names}',
    )
    # Verify the image is captured in the trace's parts_store
    self.assertIsNotNone(root_trace.parts_store)
    assert root_trace.parts_store is not None  # Type narrowing for pytype

    image_found = False
    function_call_found = False
    function_response_found = False

    # Recursively gather all events from trace and its subtraces
    def check_events(trace_obj):
      nonlocal image_found, function_call_found, function_response_found
      for event in trace_obj.events:
        if event.sub_trace:
          check_events(event.sub_trace)
          continue
        if not event.part_hash:
          continue
        part_dict = root_trace.parts_store.get(event.part_hash, {})
        part = part_dict.get('part', {})
        mimetype = part_dict.get('mimetype', '')

        # Check for image parts
        inline_data = part.get('inline_data', {})
        if mimetype.startswith('image/'):
          image_found = True
        elif inline_data.get('mime_type', '').startswith('image/'):
          image_found = True

        # Check for function call
        if part.get('function_call', {}).get('name') == 'generate_image':
          function_call_found = True

        # Check for function response
        func_resp = part.get('function_response', {})
        if func_resp.get('name') == 'generate_image':
          function_response_found = True
          # Check if function response contains image data in parts
          # Parts can be directly in func_resp['parts'] or in
          # func_resp['response']['parts']
          resp_parts = func_resp.get('parts', []) or func_resp.get(
              'response', {}
          ).get('parts', [])
          for resp_part in resp_parts:
            inline_blob = resp_part.get('inline_data', {})
            if inline_blob.get('mime_type', '').startswith('image/'):
              image_found = True

    check_events(root_trace)

    self.assertTrue(
        function_call_found,
        'Function call should be in trace events',
    )
    self.assertTrue(
        function_response_found,
        'Function response should be in trace events',
    )
    self.assertTrue(
        image_found,
        'Image returned by async function should be in trace events',
    )

  async def test_trace_max_size_bytes(self):
    trace = trace_file.SyncFileTrace(name='test_max_size', max_size_bytes=300)
    async with trace:
      p1 = content_api.ProcessorPart('part1')
      p2 = content_api.ProcessorPart(
          'part2_' + '0123456789' * 20,  # Approx 200 chars
          role='user',
          substream_name='test',
          metadata={'key': 'value'},
      )
      p3 = content_api.ProcessorPart.from_bytes(
          data=b'img',
          mimetype='image/png',
          metadata={trace_file._IMAGE_SIZE_KEY: (200, 200)},
      )
      await trace.add_input(p1)
      await trace.add_input(p2)
      await trace.add_input(p3)

    self.assertEqual(len(trace.events), 3)
    # part1 should be stored fully.
    self.assertEqual(
        trace.parts_store[trace.events[0].part_hash],
        {
            'part': {'text': 'part1'},
            'metadata': {},
            'mimetype': 'text/plain',
            'role': '',
            'substream_name': '',
        },
    )
    # part 2&3 should exceed limit and be stored as metadata + extra args only.
    self.assertEqual(
        trace.parts_store[trace.events[1].part_hash],
        {
            'mimetype': 'text/plain',
            'role': 'user',
            'substream_name': 'test',
            'metadata': {'key': 'value'},
        },
    )
    self.assertEqual(
        trace.parts_store[trace.events[2].part_hash],
        {
            'mimetype': 'image/png',
            'role': '',
            'substream_name': '',
            'metadata': {trace_file._IMAGE_SIZE_KEY: (200, 200)},
        },
    )


if __name__ == '__main__':
  absltest.main()
