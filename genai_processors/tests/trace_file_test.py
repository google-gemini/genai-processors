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
from genai_processors.dev import trace_file
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

    # We have:
    # root_trace:
    #  \__ SubTraceProcessor (call)
    #      \__ TTFTSingleStream
    #           \__ _ChainProcessor
    #               \__ log_on_close (added by TTFTSingleStream)
    #               \__ to_upper_fn
    #               \__ add_one
    #               \__ log_on_first (added by TTFTSingleStream)

    # Get SubTraceProcessor
    trace = cast(trace_file.SyncFileTrace, root_trace.events[0].sub_trace)
    self.assertFalse(trace.events[0].is_input)
    self.assertEqual(trace.events[0].relation, 'chain')
    # Get TTFTSingleStream
    sub_trace = cast(trace_file.SyncFileTrace, trace.events[0].sub_trace)
    self.assertIsNotNone(sub_trace)
    self.assertEqual(trace.events[0].relation, 'chain')
    # Get _ChainProcessor
    sub_trace = cast(trace_file.SyncFileTrace, sub_trace.events[0].sub_trace)
    # Get to_upper_fn
    sub_trace = cast(trace_file.SyncFileTrace, sub_trace.events[1].sub_trace)
    self.assertIsNotNone(sub_trace)
    self.assertIn('to_upper_fn', sub_trace.name)
    # Check the output of to_upper_fn
    self.assertFalse(sub_trace.events[1].is_input)
    self.assertEqual(
        root_trace.parts_store[sub_trace.events[1].part_hash]['part']['text'],
        'HELLO_sub_trace',
    )
    self.assertIsNotNone(sub_trace.start_time)
    self.assertIsNotNone(sub_trace.end_time)
    self.assertLess(sub_trace.start_time, sub_trace.end_time)

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

    # Create a small green image using PIL
    img = Image.new('RGB', (10, 10), color='green')
    img_bytes_io = io.BytesIO()
    img.save(img_bytes_io, format='PNG')
    img_part = content_api.ProcessorPart.from_bytes(
        data=img_bytes_io.getvalue(),
        mimetype='image/png',
    )

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
    audio_part = content_api.ProcessorPart.from_bytes(
        data=audio_bytes_io.getvalue(),
        mimetype='audio/wav',
    )
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

    html_files = [f for f in os.listdir(trace_dir) if f.endswith('.html')]
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


if __name__ == '__main__':
  absltest.main()
