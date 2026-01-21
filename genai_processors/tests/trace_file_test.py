import asyncio
from collections.abc import AsyncIterable
import io
import json
import os
import shutil
from typing import cast
import unittest

from absl.testing import absltest
from genai_processors import content_api
from genai_processors import mime_types
from genai_processors import processor
from genai_processors import streams
from genai_processors.dev import trace_file
import numpy as np
from PIL import Image
from scipy.io import wavfile


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


class SubTraceProcessor(processor.Processor):

  def __init__(self):
    super().__init__()
    self.sub_processor = to_upper_fn

  async def call(
      self, content: AsyncIterable[content_api.ProcessorPart]
  ) -> AsyncIterable[content_api.ProcessorPartTypes]:
    async for part in self.sub_processor(content):
      if isinstance(part, content_api.ProcessorPart) and mime_types.is_text(
          part.mimetype
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
    input_parts = [content_api.ProcessorPart('hello')]
    async with trace_file.SyncFileTrace(trace_dir=self.trace_dir):
      results = await streams.gather_stream(
          p(streams.stream_content(input_parts))
      )

    self.assertEqual(results[0].text, 'HELLO_sub_trace_outer')
    json_files = [f for f in os.listdir(self.trace_dir) if f.endswith('.json')]
    self.assertTrue(len(json_files), 1)
    trace_path = os.path.join(self.trace_dir, json_files[0])
    self.assertTrue(os.path.exists(trace_path.replace('.json', '.html')))

    trace = trace_file.SyncFileTrace.load(trace_path)

    # First event is a subtrace for the upper function. This is was is first
    # entered in the trace scope.
    self.assertFalse(trace.events[0].is_input)
    sub_trace = cast(trace_file.SyncFileTrace, trace.events[0].sub_trace)
    self.assertIsNotNone(sub_trace)
    self.assertIn('to_upper_fn', sub_trace.name)
    self.assertFalse(sub_trace.events[1].is_input)
    self.assertEqual(
        sub_trace.events[1].part_dict['part']['text'], 'HELLO_sub_trace'
    )
    self.assertIsNotNone(sub_trace.start_time)
    self.assertIsNotNone(sub_trace.end_time)
    self.assertLess(sub_trace.start_time, sub_trace.end_time)

    # Second input event is the input part to SubTraceProcessor.
    self.assertTrue(trace.events[1].is_input)
    self.assertEqual(trace.events[1].part_dict['part']['text'], 'hello')

    # Third event is the output event of SubTraceProcessor
    self.assertFalse(trace.events[2].is_input)
    self.assertEqual(
        trace.events[2].part_dict['part']['text'], 'HELLO_sub_trace_outer'
    )

  async def test_trace_references(self):
    p = SubTraceProcessor()
    input_part = content_api.ProcessorPart('world')
    # First call
    async with trace_file.SyncFileTrace(trace_dir=self.trace_dir):
      await streams.gather_stream(p(streams.stream_content([input_part])))

    json_files = [f for f in os.listdir(self.trace_dir) if f.endswith('.json')]
    self.assertTrue(len(json_files), 1)
    trace1_path = os.path.join(self.trace_dir, json_files[0])
    trace1 = trace_file.SyncFileTrace.load(trace1_path)
    self.assertTrue(trace1.events[1].is_input)
    self.assertEqual(trace1.events[1].part_dict['part']['text'], 'world')

    sub_trace1 = cast(trace_file.SyncFileTrace, trace1.events[0].sub_trace)
    self.assertIsNotNone(sub_trace1)
    self.assertTrue(sub_trace1.events[0].is_input)
    self.assertIsNotNone(sub_trace1.events[0].part_dict)

    # Second call with same part
    for f in os.listdir(self.trace_dir):
      os.remove(os.path.join(self.trace_dir, f))
    async with trace_file.SyncFileTrace(trace_dir=self.trace_dir):
      await streams.gather_stream(p(streams.stream_content([input_part])))
    json_files = [f for f in os.listdir(self.trace_dir) if f.endswith('.json')]
    self.assertTrue(len(json_files), 1)
    trace2_path = os.path.join(self.trace_dir, json_files[0])
    trace2 = trace_file.SyncFileTrace.load(trace2_path)
    self.assertTrue(trace2.events[1].is_input)
    self.assertIsNotNone(trace2.events[1].part_dict)

  async def test_trace_save_load(self):
    trace = trace_file.SyncFileTrace(name='test')
    await trace.add_input(content_api.ProcessorPart('in'))
    await trace.add_input(
        content_api.ProcessorPart.from_bytes(
            data=b'bytes',
            mimetype='image/jpeg',
        )
    )
    sub_trace = await trace.add_sub_trace(name='sub_test')
    await sub_trace.add_input(content_api.ProcessorPart('sub_in'))
    await sub_trace.add_output(content_api.ProcessorPart('sub_out'))
    await trace.add_output(content_api.ProcessorPart('out'))

    tmpdir = absltest.get_default_test_tmpdir()
    trace_path = os.path.join(tmpdir, 'trace.json')

    trace.save(trace_path)
    loaded_trace = trace_file.SyncFileTrace.load(trace_path)

    self.assertEqual(
        json.loads(trace.model_dump_json()),
        json.loads(loaded_trace.model_dump_json()),
    )

  async def test_save_html(self):
    p = SubTraceProcessor()
    trace_dir = os.getenv('TEST_UNDECLARED_OUTPUTS_DIR')

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
    wavfile.write(audio_bytes_io, sample_rate, audio_data)
    audio_part = content_api.ProcessorPart.from_bytes(
        data=audio_bytes_io.getvalue(),
        mimetype='audio/wav',
    )
    parts = [img_part, audio_part, content_api.ProcessorPart('hello')]
    async with trace_file.SyncFileTrace(trace_dir=trace_dir):
      await processor.apply_async(p, parts)

    html_files = [f for f in os.listdir(trace_dir) if f.endswith('.html')]
    self.assertTrue(len(html_files), 1)
    trace_path = os.path.join(trace_dir, html_files[0])
    self.assertTrue(os.path.exists(trace_path))


if __name__ == '__main__':
  absltest.main()
