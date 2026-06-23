# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import asyncio
from collections.abc import AsyncIterable
import time
import unittest.mock
from genai_processors import content_api
from genai_processors import processor
from genai_processors import streams
from genai_processors.dev import latency
import pytest


@pytest.mark.asyncio
async def test_prober_structure():
  """Tests that Prober is scheduled correctly."""

  pipeline = latency.Prober('start', interval_seconds=0.1)

  # Send a few parts to establish a baseline for FPS
  input_stream = streams.stream_content(['test'] * 11, with_delay_sec=0.02)
  results = await pipeline(input_stream).gather()

  for part in results:
    assert part.text == 'test'

  assert len(results) == 11
  assert 'prober' in results[5].metadata
  assert 'prober' in results[10].metadata
  delta_time = (
      results[10].metadata['prober']['markers'][0][1]
      - results[5].metadata['prober']['markers'][0][1]
  )
  assert delta_time == pytest.approx(0.1, abs=0.01)


@pytest.mark.asyncio
async def test_probe_checkpoint_trace():
  """Tests that ProbeCheckpoint builds a trace of markers with FPS."""
  pipeline = latency.ProbeCheckpoint('step1') + latency.ProbeCheckpoint('step2')
  start_time = 100.0
  # Prober now starts with one marker
  input_stream = [
      'initial',
      content_api.ProcessorPart(
          'test',
          metadata={
              'prober': {
                  'start_time': start_time,
                  'markers': [('start', start_time, 0, 50.0)],
              }
          },
      ),
  ]
  result = await pipeline(input_stream).gather()

  # Check that we have the original marker + the extra 2.
  markers = result[1].metadata['prober']['markers']
  assert len(markers) == 3
  assert markers[1][0] == 'step1'
  assert markers[2][0] == 'step2'
  assert markers[1][3] > 0  # FPS should be present


@pytest.mark.asyncio
async def test_fps_calculation():
  """Verify that FPS is calculated correctly based on part count."""
  pipeline = (
      latency.Prober('start', interval_seconds=1.0).to_processor()
      + latency.ProbeCheckpoint('step').to_processor()
  )
  # The first part sets the marker for the prober (start), we collect
  # the prober info after 5 parts, so in total we have 6 parts.
  input_stream = streams.stream_content(['test'] * 6, with_delay_sec=0.2)
  result = await pipeline(input_stream).gather()
  fps = result[-1].metadata['prober']['markers'][0][3]
  assert fps == pytest.approx(5.0, abs=0.1)


@pytest.mark.asyncio
async def test_probe_checkpoint_substream():
  """Tests that ProbeCheckpoint filters by substream_name."""
  pipeline = latency.ProbeCheckpoint('step', substream_name='server')

  content = [
      content_api.ProcessorPart(
          'test',
          metadata={'prober': {'start_time': time.monotonic(), 'markers': []}},
      ),
      content_api.ProcessorPart(
          'test',
          metadata={'prober': {'start_time': time.monotonic(), 'markers': []}},
          substream_name='server',
      ),
  ]
  result = await pipeline(content).gather()
  assert len(result[0].metadata['prober']['markers']) == 0
  assert len(result[1].metadata['prober']['markers']) == 1


def test_log_arrival_trace():
  """Tests log_arrival logs FPS and duration."""
  with unittest.mock.patch(
      'genai_processors.dev.latency.logging.info'
  ) as mock_log:
    part = content_api.ProcessorPart(
        'test',
        metadata={
            'prober': {
                'start_time': 100.0,
                'markers': [
                    ('step1', 100.01, 0.01, 50.0),
                    ('step2', 100.03, 0.02, 48.5),
                ],
            }
        },
    )

    latency.Prober.log_arrival(part)

    mock_log.assert_called()
    assert 'default' == mock_log.call_args[0][1]
    trace_arg = mock_log.call_args[0][2]
    assert 'step1 (+10.0ms, 50.0fps)' in trace_arg
    assert 'step2 (+20.0ms, 48.5fps)' in trace_arg


@pytest.mark.asyncio
async def test_throttler_drop_strategy():
  """Tests that Throttler protects probers and drops the oldest non-prober part."""
  # max_size=1 in the throttler + 1 in hand-off queue, i.e. we can
  # handle max_size + 2 in flight but then we'll start dropping parts.

  @processor.processor_function
  async def slow_consumer(
      content: processor.ProcessorStream,
  ) -> AsyncIterable[content_api.ProcessorPartTypes]:
    async for p in content:
      await asyncio.sleep(0.6)
      yield p

  pipeline = latency.Throttler('test', max_size=1) + slow_consumer
  input_stream = streams.stream_content(
      [
          # processed immediately, not blocking - arrives at 0s
          'p1',
          # goes to output queue - arrives at 0.1s
          content_api.ProcessorPart('p2', metadata={'prober': {}}),
          # goes to output queue - arrives at 0.2s
          'p3',
          # buffered - arrives at 0.3sec
          'p4',
          # buffered, drops p4 - arrives at 0.4s
          content_api.ProcessorPart('p5', metadata={'prober': {}}),
          # buffered, drops p5 because output queue is prober - arrives at 0.5s
          content_api.ProcessorPart('p6', metadata={'prober': {}}),
      ],
      with_delay_sec=0.1,
  )

  results = await pipeline(input_stream).gather()

  names = [p.text for p in results if p]
  # See explanation above.
  assert names == ['p1', 'p2', 'p6']


@pytest.mark.asyncio
async def test_probe_log():
  """Tests that ProbeLog logs arrival and yields a latency part."""
  probe_log = latency.ProbeLog(substream_name='server')

  part = content_api.ProcessorPart(
      'test',
      metadata={
          'prober': {'start_time': 100.0, 'markers': [('start', 0, 0, 0)]}
      },
  )
  with unittest.mock.patch.object(latency.logging, 'log') as mock_info:
    results = await probe_log(part).gather()
    # We have two parts, the original one and a new one under the
    # latency substream
    assert len(results) == 2
    assert results[1].substream_name == 'latency'
    # first arg of log info is about latency
    assert 'Latency Trace' in mock_info.call_args_list[0][0][1]


@pytest.mark.asyncio
async def test_throttler_warning_log():
  """Tests that Throttler logs a warning when dropping non-prober parts."""

  @processor.processor_function
  async def slow_consumer(
      content: processor.ProcessorStream,
  ) -> AsyncIterable[content_api.ProcessorPartTypes]:
    async for p in content:
      await asyncio.sleep(0.5)
      yield p

  throttler = latency.Throttler('test_log', max_size=1, log_interval_sec=0.1)
  pipeline = throttler + slow_consumer
  input_stream = streams.stream_content(
      [f'p{i}' for i in range(10)], with_delay_sec=0.025
  )

  with unittest.mock.patch.object(latency.logging, 'warning') as mock_warning:
    await pipeline(input_stream).gather()
    mock_warning.assert_called()
    assert 'Throttler' in mock_warning.call_args[0][0]
    # 4 = 3 drops every 0.03 secs + 1 drop after 0.1 secs
    assert 4 == mock_warning.call_args[0][2]
