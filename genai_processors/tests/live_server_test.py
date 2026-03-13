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

"""Tests for live_server."""

import asyncio
import base64
import json
from typing import Any, AsyncIterable
import unittest

from absl.testing import absltest
from absl.testing import parameterized
from genai_processors import content_api
from genai_processors import processor
from genai_processors import streams
from genai_processors.dev import live_server
from websockets.exceptions import ConnectionClosed


class MockWebsocket:
  """Mocks a websockets ServerConnection."""

  def __init__(self, inputs: list[str]):
    self.inputs = inputs
    self.outputs = []
    self.is_closed = False

  async def __aiter__(self):
    while self.inputs:
      yield self.inputs.pop(0)

  async def send(self, data: str):
    if self.is_closed:
      raise ConnectionClosed(None, None)
    self.outputs.append(data)
    if not self.inputs:
      self.is_closed = True


@processor.processor_function
async def echo_processor(
    content: processor.ProcessorStream,
) -> AsyncIterable[content_api.ProcessorPartTypes]:
  """Simple processor that echoes text and reports mime types."""
  async for part in content:
    if content_api.is_text(part.mimetype):
      yield f'echo: {part.text}'
    else:
      yield f'received: {part.mimetype}'


class LiveServerTest(parameterized.TestCase, unittest.IsolatedAsyncioTestCase):

  async def test_basic_text_exchange(self):
    """Tests basic text exchange with the live server."""
    client_input = json.dumps({
        'part': {'text': 'hello'},
        'role': 'user',
    })
    expected_outputs = [
        {
            'metadata': {},
            'mimetype': 'text/plain',
            'part': {'text': 'echo: hello'},
            'role': '',
            'substream_name': '',
        },
    ]
    ws = MockWebsocket([client_input])

    def processor_factory(config: dict[str, Any]) -> processor.Processor:
      del config  # Unused.
      return echo_processor.to_processor()

    await live_server.live_server(
        processor_factory=processor_factory,
        trace_dir=None,
        max_size_bytes=None,
        ais_websocket=ws,  # pytype: disable=wrong-arg-types
    )

    actual_outputs = [json.loads(output) for output in ws.outputs]
    self.assertEqual(actual_outputs, expected_outputs)

  async def test_image_part_base64(self):
    """Tests image part with base64 encoding."""
    client_input = json.dumps({
        'part': {
            'inline_data': {
                'data': base64.b64encode(b'fakeimage').decode('utf-8'),
                'mime_type': 'image/png',
            }
        },
        'role': 'user',
    })
    expected_outputs = [
        {
            'metadata': {},
            'mimetype': 'text/plain',
            'part': {'text': 'received: image/png'},
            'role': '',
            'substream_name': '',
        },
    ]
    ws = MockWebsocket([client_input])

    await live_server.live_server(
        processor_factory=lambda _: echo_processor,
        trace_dir=None,
        max_size_bytes=None,
        ais_websocket=ws,  # pytype: disable=wrong-arg-types
    )

    actual_outputs = [json.loads(output) for output in ws.outputs]
    self.assertEqual(actual_outputs, expected_outputs)

  async def test_config_update(self):
    """Tests config update re-initializing the processor."""
    config_msg = json.dumps({
        'mimetype': 'application/x-config',
        'metadata': {'model_name': 'gemini-pro'},
    })
    text_msg = json.dumps({'part': {'text': 'hi'}})
    ws = MockWebsocket([config_msg, text_msg])
    expected_outputs = [
        # Health check after config update as the live server restarts.
        {
            'metadata': {'health_check': True},
            'mimetype': 'text/plain',
            'part': {'text': ''},
            'role': '',
            'substream_name': '',
        },
        {
            'metadata': {},
            'mimetype': 'text/plain',
            'part': {'text': 'echo: hi'},
            'role': '',
            'substream_name': '',
        },
    ]

    configs_received = []

    def processor_factory(config: dict[str, Any]) -> processor.Processor:
      configs_received.append(config)
      return echo_processor.to_processor()

    await live_server.live_server(
        processor_factory=processor_factory,
        trace_dir=None,
        max_size_bytes=None,
        ais_websocket=ws,  # pytype: disable=wrong-arg-types
    )

    # Initial factory call (empty config)
    # Second factory call (after config update)
    self.assertEqual(configs_received, [{}, {'model_name': 'gemini-pro'}])

    # Check that it still processed the text after config update
    actual_outputs = [json.loads(output) for output in ws.outputs]
    self.assertEqual(actual_outputs, expected_outputs)

  async def test_reset_command(self):
    """Tests the reset command."""
    reset_msg = json.dumps({
        'mimetype': 'application/x-command',
        'metadata': {'command': 'reset'},
    })
    text_msg = json.dumps({'part': {'text': 'hi'}})
    ws = MockWebsocket([reset_msg, text_msg])

    factory_calls = 0

    def processor_factory(config: dict[str, Any]) -> processor.Processor:
      nonlocal factory_calls
      factory_calls += 1
      return echo_processor.to_processor()

    await live_server.live_server(
        processor_factory=processor_factory,
        trace_dir=None,
        max_size_bytes=None,
        ais_websocket=ws,  # pytype: disable=wrong-arg-types
    )

    # Called twice: initial and after reset
    self.assertEqual(factory_calls, 2)

  async def test_mic_off_signal(self):
    """Tests that mic: off translates to audio_stream_end."""
    mic_off = json.dumps({
        'mimetype': 'application/x-state',
        'metadata': {'mic': 'off'},
    })
    expected_outputs = [
        {
            'part': {'text': ''},
            'role': 'user',
            'substream_name': 'realtime',
            'mimetype': 'text/plain',
            'metadata': {'audio_stream_end': True},
        },
    ]
    ws = MockWebsocket([mic_off])

    received_parts = []

    @processor.processor_function
    async def capture_processor(
        content: processor.ProcessorStream,
    ) -> AsyncIterable[content_api.ProcessorPartTypes]:
      async for part in content:
        received_parts.append(part)
        yield part

    await live_server.live_server(
        processor_factory=lambda _: capture_processor,
        trace_dir=None,
        max_size_bytes=None,
        ais_websocket=ws,  # pytype: disable=wrong-arg-types
    )
    actual_outputs = [json.loads(output) for output in ws.outputs]
    self.assertEqual(actual_outputs, expected_outputs)

  async def test_generation_complete_metadata(self):
    """Tests special metadata handling in server responses."""

    @processor.processor_function
    async def completion_processor(
        content: processor.ProcessorStream,
    ) -> AsyncIterable[content_api.ProcessorPartTypes]:
      async for _ in content:
        # Send a part with generation_complete
        yield content_api.ProcessorPart(
            '', metadata={'generation_complete': True}
        )

    ws = MockWebsocket([json.dumps({'part': {'text': 'go'}})])
    expected_outputs = [
        {
            'metadata': {'generation_complete': True},
            'mimetype': 'application/x-state',
            'part': {'text': ''},
            'role': '',
            'substream_name': '',
        },
    ]

    await live_server.live_server(
        processor_factory=lambda _: completion_processor,
        trace_dir=None,
        max_size_bytes=None,
        ais_websocket=ws,  # pytype: disable=wrong-arg-types
    )
    actual_outputs = [json.loads(output) for output in ws.outputs]
    self.assertEqual(actual_outputs, expected_outputs)

  async def test_malformed_json_handling(self):
    """Tests that the server handles malformed JSON without crashing."""
    malformed = 'not json'
    valid = json.dumps({'part': {'text': 'ok'}})
    ws = MockWebsocket([malformed, valid])
    expected_outputs = [
        # Server restarts after malformed JSON, so we expect a health check.
        {
            'metadata': {'health_check': True},
            'mimetype': 'text/plain',
            'part': {'text': ''},
            'role': '',
            'substream_name': '',
        },
        {
            'metadata': {},
            'mimetype': 'text/plain',
            'part': {'text': 'echo: ok'},
            'role': '',
            'substream_name': '',
        },
    ]

    await live_server.live_server(
        processor_factory=lambda _: echo_processor,
        trace_dir=None,
        max_size_bytes=None,
        ais_websocket=ws,  # pytype: disable=wrong-arg-types
    )

    # The malformed JSON triggers an exception in receive(),
    # which is caught by the loop in live_server.
    # We should still see the response for the second valid message.
    # Note: live_server restarts the processor on exception.
    actual_outputs = [json.loads(output) for output in ws.outputs]
    self.assertEqual(actual_outputs, expected_outputs)


if __name__ == '__main__':
  absltest.main()
