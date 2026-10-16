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
"""Tests for the MiniMax text-to-speech processor."""

import json
import unittest

from absl.testing import parameterized
from genai_processors import content_api
from genai_processors.contrib import minimax_text_to_speech
import httpx


ProcessorPart = content_api.ProcessorPart


class MiniMaxTextToSpeechTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  async def test_non_streaming_request_and_response(self):
    def request_handler(request: httpx.Request):
      self.assertEqual(str(request.url), 'https://api.minimax.io/v1/t2a_v2')
      self.assertEqual(request.headers['authorization'], 'Bearer test-key')
      body = json.loads(request.content)
      self.assertEqual(
          body,
          {
              'model': 'speech-2.8-hd',
              'text': 'Hello',
              'stream': False,
              'output_format': 'hex',
              'audio_setting': {
                  'format': 'wav',
                  'sample_rate': 32000,
              },
              'voice_setting': {'voice_id': 'English_Graceful_Lady'},
              'language_boost': 'English',
              'pronunciation_dict': {'tone': ['read/(riːd)']},
              'voice_modify': {'pitch': 2},
              'subtitle_enable': True,
          },
      )
      return httpx.Response(
          200,
          json={
              'data': {'audio': '000102', 'status': 2},
              'base_resp': {'status_code': 0, 'status_msg': 'success'},
          },
      )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(request_handler)
    ) as client:
      tts = minimax_text_to_speech.MiniMaxTextToSpeech(
          api_key='test-key',
          client=client,
          voice_setting={'voice_id': 'English_Graceful_Lady'},
          language_boost='English',
          pronunciation_dict={'tone': ['read/(riːd)']},
          audio_setting={'format': 'wav', 'sample_rate': 32000},
          voice_modify={'pitch': 2},
          subtitle_enable=True,
          with_text_passthrough=False,
      )
      output = await tts(['Hello']).gather()

    self.assertEqual(
        output,
        [ProcessorPart(b'\x00\x01\x02', mimetype='audio/wav', role='model')],
    )

  async def test_china_endpoint_and_streaming_response(self):
    def request_handler(request: httpx.Request):
      self.assertEqual(str(request.url), 'https://api.minimaxi.com/v1/t2a_v2')
      self.assertTrue(json.loads(request.content)['stream'])
      return httpx.Response(
          200,
          text='\n'.join(
              [
                  'data: {"data":{"audio":"0102","status":1},'
                  '"base_resp":{"status_code":0}}',
                  'data: {"data":{"audio":"0304","status":1},'
                  '"base_resp":{"status_code":0}}',
                  'data: {"data":{"audio":"01020304","status":2},'
                  '"base_resp":{"status_code":0}}',
              ]
          ),
      )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(request_handler)
    ) as client:
      tts = minimax_text_to_speech.MiniMaxTextToSpeech(
          api_key='test-key',
          client=client,
          region='cn_zh',
          stream=True,
          audio_setting={'format': 'pcm'},
          with_text_passthrough=False,
      )
      output = await tts(['Hello']).gather()

    self.assertEqual(
        output,
        [
            ProcessorPart(b'\x01\x02', mimetype='audio/pcm', role='model'),
            ProcessorPart(b'\x03\x04', mimetype='audio/pcm', role='model'),
        ],
    )

  async def test_passes_through_non_text_and_optional_text(self):
    image = ProcessorPart(b'image', mimetype='image/png')
    async with httpx.AsyncClient() as client:
      tts = minimax_text_to_speech.MiniMaxTextToSpeech(
          api_key='test-key', client=client
      )
      output = await tts([image, '']).gather()
    self.assertEqual(output, [image, ProcessorPart('')])

  async def test_api_error_raises(self):
    def request_handler(_: httpx.Request):
      return httpx.Response(
          200,
          json={
              'data': None,
              'base_resp': {
                  'status_code': 1004,
                  'status_msg': 'authentication failed',
              },
          },
      )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(request_handler)
    ) as client:
      tts = minimax_text_to_speech.MiniMaxTextToSpeech(
          api_key='test-key', client=client, with_text_passthrough=False
      )
      with self.assertRaisesRegex(
          minimax_text_to_speech.MiniMaxTextToSpeechError, '1004'
      ):
        await tts(['Hello']).gather()

  @parameterized.named_parameters(
      ('bad_model', {'model': 'old-model'}, 'model'),
      ('bad_region', {'region': 'unknown'}, 'region'),
      ('bad_audio_format', {'audio_setting': {'format': 'aac'}}, 'audio'),
      ('stream_url', {'stream': True, 'output_format': 'url'}, 'hex'),
  )
  def test_rejects_invalid_configuration(self, kwargs, error_text):
    with self.assertRaisesRegex(ValueError, error_text):
      minimax_text_to_speech.MiniMaxTextToSpeech(api_key='test-key', **kwargs)


if __name__ == '__main__':
  unittest.main()
