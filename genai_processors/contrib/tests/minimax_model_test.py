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

"""Tests MiniMax text generation at the serialized HTTP boundary."""

import copy
import json
import unittest
from unittest import mock

from absl.testing import parameterized
from genai_processors import content_api
from genai_processors.contrib import minimax_model
import httpx


class MiniMaxModelTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  def response(self, message=None):
    return httpx.Response(
        200,
        json={
            'choices': [
                {
                    'message': message
                    or {
                        'role': 'assistant',
                        'content': 'Hello',
                        'reasoning_content': 'Thinking',
                    },
                    'finish_reason': 'stop',
                }
            ],
            'usage': {'total_tokens': 12},
            'model': minimax_model.DEFAULT_MODEL,
        },
    )

  async def test_default_request_uses_real_client_configuration(self):
    requests = []

    def handler(request):
      requests.append(request)
      return self.response()

    real_client = httpx.AsyncClient
    clients = []

    def create_client(**kwargs):
      client = real_client(transport=httpx.MockTransport(handler), **kwargs)
      clients.append(client)
      return client

    with mock.patch.object(httpx, 'AsyncClient', side_effect=create_client):
      async with minimax_model.MiniMaxModel(api_key='test-key') as model:
        output = await model('Hi').gather()
    self.assertTrue(clients[0].is_closed)
    self.assertLen(requests, 1)
    request = requests[0]
    self.assertEqual(request.method, 'POST')
    self.assertEqual(
        str(request.url), 'https://api.minimax.io/v1/chat/completions'
    )
    self.assertEqual(request.headers['authorization'], 'Bearer test-key')
    self.assertEqual(request.headers['content-type'], 'application/json')
    self.assertEqual(
        json.loads(request.content),
        {
            'model': 'MiniMax-M3.1-Flash',
            'messages': [{'role': 'user', 'content': 'Hi'}],
            'stream': False,
        },
    )
    self.assertEqual(output[0].text, 'Hello')
    self.assertEqual(output[0].metadata['usage'], {'total_tokens': 12})
    self.assertEqual(
        minimax_model.RECOMMENDED_MODELS, ('MiniMax-M3.1-Flash', 'MiniMax-M3')
    )

  @parameterized.parameters(
      (
          {'region': 'china'},
          'https://api.minimaxi.com/v1/chat/completions',
          'MiniMax-M3.1-Flash',
      ),
      (
          {'model_name': 'MiniMax-M3'},
          'https://api.minimax.io/v1/chat/completions',
          'MiniMax-M3',
      ),
      (
          {'model_name': 'MiniMax-M2.7'},
          'https://api.minimax.io/v1/chat/completions',
          'MiniMax-M2.7',
      ),
      (
          {
              'base_url': 'https://custom.invalid/v2/',
              'model_name': 'saved-model',
          },
          'https://custom.invalid/v2/chat/completions',
          'saved-model',
      ),
  )
  async def test_region_and_explicit_model(self, kwargs, url, model_id):
    def handler(request):
      self.assertEqual(str(request.url), url)
      self.assertEqual(json.loads(request.content)['model'], model_id)
      return self.response()

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as client:
      async with minimax_model.MiniMaxModel(
          api_key='test-key', client=client, **kwargs
      ) as model:
        await model('Hi').gather()
      self.assertFalse(client.is_closed)

  @parameterized.parameters('low', 'medium', 'high', 'xhigh', 'max')
  async def test_flash_effort(self, effort):
    def handler(request):
      body = json.loads(request.content)
      self.assertEqual(body['thinking'], {'type': 'adaptive'})
      self.assertEqual(body['reasoning_effort'], effort)
      self.assertEqual(body['max_completion_tokens'], 4096)
      self.assertNotIn('service_tier', body)
      self.assertNotIn('reasoning_split', body)
      return self.response()

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as client:
      model = minimax_model.MiniMaxModel(
          api_key='test-key',
          client=client,
          thinking='adaptive',
          reasoning_effort=effort,
          max_completion_tokens=4096,
      )
      await model('Hi').gather()

  async def test_m3_disabled_thinking(self):
    def handler(request):
      body = json.loads(request.content)
      self.assertEqual(body['model'], 'MiniMax-M3')
      self.assertEqual(body['thinking'], {'type': 'disabled'})
      self.assertNotIn('reasoning_effort', body)
      return self.response()

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as client:
      model = minimax_model.MiniMaxModel(
          api_key='test-key',
          client=client,
          model_name='MiniMax-M3',
          thinking='disabled',
      )
      await model('Hi').gather()

  async def test_full_assistant_history_round_trip(self):
    bodies = []
    message = {
        'role': 'assistant',
        'content': '<think>Legacy</think>Hello',
        'reasoning_content': 'Full reasoning',
        'custom_field': 'preserved',
    }

    def handler(request):
      bodies.append(json.loads(request.content))
      return self.response(message)

    history = [content_api.ProcessorPart('Be concise', role='system'), 'Hi']
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as client:
      model = minimax_model.MiniMaxModel(api_key='test-key', client=client)
      output = await model(history).gather()
      original_metadata = copy.deepcopy(output[0].metadata)
      await model([*history, *output, 'Continue']).gather()
    self.assertEqual(
        bodies[1]['messages'],
        [
            {'role': 'system', 'content': 'Be concise'},
            {'role': 'user', 'content': 'Hi'},
            message,
            {'role': 'user', 'content': 'Continue'},
        ],
    )
    self.assertEqual(output[0].metadata, original_metadata)

  async def test_empty_input_and_media_do_not_send_requests(self):
    def handler(request):
      self.fail('Unexpected HTTP request')

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as client:
      model = minimax_model.MiniMaxModel(api_key='test-key', client=client)
      self.assertEmpty(await model([]).gather())
      image = content_api.ProcessorPart(b'image', mimetype='image/png')
      with self.assertRaisesRegex(ValueError, 'text parts only'):
        await model(['Hi', image]).gather()
      self.assertEqual(image.bytes, b'image')

  async def test_http_error_propagates(self):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(429))
    ) as client:
      model = minimax_model.MiniMaxModel(api_key='test-key', client=client)
      with self.assertRaises(httpx.HTTPStatusError):
        await model('Hi').gather()

  @parameterized.parameters(
      {'thinking': 'disabled'},
      {'reasoning_effort': 'none'},
      {'thinking': 'enabled'},
      {'region': 'unknown'},
      {'model_name': ''},
      {'max_completion_tokens': 0},
  )
  def test_invalid_configuration_fails_before_client_creation(self, **kwargs):
    with mock.patch.object(httpx, 'AsyncClient') as create_client:
      with self.assertRaises(ValueError):
        minimax_model.MiniMaxModel(api_key='test-key', **kwargs)
      create_client.assert_not_called()


if __name__ == '__main__':
  unittest.main()
