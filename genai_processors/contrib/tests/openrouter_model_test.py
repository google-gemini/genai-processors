# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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

"""Tests for OpenRouter model processor."""

import enum
import http
import json
import unittest
from unittest import mock

from absl.testing import parameterized
from genai_processors import content_api
from genai_processors import processor
from genai_processors.contrib import openrouter_model
from google.genai import types as genai_types
import httpx


class ResponseEnum(enum.StrEnum):
  GOOD = 'good'
  EXCELLENT = 'excellent'


class OpenRouterModelTest(parameterized.TestCase):

  def test_basic_inference(self):
    def request_handler(request: httpx.Request):
      self.assertEqual(
          str(request.url), 'https://openrouter.ai/api/v1/chat/completions'
      )
      self.assertEqual(request.headers['authorization'], 'Bearer test-api-key')
      self.assertEqual(request.headers['content-type'], 'application/json')
      self.assertEqual(request.headers['user-agent'], 'genai-processors')

      request_body = json.loads(request.content.decode('utf-8'))
      self.assertEqual(request_body['model'], 'openai/gpt-4o')
      self.assertEqual(request_body['stream'], True)
      self.assertEqual(
          request_body['messages'],
          [{'role': 'user', 'content': 'Hello, how are you?'}],
      )

      response_lines = [
          (
              'data: {"choices": [{"delta": {"content": "I am"},'
              ' "finish_reason": null}]}'
          ),
          (
              'data: {"choices": [{"delta": {"content": " doing well"},'
              ' "finish_reason": null}]}'
          ),
          (
              'data: {"choices": [{"delta": {}, "finish_reason": "stop"}],'
              ' "usage": {"prompt_tokens": 5, "completion_tokens": 4,'
              ' "total_tokens": 9}, "model": "openai/gpt-4o"}'
          ),
          'data: [DONE]',
      ]
      return httpx.Response(
          http.HTTPStatus.OK, content='\n'.join(response_lines).encode('utf-8')
      )

    # Mock client with proper headers that match what OpenRouter model sets.
    mock_client = httpx.AsyncClient(
        transport=httpx.MockTransport(request_handler),
        base_url='https://openrouter.ai/api/v1',
        headers={
            'Authorization': 'Bearer test-api-key',
            'Content-Type': 'application/json',
            'User-Agent': 'genai-processors',
        },
    )

    with mock.patch.object(httpx, 'AsyncClient', return_value=mock_client):
      model = openrouter_model.OpenRouterModel(
          api_key='test-api-key',
          model_name='openai/gpt-4o',
      )
      output = processor.apply_sync(model, ['Hello, how are you?'])

    self.assertEqual(content_api.as_text(output), 'I am doing well')
    # Check that the last part has the correct metadata.
    last_part = output[-1]
    self.assertEqual(last_part.metadata['finish_reason'], 'stop')
    self.assertEqual(last_part.metadata['turn_complete'], True)
    self.assertEqual(last_part.metadata['model'], 'openai/gpt-4o')
    self.assertEqual(last_part.metadata['usage']['total_tokens'], 9)

  def test_headers_and_site_info(self):
    def request_handler(request: httpx.Request):
      self.assertEqual(
          str(request.url), 'https://openrouter.ai/api/v1/chat/completions'
      )
      self.assertEqual(request.headers['authorization'], 'Bearer test-api-key')
      self.assertEqual(request.headers['http-referer'], 'https://mysite.com')
      self.assertEqual(request.headers['x-title'], 'My App')

      response_lines = [
          'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}',
          'data: [DONE]',
      ]
      return httpx.Response(
          http.HTTPStatus.OK, content='\n'.join(response_lines).encode('utf-8')
      )

    # Mock client with site-specific headers.
    mock_client = httpx.AsyncClient(
        transport=httpx.MockTransport(request_handler),
        base_url='https://openrouter.ai/api/v1',
        headers={
            'Authorization': 'Bearer test-api-key',
            'Content-Type': 'application/json',
            'User-Agent': 'genai-processors',
            'HTTP-Referer': 'https://mysite.com',
            'X-Title': 'My App',
        },
    )

    with mock.patch.object(httpx, 'AsyncClient', return_value=mock_client):
      model = openrouter_model.OpenRouterModel(
          api_key='test-api-key',
          model_name='test-model',
          site_url='https://mysite.com',
          site_name='My App',
      )
      processor.apply_sync(model, ['test'])

  def test_config_parameters(self):
    def request_handler(request: httpx.Request):
      request_body = json.loads(request.content.decode('utf-8'))
      self.assertEqual(request_body['temperature'], 0.8)
      self.assertEqual(request_body['max_tokens'], 100)
      self.assertEqual(request_body['top_p'], 0.9)

      response_lines = [
          'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}',
          'data: [DONE]',
      ]
      return httpx.Response(
          http.HTTPStatus.OK, content='\n'.join(response_lines).encode('utf-8')
      )

    mock_client = httpx.AsyncClient(
        transport=httpx.MockTransport(request_handler),
        base_url='https://openrouter.ai/api/v1',
        headers={
            'Authorization': 'Bearer test-api-key',
            'Content-Type': 'application/json',
            'User-Agent': 'genai-processors',
        },
    )

    with mock.patch.object(httpx, 'AsyncClient', return_value=mock_client):
      model = openrouter_model.OpenRouterModel(
          api_key='test-api-key',
          model_name='test-model',
          generate_content_config=openrouter_model.GenerateContentConfig(
              temperature=0.8,
              max_tokens=100,
              top_p=0.9,
          ),
      )
      processor.apply_sync(model, ['test'])

  def test_response_schema(self):
    def request_handler(request: httpx.Request):
      request_body = json.loads(request.content.decode('utf-8'))
      self.assertIn('response_format', request_body)
      self.assertEqual(request_body['response_format']['type'], 'json_object')
      self.assertIn('schema', request_body['response_format'])

      response_lines = [
          (
              'data: {"choices": [{"delta": {"content": "{\\"result\\":'
              ' \\"good\\"}"}, "finish_reason": "stop"}]}'
          ),
          'data: [DONE]',
      ]
      return httpx.Response(
          http.HTTPStatus.OK, content='\n'.join(response_lines).encode('utf-8')
      )

    mock_client = httpx.AsyncClient(
        transport=httpx.MockTransport(request_handler),
        base_url='https://openrouter.ai/api/v1',
        headers={
            'Authorization': 'Bearer test-api-key',
            'Content-Type': 'application/json',
            'User-Agent': 'genai-processors',
        },
    )

    with mock.patch.object(httpx, 'AsyncClient', return_value=mock_client):
      model = openrouter_model.OpenRouterModel(
          api_key='test-api-key',
          model_name='test-model',
          generate_content_config=openrouter_model.GenerateContentConfig(
              response_schema=ResponseEnum,
          ),
      )
      output = processor.apply_sync(model, ['How is the weather?'])

    self.assertEqual(content_api.as_text(output), '{"result": "good"}')

  def test_function_calling(self):
    call_count = 0

    def request_handler(request: httpx.Request):
      del request  # Unsued.
      nonlocal call_count
      call_count += 1

      if call_count == 1:
        # First request - return function call.
        response_lines = [
            (
                'data: {"choices": [{"delta": {"function_call": {"name":'
                ' "get_weather"}}, "finish_reason": null}]}'
            ),
            (
                'data: {"choices": [{"delta": {"function_call": {"arguments":'
                ' "{\\"location\\": \\"Boston\\"}"}}, "finish_reason": null}]}'
            ),
            (
                'data: {"choices": [{"delta": {}, "finish_reason":'
                ' "function_call"}]}'
            ),
            'data: [DONE]',
        ]
      else:
        # Second request with function response - return text.
        response_lines = [
            (
                'data: {"choices": [{"delta": {"content": "The weather in'
                ' Boston is 72°F and sunny."}, "finish_reason": "stop"}]}'
            ),
            'data: [DONE]',
        ]

      return httpx.Response(
          http.HTTPStatus.OK, content='\n'.join(response_lines).encode('utf-8')
      )

    mock_client = httpx.AsyncClient(
        transport=httpx.MockTransport(request_handler),
        base_url='https://openrouter.ai/api/v1',
        headers={
            'Authorization': 'Bearer test-api-key',
            'Content-Type': 'application/json',
            'User-Agent': 'genai-processors',
        },
    )

    with mock.patch.object(httpx, 'AsyncClient', return_value=mock_client):
      weather_tool = genai_types.Tool(
          function_declarations=[
              genai_types.FunctionDeclaration(
                  name='get_weather',
                  description='Get the current weather',
                  parameters=genai_types.Schema(
                      type=genai_types.Type.OBJECT,
                      properties={
                          'location': genai_types.Schema(
                              type=genai_types.Type.STRING,
                              description=(
                                  'The city and state, e.g. San Francisco, CA'
                              ),
                          )
                      },
                      required=['location'],
                  ),
              )
          ]
      )

      model = openrouter_model.OpenRouterModel(
          api_key='test-api-key',
          model_name='test-model',
          generate_content_config=openrouter_model.GenerateContentConfig(
              tools=[weather_tool]
          ),
      )

      # First request.
      conversation = ['What is the weather in Boston?']
      output = processor.apply_sync(model, conversation)

      self.assertLen(output, 2)  # Function call + end marker.
      function_call_part = output[0]
      self.assertIsNotNone(function_call_part.function_call)
      self.assertEqual(function_call_part.function_call.name, 'get_weather')
      self.assertEqual(
          function_call_part.function_call.args, {'location': 'Boston'}
      )

      # Add function response and make second request.
      conversation.extend(output)
      conversation.append(
          content_api.ProcessorPart.from_function_response(
              name='get_weather',
              response={'weather': '72°F and sunny'},
          )
      )

      output = processor.apply_sync(model, conversation)
      self.assertEqual(
          content_api.as_text(output),
          'The weather in Boston is 72°F and sunny.',
      )

  def test_image_input(self):
    def request_handler(request: httpx.Request):
      request_body = json.loads(request.content.decode('utf-8'))
      message = request_body['messages'][0]

      self.assertEqual(message['role'], 'user')
      self.assertIsInstance(message['content'], list)
      self.assertLen(message['content'], 1)

      content_item = message['content'][0]
      self.assertEqual(content_item['type'], 'image_url')
      self.assertIn('image_url', content_item)
      self.assertTrue(
          content_item['image_url']['url'].startswith('data:image/png;base64,')
      )

      response_lines = [
          (
              'data: {"choices": [{"delta": {"content": "I see a test image."},'
              ' "finish_reason": "stop"}]}'
          ),
          'data: [DONE]',
      ]
      return httpx.Response(
          http.HTTPStatus.OK, content='\n'.join(response_lines).encode('utf-8')
      )

    mock_client = httpx.AsyncClient(
        transport=httpx.MockTransport(request_handler),
        base_url='https://openrouter.ai/api/v1',
        headers={
            'Authorization': 'Bearer test-api-key',
            'Content-Type': 'application/json',
            'User-Agent': 'genai-processors',
        },
    )

    with mock.patch.object(httpx, 'AsyncClient', return_value=mock_client):
      model = openrouter_model.OpenRouterModel(
          api_key='test-api-key',
          model_name='test-model',
      )

      # Create a simple PNG image bytes.
      png_bytes = b'PNG\x47\x0D\x0A\x1A\x0A'

      output = processor.apply_sync(
          model, [content_api.ProcessorPart(png_bytes, mimetype='image/png')]
      )

    self.assertEqual(content_api.as_text(output), 'I see a test image.')

  def test_error_handling(self):
    def request_handler(request: httpx.Request):
      del request  # Unused.
      error_response = {'error': {'message': 'Invalid API key', 'code': 401}}
      return httpx.Response(
          http.HTTPStatus.UNAUTHORIZED,
          content=json.dumps(error_response).encode('utf-8'),
      )

    mock_client = httpx.AsyncClient(
        transport=httpx.MockTransport(request_handler),
        base_url='https://openrouter.ai/api/v1',
        headers={
            'Authorization': 'Bearer invalid-key',
            'Content-Type': 'application/json',
            'User-Agent': 'genai-processors',
        },
    )

    with mock.patch.object(httpx, 'AsyncClient', return_value=mock_client):
      model = openrouter_model.OpenRouterModel(
          api_key='invalid-key',
          model_name='test-model',
      )

      with self.assertRaises(httpx.HTTPStatusError) as context:
        processor.apply_sync(model, ['test'])

      self.assertIn('Invalid API key', str(context.exception))


if __name__ == '__main__':
  unittest.main()
