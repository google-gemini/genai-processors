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

import unittest
from unittest import mock
from absl.testing import parameterized
from genai_processors import content_api
from genai_processors import processor
from genai_processors import streams
from genai_processors.contrib.langchain_model import LangChainModel
from langchain_core import messages as langchain_messages
from langchain_core.prompts import ChatPromptTemplate


class MockChunk:
  """Lightweight stand-in for a LangChain chunk with only a `.content` attribute."""

  def __init__(self, content: str):
    self.content = content


class LangChainModelTest(
    unittest.IsolatedAsyncioTestCase, parameterized.TestCase
):
  """End-to-end tests for the LangChainModel contrib module."""

  def test_text_only_sync(self):
    """Single-turn, non-streaming inference via `apply_sync` with a single text part."""
    mock_llm = mock.Mock(model='mock-model')
    mock_llm.astream.return_value = streams.stream_content(
        [MockChunk('Hello'), MockChunk(' world')]
    )
    proc = LangChainModel(model=mock_llm)
    output = processor.apply_sync(proc, ['Hi there'])

    self.assertLen(output, 2)
    self.assertEqual(output[0].text, 'Hello')
    self.assertEqual(output[0].role, 'model')
    self.assertEqual(output[0].metadata, {'model': 'mock-model'})
    self.assertEqual(output[1].text, ' world')
    self.assertEqual(output[1].role, 'model')
    self.assertEqual(output[1].metadata, {'model': 'mock-model'})

  def test_streaming_sync(self):
    """A turn with multiple text parts that streams three chunks in sequence."""
    mock_llm = mock.Mock(model='mock-model')
    mock_llm.astream.return_value = streams.stream_content(
        [MockChunk('A'), MockChunk('B'), MockChunk('C')]
    )
    proc = LangChainModel(model=mock_llm)
    output = processor.apply_sync(proc, ['Question 1', 'Question 2'])

    self.assertEqual(content_api.as_text(output), 'ABC')

  async def test_streaming_async(self):
    """Verify roles and image handling via astream call inputs."""
    mock_llm = mock.Mock(model='mock-model')

    mock_llm.astream.return_value = streams.stream_content(
        [MockChunk('mocked')]
    )

    proc = LangChainModel(model=mock_llm)

    async def input_stream():
      yield content_api.ProcessorPart(
          value='System prompt',
          mimetype='text/plain',
          role='system',
      )
      yield content_api.ProcessorPart(
          value='User text',
          mimetype='text/plain',
          role='user',
      )
      yield content_api.ProcessorPart(
          value=b'\x89PNG\r\n\x1a\n',
          mimetype='image/png',
          role='user',
      )

    output = []
    async for part in proc(input_stream()):
      output.append(part)

    self.assertLen(output, 1)
    self.assertEqual(output[0].text, 'mocked')
    self.assertEqual(output[0].role, 'model')

    called_args = mock_llm.astream.call_args[0][0]
    self.assertLen(called_args, 2)

    self.assertIsInstance(called_args[0], langchain_messages.SystemMessage)
    self.assertEqual(called_args[0].content, 'System prompt')

    self.assertIsInstance(called_args[1], langchain_messages.HumanMessage)
    self.assertEqual(
        called_args[1].content,
        [
            {'type': 'text', 'text': 'User text'},
            {
                'type': 'image_url',
                'image_url': {'url': 'data:image/png;base64,iVBORw0KGgo='},
            },
        ],
    )

  def test_multimodal_message_conversion(self):
    """Multiple parts of the same role are grouped into one LangChain message.

    If multiple fragments are present the order must be preserved.
    """
    mock_llm = mock.Mock(model='mock-model')
    proc = LangChainModel(model=mock_llm)

    parts = [
        content_api.ProcessorPart(
            value='Caption me',
            mimetype='text/plain',
            role='user',
        ),
        content_api.ProcessorPart(
            value=b'\x89PNG\r\n\x1a\n',
            mimetype='image/png',
            role='user',
        ),
        content_api.ProcessorPart(
            value='System instruction',
            mimetype='text/plain',
            role='system',
        ),
    ]
    content = content_api.ProcessorContent(parts)
    msgs = proc._convert_to_langchain_messages(content.all_parts)

    self.assertLen(msgs, 2)
    self.assertIsInstance(msgs[0], langchain_messages.HumanMessage)
    self.assertEqual(
        msgs[0].content,
        [
            {'type': 'text', 'text': 'Caption me'},
            {
                'type': 'image_url',
                'image_url': {'url': 'data:image/png;base64,iVBORw0KGgo='},
            },
        ],
    )
    self.assertEqual(msgs[0].additional_kwargs, {'metadata': {}})
    self.assertIsInstance(msgs[1], langchain_messages.SystemMessage)
    self.assertEqual(msgs[1].content, 'System instruction')
    self.assertEqual(msgs[1].additional_kwargs, {'metadata': {}})

  def test_single_text_message_conversion(self):
    """A single text part is converted to a string-based LangChain message."""
    mock_llm = mock.Mock(model='mock-model')
    proc = LangChainModel(model=mock_llm)

    parts = [
        content_api.ProcessorPart(
            value='Single text',
            mimetype='text/plain',
            role='user',
        )
    ]
    content = content_api.ProcessorContent(parts)
    msgs = proc._convert_to_langchain_messages(content.all_parts)

    self.assertLen(msgs, 1)
    self.assertIsInstance(msgs[0], langchain_messages.HumanMessage)
    self.assertEqual(msgs[0].content, 'Single text')
    self.assertEqual(msgs[0].additional_kwargs, {'metadata': {}})

  def test_system_instruction(self):
    """System instruction is prepended as a SystemMessage in the full pipeline."""
    mock_llm = mock.Mock(model='mock-model')
    mock_llm.astream.return_value = streams.stream_content(
        [MockChunk('Response')]
    )
    proc = LangChainModel(
        model=mock_llm, system_instruction='You are a helpful AI.'
    )

    parts = [
        content_api.ProcessorPart(
            value='User question',
            mimetype='text/plain',
            role='user',
        )
    ]
    content = content_api.ProcessorContent(parts)
    output = processor.apply_sync(proc, content.all_parts)

    self.assertLen(output, 1)
    self.assertEqual(output[0].text, 'Response')
    self.assertEqual(output[0].role, 'model')
    self.assertEqual(output[0].metadata, {'model': 'mock-model'})

  def test_prompt_template(self):
    """Prompt template is applied to messages if provided."""
    mock_llm = mock.Mock(model='mock-model')
    mock_llm.astream.return_value = streams.stream_content(
        [MockChunk('Formatted response')]
    )
    prompt_template = [('system', 'Be funny.'), ('placeholder', '{messages}')]
    # Equivalently:'User input: {messages}'
    proc = LangChainModel(
        model=mock_llm,
        prompt_template=ChatPromptTemplate(prompt_template),
    )

    parts = [
        content_api.ProcessorPart(
            value='Test input',
            mimetype='text/plain',
            role='user',
        )
    ]
    content = content_api.ProcessorContent(parts)
    output = processor.apply_sync(proc, content.all_parts)

    self.assertLen(output, 1)
    self.assertEqual(output[0].text, 'Formatted response')
    self.assertEqual(output[0].role, 'model')
    self.assertEqual(output[0].metadata, {'model': 'mock-model'})

    # Verify that the prompt template was actually applied.
    called_args = mock_llm.astream.call_args[0][0]
    self.assertEqual(
        called_args, {'input': 'System: Be funny.\nHuman: Test input'}
    )

  @parameterized.parameters(
      ('audio/mpeg', b'\x00\x01'),
      ('video/mp4', b'\x00\x02'),
  )
  def test_unsupported_mimetype_raises(self, mime_type, raw_data):
    """Unsupported MIME types should raise a ValueError."""
    mock_llm = mock.Mock(model='mock-model')
    proc = LangChainModel(model=mock_llm)

    bad_parts = [
        content_api.ProcessorPart(
            value=raw_data, mimetype=mime_type, role='user'
        )
    ]
    content = content_api.ProcessorContent(bad_parts)
    with self.assertRaises(ValueError) as cm:
      proc._convert_to_langchain_messages(content.all_parts)
    self.assertIn(f'Unsupported mimetype: {mime_type}', str(cm.exception))


if __name__ == '__main__':
  unittest.main()
