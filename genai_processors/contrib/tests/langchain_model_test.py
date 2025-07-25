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
from absl.testing import parameterized
from genai_processors import content_api
from genai_processors import processor
from genai_processors.contrib.langchain_model import LangChainModel
from langchain_core import messages as langchain_messages


class MockChunk:
  """Lightweight stand-in for a LangChain chunk with only a `.content` attribute."""

  def __init__(self, content: str):
    self.content = content


class MockChatModel:
  """Fake BaseChatModel that streams back provided MockChunk objects."""

  def __init__(self, chunks):
    self._chunks = [MockChunk(c) for c in chunks]
    self.model = 'mock-model'

  async def astream(self, payload):
    """Asynchronously stream chunks, simulating LangChain behavior.

    Args:
        payload: The input to the model (list of messages or dict with 'input').

    Yields:
        MockChunk: Each chunk in the preconfigured sequence.
    """
    # Handle prompt template case where payload is a list of messages
    if isinstance(payload, dict) and 'input' in payload:
      # No need to validate messages for string input
      pass
    else:
      # Validate that payload is a list of LangChain messages
      for msg in payload:
        assert isinstance(
            msg,
            (
                langchain_messages.HumanMessage,
                langchain_messages.SystemMessage,
                langchain_messages.AIMessage,
            ),
        ), f'Invalid message type: {type(msg)}'
    for chunk in self._chunks:
      yield chunk


class LangChainModelTest(
    unittest.IsolatedAsyncioTestCase, parameterized.TestCase
):
  """End-to-end tests for the LangChainModel contrib module."""

  def test_text_only_sync(self):
    """Single-turn, non-streaming inference via `apply_sync` with a single text part."""
    llm = MockChatModel(['Hello', ' world'])
    proc = LangChainModel(llm=llm)

    parts = [
        content_api.ProcessorPart(
            value='Hi there',
            mimetype='text/plain',
            role='user',
            metadata={'source': 'test'},
        )
    ]
    content = content_api.ProcessorContent(parts)
    output = processor.apply_sync(proc, content.all_parts)

    self.assertEqual(len(output), 2)
    self.assertEqual(output[0].text, 'Hello')
    self.assertEqual(output[0].role, 'model')
    self.assertEqual(output[0].metadata, {'model': 'mock-model'})
    self.assertEqual(output[1].text, ' world')
    self.assertEqual(output[1].role, 'model')
    self.assertEqual(output[1].metadata, {'model': 'mock-model'})

  def test_streaming_sync(self):
    """A turn with multiple text parts that streams three chunks in sequence."""
    llm = MockChatModel(['A', 'B', 'C'])
    proc = LangChainModel(llm=llm)

    parts = [
        content_api.ProcessorPart(
            value='Question 1',
            mimetype='text/plain',
            role='user',
            metadata={'id': '1'},
        ),
        content_api.ProcessorPart(
            value='Question 2',
            mimetype='text/plain',
            role='user',
            metadata={'id': '2'},
        ),
    ]
    content = content_api.ProcessorContent(parts)
    output = processor.apply_sync(proc, content.all_parts)

    self.assertEqual(len(output), 3)
    self.assertSequenceEqual([p.text for p in output], ['A', 'B', 'C'])
    self.assertTrue(all(p.role == 'model' for p in output))
    self.assertTrue(all(p.metadata == {'model': 'mock-model'} for p in output))

  async def test_streaming_async(self):
    """Asynchronous streaming with multiple parts of different roles."""
    llm = MockChatModel(['Response', ' part'])
    proc = LangChainModel(llm=llm)

    async def input_stream():
      yield content_api.ProcessorPart(
          value='System prompt',
          mimetype='text/plain',
          role='system',
          metadata={'id': 'sys1'},
      )
      yield content_api.ProcessorPart(
          value='User text',
          mimetype='text/plain',
          role='user',
          metadata={'id': 'user1'},
      )
      yield content_api.ProcessorPart(
          value=b'\x89PNG\r\n\x1a\n',
          mimetype='image/png',
          role='user',
          metadata={'id': 'image1'},
      )

    output = []
    async for part in proc(input_stream()):
      output.append(part)

    self.assertEqual(len(output), 2)
    self.assertSequenceEqual([p.text for p in output], ['Response', ' part'])
    self.assertTrue(all(p.role == 'model' for p in output))
    self.assertTrue(all(p.metadata == {'model': 'mock-model'} for p in output))

  def test_multimodal_message_conversion(self):
    """Multiple parts of the same role are grouped into one LangChain message.

    If multiple fragments are present the order must be preserved.
    """
    llm = MockChatModel([])
    proc = LangChainModel(llm=llm)

    parts = [
        content_api.ProcessorPart(
            value='Caption me',
            mimetype='text/plain',
            role='user',
            metadata={'id': 'text1'},
        ),
        content_api.ProcessorPart(
            value=b'\x89PNG\r\n\x1a\n',
            mimetype='image/png',
            role='user',
            metadata={'id': 'image1'},
        ),
        content_api.ProcessorPart(
            value='System instruction',
            mimetype='text/plain',
            role='system',
            metadata={'id': 'sys1'},
        ),
    ]
    content = content_api.ProcessorContent(parts)
    msgs = proc._convert_to_langchain_messages(content.all_parts)

    self.assertEqual(
        len(msgs), 2
    )  # One message for 'user' role, one for 'system' role
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
    self.assertEqual(msgs[0].additional_kwargs, {'metadata': {'id': 'image1'}})
    self.assertIsInstance(msgs[1], langchain_messages.SystemMessage)
    self.assertEqual(msgs[1].content, 'System instruction')
    self.assertEqual(msgs[1].additional_kwargs, {'metadata': {'id': 'sys1'}})

  def test_single_text_message_conversion(self):
    """A single text part is converted to a string-based LangChain message."""
    llm = MockChatModel([])
    proc = LangChainModel(llm=llm)

    parts = [
        content_api.ProcessorPart(
            value='Single text',
            mimetype='text/plain',
            role='user',
            metadata={'id': 'text1'},
        )
    ]
    content = content_api.ProcessorContent(parts)
    msgs = proc._convert_to_langchain_messages(content.all_parts)

    self.assertEqual(len(msgs), 1)
    self.assertIsInstance(msgs[0], langchain_messages.HumanMessage)
    self.assertEqual(msgs[0].content, 'Single text')
    self.assertEqual(msgs[0].additional_kwargs, {'metadata': {'id': 'text1'}})

  def test_system_instruction(self):
    """System instruction is prepended as a SystemMessage in the full pipeline."""
    llm = MockChatModel(['Response'])
    proc = LangChainModel(llm=llm, system_instruction='You are a helpful AI.')

    parts = [
        content_api.ProcessorPart(
            value='User question',
            mimetype='text/plain',
            role='user',
            metadata={'id': 'q1'},
        )
    ]
    content = content_api.ProcessorContent(parts)
    output = processor.apply_sync(proc, content.all_parts)

    self.assertEqual(len(output), 1)
    self.assertEqual(output[0].text, 'Response')
    self.assertEqual(output[0].role, 'model')
    self.assertEqual(output[0].metadata, {'model': 'mock-model'})

  def test_prompt_template(self):
    """Prompt template is applied to messages if provided."""
    llm = MockChatModel(['Formatted response'])
    prompt_template = 'User input: {{ messages }}'
    proc = LangChainModel(llm=llm, prompt_template=prompt_template)

    parts = [
        content_api.ProcessorPart(
            value='Test input',
            mimetype='text/plain',
            role='user',
            metadata={'id': 't1'},
        )
    ]
    content = content_api.ProcessorContent(parts)
    output = processor.apply_sync(proc, content.all_parts)

    self.assertEqual(len(output), 1)
    self.assertEqual(output[0].text, 'Formatted response')
    self.assertEqual(output[0].role, 'model')
    self.assertEqual(output[0].metadata, {'model': 'mock-model'})

  @parameterized.parameters(
      ('audio/mpeg', b'\x00\x01'),
      ('video/mp4', b'\x00\x02'),
  )
  def test_unsupported_mimetype_raises(self, mime_type, raw_data):
    """Unsupported MIME types should raise a ValueError."""
    llm = MockChatModel([])
    proc = LangChainModel(llm=llm)

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
