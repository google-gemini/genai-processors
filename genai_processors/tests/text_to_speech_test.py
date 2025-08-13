from collections.abc import AsyncIterable
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from genai_processors import content_api
from genai_processors import streams
from genai_processors.core import text_to_speech
from google.cloud import texttospeech_v1 as texttospeech


ProcessorPart = content_api.ProcessorPart


class FakeSpeechClient:
  """Fake implementation of the Text to Speech client."""

  def __init__(
      self,
      with_exception: bool = False,
      audio_bytes: list[bytes] = None,
  ):
    self.with_exception = with_exception
    self.input_stream = []
    self.idx = 0
    self.output_stream = [
        texttospeech.StreamingSynthesizeResponse(
            audio_content=audio_content,
        )
        for audio_content in audio_bytes
    ]

  async def streaming_synthesize(
      self,
      requests: AsyncIterable[texttospeech.StreamingSynthesizeRequest],
  ):
    async def iterate_requests():
      async for r in requests:
        if self.with_exception:
          raise RuntimeError('Unexpected failure')
        self.input_stream.append(r)
        if r.streaming_config:
          # Config request does not yield any output.
          continue
        if self.output_stream:
          yield self.output_stream[self.idx]
          self.idx += 1

    return iterate_requests()


class TextToSpeechTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  async def test_text_to_speech_success(self):
    """Tests successful audio generation from text."""
    fake_client = FakeSpeechClient(audio_bytes=[b'audio_data1', b'audio_data2'])
    with mock.patch.object(
        texttospeech,
        'TextToSpeechAsyncClient',
        auto_mock=True,
    ) as mock_client_class:
      mock_client_class.return_value = fake_client
      tts_processor = text_to_speech.TextToSpeech(
          project_id='test_project', with_text_passthrough=False
      )
      input_content = [
          ProcessorPart('Hello'),
          ProcessorPart('World'),
      ]
      output_content = await streams.gather_stream(
          tts_processor.call(streams.stream_content(input_content))
      )
      self.assertEqual(
          output_content,
          [
              ProcessorPart(
                  b'audio_data1', mimetype='audio/l16;rate=24000', role='model'
              ),
              ProcessorPart(
                  b'audio_data2', mimetype='audio/l16;rate=24000', role='model'
              ),
          ],
      )
      self.assertLen(fake_client.input_stream, 3)
      self.assertIsNotNone(fake_client.input_stream[0].streaming_config)
      self.assertEqual(
          fake_client.input_stream[1].input.text,
          'Hello',
      )
      self.assertEqual(
          fake_client.input_stream[2].input.text,
          'World',
      )

  async def test_text_to_speech_non_text_input(self):
    """Tests that non-text input is passed through."""
    fake_client = FakeSpeechClient(audio_bytes=[b'audio_data1', b'audio_data2'])
    with mock.patch.object(
        texttospeech,
        'TextToSpeechAsyncClient',
        auto_mock=True,
    ) as mock_client_class:
      mock_client_class.return_value = fake_client
      tts_processor = text_to_speech.TextToSpeech(
          project_id='test_project', with_text_passthrough=False
      )
      input_content = [ProcessorPart(b'image_data', mimetype='image/png')]
      output_content = await streams.gather_stream(
          tts_processor.call(streams.stream_content(input_content))
      )
      self.assertEqual(output_content, input_content)

      self.assertLen(fake_client.input_stream, 1)
      self.assertIsNotNone(fake_client.input_stream[0].streaming_config)

  async def test_text_to_speech_empty_text(self):
    """Tests that empty text input is skipped."""
    fake_client = FakeSpeechClient(audio_bytes=[])
    with mock.patch.object(
        texttospeech,
        'TextToSpeechAsyncClient',
        auto_mock=True,
    ) as mock_client_class:
      mock_client_class.return_value = fake_client
      tts_processor = text_to_speech.TextToSpeech(
          project_id='test_project', with_text_passthrough=False
      )
      input_content = [ProcessorPart('')]
      output_content = await streams.gather_stream(
          tts_processor.call(streams.stream_content(input_content))
      )
      self.assertEmpty(output_content)
      self.assertLen(fake_client.input_stream, 1)
      self.assertIsNotNone(fake_client.input_stream[0].streaming_config)

  async def test_text_to_speech_with_exception(self):
    """Tests that exceptions are handled."""
    fake_client = FakeSpeechClient(audio_bytes=[], with_exception=True)
    with mock.patch.object(
        texttospeech,
        'TextToSpeechAsyncClient',
        auto_mock=True,
    ) as mock_client_class:
      mock_client_class.return_value = fake_client
      tts_processor = text_to_speech.TextToSpeech(project_id='test_project')
      input_content = [ProcessorPart('Hello')]
      with self.assertRaises(RuntimeError):
        await streams.gather_stream(
            tts_processor.call(streams.stream_content(input_content))
        )

  async def test_text_to_speech_with_text_passthrough(self):
    """Tests that text is passed through when with_text_passthrough is True."""
    fake_client = FakeSpeechClient(audio_bytes=[b'audio_data1', b'audio_data2'])
    with mock.patch.object(
        texttospeech,
        'TextToSpeechAsyncClient',
        auto_mock=True,
    ) as mock_client_class:
      mock_client_class.return_value = fake_client
      tts_processor = text_to_speech.TextToSpeech(
          project_id='test_project', with_text_passthrough=True
      )
      input_content = [
          ProcessorPart('Hello'),
          ProcessorPart('World'),
      ]
      output_content = await streams.gather_stream(
          tts_processor.call(streams.stream_content(input_content))
      )
      self.assertEqual(
          output_content,
          [
              ProcessorPart('Hello'),
              ProcessorPart('World'),
              ProcessorPart(
                  b'audio_data1', mimetype='audio/l16;rate=24000', role='model'
              ),
              ProcessorPart(
                  b'audio_data2', mimetype='audio/l16;rate=24000', role='model'
              ),
          ],
      )


if __name__ == '__main__':
  absltest.main()
