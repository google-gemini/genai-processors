import asyncio
from collections.abc import AsyncIterable
from unittest import mock

from absl.testing import absltest
from genai_processors import content_api
from genai_processors import context as context_lib
from genai_processors import processor
from genai_processors.core import speech_to_text
from google.cloud import speech_v2
from google.cloud.speech_v2.types import cloud_speech as cloud_speech_types
import grpc


ProcessorPart = content_api.ProcessorPart


SpeechEventType = speech_v2.types.StreamingRecognizeResponse.SpeechEventType


class FakeSpeechClient:
  """Fake implementation of the Speech client."""

  def __init__(
      self,
      transcripts: list[str],
      is_final: list[bool],
      event_types: list[SpeechEventType],
      with_exception: bool = False,
  ):
    self.with_exception = with_exception
    self.input_stream = []
    self.output_stream = [
        self.read_srr(t, f, e)
        for t, f, e in zip(transcripts, is_final, event_types)
    ] + [self.read_eof()]
    self.idx = 0

  def read_eof(self):
    return grpc.aio.EOF

  def read_srr(
      self,
      transcript: str,
      is_final: bool,
      event_type: SpeechEventType,
  ) -> cloud_speech_types.StreamingRecognizeResponse:
    return cloud_speech_types.StreamingRecognizeResponse(
        results=[
            cloud_speech_types.StreamingRecognitionResult(
                alternatives=[
                    cloud_speech_types.SpeechRecognitionAlternative(
                        transcript=transcript,
                    )
                ],
                is_final=is_final,
            )
        ],
        speech_event_type=event_type,
    )

  async def streaming_recognize(
      self,
      requests: AsyncIterable[cloud_speech_types.StreamingRecognizeRequest],
  ):
    async def iterate_requests():
      async for r in requests:
        if self.with_exception:
          raise RuntimeError('Unexpected failure')
        self.input_stream.append(r)
        if len(self.output_stream) > 1:
          # output_stream is always > 1 (we add EOF during init).
          yield self.output_stream[self.idx]
          self.idx += 1

    return iterate_requests()


class SpeechToTextTest(absltest.TestCase):

  def test_empty_stream(self):
    fake_client = FakeSpeechClient(
        transcripts=[],
        is_final=[],
        event_types=[],
    )
    with mock.patch.object(
        speech_v2,
        'SpeechAsyncClient',
        auto_mock=True,
    ) as mock_client_class:
      mock_client_class.return_value = fake_client
      stt = speech_to_text.SpeechToText(project_id='unused')
      parts = processor.apply_sync(stt, [])
      self.assertEmpty(parts)
      self.assertLen(fake_client.input_stream, 1)
      self.assertEqual(
          fake_client.input_stream,
          [
              cloud_speech_types.StreamingRecognizeRequest(
                  streaming_config=cloud_speech_types.StreamingRecognitionConfig(
                      config=cloud_speech_types.RecognitionConfig(
                          explicit_decoding_config=cloud_speech_types.ExplicitDecodingConfig(
                              sample_rate_hertz=24000,
                              encoding=cloud_speech_types.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                              audio_channel_count=1,
                          ),
                          language_codes=['en-US'],
                          model='latest_long',
                      ),
                      streaming_features=cloud_speech_types.StreamingRecognitionFeatures(
                          interim_results=True,
                          enable_voice_activity_events=True,
                      ),
                  ),
                  recognizer='projects/unused/locations/global/recognizers/_',
              )
          ],
      )

  def test_final_result_with_endpointing(self):
    fake_client = FakeSpeechClient(
        transcripts=['', 'test transcript'],
        is_final=[False, True],
        event_types=[
            SpeechEventType.SPEECH_ACTIVITY_BEGIN,
            SpeechEventType.SPEECH_ACTIVITY_END,
        ],
    )
    with mock.patch.object(
        speech_v2,
        'SpeechAsyncClient',
        auto_mock=True,
    ) as mock_client_class:
      mock_client_class.return_value = fake_client
      stt = speech_to_text.SpeechToText(project_id='unused')
      input_content = [
          ProcessorPart(b'audio data', mimetype='audio/l16; rate=24000')
      ]
      parts = processor.apply_sync(stt, input_content)
      expected_parts = [
          ProcessorPart.from_dataclass(
              dataclass=speech_to_text.StartOfSpeech(),
              substream_name=speech_to_text.ENDPOINTING_SUBSTREAM_NAME,
          ),
          ProcessorPart(
              'test transcript',
              role='user',
              metadata={
                  'is_final': True,
              },
              substream_name=speech_to_text.TRANSCRIPTION_SUBSTREAM_NAME,
          ),
          ProcessorPart.from_dataclass(
              dataclass=speech_to_text.EndOfSpeech(),
              substream_name=speech_to_text.ENDPOINTING_SUBSTREAM_NAME,
          ),
      ]
      self.assertListEqual(parts, expected_parts)

  def test_interim_result(self):
    fake_client = FakeSpeechClient(
        transcripts=['', 'test transcript'],
        is_final=[False, False],
        event_types=[
            SpeechEventType.SPEECH_ACTIVITY_BEGIN,
            SpeechEventType.SPEECH_EVENT_TYPE_UNSPECIFIED,
        ],
    )
    with mock.patch.object(
        speech_v2,
        'SpeechAsyncClient',
        auto_mock=True,
    ) as mock_client_class:
      mock_client_class.return_value = fake_client
      stt = speech_to_text.SpeechToText(project_id='unused')
      input_content = [
          ProcessorPart(b'audio data', mimetype='audio/l16; rate=24000')
      ]

      parts = processor.apply_sync(stt, input_content)
      expected_parts = [
          ProcessorPart.from_dataclass(
              dataclass=speech_to_text.StartOfSpeech(),
              substream_name=speech_to_text.ENDPOINTING_SUBSTREAM_NAME,
          ),
          ProcessorPart(
              'test transcript',
              role='user',
              metadata={
                  'is_final': False,
              },
              substream_name=speech_to_text.TRANSCRIPTION_SUBSTREAM_NAME,
          ),
      ]
      self.assertListEqual(parts, expected_parts)

  def test_interim_and_final_results(self):
    fake_client = FakeSpeechClient(
        transcripts=['interim 1', 'interim 2', 'final transcript'],
        is_final=[False, False, True],
        event_types=[
            SpeechEventType.SPEECH_ACTIVITY_BEGIN,
        ]
        + [SpeechEventType.SPEECH_EVENT_TYPE_UNSPECIFIED]
        + [SpeechEventType.SPEECH_ACTIVITY_END],
    )
    with mock.patch.object(
        speech_v2,
        'SpeechAsyncClient',
        auto_mock=True,
    ) as mock_client_class:
      mock_client_class.return_value = fake_client

      stt = speech_to_text.SpeechToText(project_id='unused')
      input_content = [
          ProcessorPart(b'audio data', mimetype='audio/l16; rate=24000'),
          # Empty inputs are here to make the fake client return something.
          # It will return here the interim results.
          ProcessorPart(b'', mimetype='audio/l16; rate=24000'),
          ProcessorPart(b'', mimetype='audio/l16; rate=24000'),
      ]
      parts = processor.apply_sync(stt, input_content)
      expected_parts = [
          ProcessorPart.from_dataclass(
              dataclass=speech_to_text.StartOfSpeech(),
              substream_name=speech_to_text.ENDPOINTING_SUBSTREAM_NAME,
          ),
          ProcessorPart(
              'interim 1',
              role='user',
              metadata={
                  'is_final': False,
              },
              substream_name=speech_to_text.TRANSCRIPTION_SUBSTREAM_NAME,
          ),
          ProcessorPart(
              'interim 2',
              role='user',
              metadata={
                  'is_final': False,
              },
              substream_name=speech_to_text.TRANSCRIPTION_SUBSTREAM_NAME,
          ),
          ProcessorPart(
              'final transcript',
              role='user',
              metadata={
                  'is_final': True,
              },
              substream_name=speech_to_text.TRANSCRIPTION_SUBSTREAM_NAME,
          ),
          ProcessorPart.from_dataclass(
              dataclass=speech_to_text.EndOfSpeech(),
              substream_name=speech_to_text.ENDPOINTING_SUBSTREAM_NAME,
          ),
      ]
      self.assertListEqual(parts, expected_parts)

  def test_pass_through_non_streaming_audio_part(self):
    fake_client = FakeSpeechClient(
        transcripts=['', 'test transcript', ''],
        is_final=[False, True, False],
        event_types=[
            SpeechEventType.SPEECH_ACTIVITY_BEGIN,
            SpeechEventType.SPEECH_EVENT_TYPE_UNSPECIFIED,
            SpeechEventType.SPEECH_ACTIVITY_END,
        ],
    )
    with mock.patch.object(
        speech_v2,
        'SpeechAsyncClient',
        auto_mock=True,
    ) as mock_client_class:
      mock_client_class.return_value = fake_client
      stt = speech_to_text.SpeechToText(project_id='unused')
      input_content = [
          ProcessorPart(b'audio data', mimetype='audio/l16; rate=24000'),
          ProcessorPart('Hello', mimetype='text/plain'),
      ]
      parts = processor.apply_sync(stt, input_content)
      self.assertIn(
          ProcessorPart('Hello', mimetype='text/plain'),
          parts,
      )

  def test_sends_silent_audio_part(self):
    fake_client = FakeSpeechClient(
        transcripts=[],
        is_final=[],
        event_types=[],
    )
    with mock.patch.object(
        speech_v2,
        'SpeechAsyncClient',
        auto_mock=True,
    ) as mock_client_class:
      mock_client_class.return_value = fake_client

      stt = speech_to_text.SpeechToText(
          project_id='unused', maintain_connection_active_with_silent_audio=True
      )

      async def stream_content() -> AsyncIterable[ProcessorPart]:
        await asyncio.sleep(1.5)
        yield ProcessorPart(b'', mimetype='audio/l16; rate=24000')

      async def run():
        buffer = []
        async with context_lib.context():
          async for part in stt(stream_content()):
            buffer.append(part)
          return buffer

      parts = asyncio.run(run())
      self.assertEmpty(parts)
      self.assertLen(fake_client.input_stream, 3)
      # First request is the config.
      self.assertEqual(
          fake_client.input_stream[0],
          cloud_speech_types.StreamingRecognizeRequest(
              streaming_config=cloud_speech_types.StreamingRecognitionConfig(
                  config=cloud_speech_types.RecognitionConfig(
                      explicit_decoding_config=cloud_speech_types.ExplicitDecodingConfig(
                          sample_rate_hertz=24000,
                          encoding=cloud_speech_types.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                          audio_channel_count=1,
                      ),
                      language_codes=['en-US'],
                      model='latest_long',
                  ),
                  streaming_features=cloud_speech_types.StreamingRecognitionFeatures(
                      interim_results=True,
                      enable_voice_activity_events=True,
                  ),
              ),
              recognizer='projects/unused/locations/global/recognizers/_',
          ),
      )
      # Second request is the silent audio. There can be a slight difference
      # in the length of the silent audio. Only compares the first 1000 bytes.
      self.assertEqual(fake_client.input_stream[1].audio[:1000], b'\0' * 1000)
      # Last request is the empty input.
      self.assertEqual(
          fake_client.input_stream[2],
          cloud_speech_types.StreamingRecognizeRequest(audio=b''),
      )

  def test_stream_reconnect(self):
    speech_to_text.STREAMING_LIMIT_SEC = 1
    speech_to_text.STREAMING_HARD_LIMIT_SEC = 2
    fake_client = FakeSpeechClient(
        transcripts=['interim1', 'final1', '', 'final2', '', 'final3'],
        is_final=[False, True, False, True, False, True],
        event_types=[
            SpeechEventType.SPEECH_ACTIVITY_BEGIN,
            SpeechEventType.SPEECH_ACTIVITY_END,
        ]
        * 3,
    )
    with mock.patch.object(
        speech_v2,
        'SpeechAsyncClient',
        auto_mock=True,
    ) as mock_client_class:
      mock_client_class.return_value = fake_client
      stt = speech_to_text.SpeechToText(project_id='unused')

      async def stream_content() -> AsyncIterable[ProcessorPart]:
        yield ProcessorPart(b'audio11', mimetype='audio/l16; rate=24000')
        await asyncio.sleep(1)
        yield ProcessorPart(b'audio12', mimetype='audio/l16; rate=24000')
        await asyncio.sleep(1.5)
        yield ProcessorPart(b'audio2', mimetype='audio/l16; rate=24000')
        await asyncio.sleep(2)
        yield ProcessorPart(b'audio3', mimetype='audio/l16; rate=24000')

      async def run():
        buffer = []
        async with context_lib.context():
          async for part in stt(stream_content()):
            buffer.append(part)
          return buffer

      parts = asyncio.run(run())
      expected_parts = [
          ProcessorPart.from_dataclass(
              dataclass=speech_to_text.StartOfSpeech(),
              substream_name=speech_to_text.ENDPOINTING_SUBSTREAM_NAME,
          ),
          ProcessorPart(
              'interim1',
              role='user',
              metadata={
                  'is_final': False,
              },
              substream_name=speech_to_text.TRANSCRIPTION_SUBSTREAM_NAME,
          ),
          ProcessorPart(
              'final1',
              role='user',
              metadata={
                  'is_final': True,
              },
              substream_name=speech_to_text.TRANSCRIPTION_SUBSTREAM_NAME,
          ),
          ProcessorPart.from_dataclass(
              dataclass=speech_to_text.EndOfSpeech(),
              substream_name=speech_to_text.ENDPOINTING_SUBSTREAM_NAME,
          ),
          ProcessorPart.from_dataclass(
              dataclass=speech_to_text.StartOfSpeech(),
              substream_name=speech_to_text.ENDPOINTING_SUBSTREAM_NAME,
          ),
          ProcessorPart(
              'final2',
              role='user',
              metadata={
                  'is_final': True,
              },
              substream_name=speech_to_text.TRANSCRIPTION_SUBSTREAM_NAME,
          ),
          ProcessorPart.from_dataclass(
              dataclass=speech_to_text.EndOfSpeech(),
              substream_name=speech_to_text.ENDPOINTING_SUBSTREAM_NAME,
          ),
          ProcessorPart.from_dataclass(
              dataclass=speech_to_text.StartOfSpeech(),
              substream_name=speech_to_text.ENDPOINTING_SUBSTREAM_NAME,
          ),
          ProcessorPart(
              'final3',
              role='user',
              metadata={
                  'is_final': True,
              },
              substream_name=speech_to_text.TRANSCRIPTION_SUBSTREAM_NAME,
          ),
          ProcessorPart.from_dataclass(
              dataclass=speech_to_text.EndOfSpeech(),
              substream_name=speech_to_text.ENDPOINTING_SUBSTREAM_NAME,
          ),
      ]
      self.assertEqual(parts, expected_parts)

  def test_unexpected_failures_are_propagated(self):
    fake_client = FakeSpeechClient(
        transcripts=['test transcript'],
        is_final=[True],
        event_types=[SpeechEventType.SPEECH_EVENT_TYPE_UNSPECIFIED],
        with_exception=True,
    )
    with mock.patch.object(
        speech_v2,
        'SpeechAsyncClient',
        auto_mock=True,
    ) as mock_client_class:
      mock_client_class.return_value = fake_client
      stt = speech_to_text.SpeechToText(project_id='unused')
      input_content = [
          ProcessorPart(b'audio data', mimetype='audio/l16; rate=24000')
      ]

      with self.assertRaises(RuntimeError) as context:
        processor.apply_sync(stt, input_content)
      self.assertEqual(str(context.exception), 'Unexpected failure')

  def test_unsupported_audio_mimetype_raises_error(self):
    fake_client = FakeSpeechClient(
        transcripts=['test transcript'],
        is_final=[True],
        event_types=[SpeechEventType.SPEECH_EVENT_TYPE_UNSPECIFIED],
    )
    with mock.patch.object(
        speech_v2,
        'SpeechAsyncClient',
        auto_mock=True,
    ) as mock_client_class:
      mock_client_class.return_value = fake_client
      stt = speech_to_text.SpeechToText(project_id='unused')
      input_content = [ProcessorPart(b'audio data', mimetype='audio/wav')]
      with self.assertRaises(ValueError) as context:
        processor.apply_sync(stt, input_content)
      self.assertEqual(
          str(context.exception),
          'Unsupported audio mimetype: audio/wav. Expected'
          ' audio/l16;[.*]rate=24000.',
      )


if __name__ == '__main__':
  absltest.main()
