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
"""Handles extracting speech from streaming audio content parts.

Uses Google Cloud Speech API to transcribe audio parts into text parts.

install google cloud speech client with:

```python
pip install --upgrade google-cloud-speech
```

See the `speech_to_text_cli.py` script for a usage example and how to test it
locally. It is recommended to test the quality of the transcription with
different models and recognizers.
"""

import asyncio
from collections.abc import AsyncIterable
import time

from absl import logging
from genai_processors import content_api
from genai_processors import processor
from genai_processors import streams
from genai_processors.core import speech_events
from google.cloud import speech_v2
import grpc


_SILENT_AUDIO_DELAY_SECONDS = 1

RecognizeStream = grpc.aio.StreamStreamCall[
    speech_v2.types.StreamingRecognizeRequest,
    speech_v2.types.StreamingRecognizeResponse,
]


DEFAULT_SAMPLE_RATE_HZ = 24000
# streaming_recognize RPC has limit on the duration and has to be restarted
# periodically. Instead of waiting for the deadline we try to restart it at
# the moments when that won't cause hiccups.
STREAMING_HARD_LIMIT_SEC = (
    240  # 4 minutes / restart stream even when user is speaking.
)
STREAMING_LIMIT_SEC = (
    180  # 3 minutes / restart stream when user is not speaking.
)

ProcessorPart = content_api.ProcessorPart
ProcessorPartTypes = content_api.ProcessorPartTypes

# Re-export from speech_events for backward compatibility.
# Use speech_events variants instead.
StartOfSpeech = speech_events.StartOfSpeech
EndOfSpeech = speech_events.EndOfSpeech
TRANSCRIPTION_SUBSTREAM_NAME = speech_events.TRANSCRIPTION_SUBSTREAM_NAME
ENDPOINTING_SUBSTREAM_NAME = speech_events.ENDPOINTING_SUBSTREAM_NAME


class AddSilentPartMaybe(processor.Processor):
  """Adds silent audio parts if no activity after `silent_part_duration_sec`.

  If the stream is empty after a few seconds, the Speech API will close the
  connection. This processor adds silent audio parts to the output stream to
  keep the connection alive.
  """

  def __init__(
      self,
      silent_part_duration_sec: float = 1,
      sample_rate: int = DEFAULT_SAMPLE_RATE_HZ,
  ):
    self._silent_part_duration_sec = silent_part_duration_sec
    self._sample_rate = sample_rate

  async def call(
      self, content: processor.ProcessorStream
  ) -> AsyncIterable[ProcessorPartTypes]:

    logging.info('Transcriber: _process_audio started.')
    last_streamed_audio_time_sec = time.perf_counter()

    async def _insert_silent_audio() -> AsyncIterable[ProcessorPart]:
      """Sends silent audio to the Speech API to keep the stream alive."""
      nonlocal last_streamed_audio_time_sec

      while True:
        await asyncio.sleep(self._silent_part_duration_sec)
        delta_time_sec = time.perf_counter() - last_streamed_audio_time_sec
        if delta_time_sec > self._silent_part_duration_sec:
          yield ProcessorPart(
              value=b'\0' * round(self._sample_rate * delta_time_sec),
              mimetype=f'audio/l16; rate={self._sample_rate}',
          )
          last_streamed_audio_time_sec = time.perf_counter()

    audio_stream = streams.merge(
        [content, _insert_silent_audio()], stop_on_first=True
    )

    async for part in audio_stream:
      last_streamed_audio_time_sec = time.perf_counter()
      yield part

    logging.info('Transcriber: _process_audio finished.')


class _Transcriber(processor.Processor):
  """Transcribes streaming audio using the Cloud Speech API."""

  def __init__(
      self,
      project_id: str,
      recognition_config: speech_v2.types.RecognitionConfig,
      with_endpointing: bool = True,
      substream_endpointing: str = speech_events.ENDPOINTING_SUBSTREAM_NAME,
      strict_endpointing: bool = True,
      with_interim_results: bool = True,
      substream_transcription: str = speech_events.TRANSCRIPTION_SUBSTREAM_NAME,
      passthrough_audio: bool = False,
  ):
    """Transcribes audio parts using the Cloud Speech API.

    Args:
      project_id: The project ID to use for the Speech API.
      recognition_config: The recognition config to use for the Speech API. Set
        it up to adjust the sample rate, languages or the recognition model.
      with_endpointing: Whether to yield endpointing events. Endpointing events
        are text parts with the value set to one of the
        `speech_to_text.SpeechEventType` string enums. The endpointing events
        are yielded in the substream defined by substream_endpointing.
      substream_endpointing: The substream name to use for the endpointing
        events.
      strict_endpointing: Whether to send endpointing events only when interim
        results have been found. This avoids yielding endpointing events when
        the user speech is not recognized (e.g. does not return endpointing for
        noise or laughs or coughing, etc.).
      with_interim_results: Whether to yield interim results. If set to False,
        the processor will only yield the final transcription.
      substream_transcription: The substream name to use for the transcription.
      passthrough_audio: Whether to passthrough the audio parts to the output
        stream. The substream name is set to the default one: ''.
    """
    self._config = speech_v2.types.StreamingRecognitionConfig(
        config=recognition_config,
        streaming_features=speech_v2.types.StreamingRecognitionFeatures(
            interim_results=True,
            enable_voice_activity_events=True,
        ),
    )
    self._sample_rate = (
        self._config.config.explicit_decoding_config.sample_rate_hertz
        or DEFAULT_SAMPLE_RATE_HZ
    )
    self._with_endpointing = with_endpointing
    self._substream_endpointing = substream_endpointing
    self._strict_endpointing = strict_endpointing
    self._with_interim_results = with_interim_results
    self._substream_transcription = substream_transcription
    self._project_id = project_id
    self._passthrough_audio = passthrough_audio

  def _make_setup_request(self) -> speech_v2.types.StreamingRecognizeRequest:
    return speech_v2.types.StreamingRecognizeRequest(
        streaming_config=self._config,
        recognizer=(
            f'projects/{self._project_id}/locations/global/recognizers/_'
        ),
    )

  async def call(
      self,
      content: processor.ProcessorStream,
  ) -> AsyncIterable[ProcessorPartTypes]:
    """Transcribes streaming audio using the Cloud Speech API."""

    # The output queue is used to yield the audio parts unchanged in the output
    # stream when self._passthrough_audio is True.
    output_queue = asyncio.Queue[ProcessorPart | None]()

    stream_state: dict[str, bool | float] = {
        'start_time_sec': time.perf_counter(),
        'restart_stream': False,
        'user_speaking': False,
        'stream_is_on': True,
    }

    async def request_stream(
        request_queue: asyncio.Queue[
            speech_v2.types.StreamingRecognizeRequest | None
        ],
    ):
      try:
        request_queue.put_nowait(self._make_setup_request())
        async for part in content:
          if not content_api.is_audio(part.mimetype):
            output_queue.put_nowait(part)
            continue
          if self._passthrough_audio:
            output_queue.put_nowait(part)
          if part.part.inline_data is None:
            continue
          if not part.mimetype.lower().startswith(
              'audio/l16'
          ) or not part.mimetype.lower().endswith(f'rate={self._sample_rate}'):
            raise ValueError(
                f'Unsupported audio mimetype: {part.mimetype}. Expected'
                f' audio/l16;[.*]rate={self._sample_rate}.'
            )
          request_queue.put_nowait(
              speech_v2.types.StreamingRecognizeRequest(
                  audio=part.part.inline_data.data,
              )
          )
          delta_time_sec = time.perf_counter() - stream_state['start_time_sec']
          if (
              (delta_time_sec > STREAMING_LIMIT_SEC)
              and not stream_state['user_speaking']
          ) or (delta_time_sec > STREAMING_HARD_LIMIT_SEC):
            stream_state['restart_stream'] = True
            break
      finally:
        request_queue.put_nowait(None)

    async def send_audio_to_speech_api():
      # Instantiates a client.
      try:
        logging.debug('Transcriber: (re)creating client')
        client = speech_v2.SpeechAsyncClient()
        last_endpointing_event = None
        while stream_state['stream_is_on']:
          request_queue = asyncio.Queue[
              speech_v2.types.StreamingRecognizeRequest | None
          ]()
          populate_request_queue = processor.create_task(
              request_stream(request_queue)
          )
          response_stream = await client.streaming_recognize(
              requests=streams.dequeue(request_queue)
          )
          async for response in response_stream:
            if response == grpc.aio.EOF:
              break
            if (
                response.speech_event_type
                == speech_v2.types.StreamingRecognizeResponse.SpeechEventType.SPEECH_ACTIVITY_BEGIN
            ):
              last_endpointing_event = StartOfSpeech()
              stream_state['user_speaking'] = True
              if self._with_endpointing and not self._strict_endpointing:
                last_endpointing_event = 'SPEECH_ACTIVITY_BEGIN_SENT'
                output_queue.put_nowait(
                    ProcessorPart.from_dataclass(
                        dataclass=StartOfSpeech(),
                        substream_name=self._substream_endpointing,
                    )
                )
            if response.results and response.results[0].alternatives:
              if (
                  isinstance(last_endpointing_event, StartOfSpeech)
                  and self._strict_endpointing
              ):
                # We have not sent the SPEECH_ACTIVITY_BEGIN event yet, we
                # waited for the first transcript to appear.
                last_endpointing_event = 'SPEECH_ACTIVITY_BEGIN_SENT'
                output_queue.put_nowait(
                    ProcessorPart.from_dataclass(
                        dataclass=StartOfSpeech(),
                        substream_name=self._substream_endpointing,
                    )
                )
              if text := response.results[0].alternatives[0].transcript:
                metadata = {
                    'is_final': response.results[0].is_final,
                }
                if self._with_interim_results or response.results[0].is_final:
                  output_queue.put_nowait(
                      ProcessorPart(
                          text,
                          role='user',
                          metadata=metadata,
                          substream_name=self._substream_transcription,
                      )
                  )
            if (
                response.speech_event_type
                == speech_v2.types.StreamingRecognizeResponse.SpeechEventType.SPEECH_ACTIVITY_END
            ):
              stream_state['user_speaking'] = False
              if (
                  self._with_endpointing
                  and last_endpointing_event == 'SPEECH_ACTIVITY_BEGIN_SENT'
              ):
                output_queue.put_nowait(
                    ProcessorPart.from_dataclass(
                        dataclass=EndOfSpeech(),
                        substream_name=self._substream_endpointing,
                    )
                )
              last_endpointing_event = None
          if stream_state['restart_stream']:
            stream_state['restart_stream'] = False
            stream_state['stream_is_on'] = True
            stream_state['start_time_sec'] = time.perf_counter()
            client = speech_v2.SpeechAsyncClient()
            populate_request_queue.cancel()
          else:
            stream_state['stream_is_on'] = False
      finally:
        output_queue.put_nowait(None)

    send_task = processor.create_task(send_audio_to_speech_api())
    while part := await output_queue.get():
      yield part
    await send_task


class SpeechToText(processor.Processor):
  """Converts audio parts into text parts."""

  def __init__(
      self,
      project_id: str,
      recognition_config: speech_v2.types.RecognitionConfig | None = None,
      audio_passthrough: bool = False,
      with_endpointing: bool = True,
      substream_endpointing: str = speech_events.ENDPOINTING_SUBSTREAM_NAME,
      strict_endpointing: bool = True,
      with_interim_results: bool = True,
      substream_transcription: str = speech_events.TRANSCRIPTION_SUBSTREAM_NAME,
      maintain_connection_active_with_silent_audio: bool = False,
  ):
    """Initializes the SpeechToText processor.

    The speech processor uses the Cloud Speech API to transcribe audio parts
    into text parts. It injects silent audio parts to keep the stream alive
    when the user is not speaking and restarts the connection automatically
    after 3-4 minutes to avoid the stream being closed by the server.

    The processor yields endpointing events when the user starts and stops
    speaking. If with_endpointing is False, the endpointing events are not
    yielded. The endpointing events are yielded in the substream defined by
    substream_endpointing. When strict_endpointing is True, the endpointing
    events are yielded only when interim results have been found. This avoids
    yielding endpointing events when the user speech is not recognized (e.g.
     short noise or sound).

    Args:
      project_id: The project ID to use for the Speech API.
      recognition_config: The recognition config to use for the Speech API. Set
        it up to adjust the sample rate, languages or the recognition model.
      audio_passthrough: Whether to passthrough the audio parts to the output
        stream. The substream name is set to the default one: ''.
      with_endpointing: Whether to yield endpointing events. Endpointing events
        are text parts with the value set to one of the
        `speech_to_text.SpeechEventType` string enums. The endpointing events
        are yielded in the substream defined by substream_endpointing.
      substream_endpointing: The substream name to use for the endpointing
        events.
      strict_endpointing: Whether to send endpointing events only when interim
        results have been found. This avoids yielding endpointing events when
        the user speech is not recognized (e.g. does not return endpointing for
        noise or laughs or coughing, etc.).
      with_interim_results: Whether to yield interim results. If set to False,
        the processor will only yield the final transcription.
      substream_transcription: The substream name to use for the transcription.
      maintain_connection_active_with_silent_audio: Whether to maintain the
        connection active with silent audio. If set to True, the processor will
        inject silent audio parts to keep the stream alive when the processor
        does not receive any audio part. This can be needed if the Speech API
        closes the stream when it does not receive any audio for a long time.
    """
    recognition_config = recognition_config or speech_v2.types.RecognitionConfig(
        explicit_decoding_config=speech_v2.types.ExplicitDecodingConfig(
            sample_rate_hertz=DEFAULT_SAMPLE_RATE_HZ,
            encoding=speech_v2.types.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
            audio_channel_count=1,
        ),
        language_codes=['en-US'],
        model='latest_long',
    )
    self._processor = _Transcriber(
        project_id=project_id,
        recognition_config=recognition_config,
        with_endpointing=with_endpointing,
        substream_endpointing=substream_endpointing,
        strict_endpointing=strict_endpointing,
        with_interim_results=with_interim_results,
        substream_transcription=substream_transcription,
        passthrough_audio=audio_passthrough,
    )
    if maintain_connection_active_with_silent_audio:
      sample_rate = (
          recognition_config.explicit_decoding_config.sample_rate_hertz
          or DEFAULT_SAMPLE_RATE_HZ
      )
      self._processor = (
          AddSilentPartMaybe(
              silent_part_duration_sec=_SILENT_AUDIO_DELAY_SECONDS,
              sample_rate=sample_rate,
          )
          + self._processor
      )

  async def call(
      self,
      content: processor.ProcessorStream,
  ) -> AsyncIterable[ProcessorPartTypes]:
    async for part in self._processor(content):
      yield part
