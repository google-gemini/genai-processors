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
"""Converts text parts to audio with the MiniMax Text-to-Audio API."""

from collections.abc import AsyncIterable, Mapping
import json
from typing import Any, Literal, TypedDict

from genai_processors import content_api
from genai_processors import processor
import httpx


DEFAULT_MODEL = 'speech-2.8-hd'
SUPPORTED_MODELS = (
    'speech-2.8-hd',
    'speech-2.8-turbo',
    'speech-2.6-hd',
    'speech-2.6-turbo',
    'speech-02-hd',
    'speech-02-turbo',
    'speech-01-hd',
    'speech-01-turbo',
)

_ENDPOINTS = {
    'global_en': 'https://api.minimax.io/v1/t2a_v2',
    'cn_zh': 'https://api.minimaxi.com/v1/t2a_v2',
}
_MIME_TYPES = {
    'mp3': 'audio/mpeg',
    'wav': 'audio/wav',
    'flac': 'audio/flac',
    'pcm': 'audio/pcm',
}

Region = Literal['global_en', 'cn_zh']
AudioFormat = Literal['mp3', 'wav', 'flac', 'pcm']
OutputFormat = Literal['hex', 'url']


class VoiceSetting(TypedDict, total=False):
  """Voice controls accepted by the Text-to-Audio API."""

  voice_id: str
  speed: float
  vol: float
  pitch: int
  emotion: str
  text_normalization: bool
  latex_read: bool


class AudioSetting(TypedDict, total=False):
  """Audio encoding controls accepted by the Text-to-Audio API."""

  sample_rate: int
  bitrate: int
  format: AudioFormat
  channel: int
  force_cbr: bool


class PronunciationDictionary(TypedDict, total=False):
  """Custom pronunciation rules accepted by the Text-to-Audio API."""

  tone: list[str]


class VoiceModify(TypedDict, total=False):
  """Voice effects accepted by the Text-to-Audio API."""

  pitch: int
  intensity: int
  timbre: int
  sound_effects: str


class MiniMaxTextToSpeechError(RuntimeError):
  """Raised when the Text-to-Audio API cannot produce audio."""


class MiniMaxTextToSpeech(processor.Processor):
  """Converts each non-empty text part to a MiniMax audio part."""

  def __init__(
      self,
      *,
      api_key: str,
      voice_setting: VoiceSetting | None = None,
      model: str = DEFAULT_MODEL,
      region: Region = 'global_en',
      stream: bool = False,
      language_boost: str | None = None,
      output_format: OutputFormat = 'hex',
      pronunciation_dict: PronunciationDictionary | None = None,
      audio_setting: AudioSetting | None = None,
      voice_modify: VoiceModify | None = None,
      subtitle_enable: bool = False,
      with_text_passthrough: bool = True,
      client: httpx.AsyncClient | None = None,
  ):
    """Initializes the processor.

    Args:
      api_key: API key sent using bearer authentication.
      voice_setting: Voice selection and delivery controls.
      model: Speech synthesis model. Defaults to the current HD model.
      region: Regional API endpoint, either `global_en` or `cn_zh`.
      stream: Whether to request server-sent audio chunks.
      language_boost: Language or dialect recognition hint.
      output_format: Non-streaming response encoding (`hex` or `url`).
      pronunciation_dict: Custom pronunciation replacements.
      audio_setting: Audio encoding controls. Defaults to MP3.
      voice_modify: Optional pitch, intensity, timbre, and sound effects.
      subtitle_enable: Whether the response should include subtitles.
      with_text_passthrough: Whether input text is also yielded.
      client: Optional HTTP client, primarily useful for custom transports.

    Raises:
      ValueError: If a model, region, audio format, or output format is invalid.
    """
    if model not in SUPPORTED_MODELS:
      raise ValueError(f'Unsupported MiniMax speech model: {model}')
    if region not in _ENDPOINTS:
      raise ValueError(f'Unsupported MiniMax region: {region}')
    if output_format not in ('hex', 'url'):
      raise ValueError(f'Unsupported MiniMax output format: {output_format}')
    if stream and output_format != 'hex':
      raise ValueError('Streaming responses require hex output.')

    self._audio_setting: dict[str, Any] = {'format': 'mp3'}
    if audio_setting:
      self._audio_setting.update(audio_setting)
    audio_format = self._audio_setting['format']
    if audio_format not in _MIME_TYPES:
      raise ValueError(f'Unsupported MiniMax audio format: {audio_format}')

    self._model = model
    self._endpoint = _ENDPOINTS[region]
    self._stream = stream
    self._output_format = output_format
    self._with_text_passthrough = with_text_passthrough
    self._request_options = {
        'voice_setting': voice_setting,
        'language_boost': language_boost,
        'pronunciation_dict': pronunciation_dict,
        'voice_modify': voice_modify,
        'subtitle_enable': subtitle_enable,
    }
    self._headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
        'User-Agent': 'genai-processors',
    }
    self._client = client or httpx.AsyncClient(
        headers=self._headers,
        timeout=300,
    )

  @property
  def mimetype(self) -> str:
    """MIME type emitted by the configured audio format."""
    return _MIME_TYPES[self._audio_setting['format']]

  def _payload(self, text: str) -> dict[str, Any]:
    payload = {
        'model': self._model,
        'text': text,
        'stream': self._stream,
        'output_format': self._output_format,
        'audio_setting': self._audio_setting,
    }
    payload.update(
        key_value
        for key_value in self._request_options.items()
        if key_value[1] is not None
    )
    return payload

  def _response_audio(self, response: Mapping[str, Any]) -> tuple[str, int]:
    base_response = response.get('base_resp')
    status_code = (
        base_response.get('status_code')
        if isinstance(base_response, Mapping)
        else None
    )
    if status_code != 0:
      status_message = (
          base_response.get('status_msg')
          if isinstance(base_response, Mapping)
          else 'missing base_resp'
      )
      raise MiniMaxTextToSpeechError(
          f'Text-to-Audio request failed ({status_code}): {status_message}'
      )

    data = response.get('data')
    if not isinstance(data, Mapping):
      raise MiniMaxTextToSpeechError('Text-to-Audio response has no data.')
    audio = data.get('audio')
    status = data.get('status')
    if not isinstance(audio, str) or not audio:
      raise MiniMaxTextToSpeechError('Text-to-Audio response has no audio.')
    if not isinstance(status, int):
      raise MiniMaxTextToSpeechError('Text-to-Audio response has no status.')
    return audio, status

  async def _audio_bytes(self, audio: str) -> bytes:
    if self._output_format == 'url':
      response = await self._client.get(audio)
      response.raise_for_status()
      return response.content
    try:
      return bytes.fromhex(audio)
    except ValueError as error:
      raise MiniMaxTextToSpeechError(
          'Text-to-Audio response contains invalid hex audio.'
      ) from error

  async def _non_streaming_audio(self, text: str) -> bytes:
    response = await self._client.post(
        self._endpoint, headers=self._headers, json=self._payload(text)
    )
    response.raise_for_status()
    audio, _ = self._response_audio(response.json())
    return await self._audio_bytes(audio)

  async def _streaming_audio(self, text: str) -> AsyncIterable[bytes]:
    final_audio: str | None = None
    received_chunks = False
    async with self._client.stream(
        'POST',
        self._endpoint,
        headers=self._headers,
        json=self._payload(text),
    ) as response:
      response.raise_for_status()
      async for line in response.aiter_lines():
        if not line or line.startswith(':'):
          continue
        if line.startswith('data:'):
          line = line[5:].strip()
        if line == '[DONE]':
          continue
        try:
          event = json.loads(line)
        except json.JSONDecodeError as error:
          raise MiniMaxTextToSpeechError(
              'Text-to-Audio stream contains invalid JSON.'
          ) from error
        audio, status = self._response_audio(event)
        if status == 1:
          received_chunks = True
          yield await self._audio_bytes(audio)
        elif status == 2:
          final_audio = audio
    if not received_chunks and final_audio:
      yield await self._audio_bytes(final_audio)

  async def call(
      self, content: processor.ProcessorStream
  ) -> AsyncIterable[content_api.ProcessorPartTypes]:
    """Yields audio for text and passes non-text parts through unchanged."""
    async for part in content:
      is_text = content_api.is_text(part.mimetype)
      if not is_text or self._with_text_passthrough:
        yield part
      if not is_text or not part.text:
        continue

      if self._stream:
        async for audio in self._streaming_audio(part.text):
          yield content_api.ProcessorPart(
              audio, mimetype=self.mimetype, role='model'
          )
      else:
        audio = await self._non_streaming_audio(part.text)
        yield content_api.ProcessorPart(
            audio, mimetype=self.mimetype, role='model'
        )
