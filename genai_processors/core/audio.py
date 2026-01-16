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
"""Audio processors."""

from collections.abc import AsyncIterable
import io
import re
import wave

from genai_processors import content_api
from genai_processors import processor

_DEFAULT_SAMPLE_RATE = 24000


def _l16_to_wav(
    l16_audio_data: bytes, sample_rate: int, num_channels: int = 1
) -> bytes:
  """Converts L16 audio data to the WAV file format.

  Args:
      l16_audio_data: The raw L16 audio data.
      sample_rate: The sample rate of the audio (e.g., 8000, 16000).
      num_channels: The number of audio channels (e.g., 1 for mono, 2 for
        stereo).

  Returns:
      bytes: A byte string containing the WAV file data.
  """
  wav_stream = io.BytesIO()

  with wave.open(wav_stream, 'wb') as wav_file:
    wav_file.setnchannels(num_channels)
    wav_file.setsampwidth(2)  # L16 is 16-bit, which is 2 bytes
    wav_file.setframerate(sample_rate)
    wav_file.writeframes(l16_audio_data)

  return wav_stream.getvalue()


class AudioToWav(processor.Processor):
  """Concatenates consecutive audio parts into a single WAV part.

  Non live Gemini models don't handle lots of small audio parts well. We have to
  concatenate them into a single part to make sure they are handled correctly.
  """

  def __init__(self):
    self._buffer: list[bytes] = []
    self._mimetype: str | None = None

  async def _flush(
      self,
  ) -> AsyncIterable[content_api.ProcessorPartTypes]:
    if self._buffer:
      sample_rate = _DEFAULT_SAMPLE_RATE
      if self._mimetype:
        match = re.search(r'rate=(\d+)', self._mimetype)
        if match:
          sample_rate = int(match.group(1))

      part = content_api.ProcessorPart(
          # Non Live Gemini API models do not support audio/l16 or audio/pcm
          # mime types. So we have to convert it to audio.wav.
          _l16_to_wav(b''.join(self._buffer), sample_rate),
          mimetype='audio/wav',
      )
      self._buffer = []
      self._mimetype = None
      yield part

  async def call(
      self,
      content: AsyncIterable[content_api.ProcessorPartTypes],
  ) -> AsyncIterable[content_api.ProcessorPartTypes]:
    """Concatenates consecutive audio parts."""
    async for part in content:
      if content_api.is_audio(part.mimetype):
        if not (
            part.mimetype.startswith('audio/l16')
            or part.mimetype.startswith('audio/pcm')
        ):
          raise ValueError(
              'Only audio/l16 or audio/pcm is supported. Unsupported audio'
              f' mimetype: {part.mimetype}'
          )

        if self._buffer and part.mimetype != self._mimetype:
          async for flushed_part in self._flush():
            yield flushed_part

        self._buffer.append(part.bytes)
        self._mimetype = part.mimetype
      else:
        async for flushed_part in self._flush():
          yield flushed_part
        yield part

    async for flushed_part in self._flush():
      yield flushed_part
