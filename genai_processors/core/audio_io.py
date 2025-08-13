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
"""Audio processors."""

import asyncio
from typing import AsyncIterable, Optional

from genai_processors import content_api
from genai_processors import processor
import pyaudio

ProcessorPart = content_api.ProcessorPart

# Audio output chunk size in bytes.
AUDIO_OUT_CHUNK_SIZE = 1024

# Add accepted audio formats here.
AudioFormats = pyaudio.paInt16 | pyaudio.paInt24


@processor.source()
async def PyAudioIn(
    pya: pyaudio.PyAudio,
    substream_name: str = 'realtime',
    audio_format: AudioFormats = pyaudio.paInt16,  # 16-bit PCM.
    channels: int = 1,
    rate: int = 24000,
    use_pcm_mimetype: bool = False,
):
  """Receives audio input and inserts it into the input stream.

  The audio input is received from the default input device.

  Args:
    pya: The pyaudio object to use for capturing audio.
    substream_name: The name of the substream that will contain all the audio
      parts captured from the mic.
    audio_format: The audio format to use for the audio.
    channels: The number of channels in the audio.
    rate: The sample rate of the audio.
    use_pcm_mimetype: Whether to use PCM mimetype instead of the more specific
      l16 or l24 mimetype.
  """
  mimetype = 'audio/'
  match audio_format:
    case pyaudio.paInt16:
      mimetype += 'pcm' if use_pcm_mimetype else 'l16'
    case pyaudio.paInt24:
      mimetype += 'pcm' if use_pcm_mimetype else 'l24'
    case _:
      raise ValueError(f'Unsupported audio format: {format}')
  mimetype = f'{mimetype};rate={rate}'

  mic_info = pya.get_default_input_device_info()
  audio_stream = await asyncio.to_thread(
      pya.open,
      format=audio_format,
      channels=channels,
      rate=rate,
      input=True,
      input_device_index=mic_info['index'],
      frames_per_buffer=AUDIO_OUT_CHUNK_SIZE,
  )
  if __debug__:  # pylint: disable=undefined-variable
    kwargs = {'exception_on_overflow': False}
  else:
    kwargs = {}

  while True:
    data = await asyncio.to_thread(
        audio_stream.read, AUDIO_OUT_CHUNK_SIZE, **kwargs
    )
    yield ProcessorPart(
        data, mimetype=mimetype, substream_name=substream_name, role='user'
    )


class PyAudioOut(processor.Processor):
  """Receives audio output from a live session and talks back to the user.

  Uses pyaudio to play audio back to the user.

  All non audio parts are passed through based on the `passthrough_audio` param
  passed to the constructor.

  Combine this processor with `RateLimitAudio` to receive the audio chunks at
  the time where they need to be played back to the user.
  """

  def __init__(
      self,
      pya: pyaudio.PyAudio,
      audio_format=pyaudio.paInt16,  # 16-bit PCM.
      channels: int = 1,
      rate: int = 24000,
      passthrough_audio: bool = False,
  ):
    self._pya = pya
    self._format = audio_format
    self._channels = channels
    self._rate = rate
    self._passthrough_audio = passthrough_audio

  async def call(
      self, content: AsyncIterable[ProcessorPart]
  ) -> AsyncIterable[ProcessorPart]:
    """Receives audio output from a live session."""
    audio_output = asyncio.Queue[Optional[ProcessorPart]]()

    stream = await asyncio.to_thread(
        self._pya.open,
        format=self._format,
        channels=self._channels,
        rate=self._rate,
        output=True,
    )

    async def play_audio():  # pylint: disable=invalid-name
      while part := await audio_output.get():
        if part.part.inline_data is not None:
          await asyncio.to_thread(stream.write, part.part.inline_data.data)

    play_audio_task = processor.create_task(play_audio())

    async for part in content:
      if content_api.is_audio(part.mimetype):
        audio_output.put_nowait(part)
        if self._passthrough_audio:
          yield part
      else:
        yield part
    await audio_output.put(None)
    play_audio_task.cancel()
