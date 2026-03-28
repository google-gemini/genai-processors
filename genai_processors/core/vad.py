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
"""Voice Activity Detection (VAD) processor.

This processor detects speech and silence transitions in streaming audio using
the WebRTC VAD algorithm.

Audio parts (audio/l16 or audio/pcm) are passed through unchanged in their
original format. When a transition is detected, a VAD event part is injected
into the output stream:

  * Start of speech:
      ``ProcessorPart.from_dataclass(StartOfSpeech())``
  * End of speech:
      ``ProcessorPart.from_dataclass(EndOfSpeech())``

Non-audio parts are always passed through unchanged.

To ensure correct ordering, the processor buffers audio frames during the
detection window. When a ``StartOfSpeech`` transition is detected, the event
is emitted *before* the buffered audio that triggered it. When an
``EndOfSpeech`` transition is detected, the buffered audio is emitted first,
followed by the event.

The WebRTC VAD only supports sample rates of 8000, 16000, 32000, or 48000 Hz.
If input audio uses a different sample rate (e.g. the common 24000 Hz), the
processor automatically resamples internally for VAD analysis while passing
through the original audio unchanged.

Example usage::

  vad_processor = vad.Vad(sample_rate=16000, aggressiveness=3)

  # Chain with other processors:
  pipeline = audio_source + vad_processor + downstream_processor
"""

import collections
import math
import re
import struct
from typing import AsyncIterable

from genai_processors import content_api
from genai_processors import mime_types
from genai_processors import processor
from genai_processors.core import speech_events

ProcessorPart = content_api.ProcessorPart
ProcessorPartTypes = content_api.ProcessorPartTypes

StartOfSpeech = speech_events.StartOfSpeech
EndOfSpeech = speech_events.EndOfSpeech

_VALID_SAMPLE_RATES = (8000, 16000, 32000, 48000)
_VALID_FRAME_DURATIONS_MS = (10, 20, 30)

# The internal sample rate used for VAD analysis when the input sample rate
# is not directly supported by the VAD algorithm.
_DEFAULT_VAD_SAMPLE_RATE = 16000

import webrtcvad


def _create_vad(aggressiveness: int) -> webrtcvad.Vad:
  return webrtcvad.Vad(aggressiveness)


def _resample(
    data: bytes,
    src_rate: int,
    dst_rate: int,
) -> bytes:
  """Resamples 16-bit PCM audio using nearest-neighbor interpolation.

  Args:
    data: Input audio as raw 16-bit little-endian PCM bytes.
    src_rate: Source sample rate in Hz.
    dst_rate: Destination sample rate in Hz.

  Returns:
    Resampled audio as raw 16-bit little-endian PCM bytes.
  """
  if src_rate == dst_rate:
    return data
  src_samples = len(data) // 2
  if src_samples == 0:
    return b''
  dst_samples = int(src_samples * dst_rate / src_rate)
  if dst_samples == 0:
    return b''

  ratio = src_rate / dst_rate
  result = bytearray()
  for i in range(dst_samples):
    idx = int(i * ratio) * 2
    result.extend(data[idx : idx + 2])
  return bytes(result)


class Vad(processor.Processor):
  """Voice Activity Detection processor using WebRTC VAD.

  Processes streaming audio parts and detects speech/silence transitions.
  Audio parts are passed through unchanged in their original format,
  with VAD event parts injected when transitions are detected.

  If the input audio sample rate is not supported by the VAD algorithm
  (8000, 16000, 32000, or 48000 Hz), the audio is internally resampled
  to the configured `sample_rate` for analysis. The original audio parts
  are always passed through in their original format.

  The detection uses a ring buffer (sliding window) for debouncing. When
  a sufficient fraction of frames in the window are voiced (controlled by
  `speech_threshold`), the processor transitions to the "speaking" state
  and emits a start-of-speech event. Conversely, when enough frames are
  unvoiced (controlled by `silence_threshold`), it transitions to
  "silence" and emits end-of-speech.

  To ensure correct event ordering, audio parts are buffered during the
  detection window:

  * `StartOfSpeech` is emitted *before* the buffered audio frames that
    contributed to the decision.
  * `EndOfSpeech` is emitted *after* the buffered audio frames,
    followed by the event.

  Attributes:
    sample_rate: The VAD analysis sample rate. Input audio at other rates is
      resampled internally to this rate for VAD processing.
    aggressiveness: WebRTC VAD aggressiveness (0-3).
    frame_duration_ms: Duration of each analysis frame in milliseconds.
  """

  def __init__(
      self,
      sample_rate: int = _DEFAULT_VAD_SAMPLE_RATE,
      aggressiveness: int = 2,
      frame_duration_ms: int = 20,
      padding_duration_ms: int = 100,
      speech_threshold: float = 0.8,
      silence_threshold: float = 0.8,
  ):
    """Initializes the VAD processor.

    Args:
      sample_rate: The sample rate to use for VAD analysis in Hz. Must be one of
        8000, 16000, 32000, or 48000. Audio at other rates is automatically
        resampled to this rate for analysis.
      aggressiveness: Integer in [0, 3] controlling the aggressiveness of the
        VAD. 0 is the least aggressive about filtering out non-speech, 3 is the
        most aggressive.
      frame_duration_ms: The frame duration in milliseconds. Must be one of 10,
        20, or 30.
      padding_duration_ms: The size of the sliding window (ring buffer) for
        debouncing transitions, in milliseconds.
      speech_threshold: Fraction of voiced frames in the ring buffer required to
        trigger a start-of-speech event (0.0 to 1.0).
      silence_threshold: Fraction of unvoiced frames in the ring buffer required
        to trigger an end-of-speech event (0.0 to 1.0).

    Raises:
      ValueError: If sample_rate or frame_duration_ms are not valid.
    """
    if sample_rate not in _VALID_SAMPLE_RATES:
      raise ValueError(
          f'Invalid sample rate: {sample_rate}. Must be one of'
          f' {_VALID_SAMPLE_RATES}.'
      )
    if frame_duration_ms not in _VALID_FRAME_DURATIONS_MS:
      raise ValueError(
          f'Invalid frame duration: {frame_duration_ms}. Must be one'
          f' of {_VALID_FRAME_DURATIONS_MS}.'
      )
    self._sample_rate = sample_rate
    self._frame_duration_ms = frame_duration_ms
    self._speech_threshold = speech_threshold
    self._silence_threshold = silence_threshold

    # Ring buffer for debouncing. Stores (original_audio_part, is_speech).
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    self._ring_buffer: collections.deque[tuple[ProcessorPart, bool]] = (
        collections.deque(maxlen=num_padding_frames)
    )

    # State tracking.
    self._triggered = False
    self._audio_buffer = b''
    self._input_sample_rate: int | None = None
    self._input_mimetype: str = ''

    # Create the VAD instance.
    self._vad = _create_vad(aggressiveness)

  def _vad_frame_byte_size(self) -> int:
    """Number of bytes per frame at the VAD analysis sample rate."""
    return int(self._sample_rate * (self._frame_duration_ms / 1000.0) * 2)

  def _input_frame_byte_size(self, input_sample_rate: int) -> int:
    """Number of bytes per frame at the input sample rate."""
    return int(input_sample_rate * (self._frame_duration_ms / 1000.0) * 2)

  def _process_frame(
      self,
      vad_frame: bytes,
      original_part: ProcessorPart,
  ) -> tuple[str | None, list[ProcessorPart]]:
    """Processes a single audio frame and returns any VAD event.

    Args:
      vad_frame: Audio frame resampled to the VAD sample rate.
      original_part: The original audio part (at input sample rate) to buffer
        for emission.

    Returns:
      A tuple of (event_type, buffered_audio_parts) where event_type is
      'start_of_speech', 'end_of_speech', or None. buffered_audio_parts
      contains the original audio parts from the ring buffer that should
      be flushed.
    """
    is_speech = self._vad.is_speech(vad_frame, self._sample_rate)
    self._ring_buffer.append((original_part, is_speech))

    if not self._triggered:
      num_voiced = sum(1 for _, voiced in self._ring_buffer if voiced)
      if num_voiced > self._speech_threshold * self._ring_buffer.maxlen:
        self._triggered = True
        # Collect all buffered original parts.
        buffered = [p for p, _ in self._ring_buffer]
        self._ring_buffer.clear()
        return 'start_of_speech', buffered
    else:
      num_unvoiced = sum(1 for _, voiced in self._ring_buffer if not voiced)
      if num_unvoiced > self._silence_threshold * self._ring_buffer.maxlen:
        self._triggered = False
        # Collect buffered original parts.
        buffered = [p for p, _ in self._ring_buffer]
        self._ring_buffer.clear()
        return 'end_of_speech', buffered
    return None, []

  async def call(
      self,
      content: processor.ProcessorStream,
  ) -> AsyncIterable[ProcessorPartTypes]:
    """Processes audio parts and detects speech/silence transitions.

    All input audio is yielded in its original format. When speech
    transitions are detected, VAD event parts are injected into the
    stream at the correct position:

    * ``StartOfSpeech`` is emitted before the audio that triggered it.
    * ``EndOfSpeech`` is emitted after the buffered speech/silence audio.

    If the input audio sample rate is not natively supported by the VAD,
    it is internally resampled for analysis while the original audio is
    passed through unchanged.

    Non-audio parts are passed through immediately.

    Args:
      content: Input stream of processor parts.

    Yields:
      Input parts (possibly reordered around events) plus VAD event parts.
    """
    async for part in content:
      if not content_api.is_audio(part.mimetype):
        yield part
        continue

      if not (
          part.mimetype.startswith('audio/l16')
          or part.mimetype.startswith('audio/pcm')
      ):
        raise ValueError(
            f'Unsupported audio mimetype: {part.mimetype}. Expected'
            ' audio/l16 or audio/pcm.'
        )

      audio_data = part.bytes or b''

      # Determine the input sample rate.
      input_rate = mime_types.parse_frame_rate(part.mimetype, self._sample_rate)
      self._input_mimetype = part.mimetype

      # Determine per-frame sizes at the input sample rate.
      input_frame_size = self._input_frame_byte_size(input_rate)
      vad_frame_size = self._vad_frame_byte_size()
      needs_resample = input_rate not in _VALID_SAMPLE_RATES

      # Accumulate audio data and process complete frames.
      self._audio_buffer += audio_data

      while len(self._audio_buffer) >= input_frame_size:
        input_frame = self._audio_buffer[:input_frame_size]
        self._audio_buffer = self._audio_buffer[input_frame_size:]

        # Create the original-format part for passthrough.
        original_part = ProcessorPart(
            input_frame,
            mimetype=part.mimetype,
        )

        # Resample for VAD analysis if needed.
        if needs_resample:
          vad_frame = _resample(
              input_frame,
              input_rate,
              self._sample_rate,
          )
          # Ensure the resampled frame is the expected size.
          vad_frame = vad_frame[:vad_frame_size].ljust(vad_frame_size, b'\x00')
        else:
          vad_frame = input_frame

        if (
            not self._triggered
            and len(self._ring_buffer) == self._ring_buffer.maxlen
        ):
          # Emit the oldest part in the ring buffer when no speech is detected.
          yield self._ring_buffer[0][0]

        event, buffered_parts = self._process_frame(
            vad_frame,
            original_part,
        )

        if event == 'start_of_speech':
          # Emit start event BEFORE the buffered audio.
          yield ProcessorPart.from_dataclass(
              dataclass=StartOfSpeech(),
          )
          for buffered_part in buffered_parts:
            yield buffered_part
        elif event == 'end_of_speech':
          # The preceding buffered parts were already yielded while _triggered
          # was True. We only need to yield the current frame before the end
          # event.
          yield original_part
          yield ProcessorPart.from_dataclass(
              dataclass=EndOfSpeech(),
          )
        else:
          # No transition — if we're in triggered (speaking) state,
          # emit the frame immediately. Otherwise it stays in the
          # ring buffer for debouncing.
          if self._triggered:
            yield original_part

    # Flush any remaining partial frame data from the audio buffer.
    if self._audio_buffer:
      yield ProcessorPart(
          self._audio_buffer,
          mimetype=self._input_mimetype or 'audio/l16',
      )
      self._audio_buffer = b''

    # Flush any remaining frames in the ring buffer that have NOT
    # been yielded yet. When triggered (speaking), frames are
    # yielded immediately so only the untriggered case needs flush.
    if self._ring_buffer and not self._triggered:
      for original_part, _ in self._ring_buffer:
        yield original_part
      self._ring_buffer.clear()

    # If stream ends while in speech state, emit end of speech.
    if self._triggered:
      self._triggered = False
      self._ring_buffer.clear()
      yield ProcessorPart.from_dataclass(
          dataclass=EndOfSpeech(),
      )
