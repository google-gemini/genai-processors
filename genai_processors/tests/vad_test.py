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

import math
import struct
import unittest

from absl.testing import absltest
from absl.testing import parameterized
from genai_processors import content_api
from genai_processors import mime_types
from genai_processors.core import speech_events
from genai_processors.core import vad


def _make_silence(
    duration_sec: float,
    sample_rate: int = 16000,
) -> bytes:
  """Creates silent (zero) audio data."""
  num_samples = int(sample_rate * duration_sec)
  return b'\x00\x00' * num_samples


def _make_speech(
    duration_sec: float,
    sample_rate: int = 16000,
    frequency: float = 440.0,
    amplitude: float = 20000.0,
) -> bytes:
  """Creates a sine wave that the VAD will detect as speech."""
  num_samples = int(sample_rate * duration_sec)
  samples = []
  for i in range(num_samples):
    value = int(amplitude * math.sin(2 * math.pi * frequency * i / sample_rate))
    value = max(-32768, min(32767, value))
    samples.append(struct.pack('<h', value))
  return b''.join(samples)


def _make_audio_part(
    audio_data: bytes,
    sample_rate: int = 16000,
    mimetype: str | None = None,
) -> content_api.ProcessorPart:
  """Creates an audio ProcessorPart."""
  if mimetype is None:
    mimetype = f'audio/l16;rate={sample_rate}'
  return content_api.ProcessorPart(
      audio_data,
      mimetype=mimetype,
  )


def _is_vad_event(part: content_api.ProcessorPart) -> bool:
  return speech_events.is_start_of_speech(
      part
  ) or speech_events.is_end_of_speech(part)


def _get_vad_events(
    parts: list[content_api.ProcessorPart],
) -> list[str]:
  """Returns a list of VAD event names from output parts."""
  events = []
  for p in parts:
    if speech_events.is_start_of_speech(p):
      events.append('start_of_speech')
    elif speech_events.is_end_of_speech(p):
      events.append('end_of_speech')
  return events


class VadConstructorTest(parameterized.TestCase):

  def test_invalid_sample_rate(self):
    with self.assertRaisesRegex(ValueError, 'Invalid sample rate'):
      vad.Vad(sample_rate=24000)

  def test_invalid_frame_duration(self):
    with self.assertRaisesRegex(ValueError, 'Invalid frame duration'):
      vad.Vad(frame_duration_ms=25)

  @parameterized.parameters(8000, 16000, 32000, 48000)
  def test_valid_sample_rates(self, sample_rate):
    v = vad.Vad(sample_rate=sample_rate)
    self.assertIsNotNone(v)

  @parameterized.parameters(10, 20, 30)
  def test_valid_frame_durations(self, frame_duration_ms):
    v = vad.Vad(frame_duration_ms=frame_duration_ms)
    self.assertIsNotNone(v)


class VadProcessorTest(
    parameterized.TestCase,
    unittest.IsolatedAsyncioTestCase,
):

  async def test_empty_input(self):
    p = vad.Vad(sample_rate=16000)
    output = await p([]).gather()
    self.assertEmpty(output)

  async def test_non_audio_passthrough(self):
    p = vad.Vad(sample_rate=16000)
    input_parts = [
        content_api.ProcessorPart(
            'hello',
            mimetype='text/plain',
        ),
        content_api.ProcessorPart(
            'world',
            mimetype='text/plain',
        ),
    ]
    output = await p(input_parts).gather()
    self.assertLen(output, 2)
    self.assertEqual(output[0].text, 'hello')
    self.assertEqual(output[1].text, 'world')
    self.assertEmpty(_get_vad_events(output))

  async def test_silence_only_no_events(self):
    """All-zero audio should not trigger any VAD events."""
    p = vad.Vad(sample_rate=16000, aggressiveness=3)
    silence = _make_silence(1.0, sample_rate=16000)
    input_parts = [
        _make_audio_part(silence, sample_rate=16000),
    ]
    output = await p(input_parts).gather()
    events = _get_vad_events(output)
    self.assertEmpty(events)

  async def test_speech_triggers_start_of_speech(self):
    """Continuous speech should trigger a StartOfSpeech."""
    p = vad.Vad(
        sample_rate=16000,
        aggressiveness=1,
        frame_duration_ms=30,
        padding_duration_ms=90,
        speech_threshold=0.5,
    )
    speech = _make_speech(
        1.0,
        sample_rate=16000,
        amplitude=30000.0,
    )
    input_parts = [
        _make_audio_part(speech, sample_rate=16000),
    ]
    output = await p(input_parts).gather()
    events = _get_vad_events(output)
    self.assertIn('start_of_speech', events)

  async def test_speech_then_silence_triggers_both(self):
    """Speech then silence should trigger both events."""
    p = vad.Vad(
        sample_rate=16000,
        aggressiveness=1,
        frame_duration_ms=30,
        padding_duration_ms=90,
        speech_threshold=0.5,
        silence_threshold=0.5,
    )
    speech = _make_speech(
        0.5,
        sample_rate=16000,
        amplitude=30000.0,
    )
    silence = _make_silence(0.5, sample_rate=16000)
    audio_data = speech + silence
    input_parts = [
        _make_audio_part(audio_data, sample_rate=16000),
    ]
    output = await p(input_parts).gather()
    events = _get_vad_events(output)
    self.assertIn('start_of_speech', events)
    self.assertIn('end_of_speech', events)
    # Start should come before End.
    start_idx = events.index('start_of_speech')
    end_idx = events.index('end_of_speech')
    self.assertLess(start_idx, end_idx)

    # Ensure no duplicate frames were emitted when transitioning to silence.
    audio_parts = [
        part for part in output if content_api.is_audio(part.mimetype)
    ]
    total_output_bytes = sum(len(part.bytes) for part in audio_parts)
    self.assertEqual(total_output_bytes, len(audio_data))

  async def test_start_of_speech_before_audio(self):
    """StartOfSpeech should appear before audio parts."""
    p = vad.Vad(
        sample_rate=16000,
        aggressiveness=1,
        frame_duration_ms=30,
        padding_duration_ms=90,
        speech_threshold=0.5,
    )
    speech = _make_speech(
        1.0,
        sample_rate=16000,
        amplitude=30000.0,
    )
    input_parts = [
        _make_audio_part(speech, sample_rate=16000),
    ]
    output = await p(input_parts).gather()
    # Find the first StartOfSpeech event.
    sos_idx = None
    first_audio_after_sos = None
    for i, part in enumerate(output):
      if speech_events.is_start_of_speech(part):
        sos_idx = i
      elif (
          sos_idx is not None
          and content_api.is_audio(part.mimetype)
          and first_audio_after_sos is None
      ):
        first_audio_after_sos = i
        break
    self.assertIsNotNone(sos_idx, 'StartOfSpeech not found')
    self.assertIsNotNone(
        first_audio_after_sos,
        'No audio after StartOfSpeech',
    )
    self.assertLess(sos_idx, first_audio_after_sos)

  async def test_audio_parts_passed_through(self):
    """Audio data should appear in the output."""
    p = vad.Vad(
        sample_rate=16000,
        aggressiveness=1,
        frame_duration_ms=30,
        padding_duration_ms=90,
        speech_threshold=0.5,
    )
    speech = _make_speech(
        0.5,
        sample_rate=16000,
        amplitude=30000.0,
    )
    input_parts = [
        _make_audio_part(speech, sample_rate=16000),
    ]
    output = await p(input_parts).gather()
    audio_parts = [
        part for part in output if content_api.is_audio(part.mimetype)
    ]
    self.assertGreater(len(audio_parts), 0)
    # Total audio bytes in output should match input.
    total_output_bytes = sum(len(part.bytes) for part in audio_parts)
    self.assertEqual(total_output_bytes, len(speech))

  async def test_mixed_audio_and_text(self):
    """Non-audio parts should pass through immediately."""
    p = vad.Vad(sample_rate=16000, aggressiveness=3)
    silence = _make_silence(0.1, sample_rate=16000)
    input_parts = [
        _make_audio_part(silence, sample_rate=16000),
        content_api.ProcessorPart(
            'hello',
            mimetype='text/plain',
        ),
        _make_audio_part(silence, sample_rate=16000),
    ]
    output = await p(input_parts).gather()
    text_parts = [part for part in output if part.mimetype == 'text/plain']
    self.assertLen(text_parts, 1)
    self.assertEqual(text_parts[0].text, 'hello')

  async def test_end_of_stream_while_speaking(self):
    """Stream ending during speech emits EndOfSpeech."""
    p = vad.Vad(
        sample_rate=16000,
        aggressiveness=1,
        frame_duration_ms=30,
        padding_duration_ms=90,
        speech_threshold=0.5,
    )
    speech = _make_speech(
        1.0,
        sample_rate=16000,
        amplitude=30000.0,
    )
    input_parts = [
        _make_audio_part(speech, sample_rate=16000),
    ]
    output = await p(input_parts).gather()
    events = _get_vad_events(output)
    self.assertIn('start_of_speech', events)
    self.assertIn('end_of_speech', events)
    self.assertEqual(events[-1], 'end_of_speech')

  async def test_multiple_audio_chunks(self):
    """VAD should work across multiple small audio parts."""
    p = vad.Vad(
        sample_rate=16000,
        aggressiveness=1,
        frame_duration_ms=30,
        padding_duration_ms=90,
        speech_threshold=0.5,
    )
    speech = _make_speech(
        0.5,
        sample_rate=16000,
        amplitude=30000.0,
    )
    chunk_size = 640  # 20ms at 16kHz.
    input_parts = []
    for i in range(0, len(speech), chunk_size):
      chunk = speech[i : i + chunk_size]
      input_parts.append(
          _make_audio_part(chunk, sample_rate=16000),
      )
      input_parts.append(_make_audio_part(b'', sample_rate=16000))
    output = await p(input_parts).gather()
    events = _get_vad_events(output)
    self.assertIn('start_of_speech', events)

  async def test_pcm_mimetype_supported(self):
    """audio/pcm MIME type should be processed."""
    p = vad.Vad(sample_rate=16000, aggressiveness=3)
    silence = _make_silence(0.1, sample_rate=16000)
    part = content_api.ProcessorPart(
        silence,
        mimetype='audio/pcm',
    )
    output = await p([part]).gather()
    audio_parts = [p for p in output if content_api.is_audio(p.mimetype)]
    self.assertGreater(len(audio_parts), 0)

  async def test_unsupported_audio_mimetype_passthrough(self):
    """Non-PCM audio should raise an exception."""
    p = vad.Vad(sample_rate=16000, aggressiveness=3)
    part = content_api.ProcessorPart(
        b'\x00\x00',
        mimetype='audio/wav',
    )
    with self.assertRaisesRegex(ValueError, 'Unsupported audio mimetype'):
      output = await p([part]).gather()

  async def test_sample_rate_from_mimetype(self):
    """Sample rate should be extracted from MIME type."""
    p = vad.Vad(sample_rate=16000, aggressiveness=3)
    silence = _make_silence(0.1, sample_rate=8000)
    part = content_api.ProcessorPart(
        silence,
        mimetype='audio/l16;rate=8000',
    )
    output = await p([part]).gather()
    audio_parts = [p for p in output if content_api.is_audio(p.mimetype)]
    self.assertGreater(len(audio_parts), 0)

  async def test_uses_shared_dataclasses(self):
    """Events should be speech_events dataclasses."""
    p = vad.Vad(
        sample_rate=16000,
        aggressiveness=1,
        frame_duration_ms=30,
        padding_duration_ms=90,
        speech_threshold=0.5,
    )
    speech = _make_speech(
        1.0,
        sample_rate=16000,
        amplitude=30000.0,
    )
    input_parts = [
        _make_audio_part(speech, sample_rate=16000),
    ]
    output = await p(input_parts).gather()
    sos_parts = [p for p in output if speech_events.is_start_of_speech(p)]
    eos_parts = [p for p in output if speech_events.is_end_of_speech(p)]
    self.assertGreater(len(sos_parts), 0)
    self.assertGreater(len(eos_parts), 0)

  async def test_internal_resampling(self):
    """Audio at unsupported rates should be resampled internally but passed through in original format."""
    p = vad.Vad(
        sample_rate=16000,
        aggressiveness=1,
        frame_duration_ms=30,
        padding_duration_ms=90,
        speech_threshold=0.5,
    )
    # 24kHz is not supported by WebRTC VAD directly.
    speech = _make_speech(
        1.0,
        sample_rate=24000,
        amplitude=30000.0,
    )
    input_parts = [
        _make_audio_part(speech, sample_rate=24000),
    ]
    output = await p(input_parts).gather()
    events = _get_vad_events(output)
    self.assertIn('start_of_speech', events)

    audio_parts = [
        part for part in output if content_api.is_audio(part.mimetype)
    ]
    self.assertGreater(len(audio_parts), 0)
    self.assertTrue(
        all(part.mimetype == 'audio/l16;rate=24000' for part in audio_parts)
    )


if __name__ == '__main__':
  absltest.main()
