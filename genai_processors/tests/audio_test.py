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

import io
import wave

from absl.testing import absltest
from absl.testing import parameterized
from genai_processors import content_api
from genai_processors import processor
from genai_processors.core import audio


class AudioTest(parameterized.TestCase):

  def test_empty(self):
    p = audio.AudioToWav()
    output = processor.apply_sync(p, [])
    self.assertEmpty(output)

  @parameterized.named_parameters(
      dict(
          testcase_name='single_audio_part',
          input_parts=[
              content_api.ProcessorPart(
                  b'\x01\x02\x03\x04', mimetype='audio/l16'
              ),
          ],
          expected_framerate=24000,
          expected_readframes=b'\x01\x02\x03\x04',
      ),
      dict(
          testcase_name='single_audio_pcm_part',
          input_parts=[
              content_api.ProcessorPart(
                  b'\x01\x02\x03\x04', mimetype='audio/pcm'
              ),
          ],
          expected_framerate=24000,
          expected_readframes=b'\x01\x02\x03\x04',
      ),
      dict(
          testcase_name='multiple_audio_parts',
          input_parts=[
              content_api.ProcessorPart(b'\x01\x02', mimetype='audio/l16'),
              content_api.ProcessorPart(b'\x03\x04', mimetype='audio/l16'),
          ],
          expected_framerate=24000,
          expected_readframes=b'\x01\x02\x03\x04',
      ),
      dict(
          testcase_name='audio_with_rate',
          input_parts=[
              content_api.ProcessorPart(
                  b'\x01\x02\x03\x04', mimetype='audio/l16; rate=16000'
              ),
          ],
          expected_framerate=16000,
          expected_readframes=b'\x01\x02\x03\x04',
      ),
  )
  def test_audio_to_wav(
      self, input_parts, expected_framerate, expected_readframes
  ):
    p = audio.AudioToWav()
    output = processor.apply_sync(p, input_parts)
    self.assertLen(output, 1)
    self.assertEqual(output[0].mimetype, 'audio/wav')

    with wave.open(io.BytesIO(output[0].bytes), 'rb') as wf:
      self.assertEqual(wf.getnchannels(), 1)
      self.assertEqual(wf.getsampwidth(), 2)
      self.assertEqual(wf.getframerate(), expected_framerate)
      self.assertEqual(wf.readframes(10), expected_readframes)

  def test_interspersed_with_non_audio(self):
    p = audio.AudioToWav()
    input_parts = [
        content_api.ProcessorPart(
            b'\x01\x02', mimetype='audio/l16; rate=16000'
        ),
        content_api.ProcessorPart('hello', mimetype='text/plain'),
        content_api.ProcessorPart(b'\x03\x04', mimetype='audio/l16; rate=8000'),
    ]
    output = processor.apply_sync(p, input_parts)
    self.assertLen(output, 3)
    self.assertEqual(output[0].mimetype, 'audio/wav')
    self.assertEqual(output[1].mimetype, 'text/plain')
    self.assertEqual(output[1].text, 'hello')
    self.assertEqual(output[2].mimetype, 'audio/wav')

    with wave.open(io.BytesIO(output[0].bytes), 'rb') as wf:
      self.assertEqual(wf.getframerate(), 16000)
      self.assertEqual(wf.readframes(10), b'\x01\x02')
    with wave.open(io.BytesIO(output[2].bytes), 'rb') as wf:
      self.assertEqual(wf.getframerate(), 8000)
      self.assertEqual(wf.readframes(10), b'\x03\x04')

  def test_non_audio_only(self):
    p = audio.AudioToWav()
    input_parts = [
        content_api.ProcessorPart('hello', mimetype='text/plain'),
        content_api.ProcessorPart('world', mimetype='text/plain'),
    ]
    output = processor.apply_sync(p, input_parts)
    self.assertEqual(output, input_parts)

  def test_unsupported_audio_mimetype(self):
    p = audio.AudioToWav()
    input_parts = [
        content_api.ProcessorPart(b'\x01\x02', mimetype='audio/mpeg'),
    ]
    with self.assertRaisesRegex(
        ValueError,
        'Only audio/l16 or audio/pcm is supported. Unsupported audio'
        ' mimetype: audio/mpeg',
    ):
      processor.apply_sync(p, input_parts)


if __name__ == '__main__':
  absltest.main()
