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

import os
import time
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import cv2
from genai_processors import content_api
from genai_processors import mime_types
from genai_processors import streams
from genai_processors.core import video
import numpy as np
import PIL.Image


def mock_video_capture_read():
  # Delay it to return the image bytes after the text part.
  time.sleep(0.05)
  img = PIL.Image.new('RGB', (100, 100), color='black')
  return True, np.array(img.convert('RGB'))


class VideoInTest(parameterized.TestCase, unittest.IsolatedAsyncioTestCase):
  """Tests for the VideoIn processor."""

  def setUp(self):
    super().setUp()
    self.cv2_mock = mock.MagicMock()
    self.cv2_mock.read = mock.MagicMock()
    self.cv2_mock.read.side_effect = mock_video_capture_read

  @parameterized.named_parameters(
      ('default', 'default'),
      ('realtime', 'realtime'),
  )
  async def test_video_in(self, substream_name):
    with mock.patch.object(
        cv2,
        'VideoCapture',
        return_value=self.cv2_mock,
    ):
      input_stream = streams.stream_content(
          [
              content_api.ProcessorPart(
                  'hello',
              ),
              content_api.ProcessorPart(
                  'world',
              ),
          ],
          # This delay is added after the text part is returned to ensure the
          # stream ends after the audio part is returned.
          with_delay_sec=0.3,
      )
      video_in = video.VideoIn(substream_name=substream_name)
      output = await video_in(input_stream).gather()
      self.assertLen(output, 3)
      self.assertEqual(output[0], content_api.ProcessorPart('hello'))
      self.assertEqual(output[2], content_api.ProcessorPart('world'))
      # Compare all fields of the image part but not the image bytes. Just check
      # that the image bytes are not empty.
      self.assertEqual(output[1].mimetype, 'image/jpeg')
      self.assertEqual(output[1].substream_name, substream_name)
      self.assertEqual(output[1].role, 'user')
      self.assertIsInstance(output[1].part.inline_data.data, bytes)
      self.assertGreater(len(output[1].part.inline_data.data), 500)

  async def test_video_in_with_exception(self):
    self.cv2_mock.read.side_effect = IOError('test exception')
    with mock.patch.object(
        cv2,
        'VideoCapture',
        return_value=self.cv2_mock,
    ):
      video_in = video.VideoIn()
      with self.assertRaises(IOError):
        await video_in


class VideoExtractTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):
  """Tests for the VideoExtract processor."""

  @parameterized.named_parameters(
      dict(
          testcase_name='video_only',
          video_format=video.VideoAVFormat.VIDEO,
          expected_mime_types=['image/jpeg', 'image/jpeg'],
      ),
      dict(
          testcase_name='audio_only',
          video_format=video.VideoAVFormat.AUDIO,
          expected_mime_types=['audio/l16;rate=16000;channels=1'],
      ),
      dict(
          testcase_name='both_interleaved',
          video_format=video.VideoAVFormat.BOTH_INTERLEAVED,
          expected_mime_types=[
              'image/jpeg',
              'audio/l16;rate=16000;channels=1',
              'image/jpeg',
              'audio/l16;rate=16000;channels=1',
          ],
      ),
  )
  async def test_video_extract(self, video_format, expected_mime_types):
    """Tests the VideoExtract processor."""
    video_path = os.path.join(
        os.path.dirname(__file__), 'testdata/test_video.mp4'
    )

    with open(video_path, 'rb') as f:
      video_bytes = f.read()

    video_part = content_api.ProcessorPart(
        video_bytes,
        mimetype='video/mp4',
        substream_name='test_substream',
        role='test_role',
    )
    video_extract = video.VideoExtract(
        video_format=video_format,
        frames_per_second=1.0,
    )
    content = streams.stream_content([video_part])
    output = await streams.gather_stream(video_extract.to_processor()(content))

    self.assertLen(output, len(expected_mime_types))
    for i, part in enumerate(output):
      self.assertEqual(part.role, 'test_role')
      self.assertEqual(part.substream_name, 'test_substream')
      self.assertEqual(part.mimetype, expected_mime_types[i])

      if video_format == video.VideoAVFormat.VIDEO:
        self.assertEqual(part.metadata['video_timestamp'], float(i))
      elif video_format == video.VideoAVFormat.BOTH_INTERLEAVED:
        self.assertEqual(part.metadata['video_timestamp'], float(i // 2))

      if mime_types.is_image(part.mimetype):
        # Odd frames are red. Due to compression exact value may fluctuate.
        image_data = np.array(part.pil_image)
        self.assertGreater(image_data[0][0][0], 250)
        self.assertLess(image_data[0][0][1], 10)
        self.assertLess(image_data[0][0][2], 10)
      if mime_types.is_audio(part.mimetype):
        # Audio track is a constant 42.
        audio_data = np.frombuffer(part.bytes, dtype=np.int16)
        self.assertEqual(audio_data[1000], 42)


if __name__ == '__main__':
  absltest.main()
