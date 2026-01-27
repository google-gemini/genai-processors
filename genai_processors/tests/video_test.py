"""Tests for video processors."""

import time
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import cv2
from genai_processors import content_api
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
      input_stream = streams.stream_content(
          [
              content_api.ProcessorPart(
                  'hello',
              ),
          ],
          with_delay_sec=0.3,
      )
      video_in = video.VideoIn()
      with self.assertRaises(IOError):
        await video_in(input_stream)


if __name__ == '__main__':
  absltest.main()
