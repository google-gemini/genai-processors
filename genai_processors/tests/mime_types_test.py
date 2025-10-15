import dataclasses
import unittest

from absl.testing import absltest
from absl.testing import parameterized
from genai_processors import mime_types


@dataclasses.dataclass
class MyData:
  a: int


class MimeTypesTest(parameterized.TestCase, unittest.TestCase):

  @parameterized.parameters([
      ('text/plain', True),
      ('application/json', True),
      ('text/csv', True),
      ('text/html', True),
      ('text/xml', True),
      ('text/x-python', True),
      ('text/x-script.python', True),
      ('image/png', False),
      ('audio/wav', False),
      ('video/mp4', False),
      ('application/pdf', False),
  ])
  def test_is_text(self, mime_type, expected):
    self.assertEqual(mime_types.is_text(mime_type), expected)

  @parameterized.parameters([
      ('application/json', True),
      ('application/JSON', True),
      ('application/json; charset=utf-8', True),
      ('text/plain', False),
  ])
  def test_is_json(self, mime_type, expected):
    self.assertEqual(mime_types.is_json(mime_type), expected)

  @parameterized.parameters([
      ('application/json; type=MyData', MyData, True),
      ('APPLICATION/JSON; type=MyData', MyData, True),
      ('application/JSON; type=Mydata', MyData, False),
      ('application/json; type=OtherData', MyData, False),
      ('application/json', MyData, False),
      ('application/json; type=MyData', None, True),
      ('application/json', None, False),
  ])
  def test_is_dataclass(self, mime_type, data_class, expected):
    self.assertEqual(mime_types.is_dataclass(mime_type, data_class), expected)

  @parameterized.parameters([
      ('image/png', True),
      ('image/jpeg', True),
      ('image/webp', True),
      ('image/heic', True),
      ('image/heif', True),
      ('image/gif', True),
      ('text/plain', False),
  ])
  def test_is_image(self, mime_type, expected):
    self.assertEqual(mime_types.is_image(mime_type), expected)

  @parameterized.parameters([
      ('video/mp4', True),
      ('video/webm', True),
      ('text/plain', False),
  ])
  def test_is_video(self, mime_type, expected):
    self.assertEqual(mime_types.is_video(mime_type), expected)

  @parameterized.parameters([
      ('audio/wav', True),
      ('audio/mp3', True),
      ('text/plain', False),
  ])
  def test_is_audio(self, mime_type, expected):
    self.assertEqual(mime_types.is_audio(mime_type), expected)

  @parameterized.parameters([
      ('audio/l16', True),
      ('audio/L16', True),
      ('audio/L16; rate=16000', True),
      ('audio/wav', False),
  ])
  def test_is_streaming_audio(self, mime_type, expected):
    self.assertEqual(mime_types.is_streaming_audio(mime_type), expected)

  @parameterized.parameters([
      ('audio/wav', True),
      ('audio/mp3', False),
  ])
  def test_is_wav(self, mime_type, expected):
    self.assertEqual(mime_types.is_wav(mime_type), expected)


if __name__ == '__main__':
  absltest.main()
