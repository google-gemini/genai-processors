"""Tests for the event detection processor."""

import enum
import io
from typing import AsyncIterable
import unittest

from absl.testing import absltest
from genai_processors import content_api
from genai_processors import processor
from genai_processors import streams
from genai_processors.core import event_detection
import PIL.Image


class EventName(enum.StrEnum):
  SUNNY = enum.auto()
  CLOUDY = enum.auto()
  RAINING = enum.auto()


class MockBackend(processor.Processor):

  def __init__(self) -> None:
    super().__init__()
    self.side_effect: list[str | Exception] = []

  async def call(
      self, content: content_api.ContentStream
  ) -> AsyncIterable[content_api.ProcessorPartTypes]:
    await content
    output = self.side_effect.pop(0)
    if isinstance(output, Exception):
      raise output
    yield content_api.ProcessorPart(output)


def get_image() -> bytes:
  img = PIL.Image.new('RGB', (100, 100), color='black')
  image_io = io.BytesIO()
  img.save(image_io, format='jpeg')
  image_io.seek(0)
  return image_io.read()


class EventDetectionTest(unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    super().setUp()
    self.output_dict = {
        ('*', EventName.SUNNY): [content_api.ProcessorPart('sunny!')],
        ('*', EventName.CLOUDY): [content_api.ProcessorPart('cloudy!')],
        (EventName.CLOUDY, EventName.RAINING): [
            content_api.ProcessorPart('raining!')
        ],
        (EventName.RAINING, EventName.CLOUDY): None,
    }
    self.mock_backend = MockBackend()
    self.event_detection_processor = event_detection.EventDetection(
        backend=self.mock_backend,
        output_dict=self.output_dict,
        sensitivity={(EventName.CLOUDY, EventName.SUNNY): 3},
    )
    self.img_part = content_api.ProcessorPart(
        get_image(),
        mimetype='image/jpeg',
    )

  async def test_detections_sensitivity_lower_than_threshold(self):
    self.mock_backend.side_effect = [
        EventName.SUNNY,
        EventName.CLOUDY,
        EventName.SUNNY,
    ]
    input_stream = streams.stream_content(
        [self.img_part] * 3, with_delay_sec=0.1
    )
    output_stream = await streams.gather_stream(
        self.event_detection_processor(input_stream)
    )
    self.assertEqual(
        output_stream,
        [
            self.img_part,
            content_api.ProcessorPart('sunny!'),
            self.img_part,
            content_api.ProcessorPart('cloudy!'),
            self.img_part,
            # Last image is not detected because of sensitivity for
            # the transition (CLOUDY, SUNNY) requires 3 detections in a row.
        ],
    )

  async def test_detections_transition_not_in_output_dict(self):
    self.mock_backend.side_effect = [
        EventName.SUNNY,
        EventName.CLOUDY,
        EventName.RAINING,
        EventName.CLOUDY,
    ]
    input_stream = streams.stream_content(
        [self.img_part] * 4, with_delay_sec=0.1
    )
    output_stream = await streams.gather_stream(
        self.event_detection_processor(input_stream)
    )
    self.assertEqual(
        output_stream,
        [
            self.img_part,
            content_api.ProcessorPart('sunny!'),
            self.img_part,
            content_api.ProcessorPart('cloudy!'),
            self.img_part,
            content_api.ProcessorPart('raining!'),
            self.img_part,
            # CLOUDY is not detected because the event is already detected as
            # part of RAINING.
        ],
    )

  async def test_detection_with_sensitivity_above_threshold(self):
    self.mock_backend.side_effect = [
        EventName.SUNNY,
        EventName.CLOUDY,
        EventName.SUNNY,
        EventName.SUNNY,
        EventName.SUNNY,
        EventName.SUNNY,
    ]
    input_stream = streams.stream_content(
        [self.img_part] * 6, with_delay_sec=0.1
    )

    output_stream = await streams.gather_stream(
        self.event_detection_processor(input_stream)
    )
    self.assertEqual(
        output_stream,
        [
            self.img_part,
            content_api.ProcessorPart('sunny!'),
            self.img_part,
            content_api.ProcessorPart('cloudy!'),
            self.img_part,
            # SUNNY not detected because of sensitivity for
            # the transition (CLOUDY, SUNNY) requires > 3 detections in a row.
            self.img_part,
            self.img_part,
            self.img_part,
            content_api.ProcessorPart('sunny!'),
        ],
    )

  async def test_detection_no_output(self):
    self.mock_backend.side_effect = [
        EventName.CLOUDY,
        EventName.RAINING,
        EventName.CLOUDY,
        EventName.RAINING,
    ]
    input_stream = streams.stream_content(
        [self.img_part] * 4, with_delay_sec=0.1
    )

    output_stream = await streams.gather_stream(
        self.event_detection_processor(input_stream)
    )
    self.assertEqual(
        output_stream,
        [
            self.img_part,
            content_api.ProcessorPart('cloudy!'),
            self.img_part,
            content_api.ProcessorPart('raining!'),
            self.img_part,
            # (RAINING, SUNNY) is detected but not output because
            # the corresponding value in the output_dict is None.
            self.img_part,
            content_api.ProcessorPart('raining!'),
        ],
    )

  async def test_detection_transition_from_not_allowed(self):
    self.mock_backend.side_effect = [
        EventName.SUNNY,
        EventName.RAINING,
        EventName.CLOUDY,
        EventName.RAINING,
    ]
    input_stream = streams.stream_content(
        [self.img_part] * 4, with_delay_sec=0.1
    )

    output_stream = await streams.gather_stream(
        self.event_detection_processor(input_stream)
    )
    self.assertEqual(
        output_stream,
        [
            self.img_part,
            content_api.ProcessorPart('sunny!'),
            self.img_part,
            # RAINING is not detected because it can only transition from CLOUDY
            self.img_part,
            content_api.ProcessorPart('cloudy!'),
            self.img_part,
            content_api.ProcessorPart('raining!'),
        ],
    )

  async def test_detection_when_repeated(self):
    self.mock_backend.side_effect = [
        EventName.SUNNY,
        EventName.SUNNY,
        EventName.SUNNY,
        EventName.SUNNY,
    ]
    input_stream = streams.stream_content(
        [self.img_part] * 4, with_delay_sec=0.1
    )
    output_stream = await streams.gather_stream(
        self.event_detection_processor(input_stream)
    )
    self.assertEqual(
        output_stream,
        [
            self.img_part,
            content_api.ProcessorPart('sunny!'),
            # no detection because SUNNY is repeated
            self.img_part,
            self.img_part,
            self.img_part,
        ],
    )

  async def test_detection_with_exception(self):
    self.mock_backend.side_effect = [
        IOError('test exception'),
        EventName.RAINING,
    ]
    input_stream = streams.stream_content(
        [self.img_part] * 2, with_delay_sec=0.1
    )
    with self.assertRaises(IOError):
      await streams.gather_stream(self.event_detection_processor(input_stream))


if __name__ == '__main__':
  absltest.main()
