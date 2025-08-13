"""Tests for the event detection processor."""

import enum
import io
import unittest
from unittest import mock

from absl.testing import absltest
from genai_processors import content_api
from genai_processors import streams
from genai_processors.core import event_detection
from google.genai import models
from google.genai import types as genai_types
import PIL.Image


class EventName(enum.StrEnum):
  SUNNY = enum.auto()
  CLOUDY = enum.auto()
  RAINING = enum.auto()


def generate_response(text: str) -> genai_types.GenerateContentResponse:
  return genai_types.GenerateContentResponse(
      candidates=[
          genai_types.Candidate(
              content=genai_types.Content(
                  parts=[
                      genai_types.Part(
                          text=text,
                      )
                  ]
              )
          )
      ],
  )


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
    self.event_detection_processor = event_detection.EventDetection(
        api_key='test_api_key',
        model='test_model',
        config=genai_types.GenerateContentConfig(
            system_instruction=(
                'Determine the weather conditions under which these images'
                f' have been taken. Respond with "{EventName.SUNNY}" if the sun'
                f' is shining, "{EventName.RAINING}" if it is raining,'
                f' {EventName.CLOUDY} if it is cloudy.'
            ),
            response_mime_type='text/x.enum',
            response_schema=EventName,
        ),
        output_dict=self.output_dict,
        sensitivity={(EventName.CLOUDY, EventName.SUNNY): 3},
    )
    self.img_part = content_api.ProcessorPart(
        get_image(),
        mimetype='image/jpeg',
    )

  @mock.patch.object(
      models.AsyncModels,
      'generate_content',
      side_effect=[
          generate_response(EventName.SUNNY),
          generate_response(EventName.CLOUDY),
          generate_response(EventName.SUNNY),
      ],
  )
  async def test_detections_sensitivity_lower_than_threshold(self, _):
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

  @mock.patch.object(
      models.AsyncModels,
      'generate_content',
      side_effect=[
          generate_response(EventName.SUNNY),
          generate_response(EventName.CLOUDY),
          generate_response(EventName.RAINING),
          generate_response(EventName.CLOUDY),
      ],
  )
  async def test_detections_transition_not_in_output_dict(self, _):
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

  @mock.patch.object(
      models.AsyncModels,
      'generate_content',
      side_effect=[
          generate_response(EventName.SUNNY),
          generate_response(EventName.CLOUDY),
          generate_response(EventName.SUNNY),
          generate_response(EventName.SUNNY),
          generate_response(EventName.SUNNY),
          generate_response(EventName.SUNNY),
      ],
  )
  async def test_detection_with_sensitivity_above_threshold(self, _):
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

  @mock.patch.object(
      models.AsyncModels,
      'generate_content',
      side_effect=[
          generate_response(EventName.CLOUDY),
          generate_response(EventName.RAINING),
          generate_response(EventName.CLOUDY),
          generate_response(EventName.RAINING),
      ],
  )
  async def test_detection_no_output(self, _):
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

  @mock.patch.object(
      models.AsyncModels,
      'generate_content',
      side_effect=[
          generate_response(EventName.SUNNY),
          # This should not be detected because RAINING can only transition
          # from CLOUDY.
          generate_response(EventName.RAINING),
          # This is detected
          generate_response(EventName.CLOUDY),
          generate_response(EventName.RAINING),
      ],
  )
  async def test_detection_transition_from_not_allowed(self, _):
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

  @mock.patch.object(
      models.AsyncModels,
      'generate_content',
      side_effect=[
          generate_response(EventName.SUNNY),
          generate_response(EventName.SUNNY),
          generate_response(EventName.SUNNY),
          generate_response(EventName.SUNNY),
      ],
  )
  async def test_detection_when_repeated(self, _):
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

  @mock.patch.object(
      models.AsyncModels,
      'generate_content',
      side_effect=[
          IOError('test exception'),
          generate_response(EventName.RAINING),
      ],
  )
  async def test_detection_with_exception(self, _):
    input_stream = streams.stream_content(
        [self.img_part] * 2, with_delay_sec=0.1
    )
    with self.assertRaises(IOError):
      await streams.gather_stream(self.event_detection_processor(input_stream))


if __name__ == '__main__':
  absltest.main()
