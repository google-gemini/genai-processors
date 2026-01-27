"""Tests for the timestamp processor."""

import unittest
from absl.testing import absltest
from genai_processors import content_api
from genai_processors import streams
from genai_processors.core import timestamp


class TimestampTest(unittest.IsolatedAsyncioTestCase):

  async def test_add_timestamps_with_ms(self):
    img_part = content_api.ProcessorPart(b'image_bytes', mimetype='image/png')
    input_stream = streams.stream_content([img_part] * 3, with_delay_sec=0.2)
    timestamp_processor = timestamp.add_timestamps(with_ms=True)
    output = await timestamp_processor(input_stream).gather()
    for item in output:
      if content_api.is_text(item.mimetype):
        # Remove the xx milliseconds from the timestamp to have a deterministic
        # test (i.e. only compare up to 100ms).
        item.text = item.text[:-2]

    self.assertEqual(
        output,
        [
            content_api.ProcessorPart(
                '00:00.0', metadata={'turn_complete': False}
            ),
            img_part,
            content_api.ProcessorPart(
                '00:00.2', metadata={'turn_complete': False}
            ),
            img_part,
            content_api.ProcessorPart(
                '00:00.4', metadata={'turn_complete': False}
            ),
            img_part,
        ],
    )

  async def test_add_timestamps_without_ms(self):
    img_part = content_api.ProcessorPart(b'image_bytes', mimetype='image/png')
    input_stream = streams.stream_content([img_part] * 3, with_delay_sec=1)
    timestamp_processor = timestamp.add_timestamps(
        with_ms=False, substream_name='realtime'
    )
    output = await timestamp_processor(input_stream).gather()

    self.assertEqual(
        output,
        [
            content_api.ProcessorPart(
                '00:00',
                substream_name='realtime',
                metadata={'turn_complete': False},
            ),
            img_part,
            content_api.ProcessorPart(
                '00:01',
                substream_name='realtime',
                metadata={'turn_complete': False},
            ),
            img_part,
            content_api.ProcessorPart(
                '00:02',
                substream_name='realtime',
                metadata={'turn_complete': False},
            ),
            img_part,
        ],
    )


if __name__ == '__main__':
  absltest.main()
