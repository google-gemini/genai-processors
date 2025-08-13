import asyncio
from typing import Iterable
import unittest

from absl.testing import absltest
from absl.testing import parameterized
from genai_processors import content_api
from genai_processors import context as context_lib
from genai_processors import processor
from genai_processors.core import rate_limit_audio
import numpy as np


class RateLimitStreamingAudioWithMocks(rate_limit_audio.RateLimitAudio):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._skipped_waits = 0

  def _perf_counter(self) -> float:
    """Mock time deterministically and with delays of _asyncio_sleep."""
    self._skipped_waits += 1e-7  # To keep it strictly monotonic.
    return 123_000 + self._skipped_waits

  async def _asyncio_sleep(self, delay: float) -> None:
    """Like asyncio.sleep(), advancing simulated time with less waiting."""
    wakeup_time = self._perf_counter() + delay
    # It's hard to properly mock out the effects of asyncio event scheduling,
    # so this code contents itself with accelerating time.
    await asyncio.sleep(delay / 10)
    skipahead_sec = max(wakeup_time - self._perf_counter(), 0)
    self._skipped_waits += skipahead_sec
    assert self._perf_counter() >= wakeup_time

  async def call_with_timing(
      self,
      content_with_delays: Iterable[tuple[content_api.ProcessorPart, float]],
  ) -> Iterable[tuple[content_api.ProcessorPart, float]]:
    """Calls the processor with explicit timing of inputs and outputs.

    Args:
      content_with_delays: an iterable of (part, timestamp) pairs. Timestamps
        start from 0 and must be monotonically increasing. For each increase, an
        _asyncio_sleep() is done before yielding the input part to the
        processor,

    Returns:
      An iterable of (part, timestamp) pairs, where parts are those returned
      by the processor and timestamps are the respective current values of
      _perf_counter.
    """
    async with asyncio.TaskGroup() as tg:
      input_queue = asyncio.Queue()

      async def enqueue_input_with_delay():
        last_timestamp = 0.0
        for part, timestamp in content_with_delays:
          delay = timestamp - last_timestamp
          assert delay >= 0, f'Bad test set-up: time jumps back to {timestamp}'
          if delay > 0:
            await self._asyncio_sleep(delay)
          await input_queue.put(part)
          last_timestamp = timestamp
        await input_queue.put(None)

      tg.create_task(enqueue_input_with_delay())

      async def dequeue_input():
        while (part := await input_queue.get()) is not None:
          yield part

      start_timestamp = self._perf_counter()
      outputs_with_delays = []
      async for output_part in self(dequeue_input()):
        now_sec = self._perf_counter()
        delay = now_sec - start_timestamp
        outputs_with_delays.append((output_part, delay))
    return outputs_with_delays


def _make_streaming_audio_part(values_from, values_to):
  values = np.arange(values_from, values_to, dtype=np.int16)
  result = content_api.ProcessorPart(
      values.tobytes(), mimetype='audio/l16; rate=24000'
  )
  return result


class TestRateLimitStreamingAudio(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  def _assert_offsets(
      self, num_samples, out_times, *, log_start_index=0, expected_buffer=0.05
  ):
    actual_offsets = np.array(out_times[1:]) - out_times[0]
    expected_offsets = np.maximum(
        0.0, np.cumsum(np.array(num_samples[:-1])) - expected_buffer
    )
    for i, (expected, actual) in enumerate(
        zip(expected_offsets, actual_offsets, strict=True)
    ):
      self.assertBetween(
          actual,
          expected - 0.005,
          expected + 0.01,
          msg=f'at audio part {i + log_start_index + 1}',
      )

  @parameterized.named_parameters(
      ('with_other_parts_delayed', True),
      ('with_other_parts_bypassed', False),
  )
  async def test_rate_limit(self, delay_other_parts):
    p = RateLimitStreamingAudioWithMocks(
        sample_rate=24000, delay_other_parts=delay_other_parts
    )

    test_input = [
        # Initial slew of inputs: some text, status, and two audio parts that
        # are played back to back, each worth 8000 / 24000 = 1/3 seconds of
        # audio that get split into 7 sub-parts each.
        (content_api.ProcessorPart('first text'), 0.0),
        (_make_streaming_audio_part(123, 123 + 8000), 0.0),  # [0:7]
        (content_api.ProcessorPart('second text'), 0.0),
        (processor.status(content_api.ProcessorPart('stat stat')), 0.0),
        (_make_streaming_audio_part(456, 456 + 8000), 0.0),  # [7:14]
        # 5 seconds later: more inputs, with an audio part of 0.5 s that
        # restarts playback and requires to refill the buffer.
        (content_api.ProcessorPart('third text'), 5.0),
        (_make_streaming_audio_part(789, 789 + 12000), 5.0),  # [14:24]
    ]
    outputs = await p.call_with_timing(test_input)

    # For part types that may be reordered (e.g., status/debug parts), the
    # relative order to other types is not fully defined, so we test separately
    # per type and then compare timestamps within adequate limits.
    audio_outputs = []
    status_outputs = []
    other_outputs = []
    for output in outputs:
      part, _ = output
      if part.substream_name in context_lib.get_reserved_substreams():
        status_outputs.append(output)
      elif part.mimetype == 'audio/l16; rate=24000':
        audio_outputs.append(output)
      else:
        other_outputs.append(output)
    audio_out_parts, audio_out_times = zip(*audio_outputs)
    status_out_parts, status_out_times = zip(*status_outputs)
    other_out_parts, other_out_times = zip(*other_outputs)

    self.assertSequenceEqual(
        other_out_parts,
        [
            content_api.ProcessorPart('first text'),
            content_api.ProcessorPart('second text'),
            content_api.ProcessorPart('third text'),
        ],
    )
    self.assertSequenceEqual(
        status_out_parts,
        [processor.status(content_api.ProcessorPart('stat stat'))],
    )
    self.assertSequenceEqual(
        audio_out_parts,
        # Sub-parts [0:7] come from the first input.
        [
            _make_streaming_audio_part(123 + 1200 * i, 123 + 1200 * (i + 1))
            for i in range(6)
        ]
        + [_make_streaming_audio_part(123 + 1200 * 6, 123 + 8000)]
        +
        # Sub-parts [7:14] come from the second input.
        [
            _make_streaming_audio_part(456 + 1200 * i, 456 + 1200 * (i + 1))
            for i in range(6)
        ]
        + [_make_streaming_audio_part(456 + 1200 * 6, 456 + 8000)]
        +
        # Sub-parts [14:24] come from the third input.
        [
            _make_streaming_audio_part(789 + 1200 * i, 789 + 1200 * (i + 1))
            for i in range(10)
        ],
    )

    # The "first text" and status parts are expected to appear very soon.
    self.assertBetween(other_out_times[0], 0.0, 0.001)
    self.assertBetween(status_out_times[0], 0.0, 0.001)
    # Same for the first audio sub-part, which starts playback.
    self.assertBetween(audio_out_times[0], 0.0, 0.001)
    # The subsequent audio sub-parts of the two immediate audio parts are
    # expected to come after, properly spaced.
    self._assert_offsets(
        [
            (len(c.part.inline_data.data) // 2) / 24000
            for c in audio_out_parts[:14]
        ],
        audio_out_times[:14],
    )
    # Depending on options, the "second text" part is expected to appear...
    if delay_other_parts:
      # ...after the first audio part is done.
      self.assertBetween(other_out_times[1] - audio_out_times[6], 0.0, 0.1)
    else:
      # ...early on, if fast-tracking is enabled.
      self.assertBetween(other_out_times[0], 0.0, 0.001)
    # Then there are the parts 5 seconds later.
    self.assertBetween(other_out_times[2], 5.0, 5.001)
    self.assertBetween(audio_out_times[14], 5.0, 5.001)
    self._assert_offsets(
        [
            (len(c.part.inline_data.data) // 2) / 24000
            for c in audio_out_parts[14:]
        ],
        audio_out_times[14:],
        log_start_index=14,
    )


if __name__ == '__main__':
  absltest.main()
