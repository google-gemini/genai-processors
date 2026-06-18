import asyncio
import pytest
import time
import unittest.mock

from collections.abc import AsyncIterable
from genai_processors.dev import latency
from genai_processors import content_api
from genai_processors import processor
from genai_processors import streams
from unittest.mock import patch


@pytest.mark.asyncio
async def test_prober_structure():
    """Tests that Prober is scheduled correctly."""

    pipeline = latency.Prober("start", interval_seconds=0.1)

    # Send a few parts to establish a baseline for FPS
    input_stream = streams.stream_content(["test"] * 11, with_delay_sec=0.02)
    results = await pipeline(input_stream).gather()

    for part in results:
        assert part.text == "test"

    assert len(results) == 11
    assert "prober" in results[5].metadata
    assert "prober" in results[10].metadata
    delta_time = (
        results[10].metadata["prober"]["markers"][0][1]
        - results[5].metadata["prober"]["markers"][0][1]
    )
    assert delta_time == pytest.approx(0.1, abs=0.01)


@pytest.mark.asyncio
async def test_probe_checkpoint_trace():
    """Tests that ProbeCheckpoint builds a trace of markers with FPS."""
    pipeline = latency.ProbeCheckpoint("step1") + latency.ProbeCheckpoint("step2")
    start_time = 100.0
    # Prober now starts with one marker
    input_stream = [
        "initial",
        content_api.ProcessorPart(
            "test",
            metadata={
                "prober": {
                    "start_time": start_time,
                    "markers": [("start", start_time, 0, 50.0)],
                }
            },
        ),
    ]
    result = await pipeline(input_stream).gather()

    # Check that we have the original marker + the extra 2.
    markers = result[1].metadata["prober"]["markers"]
    assert len(markers) == 3
    assert markers[1][0] == "step1"
    assert markers[2][0] == "step2"
    assert markers[1][3] > 0  # FPS should be present


@pytest.mark.asyncio
async def test_fps_calculation():
    """Verify that FPS is calculated correctly based on part count."""
    pipeline = (
        latency.Prober("start", interval_seconds=1.0).to_processor()
        + latency.ProbeCheckpoint("step").to_processor()
    )
    # The first part sets the marker for the prober (start), we collect
    # the prober info after 5 parts, so in total we have 6 parts.
    input_stream = streams.stream_content(["test"] * 6, with_delay_sec=0.2)
    result = await pipeline(input_stream).gather()
    fps = result[-1].metadata["prober"]["markers"][0][3]
    assert fps == pytest.approx(5.0, abs=0.1)


@pytest.mark.asyncio
async def test_probe_checkpoint_substream():
    """Tests that ProbeCheckpoint filters by substream_name."""
    pipeline = latency.ProbeCheckpoint("step", substream_name="server")

    input = [
        content_api.ProcessorPart(
            "test",
            metadata={"prober": {"start_time": time.monotonic(), "markers": []}},
        ),
        content_api.ProcessorPart(
            "test",
            metadata={"prober": {"start_time": time.monotonic(), "markers": []}},
            substream_name="server",
        ),
    ]
    result = await pipeline(input).gather()
    assert len(result[0].metadata["prober"]["markers"]) == 0
    assert len(result[1].metadata["prober"]["markers"]) == 1


def test_log_arrival_trace():
    """Tests log_arrival logs FPS and duration."""
    with unittest.mock.patch("genai_processors.dev.latency.logging.info") as mock_log:
        part = content_api.ProcessorPart(
            "test",
            metadata={
                "prober": {
                    "start_time": 100.0,
                    "markers": [
                        ("step1", 100.01, 0.01, 50.0),
                        ("step2", 100.03, 0.02, 48.5),
                    ],
                }
            },
        )

        latency.Prober.log_arrival(part)

        mock_log.assert_called()
        assert "default" == mock_log.call_args[0][1]
        trace_arg = mock_log.call_args[0][2]
        assert "step1 (+10.0ms, 50.0fps)" in trace_arg
        assert "step2 (+20.0ms, 48.5fps)" in trace_arg


@pytest.mark.asyncio
async def test_throttler_drop_strategy():
    """Tests that Throttler protects probers and drops the oldest non-prober part."""
    # max_size=1 in the throttler + 1 in bridge + 1 in hand-off queue, i.e. we can
    # handle max_size + 3 in flight but then we'll start dropping parts.

    @processor.processor_function
    async def wait_some(
        content: processor.ProcessorStream,
    ) -> AsyncIterable[content_api.ProcessorPartTypes]:
        async for p in content:
            await asyncio.sleep(0.5)
            yield p

    pipeline = latency.Throttler("test", max_size=1) + wait_some
    input_stream = streams.stream_content(
        [
            # This part will be processed immediately, not blocking - arrives at 0sec
            "p1",
            # This part will not be dropped, send to output queue. - arrives at 0.1sec
            # The buffer in the throttler is empty.
            content_api.ProcessorPart("p2", metadata={"prober": {}}),
            # This part is buffered. Arrives at 0.2sec
            "p3",
            # This part is buffered. Arrives at 0.2 sec
            # This removes p3 from the buffer: p3 is sent to the output queue,
            # it is in the bridge and blocked, waiting for the output queue to be
            # empty.
            "p4",
            # This part is buffered. Arrives at 0.4 sec - before p4 is still in the
            # buffer, p4 is dropped and is replaced by p5.
            "p5",
        ],
        with_delay_sec=0.1,
    )

    results = await pipeline(input_stream).gather()

    names = [p.text for p in results if p]
    # See explanation above.
    assert names == ["p1", "p2", "p3", "p5"]


@pytest.mark.asyncio
async def test_probe_log():
    """Tests that ProbeLog logs arrival and yields a latency part."""
    probe_log = latency.ProbeLog(substream_name="server")

    part = content_api.ProcessorPart(
        "test",
        metadata={"prober": {"start_time": 100.0, "markers": [("start", 0, 0, 0)]}},
    )
    with patch.object(latency.logging, "log") as mock_info:
        results = await probe_log(part).gather()
        # We have two parts, the original one and a new one under the
        # latency substream
        assert len(results) == 2
        assert results[1].substream_name == "latency"
        # first arg of log info is about latency
        assert "Latency Trace" in mock_info.call_args_list[0][0][1]
