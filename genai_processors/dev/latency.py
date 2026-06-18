"""Latency monitoring and stream throttling processors."""

import asyncio
import time

from absl import logging
from collections.abc import AsyncIterable
from collections.abc import Callable
from genai_processors import content_api
from genai_processors import processor
from genai_processors import streams


class Prober(processor.Processor):
    """Initiates a probe sequence every `interval_seconds`.

    Adds a `prober` dictionary to the part's metadata containing the
    initiation time and an empty list of probe points.

    A probe point is a tuple of (tag, timestamp, delta, fps) and
    is typically added by a ProbeCheckpoint processor (defined below).
    """

    def __init__(
        self,
        tag: str,
        interval_seconds: float = 1.0,
    ):
        """Initializes the prober.

        Args:
            tag: The label for the initiation point.
            interval_seconds: Frequency of probes in seconds.
        """
        self.tag = tag
        self.interval_seconds = interval_seconds

    async def call(
        self, content: processor.ProcessorStream
    ) -> AsyncIterable[content_api.ProcessorPartTypes]:
        last_probe_time = time.monotonic()
        part_counter = 0

        async for part in content:
            now = time.monotonic()
            if now - last_probe_time >= self.interval_seconds:
                # We count the current part into the next interval.
                # This can reduce a bit the FPS for the first interval
                # but should even out over time.
                duration = now - last_probe_time
                fps = part_counter / duration if duration > 0 else 0

                # Initialize the probe structure
                part.metadata["prober"] = {
                    "start_time": now,
                    "markers": [
                        (self.tag, now, 0, fps)
                    ],  # Start marker with its own FPS
                }
                last_probe_time = now
                part_counter = 0
            part_counter += 1

            yield part

    @staticmethod
    def log_arrival(part: content_api.ProcessorPart):
        """Logs the full probe sequence latency results."""
        prober = part.metadata.get("prober")
        if prober:
            start_time = prober.get("start_time", 0)
            markers = prober.get("markers", [])

            if not markers:
                return

            # Build a trace string: Stage1 (+10ms, 50fps) -> Stage2 ...
            trace_parts = []
            for m in markers:
                tag, ts, delta = m[0], m[1], m[2]
                fps = m[3] if len(m) > 3 else 0
                trace_parts.append(f"{tag} (+{delta*1000:.1f}ms, {fps:.1f}fps)")

            trace = " -> ".join(trace_parts)
            total_ms = (time.monotonic() - start_time) * 1000
            substream_name = part.substream_name or "default"

            logging.info(
                "Latency Trace [%s]: %s | Total: %.1fms",
                substream_name,
                trace,
                total_ms,
            )


class ProbeCheckpoint(processor.Processor):
    """Records a latency checkpoint for parts being probed.

    Appends timing information to the `prober` metadata dictionary.
    """

    def __init__(self, tag: str, substream_name: str = ""):
        """Initializes the probe checkpoint.

        Args:
            tag: Name of the checkpoint/stage.
            substream_name: Only parts in this substream are considered
                to compute the metrics and to add the next marker to
                the probe info. Default substream is used if empty string.
        """
        self.tag = tag
        self.substream_name = substream_name

    async def call(
        self, content: processor.ProcessorStream
    ) -> AsyncIterable[content_api.ProcessorPartTypes]:
        part_counter = 0
        last_probed_time = time.monotonic()
        async for part in content:
            if part.substream_name == self.substream_name:
                prober = part.metadata.get("prober")
                if prober:
                    now = time.monotonic()
                    markers = prober.get("markers", [])

                    # Calculate delta from previous marker or start_time
                    prev_time = (
                        markers[-1][1] if markers else prober.get("start_time", now)
                    )
                    delta = now - prev_time

                    # Calculate FPS since last probed part
                    duration = now - last_probed_time
                    fps = part_counter / duration if duration > 0 else 0

                    markers.append((self.tag, now, delta, fps))

                    # Reset counters
                    last_probed_time = now
                    part_counter = 0
                part_counter += 1
            yield part


class ProbeLog(processor.PartProcessor):
    """Logs the full probe sequence latency results and yields an extra part.

    The extra part contains the same prober metadata as the input part and
    is sent to a substream (default to: "latency"). This allows one to filter
    only the latency parts, for example in order to save the full probe trace
    for later analysis.
    """

    def __init__(
        self,
        substream_name: str = "",
        msg: Callable[[content_api.ProcessorPart], str] = None,
        latency_substream: str = "latency",
    ):
        """Initializes the probe log.

        Args:
            substream_name: Parts will be output on this substream.
              Note that parts from all substreams are considered.
            msg: A callback that takes a part and returns its string
              representation.
            latency_substream: The name of the substream to emit the latency part on.
              Default is "latency".
        """
        self._substream_name = substream_name
        self._msg = msg
        self._latency_substream = latency_substream

    async def call(
        self, part: content_api.ProcessorPart
    ) -> AsyncIterable[content_api.ProcessorPartTypes]:
        yield part
        prober = part.metadata.get("prober")
        if prober:
            Prober.log_arrival(part)
            if self._msg:
                msg_str = self._msg(part).strip()
                if msg_str:
                    logging.info("Latency msg: %s", msg_str)
            yield content_api.ProcessorPart(
                "",
                metadata={"prober": prober},
                substream_name=self._latency_substream,
            )


class Throttler(processor.Processor):
    """Throttles the stream by dropping parts if the consumer is too slow.

    This processor maintains an internal buffer and drops the oldest non-prober
    part when capacity is reached. Parts containing prober information are
    never dropped; instead, they are prioritized and pushed directly to the
    output when congestion occurs, ensuring complete latency traces.
    """

    def __init__(self, tag: str, max_size: int = 2):
        """Initializes the throttler.

        Args:
            tag: A label for this throttler used in log messages.
            max_size: Maximum number of parts to buffer before dropping.
        """
        self.tag = tag
        self.max_size = max_size

    async def call(
        self, content: processor.ProcessorStream
    ) -> AsyncIterable[content_api.ProcessorPartTypes]:
        # input_queue stores parts until max_size is reached.
        input_queue = asyncio.Queue(maxsize=self.max_size)
        # output_queue is used to hand off parts to the consumer.
        output_queue = asyncio.Queue(maxsize=1)

        async def bridge():
            """Bridges input_queue to output_queue."""
            while (part := await input_queue.get()) is not None:
                await output_queue.put(part)
            await output_queue.put(None)

        async def producer():
            async for part in content:
                if input_queue.full():
                    oldest_part = input_queue.get_nowait()
                    prober = oldest_part.metadata.get("prober")
                    if prober:
                        # We block on probers, since we want to see the full trace.
                        await output_queue.put(oldest_part)
                    else:
                        part_type = (
                            oldest_part.text
                            if content_api.is_text(oldest_part.mimetype)
                            and oldest_part.text
                            else oldest_part.mimetype
                        )
                        logging.warning(
                            "Throttler [%s] queue full, dropping oldest non-prober part: %s",
                            self.tag,
                            part_type,
                        )
                input_queue.put_nowait(part)
            await input_queue.put(None)

        bridge_task = processor.create_task(bridge())
        producer_task = processor.create_task(producer())
        try:
            async for part in streams.dequeue(output_queue):
                yield part
        finally:
            producer_task.cancel()
            bridge_task.cancel()
