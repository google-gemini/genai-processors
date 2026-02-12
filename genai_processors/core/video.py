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
"""Video processors."""

import asyncio
import enum
import functools
import tempfile
from typing import AsyncIterable

import av
import cv2
from genai_processors import content_api
from genai_processors import context
from genai_processors import mime_types
from genai_processors import processor
from genai_processors import streams
import PIL.Image

DEFAULT_SAMPLE_RATE = 16000

ProcessorPart = content_api.ProcessorPart


class VideoMode(enum.Enum):
  """Video mode for the VideoIn processor."""

  CAMERA = 'camera'
  SCREEN = 'screen'


def _get_single_camera_frame(
    cap: cv2.VideoCapture, substream_name: str
) -> ProcessorPart:
  """Get a single frame from the camera."""
  # Read the frame queue
  ret, frame = cap.read()
  if not ret:
    raise RuntimeError("Couldn't captrue a frame.")
  # Fix: Convert BGR to RGB color space
  # OpenCV captures in BGR but PIL expects RGB format
  # This prevents the blue tint in the video feed
  frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  img = PIL.Image.fromarray(frame_rgb)  # Now using RGB frame
  img.format = 'JPEG'

  return ProcessorPart(img, substream_name=substream_name, role='user')


def _get_single_screen_frame(substream_name: str) -> ProcessorPart:
  """Get a single frame from the screen."""
  try:
    import mss  # pytype: disable=import-error # pylint: disable=g-import-not-at-top
  except ImportError as e:
    raise ImportError(
        "Please install mss package using 'pip install mss'"
    ) from e
  sct = mss.mss()
  monitor = sct.monitors[0]

  i = sct.grab(monitor)
  img = PIL.Image.frombuffer('RGB', i.size, i.rgb)
  img.format = 'JPEG'

  return ProcessorPart(img, substream_name=substream_name, role='user')


@processor.source()
async def VideoIn(  # pylint: disable=invalid-name
    substream_name: str = 'realtime', video_mode: VideoMode = VideoMode.CAMERA
) -> AsyncIterable[ProcessorPart]:
  """Yields image parts from a camera or a computer screen.

  Args:
    substream_name: The name of the substream to use for the generated images.
    video_mode: The video mode to use for the video. Can be CAMERA or SCREEN.
  """
  if video_mode == VideoMode.CAMERA:
    # This takes about a second, and will block the whole program
    # causing the audio pipeline to overflow if you don't to_thread it.
    cap = await asyncio.to_thread(
        cv2.VideoCapture, 0
    )  # 0 represents the default camera

    try:
      # The coroutine will be cancelled when we are done, breaking the loop.
      while True:
        yield await asyncio.to_thread(
            _get_single_camera_frame, cap, substream_name
        )
        await asyncio.sleep(1.0)
    finally:
      # Release the VideoCapture object
      cap.release()
  elif video_mode == VideoMode.SCREEN:
    while True:
      yield await asyncio.to_thread(_get_single_screen_frame, substream_name)
      await asyncio.sleep(1.0)
  else:
    raise ValueError(f'Unsupported video mode: {video_mode}')


class VideoAVFormat(enum.Enum):
  """Controls how video should be formatted into individual frames."""

  # Expand video into image frames, each frame in its own ProcessorPart.
  VIDEO = 1
  # Extract audio from the video file and insert it as a single audio/l16
  # ProcessorPart.
  AUDIO = 2
  # Both video and audio are extracted and interleaved as separate
  # ProcessorPart objects. This representation is similar to live video streams.
  # The audio part duration is the period between two images: 1/FPS.
  BOTH_INTERLEAVED = 3


class VideoExtract(processor.PartProcessor):
  """Extracts video and audio frames from a video file.

  This processor takes a stream of `ProcessorPart` objects, and if it
  finds a part that contains a video, it extracts the video and audio
  frames and yields them as individual `ProcessorPart` objects.

  Usually it is easier and better to send the whole video to the model and let
  Gemini API do the formatting. However sometimes we may want to apply
  core.window.Window to it or simulate live audio/video stream for unit tests or
  use a model that doesn't support video natively, like Gemma.
  """

  def __init__(
      self,
      frames_per_second: float = 1.0,
      video_format: VideoAVFormat = VideoAVFormat.BOTH_INTERLEAVED,
  ):
    """Initializes the VideoExtract processor.

    Args:
      frames_per_second: The number of frames per second to extract from the
        video.
      video_format: The way to represent extracted frames.
    """
    self._frames_per_second = frames_per_second
    self._video_format = video_format

  def match(self, part: ProcessorPart) -> bool:
    """Returns True if the part contains a video file."""
    return mime_types.is_video(part.mimetype)

  def _process_part(
      self,
      part: ProcessorPart,
      queue: asyncio.Queue[ProcessorPart | None],
      loop: asyncio.AbstractEventLoop,
  ) -> None:
    """Sync function to process video part in a thread."""
    try:
      audio_mime_type = f'audio/l16;rate={DEFAULT_SAMPLE_RATE};channels=1'

      with tempfile.NamedTemporaryFile('wb', suffix='.mp4') as temp_file:
        temp_file.write(part.bytes)
        temp_file.flush()

        with av.open(temp_file.name) as container:
          # Audio processing
          audio_data = b''
          if (
              self._video_format
              in [
                  VideoAVFormat.AUDIO,
                  VideoAVFormat.BOTH_INTERLEAVED,
              ]
              and container.streams.audio
          ):
            audio_stream = container.streams.audio[0]
            resampler = av.AudioResampler(
                format='s16', layout='mono', rate=16000
            )
            for frame in container.decode(audio_stream):
              for resampled_frame in resampler.resample(frame):
                audio_data += resampled_frame.to_ndarray().tobytes()

          if self._video_format == VideoAVFormat.AUDIO:
            loop.call_soon_threadsafe(
                queue.put_nowait,
                ProcessorPart(
                    audio_data,
                    mimetype=audio_mime_type,
                    substream_name=part.substream_name,
                    role=part.role,
                ),
            )
            return

          # Video processing
          container.seek(0)
          if (
              self._video_format
              in [
                  VideoAVFormat.VIDEO,
                  VideoAVFormat.BOTH_INTERLEAVED,
              ]
              and container.streams.video
          ):
            video_stream = container.streams.video[0]
            graph = av.filter.Graph()
            buffer = graph.add_buffer(template=video_stream)
            fps_filter = graph.add(
                'fps', fps=str(self._frames_per_second), round='up'
            )
            sink = graph.add('buffersink')
            buffer.link_to(fps_filter)
            fps_filter.link_to(sink)
            graph.configure()

            audio_offset_bytes = 0

            frame_idx = 0
            for frame in container.decode(video_stream):
              buffer.push(frame)

              while True:
                try:
                  filtered_frame = sink.pull()
                except (av.FFmpegError, EOFError):
                  break

                timestamp = frame_idx / self._frames_per_second
                img = filtered_frame.to_image()
                img.format = 'JPEG'
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    ProcessorPart(
                        img,
                        substream_name=part.substream_name,
                        role=part.role,
                        metadata={'video_timestamp': timestamp},
                    ),
                )

                if self._video_format == VideoAVFormat.BOTH_INTERLEAVED:
                  audio_offset_samples_end = int(
                      16000 * (frame_idx + 1) / self._frames_per_second
                  )
                  audio_offset_bytes_end = audio_offset_samples_end * 2
                  audio_chunk = audio_data[
                      audio_offset_bytes : audio_offset_bytes_end
                  ]
                  audio_offset_bytes = audio_offset_bytes_end
                  loop.call_soon_threadsafe(
                      queue.put_nowait,
                      ProcessorPart(
                          audio_chunk,
                          mimetype=audio_mime_type,
                          substream_name=part.substream_name,
                          role=part.role,
                          metadata={'video_timestamp': timestamp},
                      ),
                  )
                frame_idx += 1
    finally:
      loop.call_soon_threadsafe(queue.put_nowait, None)

  async def call(
      self,
      part: ProcessorPart,
  ) -> AsyncIterable[ProcessorPart]:
    """Extracts video and audio frames from a video file.

    Args:
      part: A `ProcessorPart` object.

    Yields:
      `ProcessorPart` objects, with video exploded into individual images and
      audio frames.
    """
    queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    context.create_task(
        asyncio.to_thread(self._process_part, part, queue, loop)
    )
    async for part in streams.dequeue(queue):
      yield part
