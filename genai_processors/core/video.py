# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
from typing import AsyncIterable
import cv2
from genai_processors import content_api
from genai_processors import processor
import PIL.Image

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
    raise ImportError("Please install mss package using 'pip install mss'") from e
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
