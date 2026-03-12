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

r"""Live commentator agent based on GenAI Processors.

Agent commentating on the video and audio stream from the default device inputs.
 This Agent can be run from a CLI directly.

## Setup

To install the dependencies for this script, run:

```
pip install pyaudio genai-processors
```

Before running this script, ensure the `GOOGLE_API_KEY` environment
variable is set to the api-key you obtained from Google AI Studio.

Important: **Use headphones**. This script uses the system default audio
input and output, which often won't include echo cancellation. So to prevent
the model from interrupting itself it is important that you use headphones.

## Run

To run the script:

```shell
python3 ./commentator_cli.py
```

The script takes a video-mode flag `--mode`, this can be "camera", "screen". The
default is "camera". To share your screen run:

```shell
python3 ./commentator_cli.py --mode=screen
```
"""

import argparse
import asyncio
import os

from absl import logging
from genai_processors.core import audio_io
from genai_processors.core import text
from genai_processors.core import video
import commentator
import pyaudio

# You need to define the API key in the environment variables.
# export GOOGLE_API_KEY=...
API_KEY = os.environ['GOOGLE_API_KEY']


async def run_commentator(video_mode: str) -> None:
  r"""Runs a live commentator in a CLI environment.

  The commentator is run from a CLI environment. The audio and video input and
  output are connected to the local machine's default input and output devices.


  Args:
    video_mode: The video mode to use for the video. Can be CAMERA or SCREEN.
  """
  pya = pyaudio.PyAudio()
  video_mode_enum = video.VideoMode(video_mode)
  input_processor = video.VideoIn(
      video_mode=video_mode_enum
  ) + audio_io.PyAudioIn(pya, use_pcm_mimetype=True)

  commentator_processor = commentator.create_live_commentator(API_KEY)

  consume_output = audio_io.PyAudioOut(pya)

  live_commentary_agent = (
      input_processor
      + commentator_processor
      + consume_output
  )

  print('Use ctrl+D to quit.')
  async for _ in live_commentary_agent(text.terminal_input()):
    pass


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--mode',
      type=str,
      default='camera',
      help='pixels to stream from',
      choices=['camera', 'screen'],
  )
  parser.add_argument(
      '--debug',
      type=bool,
      default=False,
      help='Enable debug logging.',
  )
  args = parser.parse_args()
  if args.debug:
    logging.set_verbosity(logging.DEBUG)
  asyncio.run(run_commentator(video_mode=args.mode))
