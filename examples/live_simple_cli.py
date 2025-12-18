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
r"""Command Line Interface to run a simple live agent connected to the Live API.

## Setup

To install the dependencies for this script, run:

```
pip install --upgrade pyaudio genai-processors google-genai
```

Before running this script, ensure the `GOOGLE_API_KEY` environment
variable is set to the api-key you obtained from Google AI Studio.

Important: **Use headphones**. This script uses the system default audio
input and output, which often won't include echo cancellation. So to prevent
the model from interrupting itself it is important to use headphones.

## Run

To run the script:

```shell
python3 ./live_simple_cli.py
```

The script takes a video-mode flag `--mode`, this can be "camera", "screen". The
default is "camera". To share your screen run:

```shell
python3 ./live_simple_cli.py --mode=screen
```
"""


import argparse
import asyncio
import os

from absl import logging
from genai_processors.core import audio_io
from genai_processors.core import live_model
from genai_processors.core import text
from genai_processors.core import video
from google.genai import types as genai_types
import pyaudio

# You need to define the API key in the environment variables.
API_KEY = os.environ['GOOGLE_API_KEY']

INSTRUCTION_PARTS = [
    'You are an agent that interacts with the user in a conversation. Make'
    ' the conversation lively and interesting for the user. Approach the'
    ' conversation as if you were on TV anchor. You can make jokes, explain'
    ' interesting facts related to what you see and hear, predict what'
    ' could happen, judge some actions or reactions, etc. Respond to the'
    ' user in a few sentences maximum: keep it short and engaging. Avoid'
    ' long monologues. Feel free to commentate on anything you see. It is'
    ' up to you to decide to commentate or not. You can also ask questions'
    ' to the user. If you do not get any answer, you can ask again after a '
    ' while as in a normal conversation when the counterpart does not'
    ' listen or is distracted by something else.'
]


async def run_live(video_mode: str) -> None:
  r"""Runs a simple live agent taking audio/video streams as input.

  The audio and video input and output are connected to the local machine's
  default input and output devices.

  Args:
    video_mode: The video mode to use for the video. Can be CAMERA or SCREEN.
  """
  pya = pyaudio.PyAudio()
  video_mode_enum = video.VideoMode(video_mode)
  # input processor = camera/screen streams + audio streams
  # Note that the Live API requires audio/pcm mimetype (not audio/l16).
  input_processor = video.VideoIn(
      video_mode=video_mode_enum
  ) + audio_io.PyAudioIn(pya, use_pcm_mimetype=True)

  # Calls the Live API. If you define your own live agent, this is the processor
  # that will likely be replaced. See live/commentator_cli.py for a more
  # advanced example.
  live_processor = live_model.LiveProcessor(
      api_key=API_KEY,
      model_name='gemini-2.5-flash-native-audio-preview-12-2025',
      realtime_config=genai_types.LiveConnectConfig(
          system_instruction=INSTRUCTION_PARTS,
          # Ground with Google Search
          tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())],
          # Return the transcription of the model audio.
          output_audio_transcription={},
          # Enable affective dialog (only available for native audio out models)
          enable_affective_dialog=True,
          response_modalities=['AUDIO'],
          # Set the language for the Live API.
          speech_config={'language_code': 'en-US'},
      ),
      http_options=genai_types.HttpOptions(api_version='v1alpha'),
  )

  # Plays the audio parts. This processor also handles interruptions and makes
  # sure the audio output stops when the user is speaking.
  play_output = audio_io.PyAudioOut(pya)

  # Creates an agent as: mic+camera -> Live API -> play audio
  live_agent = input_processor + live_processor + play_output

  print('Use ctrl+D to quit.')
  async for part in live_agent(text.terminal_input()):
    # Print the transcription and the output of the model (should include status
    # parts and other metadata parts)
    print(part)


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
  if not API_KEY:
    raise ValueError(
        'API key is not set. Define a GOOGLE_API_KEY environment variable with'
        ' a key obtained from AI Studio.'
    )
  if args.debug:
    logging.set_verbosity(logging.DEBUG)
  asyncio.run(run_live(video_mode=args.mode))
