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
r"""Command Line Interface to test Voice Activity Detection + GenAI Model.

## Setup

To install the dependencies for this script, run:

```
pip install genai-processors pyaudio
```

Before running this script, ensure the `GOOGLE_API_KEY` environment
variable is set to your Gemini API key.

## Run

To run the script:

```shell
python3 ./vad_cli.py
```
"""

import asyncio
import os
import time

from genai_processors import content_api
from genai_processors import mime_types
from genai_processors.core import audio
from genai_processors.core import audio_io
from genai_processors.core import genai_model
from genai_processors.core import realtime
from genai_processors.core import speech_events
from genai_processors.core import text
from genai_processors.core import vad
import pyaudio


# You need to define the api key in the environment variables.
# export GOOGLE_API_KEY=...
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')


async def run_vad_cli() -> None:
  """Runs the VAD + Gemini model in a CLI environment."""

  pya = pyaudio.PyAudio()

  # The base model that we want to query.
  base_model = genai_model.GenaiModel(
      api_key=GOOGLE_API_KEY,
      model_name='gemini-2.5-flash',
  )

  # The LiveProcessor manages conversation turns. It accumulates all the
  # content until EndOfSpeech (emitted by vad.Vad()) and then sends the
  # accumulated audio (which AudioToWav converts into a single .wav part)
  # to the GenAI model.
  live_processor = realtime.LiveProcessor(
      turn_processor=(audio.AudioToWav() + base_model),
      trigger_model_mode=realtime.AudioTriggerMode.END_OF_SPEECH,
  )

  pipeline = (
      audio_io.PyAudioIn(pya)
      # Aggressive VAD to detect speech activity (less false positives). User
      # needs to speak loud and clear.
      + vad.Vad()
      # Adds status parts to provide feedback to the user.
      + add_speech_event_status
      + live_processor
  )

  print(
      f'{time.perf_counter():.2f} - VAD + Model pipeline ready: start talking!'
  )
  try:
    await text.terminal_output(pipeline(text.terminal_input()))
  finally:
    pya.terminate()


if __name__ == '__main__':
  asyncio.run(run_vad_cli())
