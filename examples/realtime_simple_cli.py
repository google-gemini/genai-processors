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
r"""Command Line Interface to run a simple live audio agent.

This agent uses the Gemini API with live/conversation.py to use a turn-based
Gemini model to interact with the user in a real-time conversation similar to
a Live API setup. This is audio only.

## Setup

To install the dependencies for this script, run:

```
pip install --upgrade pyaudio genai-processors google-genai termcolor
```

Before running this script, ensure the `GOOGLE_API_KEY` and `GOOGLE_PROJECT_ID`
environment variables are set to the api-key you obtained from Google AI Studio.

Important: **Use headphones**. This script uses the system default audio
input and output, which often won't include echo cancellation. So to prevent
the model from interrupting itself it is important to use headphones.

## Run

To run the script:

```shell
python3 ./realtime_simple_cli.py
```
"""

import asyncio
import os
from typing import Sequence

from absl import app
from genai_processors import content_api
from genai_processors import context
from genai_processors import processor
from genai_processors.core import audio_io
from genai_processors.core import rate_limit_audio
from genai_processors.core import realtime
from genai_processors.core import speech_to_text
from genai_processors.core import text
from genai_processors.core import text_to_speech
from genai_processors.examples import models
import pyaudio

# You need to define the project id in the environment variables.
GOOGLE_PROJECT_ID = os.environ['GOOGLE_PROJECT_ID']

INSTRUCTION_PARTS = [
    'You are an agent that interacts with the user in a conversation. Make'
    ' the conversation lively and interesting for the user. You can make jokes,'
    ' explain interesting facts related to what you see and hear, predict what'
    ' could happen, judge some actions or reactions, etc. Respond to the'
    ' user in a few sentences maximum: keep it short and engaging. Avoid'
    ' long monologues. You can use Google search to add extra information to'
    ' the user questions or to come up with interesting news or facts.'
]


@processor.create_filter
def _filter_parts(part: content_api.ProcessorPart) -> bool:
  """Filters out parts that are not relevant to a conversation.

  Removes audio responses from previous turns (model only) and reserved
  substreams. This is needed when working with an LLM that takes text only.
  It is also recommended (but not required) for Audio-In models to avoid
  re-tokenizing the same audio again and to use the transcription as input
  instead.

  Args:
    part: the part to filter.

  Returns:
    True if this filter should be passed through, False if it is filtered out.
  """
  if context.is_reserved_substream(part.substream_name):
    return False
  # Filters out audio responses from previous turns (model only)
  if content_api.is_audio(part.mimetype) and part.role.lower() == 'model':
    return False
  return True


async def run_conversation() -> None:
  r"""Runs a simple conversation agent taking an audio stream as input.

  The audio input and output is connected to the local machine's default input
  and output devices.
  """

  # input processor = audio stream from the default input device/mic + STT
  # The STT processor is used to convert the audio stream to text. This will
  # be used to store the conversation history in the prompt.
  pya = pyaudio.PyAudio()
  input_processor = audio_io.PyAudioIn(pya) + speech_to_text.SpeechToText(
      project_id=GOOGLE_PROJECT_ID,
      with_interim_results=False,
  )

  # Main model that will be used to generate the response. Note that filter
  # before the genai model that will remove the audio parts and will make sure
  # only text is sent to the model.
  genai_processor = _filter_parts + models.turn_based_model(
      system_instruction=INSTRUCTION_PARTS
  )

  # TTS processor that will be used to convert the text response to audio. Note
  # the rate limit audio processor that will be used to stream back small audio
  # chunks to the client at the same rate as how they are played back. This is
  # needed to stop the audio when the user is speaking: the rate limit audio
  # processor will then stop and the audio will not be played anymore.
  tts = text_to_speech.TextToSpeech(
      project_id=GOOGLE_PROJECT_ID
  ) + rate_limit_audio.RateLimitAudio(
      sample_rate=24000,
      delay_other_parts=True,
  )

  # Plays the audio parts. This processor also handles interruptions and makes
  # sure the audio output stops when the user is speaking.
  play_output = audio_io.PyAudioOut(pya)

  # Creates an agent as:
  # mic -> speech to text -> text conversation -> text to speech -> play audio
  conversation_agent = (
      input_processor
      + realtime.LiveProcessor(turn_processor=genai_processor + tts)
      + play_output
  )
  prompt = 'USER (ctrl+D to end)> '
  await text.terminal_output(
      conversation_agent(text.terminal_input(prompt=prompt)), prompt=prompt
  )


def main(argv: Sequence[str]):
  del argv  # Unused.
  if not GOOGLE_PROJECT_ID:
    raise ValueError(
        'Project ID is not set. Define a GOOGLE_PROJECT_ID environment variable'
        ' obtained from your Cloud project.'
    )
  asyncio.run(run_conversation())


if __name__ == '__main__':
  app.run(main)
