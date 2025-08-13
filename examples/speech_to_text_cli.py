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
r"""Command Line Interface to test the Speech-to-Text GenAI Processor.

## Setup

To install the dependencies for this script, run:

```
pip install --upgrade google-cloud-speech genai-processors
```

Before running this script, ensure the `GOOGLE_PROJECT_ID` environment
variable is set to your project_id (from Google Cloud Console).

## Run

To run the script:

```shell
python3 ./speech_to_text_cli.py
```

Example of outputs:
```
# By default, endpointing is enabled.
ProcessorPart(
    {'text': 'SPEECH_ACTIVITY_BEGIN'},
    mimetype='text/plain'),
    substream_name='input_endpointing'
)
# These are the first interim result. They are not final and are only for info.
ProcessorPart(
    {'text': 'hi'}, mimetype='text/plain',
    substream_name='input_transcription',
    metadata={'is_final': False},
    role='user'
)
ProcessorPart(
    {'text': 'hi there'},
    `mimetype='text/plain',
    substream_name='input_transcription',
    metadata={'is_final': False},
    role='user'
)
ProcessorPart(
    {'text': 'hi'},
    mimetype='text/plain',
    substream_name='input_transcription',
    metadata={'is_final': False},
    role='user'
)
# Endpointing detected the end of speech. All audio parts between this part
# and the previous SPEECH_ACTIVITY_BEGIN are the ones corresponding to the
# transcript.
ProcessorPart(
    {'text': 'SPEECH_ACTIVITY_END'},
    mimetype='text/plain'
    substream_name='input_endpointing'
)
# Final transcript. It is normal to happen after the SPEECH_ACTIVITY_END here.
ProcessorPart(
    {'text': 'hi there'},
    mimetype='text/plain',
    substream_name='input_transcription',
    metadata={'is_final': True},
    role='user'
)
```
"""

import asyncio
import os
import time

from genai_processors.core import audio_io
from genai_processors.core import speech_to_text
from genai_processors.core import text
import pyaudio


# You need to define the project id in the environment variables.
# export GOOGLE_PROJECT_ID=...
GOOGLE_PROJECT_ID = os.environ['GOOGLE_PROJECT_ID']


async def run_stt() -> None:
  """Runs speech-to-text in a CLI environment."""
  pya = pyaudio.PyAudio()
  stt_processor = audio_io.PyAudioIn(pya) + speech_to_text.SpeechToText(
      project_id=GOOGLE_PROJECT_ID, with_interim_results=True
  )

  print(f'{time.perf_counter()} - STT Processor ready: start talking anytime.')
  async for parts in stt_processor(text.terminal_input()):
    print(f'{time.perf_counter()} - STT Parts: {parts}')


if __name__ == '__main__':
  asyncio.run(run_stt())
