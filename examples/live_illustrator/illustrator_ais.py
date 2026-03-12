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

r"""Live Illustrator WebSocket server for AI Studio.

A server to run the live illustrator agent on AI Studio. The applet connects
to this server and handles the UI.

See illustrator.py for the actual implementation.

To run the server locally:

 * Install the dependencies with `pip install genai-processors`.
 * Define a GOOGLE_API_KEY environment variable with your API key.
 * Launch the illustrator agent: `python3 ./illustrator_ais.py
 --alsologtostderr` (remove the `alsologtostderr` if you don't want the info
 logs). You can set the log verbosity flag with `--verbosity=1` (for debug
 logging) if you want to see more log information.

 To use the applet with the server:

 * Access the applet at
 https://aistudio.google.com/app/apps/github/google-gemini/genai-processors/tree/main/examples/live_illustrator/ais_app.
 * Allow the applet to use a camera and enable one of the video sources.
 * Set the period of the image generation (recommended at least 20 seconds).
 * Set the system instruction for the model (e.g. "create images in a cartoon
 style")
 * Open mic and start talking or having a conversation with a friend.
"""

import asyncio
import os
from typing import Any

from absl import app
from absl import flags
from genai_processors import processor
from genai_processors.examples import live_server
import illustrator

_PORT = flags.DEFINE_integer(
    'port',
    8765,
    'Port to run this WebSocket server on.',
)
_TRACE_DIR = flags.DEFINE_string(
    'trace_dir',
    None,
    'If set, enable tracing and write traces to this directory.',
)

_MAX_SIZE_BYTES = flags.DEFINE_integer(
    'max_size_bytes',
    None,
    'If set, limit the size of the trace file to this value.',
)


def create_live_illustrator(
    config: dict[str, Any],
) -> processor.Processor:
  """Creates a live commentator processor."""
  api_key = os.environ['GOOGLE_API_KEY']
  system_instruction = config.get('image_system_instruction', None)
  image_period_sec = config.get('image_period_sec', 20)
  return illustrator.create_live_illustrator(
      api_key=api_key,
      system_instruction=system_instruction,
      image_period_sec=image_period_sec,
  )


def main(argv):
  del argv
  asyncio.run(
      live_server.run_server(
          create_live_illustrator,
          port=_PORT.value,
          trace_dir=_TRACE_DIR.value,
          max_size_bytes=_MAX_SIZE_BYTES.value,
      )
  )


if __name__ == '__main__':
  app.run(main)
