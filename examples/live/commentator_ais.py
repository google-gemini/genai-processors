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

r"""Live Commentator WebSocket server for AI Studio.

A server to run the live commentator agent on AI Studio. The applet connects
to this server and handles the UI.

See commentator.py for the actual implementation.

To run the server locally:

 * Install the dependencies with `pip install genai-processors`.
 * Access the applet at
 https://aistudio.google.com/app/apps/github/google-gemini/genai-processors/tree/main/examples/live/ais_app.
* Define a GOOGLE_API_KEY environment variable with your API key.
 * Launch the commentator agent: `python3 ./commentator_ais.py`.
 * Allow the applet to use a camera and enable one of the video sources.
"""

import asyncio
import os
from typing import Any

from absl import app
from absl import flags
from absl import logging
from genai_processors import processor
from genai_processors.examples import live_server
import commentator

_PORT = flags.DEFINE_integer(
    'port',
    8765,
    'Port to run this WebSocket server on.',
)
_DEBUG = flags.DEFINE_bool(
    'debug',
    False,
    'Enable debug logging.',
)


def create_live_commentator(
    config: dict[str, Any],
) -> processor.Processor:
  """Creates a live commentator processor."""
  chattiness = config.get('chattiness', 0.5)
  api_key = os.environ['GOOGLE_API_KEY']
  return commentator.create_live_commentator(
      api_key=api_key,
      chattiness=chattiness,
      unsafe_string_list=None,
  )


def main(argv):
  if len(argv) > 2:
    raise app.UsageError('Too many command-line arguments.')
  if _DEBUG.value:
    logging.set_verbosity(logging.DEBUG)
  asyncio.run(live_server.run_server(create_live_commentator, port=_PORT.value))


if __name__ == '__main__':
  app.run(main)
