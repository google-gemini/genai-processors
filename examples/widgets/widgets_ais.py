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

r"""Dynamic Widgets demo WebSocket server for AI Studio.

A server / backend to run the Dynamic Widgets demo on AI Studio. Can be used
from this applet:
https://aistudio.google.com/app/apps/github/google-gemini/genai-processors/tree/main/examples/widgets/ais_app

See widgets.py for the actual implementation.

To run the server locally:

 * Install the dependencies with `pip install genai-processors[live]`.
 * Define a GOOGLE_API_KEY environment variable with your API key.
 * Launch the widgets agent: `python3 ./widgets_ais.py --alsologtostderr`.
 * Access the applet at
 https://aistudio.google.com/app/apps/github/google-gemini/genai-processors/tree/main/examples/widgets/ais_app.

A good prompt to test the server is:
  Describe what conic sections are illustrating each of them with a plot.
"""

import asyncio
import os
from typing import Any

from absl import app
from absl import flags
from genai_processors import processor
from genai_processors.examples import live_server
import widgets

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


def create_dr_widget(
    config: dict[str, Any],
) -> processor.Processor:
  """Creates a dynamic widgets processor."""
  del config  # Unused.
  api_key = os.environ['GOOGLE_API_KEY']
  return widgets.create_dr_widget(
      api_key=api_key,
  )


def main(argv):
  del argv
  asyncio.run(
      live_server.run_server(
          create_dr_widget,
          port=_PORT.value,
          trace_dir=_TRACE_DIR.value,
      )
  )


if __name__ == '__main__':
  app.run(main)
