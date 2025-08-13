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

An applet within AI Studio provides a UI. It gets audio and video from a camera
or screencast, sends it to this agent and playbacks the audio produced by the
agent.

The agent uses Genai Processors to transform incoming streams of parts and
passes them to Gemini Live API to generate the commentary.

This file contains plumbing to connect the agent to AI Studio.

See commentator.py for the actual implementation.

To run the server locally:

 * Install the dependencies with `pip install genai-processors`.
 * Access the applet at
 https://aistudio.google.com/app/apps/github/google-gemini/genai-processors/tree/main/examples/live/ais_app.
* Define a GOOGLE_API_KEY environment variable with your API key.
 * Launch the commentator agent: `python3 ./commentator_ais.py`.
 * Allow the applet to use a camera and enable one of the video sources.
"""

import argparse
import asyncio
import base64
import dataclasses
import json
import os
import time
from typing import AsyncIterable

from absl import logging
from genai_processors import content_api
import commentator
from websockets.asyncio.server import serve
from websockets.asyncio.server import ServerConnection
from websockets.exceptions import ConnectionClosed

# You need to define the API key in the environment variables.
# export GOOGLE_API_KEY=...
API_KEY = os.environ['GOOGLE_API_KEY']

# Mimetype for a command part. A command represents a specific instruction for
# the server to trigger actions or modify its state.
_COMMAND_MIMETYPE = 'application/x-command'

# Config parts can be sent to this server to configure the live commentator.
_CONFIG_MIMETYPE = 'application/x-config'

# Mimetype to represent the state of either the client or the server.
_STATE_MIMETYPE = 'application/x-state'


@dataclasses.dataclass(frozen=True)
class MediaPart:
  """A part of media data."""

  base64data: str
  mime_type: str

  @classmethod
  def from_json(cls, json_part: str) -> 'MediaPart':
    """Creates a Media Part from a JSON part."""
    json_dict = json.loads(json_part)
    return MediaPart(
        base64data=json_dict['data'],
        mime_type=json_dict['mime_type'],
    )

  def is_image(self) -> bool:
    """Returns whether the part is an image."""
    return self.mime_type.startswith('image/')

  def is_audio(self) -> bool:
    """Returns whether the part is audio."""
    return self.mime_type.startswith('audio/')

  def is_reset_command(self) -> bool:
    """Returns whether the part is a reset command."""
    return self.mime_type == _COMMAND_MIMETYPE and self.base64data == 'RESET'

  def is_config(self) -> bool:
    """Returns whether the part is a config."""
    return self.mime_type == _CONFIG_MIMETYPE

  def is_mic_off(self) -> bool:
    """Returns whether the part indicates the client has turned off the mic."""
    return self.mime_type == _STATE_MIMETYPE and self.base64data == 'MIC_OFF'


@dataclasses.dataclass
class LiveCommentatorConfig:
  """Config for the live commentator."""

  chattiness: float = 0.5

  @classmethod
  def from_json(cls, json_config: str) -> 'LiveCommentatorConfig':
    """Creates a LiveCommentatorConfig from a JSON config."""
    json_dict = json.loads(json_config)
    config = LiveCommentatorConfig()
    if (chattiness := json_dict.get('chattiness', None)) is not None:
      config.chattiness = chattiness
    return config


class AIStudioConnection:
  """A WebSocket connection with AI Studio."""

  def __init__(self, ais_ws: ServerConnection):
    self._ais_ws = ais_ws
    self.is_resetting = False
    self.live_commentator_config = LiveCommentatorConfig()

  async def send(
      self,
      output_stream: AsyncIterable[content_api.ProcessorPart],
  ):
    """Sends audio to AIS."""
    async for part in output_stream:
      if self.is_resetting:
        return

      if content_api.is_audio(part.mimetype):
        await self._ais_ws.send(
            json.dumps({
                'data': base64.b64encode(part.part.inline_data.data).decode(),
                'mime_type': part.mimetype,
            })
        )
      elif part.text:
        await self._ais_ws.send(
            json.dumps({
                'data': part.text,
                'mime_type': 'text/plain',
            })
        )
      elif part.get_metadata('generation_complete', False):
        await self._ais_ws.send(
            json.dumps({
                'data': 'GENERATION_COMPLETE',
                'mime_type': _STATE_MIMETYPE,
            })
        )
      elif part.get_metadata('interrupted', False):
        await self._ais_ws.send(
            json.dumps({
                'data': 'INTERRUPTED',
                'mime_type': _STATE_MIMETYPE,
            })
        )
      else:
        logging.debug(
            '%s - Chunk not sent to AIS: %s', time.perf_counter(), part
        )

  async def receive(self) -> AsyncIterable[content_api.ProcessorPart]:
    """Reads chunks from AIS."""
    async for json_part in self._ais_ws:
      part = MediaPart.from_json(json_part)
      if part.is_image() or part.is_audio():
        yield content_api.ProcessorPart(
            base64.b64decode(part.base64data),
            mimetype=part.mime_type,
            substream_name='realtime',
            role='user',
        )
      elif part.is_mic_off():
        yield content_api.ProcessorPart(
            '',
            substream_name='realtime',
            role='user',
            metadata={'audio_stream_end': True},
        )
      elif part.is_reset_command():
        # Stop reading from the WebSocket until the agent has been reset.
        logging.debug(
            "%s - RESET command received. Resetting the agent's state.",
            time.perf_counter(),
        )
        self.is_resetting = True
        return
      elif part.is_config():
        self.live_commentator_config = LiveCommentatorConfig.from_json(
            part.base64data
        )
        logging.debug(
            '%s - Config received: %s',
            time.perf_counter(),
            self.live_commentator_config,
        )
        self.is_resetting = True
        return
      else:
        logging.warning('Unknown input type: %s', part.mime_type)


async def live_commentary(ais_websocket: ServerConnection):
  """Runs the live commentary agent on AI Studio input/output streams."""
  ais = AIStudioConnection(ais_websocket)

  # Running in a loop as the agent can receive a RESET command from AIS, in
  # which case the live commentary loop needs to be reinitialized.
  while True:
    commentator_processor = commentator.create_live_commentator(
        API_KEY,
        chattiness=ais.live_commentator_config.chattiness,
        unsafe_string_list=None,
    )
    try:
      await ais.send(commentator_processor(ais.receive()))
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.debug(
          '%s - Resetting live commentary after receiving error : %s',
          time.perf_counter(),
          e,
      )

    ais.is_resetting = False

    # Exit the loop if the connection is closed.
    try:
      await ais_websocket.send(
          json.dumps({
              'data': 'HEALTH_CHECK',
              'mime_type': _STATE_MIMETYPE,
          })
      )
    except ConnectionClosed:
      logging.debug('Connection between AIS and agent has been closed.')
      break


async def run_server(port: int) -> None:
  """Starts the WebSocket server."""
  async with serve(
      handler=live_commentary,
      host='localhost',
      port=port,
      max_size=2 * 1024 * 1024,  # 2 MiB
  ) as server:
    print(f'Server started on port {port}')
    await server.serve_forever()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--port',
      type=int,
      default=8765,
      help='Port to run this WebSocket server on.',
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
  asyncio.run(run_server(args.port))
