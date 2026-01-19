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

r"""Live WebSocket server for AI Studio.

This server takes a GenAI processor factory as input and wrap it around a
WebSocket server. This allows any GenAI processor to be used with external
applications such as AI Studio.

The server expects a stringified json dictionary of content_api.ProcessorPart as
input from the client. The bytes field should be base64 encoded. The mime_type
field should be set to the correct mime_type. The substream_name field should be
set to 'realtime' for images and audio streamed live from the UI.

Empty ProcessorPart with the following metadata fields can also be sent:

-  `'reset' : True` - This is sent when to reset the server.
-  `'mic: 'off'` - This is sent when the client has turned off the mic.

The server returns a stringified json dictionary of content_api.ProcessorPart as
output. The server can also send the following empty ProcessorPart of mime type
`application/x-state` with metadata:

-  `'generation_complete' : True` - This is sent when the processor has
    finished generating a response.
-  `'interrupted' : True` - This is sent when the processor has been
    interrupted, e.g. due to a `start of speech` signal.
-  `'health_check' : True` - This is sent periodically to check if the
    connection is alive.


Example of messages going back and forth between the client and the server:

## From the client
{
  "part": {
    "text": "Hello World",
    }
  "role": "user",
  "substream_name": "realtime",
  }
}

## From the client
{
  "part": {
    "inline_data": {
      "data": "SGVsbG8gV29ybGQ=", # Base64 encoded
      "mime_type": "audio/webm",
    }
  "role": "user",
  "metadata": {
    "audio_stream_end": true,
    }
  }
}

## From the server
{
  "part": {
    "text": "Hello World",
    }
  "role": "model",
  "metadata": {
    "generation_complete": true,
    }
  }
}

## From the server
{
  "metadata": {
    "health_check": true,
    }
  }
  "mimetype": "application/x-state",
}

__WARNING__: this is a work in progress and is provided here for convenience and
for prototyping quickly. We will likely make backwards incompatible changes to
comply with a more stable GenAI API between clients and servers.
"""
import base64
import functools
import json
import time
import traceback
from typing import Any, AsyncIterable, Callable

from absl import logging
from genai_processors import content_api
from genai_processors import processor
import pydantic
from websockets.asyncio.server import serve
from websockets.asyncio.server import ServerConnection
from websockets.exceptions import ConnectionClosed

# Mimetype for a command part. A command represents a specific instruction for
# the server to trigger actions or modify its state.
_COMMAND_MIMETYPE = 'application/x-command'

# Config parts can be sent to this server to configure the server.
_CONFIG_MIMETYPE = 'application/x-config'

# Mimetype to represent the state of either the client or the server.
_STATE_MIMETYPE = 'application/x-state'


def is_reset_command(part: content_api.ProcessorPart) -> bool:
  """Returns whether the part is a reset command."""
  return (
      part.mimetype == _COMMAND_MIMETYPE
      and part.get_metadata('command', None) == 'reset'
  )


def is_config(part: content_api.ProcessorPart) -> bool:
  """Returns whether the part is a config."""
  return part.mimetype == _CONFIG_MIMETYPE


def is_mic_off(part: content_api.ProcessorPart) -> bool:
  """Returns whether the part indicates the client has turned off the mic."""
  return (
      part.mimetype == _STATE_MIMETYPE
      and part.get_metadata('mic', None) == 'off'
  )


pydantic_converter = pydantic.TypeAdapter(Any)


def clean_encoder(o: Any) -> Any:
  """A custom encoder for json.dumps that handles bytes with base64 encoding.

  The from_dict(mode='json') in ProcessorPart does not encode bytes in utf-8
  mode and this causes issues when sending ProcessorPart to JS clients (wrong
  padding, etc.). This function is used to handle bytes to base64 encoding to
  match the behaviour of the JS side. It can be used as an argument for
  json.dumps. Use it with JS dictionaries obtained from the ProcessorPart
  to_dict(mode='python') method (leaving bytes as is).

  Args:
    o: The object to be encoded. Uses pydantic to convert to json excpet when it
      is bytes. Then uses base64 encoding for bytes followed by utf-8 decoding.

  Returns:
    The object encoded as a string.
  """
  if isinstance(o, bytes):
    # Convert to Standard Base64 (happy path for JS atob)
    return base64.b64encode(o).decode('utf-8')
  return pydantic_converter.dump_python(o, mode='json')


class AIStudioConnection:
  """A WebSocket connection with AI Studio."""

  def __init__(self, ais_ws: ServerConnection):
    self._ais_ws = ais_ws
    self.is_resetting = False
    # Config dictionary directly transferred to the genai processor.
    self.processor_config = {}

  async def send(
      self,
      output_stream: AsyncIterable[content_api.ProcessorPart],
  ):
    """Sends audio to AIS."""
    async for part in output_stream:
      if self.is_resetting:
        return
      if part.get_metadata('generation_complete', False) or part.get_metadata(
          'interrupted', False
      ):
        part = content_api.ProcessorPart(
            '',
            mimetype=_STATE_MIMETYPE,
            metadata=part.metadata,
        )
        await self._ais_ws.send(json.dumps(part.to_dict()))
      else:
        await self._ais_ws.send(
            json.dumps(part.to_dict(mode='python'), default=clean_encoder)
        )

  async def receive(self) -> AsyncIterable[content_api.ProcessorPart]:
    """Reads chunks from AIS."""
    async for part_dict in self._ais_ws:
      json_dict = json.loads(part_dict)
      if 'part' not in json_dict:
        json_dict['part'] = {'text': ''}
      part = content_api.ProcessorPart.from_dict(
          data=json_dict,
      )
      if content_api.is_image(part.mimetype) or content_api.is_audio(
          part.mimetype
      ):
        part.substream_name = 'realtime'
        part.role = 'user'
        yield part
      elif is_mic_off(part):
        yield content_api.ProcessorPart(
            '',
            substream_name='realtime',
            role='user',
            metadata={'audio_stream_end': True},
        )
      elif is_reset_command(part):
        # Stop reading from the WebSocket until the processor has been reset.
        logging.debug(
            "%s - RESET command received. Resetting the processor's state.",
            time.perf_counter(),
        )
        self.is_resetting = True
        return
      elif is_config(part) and part.metadata:
        self.processor_config = part.metadata
        logging.info(
            '%s - Config received: %s',
            time.perf_counter(),
            self.processor_config,
        )
        self.is_resetting = True
        return
      else:
        logging.warning('Unknown input part type: %s', part.mimetype)


async def live_server(
    processor_factory: Callable[[dict[str, Any]], processor.Processor],
    ais_websocket: ServerConnection,
):
  """Runs the processor on AI Studio input/output streams."""
  ais = AIStudioConnection(ais_websocket)

  # Running in a loop as the agent can receive a RESET command from AIS, in
  # which case the live loop needs to be reinitialized.
  while True:
    try:
      live_processor = processor_factory(ais.processor_config)
      await ais.send(live_processor(ais.receive()))
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.info(
          '%s - Resetting live server after receiving error :'
          ' %s\n\nTraceback:\n%s',
          time.perf_counter(),
          e,
          traceback.format_exc(),
      )

    ais.is_resetting = False

    # Exit the loop if the connection is closed.
    try:
      await ais_websocket.send(
          json.dumps(
              content_api.ProcessorPart(
                  '', metadata={'health_check': True}
              ).to_dict()
          )
      )
    except ConnectionClosed:
      logging.debug('Connection between AIS and agent has been closed.')
      break


async def run_server(
    processor_factory: Callable[[dict[str, Any]], processor.Processor],
    port: int = 8765,
) -> None:
  """Starts the WebSocket server."""

  async with serve(
      handler=functools.partial(live_server, processor_factory),
      host='localhost',
      port=port,
      max_size=2 * 1024 * 1024,  # 2 MiB
  ) as server:
    print(f'Server started on port {port}')
    await server.serve_forever()
