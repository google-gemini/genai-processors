"""Trace implementation that stores traces in two files: JSON and HTML.

__WARNING__: This is an incubating feature. The trace format is subject to
changes and we do not guarantee backward compatibility at this stage.
"""

from __future__ import annotations

import asyncio
import base64
import datetime
import json
import os
from typing import Any, override

from genai_processors import content_api
from genai_processors.dev import trace
import pydantic
import shortuuid

HTML_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), 'trace.tpl.html')
with open(HTML_TEMPLATE_PATH, 'r') as f:
  HTML_TEMPLATE = f.read()

pydantic_converter = pydantic.TypeAdapter(Any)


def _bytes_encoder(o: Any) -> Any:
  """Encodes bytes in parts based on mime type.

  The dump_python(mode='json') in Pydantic does not encode bytes in utf-8
  mode and this causes issues when sending ProcessorPart to JS/HTML clients
  (wrong padding, etc.). This function is used to handle bytes to base64
  encoding to match the behaviour of the JS/HTML side.

  Args:
    o: The object to encode.

  Returns:
    The encoded object.
  """
  if isinstance(o, bytes):
    return base64.b64encode(o).decode('utf-8')
  else:
    return pydantic_converter.dump_python(o, mode='json')


# TODO(elisseeff): Adjust the logic to make it less brittle. If a new bytes
# field is added to the ProcessorPart in the future this function will not
# decode it while it should.
def _bytes_decoder(dct: dict[str, Any]) -> Any:
  """Decodes base64 encoded bytes in parts based on mime type."""
  if 'data' in dct and 'mime_type' in dct and isinstance(dct['data'], str):
    mime_type = dct['mime_type']
    if not mime_type.startswith('text/') and not mime_type.startswith(
        'application/json'
    ):
      try:
        dct['data'] = base64.b64decode(dct['data'])
      except (ValueError, TypeError):
        pass
  return dct


class TraceEvent(pydantic.BaseModel):
  """A single event in a trace.

  An event represents an input/output part or a sub-trace from a nested
  processor call.

  This class is not used in this abstract base class, but is recommend to be
  used in the implementations of the trace.
  """

  model_config = {'arbitrary_types_allowed': True}

  # A unique ID for this event.
  id: str = pydantic.Field(default_factory=lambda: str(shortuuid.uuid()))
  # The timestamp when the event was stored in the trace.
  timestamp: datetime.datetime = pydantic.Field(
      default_factory=datetime.datetime.now
  )
  # Whether the event is an input part to the processor or an output part.
  is_input: bool = False

  # The part of the event (as dictionary). None if sub_trace is provided.
  # By serializing the part into a dict we ensure that even if the part is
  # mutated later, the logged value won't change.
  part_dict: dict[str, Any] | None = None
  # If set, this event represents a nested processor call via its trace.
  sub_trace: SyncFileTrace | None = None


class SyncFileTrace(trace.Trace):
  """A trace storing events in a file.

  The trace collects all events first in memory and then writes them to a file
  when the finalize method is called.
  """

  # Where to store the trace. Required only of the root trace.
  trace_dir: str | None = None

  # The events in the trace. Collected in memory.
  events: list[TraceEvent] = []

  def to_json_str(self) -> str:
    """Converts the trace to a JSON string."""
    try:
      return json.dumps(
          self.model_dump(mode='python'),
          default=_bytes_encoder,
      )
    except TypeError as e:
      raise TypeError(
          'Failed to serialize trace to JSON. This might be due to'
          ' non-serializable types in ProcessorPart metadata. Ensure parts'
          ' added to traces are JSON-serializable.'
      ) from e

  @classmethod
  def from_json_str(cls, json_str: str) -> trace.Trace:
    """Initializes the trace from a JSON string.

    Args:
      json_str: The JSON string to initialize the trace from. The bytes field
        for audio and image parts are expected to be base64 + utf-8 encoded.

    Returns:
      The trace initialized from the JSON string.
    """
    return cls.model_validate(json.loads(json_str, object_hook=_bytes_decoder))

  def _add_part(self, part: content_api.ProcessorPart, is_input: bool) -> None:
    """Adds an input or output part to the trace events."""
    event = TraceEvent(
        part_dict=part.to_dict(mode='python'),
        is_input=is_input,
    )
    self.events.append(event)

  @override
  async def add_input(self, part: content_api.ProcessorPart) -> None:
    """Adds an input part to the trace events."""
    self._add_part(part, is_input=True)

  @override
  async def add_output(self, part: content_api.ProcessorPart) -> None:
    """Adds an output part to the trace events."""
    self._add_part(part, is_input=False)

  @override
  async def add_sub_trace(self, name: str) -> SyncFileTrace:
    """Adds a sub-trace from a nested processor call to the trace events."""
    t = SyncFileTrace(name=name)
    self.events.append(TraceEvent(sub_trace=t, is_input=False))
    return t

  @override
  async def _finalize(self) -> None:
    """Saves the trace to a file."""
    if not self.trace_dir:
      return
    trace_filename = os.path.join(
        self.trace_dir, f'{self.name}_{self.trace_id}'
    )
    await asyncio.to_thread(self.save, trace_filename + '.json')
    await asyncio.to_thread(self.save_html, trace_filename + '.html')

  def save_html(self, path: str) -> None:
    """Saves an HTML rendering of the trace to a file."""
    html = HTML_TEMPLATE.format(trace_json=self.to_json_str())
    print(f'html: {path}')
    with open(path, 'w') as html_file:
      html_file.write(html)

  def save(self, path: str) -> None:
    """Saves a trace to a file in JSON format.

    Args:
      path: The path to the file.
    """
    with open(path, 'w') as html_file:
      html_file.write(self.to_json_str())

  @classmethod
  def load(cls, path: str) -> 'SyncFileTrace':
    """Reads a trace from a JSON file.

    Args:
      path: The path to the file.

    Returns:
      The trace.
    """
    with open(path, 'r') as html_file:
      return SyncFileTrace.model_validate(
          json.loads(html_file.read(), object_hook=_bytes_decoder)
      )


SyncFileTrace.model_rebuild(force=True)
