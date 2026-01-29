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
from typing import Any, Tuple

from genai_processors import content_api
from genai_processors.dev import trace
import pydantic
import shortuuid
from typing_extensions import override
import xxhash

HTML_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), 'trace.tpl.html')
with open(HTML_TEMPLATE_PATH, 'r') as f:
  HTML_TEMPLATE = f.read()

pydantic_converter = pydantic.TypeAdapter(Any)

_QUEUE_MAX_SIZE = 1000


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


def _resize_image_part(
    part: content_api.ProcessorPart, image_size: tuple[int, int]
) -> content_api.ProcessorPart:
  """Resizes image part."""
  try:
    img = part.pil_image
    img.thumbnail(image_size)
    part = content_api.ProcessorPart(
        img,
        role=part.role,
        substream_name=part.substream_name,
        mimetype=part.mimetype,
        metadata=part.metadata,
    )
  except Exception:  # pylint: disable=broad-except
    # If resizing fails, we just use the original part.
    pass
  return part


def _compute_part_hash(part_dict: dict[str, Any]) -> str:
  """Computes a hash for a part_dict for efficient deduplication.

  Args:
    part_dict: The part dictionary to hash.

  Returns:
    A hex string hash of the part_dict content.
  """
  json_str = json.dumps(part_dict, sort_keys=True, default=str)
  return xxhash.xxh128(json_str.encode('utf-8')).hexdigest()


class TraceEvent(pydantic.BaseModel):
  """A single event in a trace.

  An event represents an input/output part or a sub-trace from a nested
  processor call.
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

  # Hash of the part (used for deduplication with the parts store in the root
  # trace).
  part_hash: str | None = None
  # If set, this event represents a nested processor call via its trace.
  sub_trace: SyncFileTrace | None = None
  # The relation between this trace and sub_trace. E.g. if it is a chain.
  relation: str | None = None


class SyncFileTrace(trace.Trace):
  """A trace storing events in a file.

  The trace collects all events first in memory and then writes them to a file
  when the finalize method is called. Parts are deduplicated online as they are
  added, reducing memory footprint for traces with repeated content.
  """

  # Where to store the trace. Required only of the root trace.
  trace_dir: str | None = None

  # The events in the trace. Collected in memory.
  events: list[TraceEvent] = pydantic.Field(default_factory=list)
  _queue: asyncio.Queue[Tuple[content_api.ProcessorPart, bool] | None] = (
      pydantic.PrivateAttr()
  )
  _worker: asyncio.Task[None] = pydantic.PrivateAttr()

  # The size to resize images to when storing them in the trace.
  # If None, images are not resized.
  image_size: tuple[int, int] | None = (200, 200)

  # Parts store for deduplication: dictionary of a part's hash to the dict
  # representing the part. For sub-traces, this should remain empty; they use
  # root's store.
  parts_store: dict[str, dict[str, Any]] = pydantic.Field(default_factory=dict)

  # Reference to root trace for sharing parts_store. None means this is root.
  _root_trace: 'SyncFileTrace | None' = pydantic.PrivateAttr(default=None)

  def _get_root(self) -> 'SyncFileTrace':
    """Returns the root trace for parts_store access."""
    return self._root_trace if self._root_trace is not None else self

  def model_post_init(self, __context: Any) -> None:  # pylint: disable=invalid-name
    self._queue = asyncio.Queue(maxsize=_QUEUE_MAX_SIZE)
    self._worker = asyncio.create_task(self._event_worker())

  async def _event_worker(self):
    """Worker task to process parts from queue and create events."""
    while item := await self._queue.get():
      part, is_input = item
      event = await self._add_part(part, is_input=is_input)
      self.events.append(event)

  def to_json_str(self) -> str:
    """Converts the trace to a JSON string with deduplicated parts.

    The output format is:
    {
      "parts_store": {"0": {...}, "1": {...}, ...},
      "trace": { ... trace data with part_ref ... }
    }

    Returns:
      A JSON string representing the trace with deduplicated parts.
    """
    try:
      trace_data = self.model_dump(
          mode='python',
          exclude_none=True,
      )
      output = {
          'parts_store': self.parts_store,
          'trace': trace_data,
      }
      return json.dumps(
          output,
          default=_bytes_encoder,
          indent=2,
      )
    except TypeError as e:
      raise TypeError(
          'Failed to serialize trace to JSON. This might be due to'
          ' non-serializable types in ProcessorPart metadata. Ensure parts'
          ' added to traces are JSON-serializable.'
      ) from e

  @classmethod
  def from_json_str(cls, json_str: str) -> 'SyncFileTrace':
    """Initializes the trace from a JSON string.

    Args:
      json_str: The JSON string to initialize the trace from. The bytes field
        for audio and image parts are expected to be base64 + utf-8 encoded.

    Returns:
      The trace initialized from the JSON string. Note that the parts store
      cannot be reused for deduplication: a part added to the trace after
      deserialization will not be deduplicated against parts added before
      deserialization.
    """
    data = json.loads(json_str, object_hook=_bytes_decoder)
    parts_store = data['parts_store']
    trace_data = data['trace']
    trace_data['parts_store'] = parts_store
    trace_obj = cls.model_validate(trace_data)
    # Set _root_trace for all sub-traces
    cls._init_sub_traces(trace_obj, trace_obj)
    return trace_obj

  @classmethod
  def _init_sub_traces(
      cls, parent: 'SyncFileTrace', root: 'SyncFileTrace'
  ) -> None:
    """Recursively sets _root_trace for all sub-traces."""
    for event in parent.events:
      if event.sub_trace is not None:
        event.sub_trace._root_trace = root
        cls._init_sub_traces(event.sub_trace, root)

  async def _add_part(
      self, part: content_api.ProcessorPart, is_input: bool
  ) -> TraceEvent:
    """Adds an input or output part to the trace events."""
    if self.image_size and content_api.is_image(part.mimetype):
      part = await asyncio.to_thread(_resize_image_part, part, self.image_size)

    part_dict = part.to_dict(mode='python')
    part_hash = _compute_part_hash(part_dict)

    # Use root's store for global deduplication across all sub-traces
    root = self._get_root()

    # O(1) deduplication lookup
    if part_hash not in root.parts_store:
      root.parts_store[part_hash] = part_dict

    event = TraceEvent(
        part_hash=part_hash,
        is_input=is_input,
    )
    return event

  @override
  async def add_input(self, part: content_api.ProcessorPart) -> None:
    """Adds an input part to the trace events."""

    await self._queue.put((part, True))

  @override
  async def add_output(self, part: content_api.ProcessorPart) -> None:
    """Adds an output part to the trace events."""

    await self._queue.put((part, False))

  @override
  def add_sub_trace(self, name: str, relation: str) -> SyncFileTrace:
    """Adds a sub-trace from a nested processor call to the trace events."""
    # This method must not block.
    t = SyncFileTrace(name=name, image_size=self.image_size)
    # Sub-traces share the root's parts_store for global deduplication
    t._root_trace = self._get_root()
    event = TraceEvent(sub_trace=t, is_input=False, relation=relation)
    self.events.append(event)
    return t

  @override
  async def _finalize(self) -> None:
    """Saves the trace to a file."""
    await self._queue.put(None)  # Sentinel to stop worker.
    await asyncio.shield(self._worker)

    for event in self.events:
      if sub_trace := event.sub_trace:
        await sub_trace._finalize()

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
    with open(path, 'r') as trace_file:
      return cls.from_json_str(trace_file.read())


SyncFileTrace.model_rebuild(force=True)
