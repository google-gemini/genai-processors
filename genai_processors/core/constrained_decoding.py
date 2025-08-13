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
"""Processors for handling constrained decoding and structured outputs."""

import dataclasses
import enum
import json
from typing import Any, AsyncIterable, Callable, Type, get_args, get_origin

from genai_processors import content_api
from genai_processors import processor


def _item_parser(
    item_type: Type[Any],
) -> Callable[[Any], content_api.ProcessorPart]:
  """Returns a parser for a single a Enum or Dataclass of type `item_type`."""
  if not isinstance(item_type, type):
    item_type = item_type.__class__

  if hasattr(item_type, 'from_dict') and dataclasses.is_dataclass(item_type):

    def parse_dataclass(data: Any) -> content_api.ProcessorPart:
      instance = item_type.from_dict(data)
      return content_api.ProcessorPart.from_dataclass(dataclass=instance)

    return parse_dataclass

  elif issubclass(item_type, enum.Enum):

    def parse_enum(data: Any) -> content_api.ProcessorPart:
      instance = item_type(data)
      return content_api.ProcessorPart(instance.value)

    return parse_enum
  else:
    raise TypeError(
        f'{item_type.__name__} must be a dataclass with the'
        ' `dataclasses_json` mixin or an Enum.'
    )


class StructuredOutputParser(processor.Processor):
  """Buffers a stream of text parts to parse a JSON object or list.

  This processor is designed to be placed after a model processor that
  streams a JSON response. It consumes the entire input stream, buffers
  the text content, and then attempts to parse it as JSON.

  If the target `schema` is a list (e.g., `list[MyData]`), this processor will
  yield each item of the list as a separate `ProcessorPart` containing an
  instance of `MyData`. This is useful for chaining with PartProcessors that can
  then operate on each item concurrently.
  """

  def __init__(self, schema: type[Any]):
    """Initializes the processor.

    Args:
      schema: The schema to parse the JSON into. The schema must be a singleton
        enum or dataclass, or be a list of such types.
    """
    origin = get_origin(schema)
    is_list = origin is list

    if is_list:
      item_type = get_args(schema)[0]
      item_parser = _item_parser(item_type)

      async def parse_list(
          buffer: str,
      ) -> AsyncIterable[content_api.ProcessorPartTypes]:
        parsed_data = json.loads(buffer)
        if not isinstance(parsed_data, list):
          raise TypeError(
              'Model output was not a list, but expected'
              f' list[{item_type.__name__}].'
          )
        for item_value in parsed_data:
          yield item_parser(item_value)

      self._parser = parse_list
    else:
      item_type = schema
      item_parser = _item_parser(item_type)

      async def parse_single(
          buffer: str,
      ) -> AsyncIterable[content_api.ProcessorPartTypes]:
        try:
          parsed_data = json.loads(buffer)
        except json.JSONDecodeError:
          type_to_check = (
              item_type if isinstance(item_type, type) else item_type.__class__
          )
          if issubclass(type_to_check, enum.Enum):
            parsed_data = buffer
          else:
            raise
        yield item_parser(parsed_data)

      self._parser = parse_single

  @processor.yield_exceptions_as_parts
  async def call(
      self, content: AsyncIterable[content_api.ProcessorPart]
  ) -> AsyncIterable[content_api.ProcessorPartTypes]:
    buffer = ''
    async for part in content:
      if content_api.is_text(part.mimetype):
        buffer += part.text
      else:
        yield part

    # Empty string is not a valid JSON. But we want to treat it as empty output.
    buffer = buffer.strip()
    if not buffer:
      return

    async for parsed_part in self._parser(buffer):
      yield parsed_part
