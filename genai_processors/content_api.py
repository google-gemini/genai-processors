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
"""Syntax sugar for working with Processor `Content` and `Part` wrappers."""

from collections.abc import Callable, Iterable, Iterator
import dataclasses
import functools
import io
import itertools
import json
from typing import Any, TypeVar

from absl import logging
from genai_processors import mime_types
from google.genai import types as genai_types
import PIL.Image


class ProcessorPart:
  """A wrapper around `Part` with additional metadata.

  Represents a single piece of content that can be processed by an agentic
  system.

  Includes metadata such as the producer of the content, the substream the part
  belongs to, the MIME type of the content, and arbitrary metadata.
  """

  def __init__(
      self,
      value: 'ProcessorPartTypes',
      *,
      role: str = '',
      substream_name: str = '',
      mimetype: str | None = None,
      metadata: dict[str, Any] | None = None,
  ) -> None:
    """Constructs a ProcessorPart using a `Part` or `ProcessorPart`.

    Args:
      value: The content to use to construct the ProcessorPart.
      role: Optional. The producer of the content. In Genai models, must be
        either 'user' or 'model', but the user can set their own semantics.
        Useful to set for multi-turn conversations, otherwise can be empty.
      substream_name: (Optional) ProcessorPart stream can be split into multiple
        independent streams. They may have specific semantics, e.g. a song and
        its lyrics, or can be just alternative responses. Prefer using a default
        substream with an empty name. If the `ProcessorPart` is created using
        another `ProcessorPart`, this ProcessorPart inherits the existing
        substream_name, unless it is overridden in this argument.
      mimetype: Mime type of the data.
      metadata: (Optional) Auxiliary information about the part. If the
        `ProcessorPart` is created using another `ProcessorPart`, this
        ProcessorPart inherits the existing metadata, unless it is overridden in
        this argument.
    """
    super().__init__()
    self._metadata = {}

    match value:
      case genai_types.Part():
        self._part = value
      case ProcessorPart():
        self._part = value.part
        role = role or value.role
        substream_name = substream_name or value.substream_name
        mimetype = mimetype or value.mimetype
        self._metadata.update(value.metadata)
      case str():
        self._part = genai_types.Part(text=value)
      case bytes():
        if not mimetype:
          raise ValueError(
              'MIME type must be specified when constructing a ProcessorPart'
              ' from bytes.'
          )
        if is_text(mimetype):
          self._part = genai_types.Part(text=value.decode('utf-8'))
        else:
          self._part = genai_types.Part.from_bytes(
              data=value, mime_type=mimetype
          )
      case PIL.Image.Image():
        if mimetype:
          # If the mimetype is explicitly specified, ensure it is an image.
          if not mimetype.startswith('image/'):
            raise ValueError(f"Can't convert image of mimetype {mimetype}.")
          suffix = mimetype[len('image/') :]
          # Ensure it matches the Image format.
          if value.format:
            if suffix != value.format.lower():
              raise ValueError(
                  f'The image format {value.format} and does not match the'
                  f' mimetype {suffix}.'
              )
        else:
          # If no mimetype is specified, get it from the Image object.
          # If no format is provided, default to webp.
          suffix = value.format.lower() if value.format else 'webp'
          mimetype = f'image/{suffix}'
        bytes_io = io.BytesIO()
        value.save(bytes_io, suffix.upper())
        self._part = genai_types.Part.from_bytes(
            data=bytes_io.getvalue(), mime_type=mimetype
        )
      case _:
        raise ValueError(f"Can't construct ProcessorPart from {type(value)}.")

    self._role = role
    self._substream_name = substream_name
    self._metadata.update(metadata or {})

    # Set the MIME type.
    if mimetype:
      self._mimetype = mimetype
    # Otherwise, if MIME type is specified using inline data, use that.
    elif self._part.inline_data and self._part.inline_data.mime_type:
      self._mimetype = self._part.inline_data.mime_type
    # Otherwise, if text is not empty, assume 'text/plain' MIME type.
    elif self._part.text:
      self._mimetype = 'text/plain'
    else:
      self._mimetype = ''

  def __repr__(self) -> str:
    optional_args = ''
    if self.substream_name:
      optional_args += f', substream_name={self.substream_name!r}'
    if self.metadata:
      optional_args += f', metadata={self.metadata}'.rstrip('\n')
    if self.role:
      optional_args += f', role={self.role!r}'
    return (
        f'ProcessorPart({self.part.to_json_dict()!r},'
        f' mimetype={self.mimetype!r}{optional_args})'
    )

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, ProcessorPart):
      return False
    return (
        self._part == other._part
        and self._role.lower() == other._role.lower()
        and self._substream_name.lower() == other._substream_name.lower()
        and self._metadata == other._metadata
    )

  @property
  def part(self) -> genai_types.Part:
    """Returns the underlying Genai Part."""
    return self._part

  @property
  def role(self) -> str:
    """Optional. The producer of the content.

    Useful to set for multi-turn conversations, otherwise can be left blank or
    unset.

    Default value is an empty string. It is up to the user to set their own
    semantics for the role.
    """
    return self._role

  @role.setter
  def role(self, value: str) -> None:
    self._role = value

  @property
  def bytes(self) -> bytes | None:
    """Returns part contents as bytes.

    Returns:
      Text encoded into bytes or bytes from inline data if the underlying part
      is a Blob.
    """
    if self.part.text:
      return self.text.encode()
    if isinstance(self.part.inline_data, genai_types.Blob):
      return self.part.inline_data.data
    return None

  @property
  def substream_name(self) -> str:
    """Returns the stream this part belongs to.

    Empty for the default stream.
    """
    return self._substream_name

  @substream_name.setter
  def substream_name(self, value: str) -> None:
    self._substream_name = value

  @property
  def mimetype(self) -> str:
    """Returns part MIME type.

    Note: Empty MIME in the underlying `Part` is assumed to be text.
    """
    return self._mimetype or 'text/plain'

  @property
  def text(self) -> str:
    """Returns part text as string.

    Returns:
      The text of the part.

    Raises:
      ValueError if part has no text.
    """
    if not mime_types.is_text(self.mimetype):
      raise ValueError('Part is not text.')
    return self.part.text or ''

  @text.setter
  def text(self, value: str) -> None:
    """Sets part to a text part."""
    self._part = genai_types.Part(text=value)

  @property
  def metadata(self) -> dict[str, Any]:
    """Returns metadata."""
    return self._metadata

  @metadata.setter
  def metadata(self, value: dict[str, Any]) -> None:
    """Sets metadata."""
    self._metadata = value

  def get_metadata(self, key: str, default=None) -> Any:
    """Returns metadata for a given key."""
    return self._metadata.get(key, default)

  @property
  def function_call(self) -> genai_types.FunctionCall | None:
    """Returns function call."""
    return self.part.function_call

  @property
  def function_response(self) -> genai_types.FunctionResponse | None:
    """Returns function response."""
    return self.part.function_response

  @property
  def tool_cancellation(self) -> str | None:
    """Returns an id of a function call to be cancelled.

    If the part is not a tool cancellation request, returns None.

    Returns:
      The id of the function call to be cancelled or None if this part is not a
      tool cancellation from the model.
    """
    if not self.part.function_response:
      return None
    if self.part.function_response.name != 'tool_cancellation':
      return None
    if not self.part.function_response.response:
      return None
    return self.part.function_response.response.get('function_call_id', None)

  T = TypeVar('T')

  def get_dataclass(self, json_dataclass: type[T]) -> T:
    """Returns representation of the Part as a given dataclass.

    Args:
      json_dataclass: A dataclass that can be converted to/from JSON.

    Returns:
      The dataclass representation of the Part.
    """
    if not mime_types.is_dataclass(self.mimetype):
      raise ValueError('Part is not a dataclass.')
    try:
      # JSON conversions are provided by the dataclass_json decorator.
      return json_dataclass.from_json(self.text)  # pytype: disable=attribute-error
    except AttributeError as e:
      raise ValueError(
          f'{json_dataclass.__name__} is not a valid json dataclass'
      ) from e

  @property
  def pil_image(self) -> PIL.Image.Image:
    """Returns PIL.Image representation of the Part."""
    if not mime_types.is_image(self.mimetype):
      raise ValueError(f'Part is not an image. Mime type is {self.mimetype}.')
    bytes_io = io.BytesIO()
    if self.part.inline_data is not None:
      bytes_io.write(self.part.inline_data.data)
    bytes_io.seek(0)
    return PIL.Image.open(bytes_io)

  # Class methods that make use of underlying Genai `Part` class methods.
  @classmethod
  def from_uri(
      cls, *, file_uri: str, mimetype: str, **kwargs
  ) -> 'ProcessorPart':
    """Constructs a ProcessorPart from URI & mimetype."""
    part = genai_types.Part.from_uri(file_uri=file_uri, mime_type=mimetype)
    return cls(part, **kwargs)

  @classmethod
  def from_function_call(
      cls, *, name: str, args: dict[str, Any], **kwargs
  ) -> 'ProcessorPart':
    """Constructs a ProcessorPart from bytes & mimetype."""
    part = genai_types.Part.from_function_call(name=name, args=args)
    return cls(part, **kwargs)

  @classmethod
  def from_function_response(
      cls,
      *,
      name: str,
      response: dict[str, Any],
      function_call_id: str | None = None,
      will_continue: bool = False,
      scheduling: genai_types.FunctionResponseScheduling | None = None,
      **kwargs,
  ) -> 'ProcessorPart':
    """Constructs a ProcessorPart as a function response."""
    part = genai_types.Part(
        function_response=genai_types.FunctionResponse(
            id=function_call_id,
            name=name,
            response=response,
            will_continue=will_continue,
            scheduling=scheduling,
        )
    )
    return cls(part, **kwargs)

  @classmethod
  def from_executable_code(
      cls, *, code: str, language: genai_types.Language, **kwargs
  ) -> 'ProcessorPart':
    """Constructs a ProcessorPart as an executable code part."""
    part = genai_types.Part.from_executable_code(code=code, language=language)
    return cls(part, **kwargs)

  @classmethod
  def from_code_execution_result(
      cls, *, outcome: genai_types.Outcome, output: str, **kwargs
  ) -> 'ProcessorPart':
    """Constructs a ProcessorPart as a code execution result part."""
    part = genai_types.Part.from_code_execution_result(
        outcome=outcome, output=output
    )
    return cls(part, **kwargs)

  @classmethod
  def from_tool_cancellation(
      cls, *, function_call_id: str, **kwargs
  ) -> 'ProcessorPart':
    """Constructs a ProcessorPart from a tool cancellation id.

    The role is overridden to 'model'.

    Args:
      function_call_id: The id of the function call to be cancelled.
      **kwargs: Additional arguments for the ProcessorPart constructor.

    Returns:
      A ProcessorPart of type tool cancellation.
    """
    part = genai_types.Part.from_function_response(
        name='tool_cancellation',
        response={'function_call_id': function_call_id},
    )
    if 'role' in kwargs and kwargs['role'].lower() != 'model':
      logging.warning(
          'Role {kwargs["role"]} is not supported for tool cancellation.'
          ' Overriding it with the model role.'
      )
    extra_args = kwargs
    extra_args['role'] = 'model'
    return cls(part, **extra_args)

  @classmethod
  def from_dataclass(cls, *, dataclass: Any, **kwargs) -> 'ProcessorPart':
    """Constructs a ProcessorPart from a dataclass."""
    part = ProcessorPart(
        json.dumps(dataclasses.asdict(dataclass)),
        mimetype=f'application/json; type={type(dataclass).__name__}',
    )
    return cls(part, **kwargs)

  @classmethod
  def end_of_turn(cls) -> 'ProcessorPart':
    return ProcessorPart('', role='user', metadata={'turn_complete': True})

  @classmethod
  def from_dict(cls, *, data: dict[str, Any]) -> 'ProcessorPart':
    """Deserializes a ProcessorPart from a JSON-compatible dictionary.

    This method reconstructs a ProcessorPart instance from a dictionary
    that was typically generated by the `to_dict()` method of another
    ProcessorPart instance.

    Args:
      data: A JSON-compatible dictionary containing the serialized data for the
        ProcessorPart.  It is expected to have the following keys:
        * 'part' (dict): A dictionary representing the underlying
          `google.genai.types.Part` object.
        * 'role' (str): The role of the part (e.g., 'user', 'model').
        * 'substream_name' (str): The substream name.
        * 'mimetype' (str): The MIME type of the part.
        * 'metadata' (dict[str, Any]): Auxiliary metadata.

    Returns:
      A new ProcessorPart instance.

    Raises:
      pydantic.ValidationError: If the `part` field in `data` is not a valid
        dictionary representation of a GenAI part.
      KeyError: If 'part' is missing from `data`.

    Example:
    ```py
    text_part = ProcessorPart("Hello", role="user")
    part_as_dict = text_part.to_dict()
    reconstructed = ProcessorPart.from_dict(data=part_as_dict)
    print(reconstructed)
    ```
    """  # fmt: skip
    return cls(
        genai_types.Part.model_validate(data['part']),
        role=data.get('role', ''),
        substream_name=data.get('substream_name', ''),
        mimetype=data.get('mimetype'),
        metadata=data.get('metadata'),
    )

  def to_dict(self) -> dict[str, Any]:
    """Serializes this ProcessorPart to a JSON-compatible dictionary.

    The resulting dictionary can be used with `ProcessorPart.from_dict()`
    to reconstruct an equivalent ProcessorPart instance.

    Returns:
      A dictionary representing the ProcessorPart.

      It is expected to have the following keys:
        * 'part' (dict): A dictionary representing the underlying
          `google.genai.types.Part` object.
        * 'role' (str): The role of the part (e.g., 'user', 'model').
        * 'substream_name' (str): The substream name.
        * 'mimetype' (str): The MIME type of the part.
        * 'metadata' (dict[str, Any]): Auxiliary metadata.


    Example:

    ```py
    text_part = ProcessorPart("Hello", role="user")
    part_as_dict = text_part.to_dict()
    print(part_as_dict)
    ```
    """
    return {
        'part': self.part.model_dump(mode='json', exclude_none=True),
        'role': self.role,
        'substream_name': self.substream_name,
        'mimetype': self.mimetype,
        'metadata': self.metadata,
    }


class ProcessorContent:
  """A wrapper around `Content` with additional metadata.

  Serves as a convenience adaptor between various native representations and
  underlying data structures. ProcessorContent can be created from a string,
  image, ..., or a sequence of these.

  Users can narrow it down to a more convenient format using content_api.as_text
  or content_api.as_markdown. Or they can iterate over parts using .items()
  method.
  """

  _all_parts: list[ProcessorPart]

  def __init__(
      self,
      *parts: 'ProcessorContentTypes',
  ) -> None:
    """Constructs a new Content object from the given inputs."""

    self.replace_parts(*parts)

    self.as_text = functools.partial(as_text, self)
    self.as_text_with_reasoning = functools.partial(
        as_text_with_reasoning, self
    )
    self.as_images = functools.partial(as_images, self)

  def __iadd__(self, other: 'ProcessorContentTypes') -> 'ProcessorContent':
    """Appends other to the content."""
    if isinstance(other, ProcessorContent):
      self += other.all_parts
    elif isinstance(other, genai_types.Content):
      if other.parts:
        if other.role:
          parts = [ProcessorPart(part, role=other.role) for part in other.parts]
        else:
          parts = other.parts
        self += parts
    elif isinstance(other, ProcessorPartTypes):
      part = ProcessorPart(other)
      self._all_parts.append(part)
    elif isinstance(other, Iterable):
      for part in other:
        self += part
    else:
      raise ValueError(f"Can't append {type(other)} to ProcessorContent.")
    return self

  def __add__(self, other: 'ProcessorContentTypes') -> 'ProcessorContent':
    """Returns concatenation of two contents."""
    result = ProcessorContent()
    result += self
    result += other
    return result

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, ProcessorContent):
      return False
    for lhs, rhs in zip(self, other, strict=True):
      if lhs != rhs:
        return False
    return True

  def items(self) -> Iterator[tuple[str, ProcessorPart]]:
    """Yields tuples of mime_type and part.

    It is allowed to modify parts inplace except changing their IDs. Though
    bear in mind that like with most of Python containers that would change the
    part in all ProcessorContent containers which hold it.

    Yields:
      Tuples of mime_type, part.
    """
    for p in self.all_parts:
      yield p.mimetype, p

  def __iter__(self) -> Iterator[ProcessorPart]:
    """Yields each of the parts from this ProcessorContent.

    It is allowed to modify parts inplace.

    Bear in mind that like with most of Python containers that would change the
    part in all ProcessorContent containers which hold it.
    """
    for _, part in self.items():
      yield part

  @property
  def all_parts(self) -> Iterator[ProcessorPart]:
    """Yields all ProcessorParts from this ProcessorContent."""
    yield from self._all_parts

  def replace_parts(self, *parts: 'ProcessorContentTypes') -> None:
    """Replaces this ProcessorContent's parts."""
    self._all_parts: list[ProcessorPart] = []
    for part in parts:
      self += part

  def __repr__(self) -> str:
    parts = ', '.join(repr(part) for part in self.all_parts)
    return f'ProcessorContent({parts})'

  def __len__(self) -> int:
    """Returns the number of parts in this ProcessorContent."""
    return sum(1 for _ in self)


# Prefer using ProcessorPart.end_of_turn() instead: it is too easy to mutate
# this global object.
END_OF_TURN = ProcessorPart.end_of_turn()


def is_end_of_turn(part: ProcessorPart) -> bool:
  """Returns whether the part is an end of turn event."""
  if part.get_metadata('turn_complete'):
    return True
  return False


# Types that can be converted to a ProcessorPart.
ProcessorPartTypes = (
    genai_types.Part
    | ProcessorPart
    | str
    | bytes
    | PIL.Image.Image
)

# Types that can be appended to ProcessorContent.
ProcessorContentTypes = (
    ProcessorContent
    | ProcessorPartTypes
    | Iterable[ProcessorContent]
    | Iterable[ProcessorPartTypes]
    | genai_types.Content
    | Iterable[genai_types.Content]
)

# Helper functions for building content.

# Helper functions for mime type dispatching.
is_text = mime_types.is_text
is_json = mime_types.is_json
is_image = mime_types.is_image
is_video = mime_types.is_video
is_audio = mime_types.is_audio
is_streaming_audio = mime_types.is_streaming_audio
is_wav = mime_types.is_wav
is_source_code = mime_types.is_source_code
is_pdf = mime_types.is_pdf
is_csv = mime_types.is_csv
is_python = mime_types.is_python
is_dataclass = mime_types.is_dataclass


def mime_type(part: ProcessorPart) -> str:
  """Returns the mimetype of the part."""
  return part.mimetype


def get_substream_name(
    part: ProcessorPart,
) -> str:
  """Returns the substream name of the part."""
  return part.substream_name


def group_by_mimetype(content: ProcessorContent) -> dict[str, ProcessorContent]:
  """Groups content by mimetype.

  The order of parts within each mimetype grouping is preserved, maintaining the
  same order as they appeared in the original input `content`.

  Args:
    content: The content to group.

  Returns:
    A dictionary mapping mimetypes to ProcessorContent objects, with the same
    order as in the original input `content`.
  """
  grouped_content = {}
  for mimetype, part in content.items():
    if mimetype not in grouped_content:
      grouped_content[mimetype] = ProcessorContent()
    grouped_content[mimetype] += part
  return grouped_content


# Functions that reduce ProcessorContent to well known formats.
def as_text(
    content: ProcessorContentTypes,
    *,
    strict: bool = False,
    substream_name: str | None = None,
) -> str:
  """Returns a text representation of the content.

  The returned text is a concatenation of all text parts in the content.

  Args:
    content: The content to process. This can be of various types as defined by
      `ProcessorContentTypes`.
    strict: If True, unsupported content types will raise a ValueError.
      Otherwise, they will be ignored.
    substream_name: If set, only text parts with the given substream name will
      be returned.
  """
  text_parts = []
  for mime, part in ProcessorContent(content).items():
    if substream_name is not None and part.substream_name != substream_name:
      continue
    if is_text(mime):
      text_parts.append(part.text)
    elif strict:
      raise ValueError(f'Unsupported content type {mime}.')

  return ''.join(text_parts)


def as_text_with_reasoning(
    content: ProcessorContentTypes,
    *,
    strict: bool = False,
) -> tuple[str, str]:
  """Returns a tuple of the final and reasoning text representing content.

  The returned tuple contains two elements:
    - The first element (index 0) is a string representing the main text
    extracted
      from the input `content`.
    - The second element (index 1) is a string representing the reasoning or
      thoughts associated with the input `content`.

  Args:
    content: The content to process. This can be of various types as defined by
      `ProcessorContentTypes`.
    strict: If True, unsupported content types will raise a ValueError.
      Otherwise, they will be ignored.

  Returns:
    A tuple containing two strings: (text, reasoning).
  """
  text_parts = []
  thought_parts = []
  for mime, p in ProcessorContent(content).items():
    if is_text(mime):
      if p.part.thought:
        thought_parts.append(p.text)
      else:
        text_parts.append(p.text)
    elif strict:
      raise ValueError(f'Unsupported content type {mime}.')

  return ''.join(text_parts), ''.join(thought_parts)


def _as_format_helper(
    content: ProcessorContentTypes,
    mime_check: Callable[[str], bool],
    ignore_unsupported_types: bool,
) -> list[ProcessorPart]:
  """Helper function to extract parts from the content based on MIME type."""
  if isinstance(content, ProcessorPart):
    # Fast path for singular parts.
    content = [content]
  elif not isinstance(content, ProcessorContent):
    content = ProcessorContent(content)

  parts = []
  for p in content:
    if mime_check(p.mimetype):
      parts.append(p)
    elif not ignore_unsupported_types:
      raise ValueError(f'Unsupported MIME type: {p.mimetype}.')
  return parts


def as_images(
    content: ProcessorContentTypes, *, ignore_unsupported_types: bool = False
) -> list[ProcessorPart]:
  """Returns the image parts from the content.

  Args:
    content: Input content.
    ignore_unsupported_types: By default if content contains non-image parts a
      ValueError would be risen. This argument allows to ignore such parts.

  Returns:
    A list of image parts, with the same order as in the input content.
  """
  return _as_format_helper(
      content, mime_types.is_image, ignore_unsupported_types
  )


def as_videos(
    content: ProcessorContentTypes, *, ignore_unsupported_types: bool = False
) -> list[ProcessorPart]:
  """Returns the video parts from the content.

  Args:
    content: Input content.
    ignore_unsupported_types: By default if content contains non-video parts a
      ValueError would be raised. This argument allows ingoring such parts.

  Returns:
    A list of video parts.
  """
  return _as_format_helper(
      content, lambda mime: mime.startswith('video/'), ignore_unsupported_types
  )


def to_genai_part(
    part_content: ProcessorPartTypes,
    mimetype: str | None = None,
) -> genai_types.Part:
  """Converts object of type `ProcessorPartTypes` to a Genai Part.

  Args:
    part_content: The content to convert.
    mimetype: (Optional) The mimetype of the content. Must be specified if
      part_content is bytes.

  Returns:
    The Genai Part representation of the content.
  """
  if isinstance(part_content, str):
    return genai_types.Part(text=part_content)
  elif isinstance(part_content, bytes):
    if mimetype is None:
      raise ValueError(
          'Mimetype must be specified for bytes to_genai_part conversion.'
      )
    p = ProcessorPart(part_content, mimetype=mimetype)
    return p.part
  elif isinstance(part_content, PIL.Image.Image):
    p = ProcessorPart(part_content)
    return p.part
  elif isinstance(part_content, ProcessorPart):
    return part_content.part
  elif isinstance(part_content, genai_types.Part):
    return part_content
  else:
    raise ValueError(
        f'Unsupported type for to_genai_part: {type(part_content)}'
    )


def to_genai_contents(
    content: ProcessorContentTypes,
) -> list[genai_types.Content]:
  """Converts a list of ProcessorParts into a list of Genai Content objects.

  Consecutive parts with the same role are grouped together into a single
  `genai_types.Content` object.

  Args:
    content: Processor content, e.g. a list of `ProcessorPartTypes`.

  Returns:
    A list of `genai_types.Content` objects, where each object represents
    content from a single role.
  """
  processor_content = ProcessorContent(content)
  contents = []
  for role, content_part in itertools.groupby(
      processor_content, lambda p: p.role
  ):
    content_parts = [p.part for p in content_part]
    contents.append(genai_types.Content(parts=content_parts, role=role))

  return contents
