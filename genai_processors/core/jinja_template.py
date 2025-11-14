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
"""Processor for rendering Jinja templates with multimodal contents."""

import abc
from collections.abc import AsyncIterable
import json
from typing import Any
import uuid

from genai_processors import content_api
from genai_processors import processor
from genai_processors import streams
from google.protobuf import json_format
from google.protobuf import message as pb_message
import jinja2


class JinjaTemplate(processor.Processor):
  """Processor for rendering a Jinja template with multimodal contents.

  Example usage:

  ```python
  from genai_processors import content_api
  from genai_processors import processor
  from genai_processors.core import jinja_template

  p = jinja_template.JinjaTemplate(
      template_str='Hello {{ name }}, answer this question: {{ content }}',
      content_varname='content',
      role='user',
      name='World',
  )
  output = processor.apply_sync(
      p,
      [
          content_api.ProcessorPart(
              'What is this landmark?',
              mimetype='text/plain',
          ),
          content_api.ProcessorPart(
              <image_bytes>,
              mimetype='image/png',
          ),
      ],
  )
  print(content_api.as_text(output))
  ```
  """

  def __init__(
      self,
      template_str: str,
      content_varname: str = 'content',
      role: str = 'user',
      *args,
      **kwargs,
  ) -> None:
    """Initializes the processor.

    Accepts the same args and kwargs as Jinja's `render()` method
    https://jinja.palletsprojects.com/en/stable/api/#jinja2.Template.render.

    Args:
      template_str: The Jinja template string.
      content_varname: The name of the Jinja variable to render the content.
      role: The role to use when outputting the rendered template.
      *args: Positional arguments to pass to Jinja's `render()` method.
      **kwargs: Keyword arguments to pass to Jinja's `render()` method.

    Raises:
      ValueError: If `content_varname` is passed in **kwargs.
    """
    if content_varname in kwargs:
      raise ValueError(
          f"{content_varname!r} is set to render the processor's content and"
          ' must not be passed as a variable to the Jinja template.'
      )

    # Render the template using a placeholder value for the processor's content
    # variable so the processor's content location can be found in the next
    # step. We use a UUID to ensure the placeholder value is not already present
    # in the template.
    content_placeholder = str(uuid.uuid4())
    kwargs.update({content_varname: content_placeholder})
    rendered_template = jinja2.Template(template_str).render(*args, **kwargs)

    # Split the template using the placeholder value as a delimiter, meaning
    # that the processor's content needs to be inserted between each element.
    # Splitting the template allows us to inject not only text but also
    # multi-part and multimodal content.
    self._template_split = rendered_template.split(content_placeholder)

    self._role = role

  async def call(
      self,
      content: AsyncIterable[content_api.ProcessorPart],
  ) -> AsyncIterable[content_api.ProcessorPartTypes]:
    # If the template was split into a single part, then the template did not
    # contain a variable to render the processor's content and should be
    # returned as is.
    if len(self._template_split) == 1:
      yield content_api.ProcessorPart(
          self._template_split[0],
          role=self._role,
      )
      return

    # `content` is a stream that can only be iterated once, so we duplicate it
    # into identical streams to insert `content_streams[i]` between
    # `self._template_split[i]` and `self._template_split[i+1]`.
    content_streams = streams.split(
        content,
        n=len(self._template_split) - 1,
        with_copy=False,
    )

    for i, template_part in enumerate(self._template_split):
      # Yield the template part. Empty parts are skipped as they correspond to
      # where the content variable was located.
      if template_part:
        yield content_api.ProcessorPart(
            template_part,
            role=self._role,
        )

      # Yield the processor's content between two consecutive elements of the
      # template split.
      if i < len(content_streams):
        async for part in content_streams[i]:
          yield part


class _Render(processor.PartProcessor):
  r"""Abstract class for rendering a Jinja template from different data objects.

  The jinja template must reference the data object containing the data to
  render by the name `data`, i.e. `{{ data.first_name }}`.

  ```python
  template_str=(
          'Hello {{ data.first_name }},\n'
          'This is your shopping list:\n'
          '{% for item in data.your_list %}This is item: {{ item }}\n'
          '{% endfor %}'
      ),
  ```

  With a data object containing:
  {'first_name': 'John', your_list: ['A', 'B', 'C']}

  The expected output is:
  ```
  Hello John,
  This is your shopping list:
  This is item: A
  This is item: B
  This is item: C
  ```
  """

  def __init__(
      self,
      template_str: str,
      **kwargs,
  ):
    """Initializes the processor.

    Args:
      template_str: The Jinja template string.
      **kwargs: Keyword arguments to pass to the Jinja template.
    """
    self._environment = jinja2.Environment()
    self._environment.globals.update(**kwargs)
    self._template = self._environment.from_string(template_str)

  @abc.abstractmethod
  def get_data(self, part: content_api.ProcessorPart) -> Any:
    """Returns the data to render in the Jinja template.

    Args:
      part: The part containing the data to render.

    Returns:
      The data to render in the Jinja template.
    """
    raise NotImplementedError()

  async def call(
      self, part: content_api.ProcessorPart
  ) -> AsyncIterable[content_api.ProcessorPart]:
    """Renders a dataclass part in a Jinja template."""
    yield content_api.ProcessorPart(
        self._template.render(data=self.get_data(part)),
        role=part.role,
        metadata=part.metadata,
        substream_name=part.substream_name,
    )


class RenderDataClass(_Render):
  r"""PartProcessor for rendering a dataclass part using a Jinja template."""

  def __init__(
      self,
      template_str: str,
      data_class: type[Any],
      **kwargs,
  ):
    """Initializes the processor.

    Args:
      template_str: The Jinja template string.
      data_class: The type of the dataclass to render.
      **kwargs: Keyword arguments to pass to the Jinja template.
    """
    super().__init__(template_str, **kwargs)
    self._data_class = data_class

  def match(self, part: content_api.ProcessorPart) -> bool:
    return content_api.is_dataclass(part.mimetype, self._data_class)

  def get_data(self, part: content_api.ProcessorPart) -> Any:
    """Returns the data to render in the Jinja template."""
    return part.get_dataclass(self._data_class)


class RenderJson(_Render):
  r"""PartProcessor for rendering a JSON dictionary using a Jinja template.

  The JSON dictionary must be referenced by the name `data` in the jinja
  template, i.e. `{{ data.key_name }}`.
  """

  def match(self, part: content_api.ProcessorPart) -> bool:
    return content_api.is_json(part.mimetype)

  def get_data(self, part: content_api.ProcessorPart) -> Any:
    """Returns the data to render in the Jinja template."""
    return json.loads(part.text)


class RenderProtoMessage(_Render):
  r"""PartProcessor for rendering a Proto message a Jinja template.

  The Proto message must be referenced by the name `data` in the jinja
  template, i.e. `{{ data.key_name }}`.

  WARNING: well known message types like Timestamp and Duration will be rendered
  as plain string directly, i.e. if one proto field is a duration, it will be
  rendered as a string like '10.000000500s' instead of the dictionary:
  ```python
  {
      'seconds': '10',
      'nanos': '500',
  }
  ```
  Make sure you use msg.duration directly instead of msg.duration.seconds. This
  is due to the peculiarity of the MessageToDict() for std message types.

  ANOTHER NOTE:
  `Struct` message will be rendered as a dictionary with the proto field names
  as the dictionary key, i.e.
  ```
  Struct(
    fields={
      'name': struct_pb2.Value(string_value='John'),
      'age': struct_pb2.Value(number_value=25),
    }
  )
  ```

  will be rendered assuming data is the dictionary:

  ```python
  {
      'name': 'John',
      'age': 25,
  }
  ```
  """

  def __init__(
      self,
      proto_message: type[pb_message.Message],
      template_str: str,
      **kwargs,
  ):
    super().__init__(template_str, **kwargs)
    self._proto_message = proto_message

  def match(self, part: content_api.ProcessorPart) -> bool:
    return content_api.is_proto_message(part.mimetype, self._proto_message)

  def get_data(self, part: content_api.ProcessorPart) -> Any:
    """Returns the data to render in the Jinja template."""
    return json_format.MessageToDict(
        self._proto_message.FromString(part.bytes),
        preserving_proto_field_name=True,
    )
