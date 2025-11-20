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
"""Utilities for implementing tools."""

from typing import Any, Callable, Collection, Union

import docstring_parser
from google.genai import _transformers
from google.genai import types as genai_types


def raise_for_gemini_server_side_tools(
    tools: list[genai_types.Tool], *, allow_list: Collection[str] = ()
) -> None:
  """Raises ValueError if the tool list contains a server-side tool.

  Gemini API provides a set of server-side tools. In general, they are not
  available on other LLM implementations. This helper method allows to warn the
  developer if they try use this unimplemented functionality.

  Args:
    tools: List of tools.
    allow_list: List of server-side tools to allow.
  """
  for tool in tools:
    for tool_name in (
        'retrieval',
        'google_search',
        'google_search_retrieval',
        'enterprise_web_search',
        'google_maps',
        'url_context',
        'code_execution',
        'computer_use',
    ):
      if getattr(tool, tool_name) is not None and tool_name not in allow_list:
        raise ValueError(f'Tool {tool_name} is not supported.')


def to_schema(
    schema: Union[genai_types.SchemaUnion, genai_types.SchemaUnionDict],
) -> genai_types.Schema:
  """Returns a JSON schema given a Python object representing the schema.

  GenAI SDK accepts a variety of Python objects, such as Enum, dataclass or a
  dictionary of Schema objects as schemas. This provides an easy way to
  constrain the LLM to generate JSON compatible with the needed type. This
  utility is useful for defining the expected structure of data for tool calling
  or constrained decoding with a model.

  Usage:
    json_schema = to_schema(...).json_schema.model_dump(
        mode='json', exclude_unset=True)

  Args:
    schema: The Python object representing the schema to convert. See
      https://ai.google.dev/gemini-api/docs/structured-output for the supported
        types.

  Returns:
    A `genai_types.Schema` object representing the schema.
  """
  return _transformers.t_schema(  # pytype: disable=wrong-arg-types
      _FakeClient(), schema
  )


def to_json_schema(
    schema: Union[genai_types.SchemaUnion, genai_types.SchemaUnionDict, None],
) -> str | None:
  """Returns a JSON schema given a Python object representing the schema.

  GenAI SDK accepts a variety of Python objects, such as Enum, dataclass or a
  dictionary of Schema objects as schemas. This provides an easy way to
  constrain the LLM to generate JSON compatible with the needed type. This
  utility is useful for defining the expected structure of data for tool calling
  or constrained decoding with a model.

  Args:
    schema: The Python object representing the schema to convert. See
      https://ai.google.dev/gemini-api/docs/structured-output for the supported
        types.

  Returns:
    A `genai_types.Schema` object representing the schema or None if the given
    `schema` was None.
  """

  if schema:
    return to_schema(schema).json_schema.model_dump(
        mode='json', exclude_unset=True
    )
  else:
    return None


def to_function_declarations(
    tool_list: list[genai_types.Tool | Callable[..., Any]],
) -> list[genai_types.FunctionDeclaration]:
  """Converts a list of tools to a list of function declarations."""
  tools: list[genai_types.Tool] = []
  for t in tool_list:
    if callable(t):
      fdecl = genai_types.FunctionDeclaration.from_callable_with_api_option(
          callable=t, api_option='GEMINI_API'
      )
      parsed_docstring = docstring_parser.parse(t.__doc__)
      fdecl.description = ''
      if parsed_docstring.short_description:
        fdecl.description += f'{parsed_docstring.short_description}'
      if parsed_docstring.long_description:
        fdecl.description += f'\n\n{parsed_docstring.long_description}'
      if fdecl.parameters:
        for param in parsed_docstring.params:
          if param.arg_name in fdecl.parameters.properties:
            fdecl.parameters.properties[param.arg_name].description = (
                param.description
            )
        tools.append(genai_types.Tool(function_declarations=[fdecl]))
    else:
      tools.append(t)
  raise_for_gemini_server_side_tools(tools)

  function_declarations: list[genai_types.FunctionDeclaration] = []
  for tool in tools:
    function_declarations.extend(tool.function_declarations or ())

  return function_declarations


def function_declaration_to_json(
    fdecl: genai_types.FunctionDeclaration,
) -> dict[str, Any]:
  """Returns a JSON representation of a FunctionDeclaration."""
  return {
      'type': 'function',
      'function': {
          'name': fdecl.name,
          'description': fdecl.description,
          'parameters': to_json_schema(fdecl.parameters),
      },
  }


# TODO(kibergus): Remove this hack once Genai SDK allows None as the client.
class _FakeClient:
  """A fake genai client to invoke t_schema."""

  def __init__(self):
    self.vertexai = False
