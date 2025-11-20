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

from absl.testing import absltest
from absl.testing import parameterized
from genai_processors import tool_utils
from google.genai import types as genai_types


def example_func_1(param1: str, param2: int, param3: bool) -> str:
  # pylint: disable=g-doc-return-or-yield
  """This is func 1.

  Args:
    param1: The first parameter.
    param2: The second parameter. With a long description that spans multiple
      lines.
    param3: The third parameter.
  """
  # pylint: enable=g-doc-return-or-yield
  return f'{param1} {param2} {param3}'


def example_func_2(param_a: bool, param_b: str) -> bool:
  # pylint: disable=g-doc-return-or-yield
  # pylint: disable=g-doc-args
  """This is func 2.

  With no Args section. It contains an explanation: like this one.

  :param param_a: A boolean parameter.
  """
  del param_b
  return not param_a
  # pylint: enable=g-doc-args
  # pylint: enable=g-doc-return-or-yield


def func_no_docstring(p1: int):
  del p1
  pass


class ToolUtilsTest(parameterized.TestCase):

  def test_to_function_declaration_no_docstring(self):
    declarations = tool_utils.to_function_declarations([func_no_docstring])
    expected_declarations = [
        genai_types.FunctionDeclaration(
            name='func_no_docstring',
            description='',
            parameters=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    'p1': genai_types.Schema(
                        type=genai_types.Type.INTEGER,
                    ),
                },
                required=['p1'],
            ),
        )
    ]
    self.assertEqual(declarations, expected_declarations)

  def test_to_function_declaration_no_arg_section(self):
    declarations = tool_utils.to_function_declarations([example_func_2])
    expected_declarations = [
        genai_types.FunctionDeclaration(
            name='example_func_2',
            description=(
                'This is func 2.\n\nWith no Args section. It contains an'
                ' explanation: like this one.'
            ),
            parameters=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    'param_a': genai_types.Schema(
                        type=genai_types.Type.BOOLEAN,
                        description='A boolean parameter.',
                    ),
                    'param_b': genai_types.Schema(
                        type=genai_types.Type.STRING,
                    ),
                },
                required=['param_a', 'param_b'],
            ),
        )
    ]
    self.assertEqual(declarations, expected_declarations)

  def test_to_function_declarations_with_callable(self):
    declarations = tool_utils.to_function_declarations([example_func_1])
    expected_declarations = [
        genai_types.FunctionDeclaration(
            name='example_func_1',
            description='This is func 1.',
            parameters=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    'param1': genai_types.Schema(
                        type=genai_types.Type.STRING,
                        description='The first parameter.',
                    ),
                    'param2': genai_types.Schema(
                        type=genai_types.Type.INTEGER,
                        description=(
                            'The second parameter. With a long description that'
                            ' spans multiple\nlines.'
                        ),
                    ),
                    'param3': genai_types.Schema(
                        type=genai_types.Type.BOOLEAN,
                        description='The third parameter.',
                    ),
                },
                required=['param1', 'param2', 'param3'],
            ),
        )
    ]
    self.assertEqual(declarations, expected_declarations)


if __name__ == '__main__':
  absltest.main()
