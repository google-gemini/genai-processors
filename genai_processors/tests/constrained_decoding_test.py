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
import dataclasses
import enum
import unittest

from absl.testing import absltest
from absl.testing import parameterized
import dataclasses_json
from genai_processors import content_api
from genai_processors import mime_types
from genai_processors.core import constrained_decoding


@dataclasses_json.dataclass_json
@dataclasses.dataclass(frozen=True)
class SimpleData:
  """A simple dataclass for testing."""

  name: str
  value: int


class Color(enum.StrEnum):
  """A simple enum for testing."""

  RED = 'RED'
  GREEN = 'GREEN'


class ConstrainedDecodingTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  @parameterized.named_parameters(
      (
          'single_dataclass',
          SimpleData,
          ['{"name": "test", "value": 123}'],
          [
              content_api.ProcessorPart.from_dataclass(
                  dataclass=SimpleData('test', 123)
              )
          ],
      ),
      (
          'list_of_dataclasses',
          list[SimpleData],
          ['[{"name": "a", "value": 1}, {"name": "b", "value": 2}]'],
          [
              content_api.ProcessorPart.from_dataclass(
                  dataclass=SimpleData('a', 1)
              ),
              content_api.ProcessorPart.from_dataclass(
                  dataclass=SimpleData('b', 2)
              ),
          ],
      ),
      (
          'single_enum_from_json_string',
          Color,
          ['"GREEN"'],
          [content_api.ProcessorPart('GREEN')],
      ),
      (
          'single_enum_from_raw_string',
          Color,
          ['GREEN'],
          [content_api.ProcessorPart('GREEN')],
      ),
      (
          'list_of_enums',
          list[Color],
          ['["RED", "GREEN"]'],
          [
              content_api.ProcessorPart('RED'),
              content_api.ProcessorPart('GREEN'),
          ],
      ),
      (
          'streaming_input_for_single_dataclass',
          SimpleData,
          ['{"name": "stream",', ' "value": 456}'],
          [
              content_api.ProcessorPart.from_dataclass(
                  dataclass=SimpleData('stream', 456)
              )
          ],
      ),
      ('empty_input_yields_nothing', SimpleData, ['   '], []),
      (
          'non_text_parts_are_passed_through',
          SimpleData,
          [
              content_api.ProcessorPart(b'image_bytes', mimetype='image/png'),
              '{"name": "test", "value": 123}',
          ],
          [
              content_api.ProcessorPart(b'image_bytes', mimetype='image/png'),
              content_api.ProcessorPart.from_dataclass(
                  dataclass=SimpleData('test', 123)
              ),
          ],
      ),
      (
          'instance_as_type',
          SimpleData('ignore', 0),
          '{"name": "test", "value": 123}',
          [
              content_api.ProcessorPart.from_dataclass(
                  dataclass=SimpleData('test', 123)
              )
          ],
      ),
  )
  async def test_json_to_dataclass_success(
      self, dc_type, input_parts, expected
  ):
    """Tests successful parsing for various types and inputs."""
    p = constrained_decoding.StructuredOutputParser(dc_type)
    self.assertEqual(await p(input_parts).gather(), expected)

  @parameterized.named_parameters(
      (
          'malformed_json',
          SimpleData,
          '{"name": "test", "value":}',
          'An unexpected error occurred',
      ),
      (
          'type_mismatch_not_a_list',
          list[SimpleData],
          '{"name": "a", "value": 1}',
          'Model output was not a list',
      ),
      (
          'type_mismatch_not_a_dict',
          SimpleData,
          '["a", 1]',
          "object has no attribute 'items'",
      ),
      (
          'invalid_enum_value',
          Color,
          '"BLUE"',
          "'BLUE' is not a valid Color",
      ),
  )
  async def test_json_to_dataclass_failure(
      self, dc_type, input_str, error_msg_substr
  ):
    p = constrained_decoding.StructuredOutputParser(dc_type)
    output = await p(input_str).gather()

    self.assertLen(output, 1)
    part = output[0]
    self.assertTrue(mime_types.is_exception(part.mimetype))
    self.assertIn(error_msg_substr, part.text)

  def test_unsupported_type_in_constructor(self):
    with self.assertRaisesRegex(
        TypeError,
        'must be a dataclass with the `dataclasses_json` mixin or an Enum',
    ):
      constrained_decoding.StructuredOutputParser(int)


if __name__ == '__main__':
  absltest.main()
