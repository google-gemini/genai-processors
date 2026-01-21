import dataclasses
import io
import json
import textwrap

from absl.testing import absltest
from absl.testing import parameterized
import dataclasses_json
from genai_processors import content_api
from genai_processors import mime_types
from google.genai import types as genai_types
import PIL.Image

from google.protobuf import struct_pb2


def _png_image_bytes() -> bytes:
  as_bytes = io.BytesIO()
  as_image = PIL.Image.new('RGB', (100, 100), color='black')
  as_image.save(as_bytes, format='PNG')
  return as_bytes.getvalue()


def _png_image_pil() -> PIL.Image.Image:
  as_bytes = _png_image_bytes()
  return PIL.Image.open(io.BytesIO(as_bytes))


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class Dataclass:
  foo: str
  bar: int


class ProcessorPartTest(parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name='image_bytes',
          bytes_data=b'foo',
          mimetype='image/png',
      ),
      dict(
          testcase_name='audio_bytes',
          bytes_data=b'bar',
          mimetype='audio/wav',
      ),
      dict(
          testcase_name='text_bytes',
          bytes_data='baz',
          mimetype='text/plain',
      ),
  ])
  def test_from_and_to_bytes(self, bytes_data, mimetype):
    part = content_api.ProcessorPart(bytes_data, mimetype=mimetype)
    if isinstance(bytes_data, str):
      bytes_data = bytes_data.encode()
    self.assertEqual(part.bytes, bytes_data)

  def test_from_tool_cancellation(self):
    part = content_api.ProcessorPart.from_tool_cancellation(
        function_call_id='foo',
        role='user',
    )
    # Enforce that the role is always MODEL for tool cancellation.
    self.assertEqual(part.role, 'model')
    self.assertEqual(part.tool_cancellation, 'foo')

    part = content_api.ProcessorPart('bar')
    self.assertIsNone(part.tool_cancellation)

  def test_from_bytes_with_text_mimetype(self):
    bytes_data = b'hello'
    mimetype = 'text/plain'
    part = content_api.ProcessorPart(bytes_data, mimetype=mimetype)
    self.assertEqual(part.text, 'hello')
    self.assertEqual(part.part.text, 'hello')

  def test_from_bytes_with_non_text_mimetype(self):
    bytes_data = b'hello'
    mimetype = 'application/octet-stream'
    part = content_api.ProcessorPart(bytes_data, mimetype=mimetype)
    self.assertEqual(part.bytes, bytes_data)
    self.assertEqual(part.part.inline_data.data, bytes_data)
    self.assertEqual(part.part.inline_data.mime_type, mimetype)

  def test_from_function_response_with_text_content(self):
    part = content_api.ProcessorPart.from_function_response(
        name='foo',
        response=content_api.ProcessorContent('bar'),
    )
    self.assertEqual(part.function_response.name, 'foo')
    self.assertEqual(part.function_response.response, {'result': 'bar'})
    self.assertIsNone(part.function_response.parts)

  def test_from_function_response_with_dataclass(self):
    part = content_api.ProcessorPart.from_function_response(
        name='foo',
        response=Dataclass(foo='foo', bar=1),
    )
    self.assertEqual(part.function_response.name, 'foo')
    self.assertEqual(
        part.function_response.response, {'result': Dataclass(foo='foo', bar=1)}
    )
    self.assertIsNone(part.function_response.parts)

  def test_from_function_response_with_image_content(self):
    image_bytes = _png_image_bytes()
    part = content_api.ProcessorPart.from_function_response(
        name='foo',
        response=content_api.ProcessorPart(image_bytes, mimetype='image/png'),
    )
    self.assertEqual(part.function_response.name, 'foo')
    self.assertIsNone(part.function_response.response)
    self.assertLen(part.function_response.parts, 1)
    self.assertEqual(
        part.function_response.parts[0].inline_data.mime_type, 'image/png'
    )
    self.assertEqual(
        part.function_response.parts[0].inline_data.data, image_bytes
    )

  def test_from_function_response_with_mixed_content(self):
    image_bytes = _png_image_bytes()
    part = content_api.ProcessorPart.from_function_response(
        name='foo',
        response=[
            'Here is a black cat in a black room: ',
            content_api.ProcessorPart(image_bytes, mimetype='image/png'),
        ],
    )
    self.assertEqual(
        part.function_response,
        genai_types.FunctionResponse(
            name='foo',
            parts=[
                genai_types.FunctionResponsePart(
                    inline_data=genai_types.FunctionResponseBlob(
                        data=b'Here is a black cat in a black room: ',
                        mime_type='text/plain',
                    ),
                ),
                genai_types.FunctionResponsePart(
                    inline_data=genai_types.FunctionResponseBlob(
                        data=image_bytes,
                        mime_type='image/png',
                    ),
                ),
            ],
        ),
    )

  def test_bytes_from_text(self):
    part = content_api.ProcessorPart('foo', mimetype='text/plain')
    self.assertEqual(part.bytes, b'foo')
    self.assertEqual(part.text, 'foo')
    part = content_api.ProcessorPart('foo', mimetype='application/x-command')
    self.assertEqual(part.bytes, b'foo')
    self.assertRaises(ValueError, lambda: part.text)

  def test_eq_part_and_non_part(self):
    part = content_api.ProcessorPart('foo')
    self.assertNotEqual(part, object())


class ProcessorContentTest(parameterized.TestCase):

  def test_construct_content_from_genai_parts(self):
    content = content_api.ProcessorContent(
        genai_types.Part(text='foo: '),
        genai_types.Part(text='bar'),
    )
    self.assertEqual('foo: bar', content_api.as_text(content))

  def test_construct_content_from_genai_content(self):
    content = content_api.ProcessorContent(
        genai_types.Content(
            parts=[
                genai_types.Part(text='foo: '),
                genai_types.Part(text='bar'),
            ],
            role='model',
        )
    )
    self.assertEqual('foo: bar', content_api.as_text(content))
    for part in content.all_parts:
      self.assertEqual(part.role, 'model')

  def test_construct_content_from_processor_parts(self):
    content = content_api.ProcessorContent(
        content_api.ProcessorPart('foo: '),
        content_api.ProcessorPart('bar'),
    )
    self.assertEqual('foo: bar', content_api.as_text(content))

  def test_text_to_text_with_reasoning(self):
    c = content_api.ProcessorContent(
        genai_types.Content(
            parts=[
                genai_types.Part(text='I am a thought', thought=True),
                genai_types.Part(text='Hi there!'),
            ],
            role='model',
        )
    )
    response, reasoning = content_api.as_text_with_reasoning(c)
    self.assertEqual(response, 'Hi there!')
    self.assertEqual(reasoning, 'I am a thought')
    for content_part in c.all_parts:
      self.assertEqual(content_part.role, 'model')

  def test_as_text(self):
    content = content_api.ProcessorContent(
        content_api.ProcessorPart('foo:', substream_name='foo'),
        content_api.ProcessorPart('bar:', substream_name='bar'),
        content_api.ProcessorPart('baz', substream_name='bar'),
        content_api.ProcessorPart('bar', substream_name='foo'),
    )
    self.assertEqual(
        content_api.as_text(content, substream_name='foo'), 'foo:bar'
    )
    self.assertEqual(
        content_api.as_text(content, substream_name='bar'), 'bar:baz'
    )
    self.assertEqual(content_api.as_text(content), 'foo:bar:bazbar')

  def test_metadata(self):
    content = content_api.ProcessorContent([
        content_api.ProcessorPart(
            'You must obey three Laws of Robotics.',
            metadata={'source': 'system'},
        ),
        content_api.ProcessorPart(
            'There is a fire in a room, break through a wall and make passage.',
            metadata={'source': 'user'},
        ),
        content_api.ProcessorPart(
            'Heat sensors: no fire detected.',
            metadata={'source': 'tool'},
        ),
        content_api.ProcessorPart(
            'It is safe to enter the room.',
            metadata={'source': 'agent'},
        ),
    ])

    prompt_parts = []
    for _, part in content.items():
      source = part.metadata['source']
      prefix = f'{source}: ' if source else ''
      prompt_parts.append(prefix + content_api.as_text(part))

    self.assertEqual(
        '\n\n'.join(prompt_parts),
        textwrap.dedent("""
            system: You must obey three Laws of Robotics.

            user: There is a fire in a room, break through a wall and make passage.

            tool: Heat sensors: no fire detected.

            agent: It is safe to enter the room.""".strip('\n')),
    )

  @parameterized.named_parameters([
      dict(
          testcase_name='single_text_part',
          parts=[genai_types.Part(text='ab')],
          expected_summary=(
              "ProcessorContent(ProcessorPart({'text': 'ab'},"
              " mimetype='text/plain'))"
          ),
      ),
      dict(
          testcase_name='single_non_text_part',
          parts=[
              genai_types.Part(
                  inline_data=genai_types.Blob(
                      data=b'ab', mime_type='image/png'
                  )
              )
          ],
          expected_summary=(
              "ProcessorContent(ProcessorPart({'inline_data': {'data': 'YWI=',"
              " 'mime_type': 'image/png'}}, mimetype='image/png'))"
          ),
      ),
      dict(
          testcase_name='multiple_parts',
          parts=[
              genai_types.Part(
                  inline_data=genai_types.Blob(
                      data=b'ab', mime_type='image/png'
                  )
              ),
              genai_types.Part(text='ab'),
          ],
          expected_summary=(
              "ProcessorContent(ProcessorPart({'inline_data': {'data': 'YWI=',"
              " 'mime_type': 'image/png'}}, mimetype='image/png'),"
              " ProcessorPart({'text': 'ab'}, mimetype='text/plain'))"
          ),
      ),
      dict(
          testcase_name='single_text_part_with_substream',
          parts=[
              content_api.ProcessorPart(
                  genai_types.Part(text='ab'), substream_name='a'
              ),
          ],
          expected_summary=(
              "ProcessorContent(ProcessorPart({'text': 'ab'},"
              " mimetype='text/plain', substream_name='a'))"
          ),
      ),
  ])
  def test_repr(self, parts, expected_summary):
    content = content_api.ProcessorContent(parts)
    self.assertEqual(repr(content), expected_summary)

  def test_concatenating_content_with_self(self):
    content = content_api.ProcessorContent('A')
    self.assertEqual(content_api.as_text([content] * 10), 'A' * 10)

  # TODO(b/403621093): Add tests for multimodal content including images.
  def test_custom_mimetype(self):
    content = content_api.ProcessorContent(
        content_api.ProcessorPart('foo', mimetype='application/json')
    )
    self.assertEqual(list(content.all_parts)[0].mimetype, 'application/json')

  def test_custom_mimetype_with_genai_part(self):
    content = content_api.ProcessorContent(
        genai_types.Part(
            text='foo',
            inline_data=genai_types.Blob(
                data=b'foo', mime_type='application/json'
            ),
        )
    )
    self.assertEqual(list(content.all_parts)[0].mimetype, 'application/json')

  def test_infer_text_mimetype_from_text(self):
    content = content_api.ProcessorContent('hello')
    self.assertEqual(list(content.all_parts)[0].mimetype, 'text/plain')

  def test_infer_text_mimetype_from_text_with_genai_part(self):
    content = content_api.ProcessorContent(genai_types.Part(text='hello'))
    self.assertEqual(list(content.all_parts)[0].mimetype, 'text/plain')

  @parameterized.named_parameters([
      dict(
          testcase_name='default_pil_image_is_webp',
          image_part=content_api.ProcessorPart(
              PIL.Image.new('RGB', (100, 100), color='black')
          ),
          mimetype='image/webp',
          image_format='WEBP',
      ),
      dict(
          testcase_name='specify_custom_mime_pil_image',
          image_part=content_api.ProcessorPart(
              PIL.Image.new('RGB', (100, 100), color='black'),
              mimetype='image/jpeg',
          ),
          mimetype='image/jpeg',
          image_format='JPEG',
      ),
      dict(
          testcase_name='specify_mime_for_images_as_bytes',
          image_part=content_api.ProcessorPart(
              _png_image_bytes(), mimetype='image/png'
          ),
          mimetype='image/png',
          image_format='PNG',
      ),
      dict(
          testcase_name='infer_pil_format',
          image_part=content_api.ProcessorPart(_png_image_pil()),
          mimetype='image/png',
          image_format='PNG',
      ),
  ])
  def test_custom_image_content(
      self,
      image_part: content_api.ProcessorPart,
      mimetype: str,
      image_format: str,
  ):
    self.assertEqual(image_part.mimetype, mimetype)
    self.assertEqual(image_part.pil_image.format, image_format)

  def test_to_and_from_dataclass(self):
    test_dataclass = Dataclass(foo='foo', bar=1)
    part = content_api.ProcessorPart.from_dataclass(
        dataclass=test_dataclass, substream_name='foo'
    )
    self.assertEqual(part.substream_name, 'foo')
    self.assertEqual(part.get_dataclass(Dataclass), test_dataclass)
    self.assertTrue(mime_types.is_json(part.mimetype))
    self.assertTrue(mime_types.is_dataclass(part.mimetype))

  def test_get_dataclass_raises_error_with_incorrect_mimetype(self):
    test_part = content_api.ProcessorPart(
        '{"foo": "hello", "bar": 1}', mimetype='application/json'
    )
    with self.assertRaises(ValueError):
      test_part.get_dataclass(Dataclass)

  def test_to_and_from_proto_message(self):
    test_proto = struct_pb2.Struct(
        fields={'foo': struct_pb2.Value(string_value='bar')}
    )
    part = content_api.ProcessorPart.from_proto_message(
        proto_message=test_proto
    )
    self.assertEqual(part.get_proto_message(struct_pb2.Struct), test_proto)
    self.assertTrue(
        mime_types.is_proto_message(part.mimetype, struct_pb2.Struct)
    )

  def test_get_proto_message_raises_error_with_incorrect_mimetype(self):
    test_part = content_api.ProcessorPart(
        '{"foo": "hello", "bar": 1}', mimetype='application/json'
    )
    with self.assertRaises(ValueError):
      test_part.get_proto_message(struct_pb2.Struct)

  def test_get_proto_message_from_bytes(self):
    test_proto = struct_pb2.Struct(
        fields={'foo': struct_pb2.Value(string_value='bar')}
    )
    part = content_api.ProcessorPart(
        test_proto.SerializeToString(),
        mimetype=mime_types.proto_message_mime_type(struct_pb2.Struct),
    )
    self.assertEqual(part.get_proto_message(struct_pb2.Struct), test_proto)

  def test_get_proto_message_raises_error_with_incorrect_proto_type(self):
    test_proto = struct_pb2.Struct(
        fields={'foo': struct_pb2.Value(string_value='bar')}
    )
    part = content_api.ProcessorPart.from_proto_message(
        proto_message=test_proto
    )
    with self.assertRaisesRegex(
        ValueError, 'Part is not a Duration proto message.'
    ):
      part.get_proto_message(duration_pb2.Duration)

  def test_is_text(self):
    text_part = content_api.ProcessorPart('hello')
    self.assertTrue(mime_types.is_text(text_part.mimetype))
    self.assertFalse(mime_types.is_json(text_part.mimetype))

  @parameterized.named_parameters([
      dict(
          testcase_name='simple_text',
          original_part=content_api.ProcessorPart(
              'hello world',
              role='user',
              substream_name='chat',
              mimetype='text/plain',
              metadata={'id': 123, 'source': 'test'},
          ),
      ),
      dict(
          testcase_name='function_call',
          original_part=content_api.ProcessorPart.from_function_call(
              name='get_weather',
              args={'location': 'London', 'unit': 'celsius'},
              role='model',
          ),
      ),
      dict(
          testcase_name='function_response',
          original_part=content_api.ProcessorPart.from_function_response(
              name='get_weather',
              response={'temp': '10'},
              function_call_id='fc001',
              role='model',
          ),
      ),
      dict(
          testcase_name='inline_data',
          original_part=content_api.ProcessorPart(
              b'\x08\x01\x02\x03\x0a\xbc',
              role='user',
              mimetype='application/octet-stream',
              metadata={'source_file': 'file.bin'},
          ),
      ),
      dict(
          testcase_name='dataclass_part',
          original_part=content_api.ProcessorPart.from_dataclass(
              dataclass=Dataclass(foo='foo', bar=1),
              substream_name='foo',
          ),
      ),
      dict(
          testcase_name='image_part',
          original_part=content_api.ProcessorPart(
              _png_image_bytes(), mimetype='image/png'
          ),
      ),
  ])
  def test_to_dict_from_dict_symmetry(
      self, original_part: content_api.ProcessorPart
  ):
    reconstructed_part = content_api.ProcessorPart.from_dict(
        data=original_part.to_dict(mode='json')
    )
    self.assertEqual(original_part, reconstructed_part)
    reconstructed_part = content_api.ProcessorPart.from_dict(
        data=original_part.to_dict(mode='python')
    )
    self.assertEqual(original_part, reconstructed_part)

  def test_from_dict_invalid_part_dict_structure(self):
    invalid_data = {
        'part': {'this_is_not_a_part_field': 'some_value'},
        'role': 'user',
    }
    with self.assertRaises(Exception):
      content_api.ProcessorPart.from_dict(data=invalid_data)

  @parameterized.named_parameters([
      dict(
          testcase_name='json_mode',
          mode='json',
          expected_data='YWI=',  # Base64 encoding of 'ab' is 'YWI='.
      ),
      dict(
          testcase_name='python_mode',
          mode='python',
          expected_data=b'ab',
      ),
  ])
  def test_to_dict_base64_encoding_of_image_bytes(self, mode, expected_data):
    part = content_api.ProcessorPart(
        b'ab',
        role='user',
        substream_name='chat',
        mimetype='image/png',
        metadata={'id': 123, 'source': 'test'},
    )
    part_as_dict = part.to_dict(mode=mode)
    self.assertEqual(part_as_dict['part']['inline_data']['data'], expected_data)

  def test_eq_content_and_non_content(self):
    content = content_api.ProcessorContent('foo')
    self.assertNotEqual(content, object())

  @parameterized.named_parameters([
      dict(
          testcase_name='simple_text',
          parts=[
              content_api.ProcessorPart('part1', role='user'),
              content_api.ProcessorPart('part2', role='user'),
              content_api.ProcessorPart('part3', role='model'),
              content_api.ProcessorPart('part4', role='user'),
          ],
          expected_genai_contents=[
              genai_types.Content(
                  parts=[
                      genai_types.Part(text='part1'),
                      genai_types.Part(text='part2'),
                  ],
                  role='user',
              ),
              genai_types.Content(
                  parts=[genai_types.Part(text='part3')],
                  role='model',
              ),
              genai_types.Content(
                  parts=[genai_types.Part(text='part4')],
                  role='user',
              ),
          ],
      ),
      dict(
          testcase_name='empty_parts',
          parts=[],
          expected_genai_contents=[],
      ),
      dict(
          testcase_name='single_part',
          parts=[content_api.ProcessorPart('part1', role='user')],
          expected_genai_contents=[
              genai_types.Content(
                  parts=[genai_types.Part(text='part1')],
                  role='user',
              ),
          ],
      ),
      dict(
          testcase_name='with_file',
          parts=[
              content_api.ProcessorPart('part1', role='user'),
              content_api.ProcessorPart('part2', role='user'),
              content_api.ProcessorPart(
                  genai_types.File(
                      name='file1', mime_type='image/png', uri='uri1'
                  )
              ),
              content_api.ProcessorPart('part4', role='user'),
          ],
          expected_genai_contents=[
              genai_types.Content(
                  parts=[
                      genai_types.Part(text='part1'),
                      genai_types.Part(text='part2'),
                  ],
                  role='user',
              ),
              genai_types.File(name='file1', mime_type='image/png', uri='uri1'),
              genai_types.Content(
                  parts=[genai_types.Part(text='part4')],
                  role='user',
              ),
          ],
      ),
  ])
  def test_to_genai_contents(self, parts, expected_genai_contents):
    genai_contents = content_api.to_genai_contents(parts)
    self.assertEqual(genai_contents, expected_genai_contents)


if __name__ == '__main__':
  absltest.main()
