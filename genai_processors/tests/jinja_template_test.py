import dataclasses
import unittest

from absl.testing import absltest
import dataclasses_json
from genai_processors import content_api
from genai_processors.core import jinja_template

from google.protobuf import struct_pb2


class JinjaTemplateTest(unittest.IsolatedAsyncioTestCase):

  async def test_empty_template(self):
    p = jinja_template.JinjaTemplate('')
    self.assertEqual(await p([]).text(), '')

  async def test_empty_template_with_processor_content(self):
    p = jinja_template.JinjaTemplate('')
    self.assertEqual(await p('Hello World').text(), '')

  async def test_template_without_content_variable(self):
    p = jinja_template.JinjaTemplate(
        'Hello {{ name }}',
        content_varname='content',
        name='World',
    )
    self.assertEqual(await p([]).text(), 'Hello World')

  async def test_template_with_content_variable_only(self):
    p = jinja_template.JinjaTemplate(
        '{{ content }}',
        content_varname='content',
    )
    self.assertEqual(await p(['Hello ', 'World']).text(), 'Hello World')

  async def test_empty_content_value(self):
    p = jinja_template.JinjaTemplate(
        '{{ content }}',
        content_varname='content',
    )
    self.assertEqual(await p([]).text(), '')

  async def test_template_starting_with_content(self):
    p = jinja_template.JinjaTemplate(
        '{{ content }} is amazing',
        content_varname='content',
    )
    self.assertEqual(await p(['The ', 'world']).text(), 'The world is amazing')

  async def test_template_ending_with_content(self):
    p = jinja_template.JinjaTemplate(
        'Amazing is {{ content }}',
        content_varname='content',
    )
    self.assertEqual(await p(['the ', 'world']).text(), 'Amazing is the world')

  async def test_template_with_multiple_content_variables(self):
    p = jinja_template.JinjaTemplate(
        '{{ content }} = {{ content }} = {{ content }}',
        content_varname='content',
    )
    self.assertEqual(await p(['4', '2']).text(), '42 = 42 = 42')

  async def test_template_with_consecutive_content_variables(self):
    p = jinja_template.JinjaTemplate(
        '{{ content }}{{ content }}{{ content }}',
        content_varname='content',
    )
    self.assertEqual(await p(['4', '2']).text(), '424242')

  async def test_template_with_content_and_custom_variables(self):
    p = jinja_template.JinjaTemplate(
        'Hello {{ name }}, answer this question: {{ content }}',
        content_varname='content',
        name='World',
    )
    self.assertEqual(
        await p('What is this landmark?').text(),
        'Hello World, answer this question: What is this landmark?',
    )

  def test_content_variable_in_kwargs(self):
    with self.assertRaisesRegex(
        ValueError,
        "'content' is set to render the processor's content and must not be"
        ' passed as a variable to the Jinja template.',
    ):
      jinja_template.JinjaTemplate(
          '',
          content_varname='content',
          content='',
      )


@dataclasses_json.dataclass_json
@dataclasses.dataclass(frozen=True)
class ExampleDataClass:
  first_name: str
  last_name: str


class RenderDataClassTest(unittest.IsolatedAsyncioTestCase):

  async def test_render_basic_dataclass(self):

    p = jinja_template.RenderDataClass(
        template_str='Hello {{ data.first_name }} {{ data.last_name }}!',
        data_class=ExampleDataClass,
    )
    self.assertEqual(
        await p(
            content_api.ProcessorPart.from_dataclass(
                dataclass=ExampleDataClass(first_name='John', last_name='Doe')
            ),
        ).text(),
        'Hello John Doe!',
    )

  async def test_render_dataclass_with_additional_variables(self):

    shopping_list = ['A', 'B', 'C']
    p = jinja_template.RenderDataClass(
        template_str=(
            'Hello {{ data.first_name }}, This is your shopping list:\n{%'
            ' for item in your_list %}This is item: {{ item }}\n{% endfor %}'
        ),
        data_class=ExampleDataClass,
        your_list=shopping_list,
    )
    self.assertEqual(
        await p(
            content_api.ProcessorPart.from_dataclass(
                dataclass=ExampleDataClass(first_name='John', last_name='Doe')
            ),
        ).text(),
        (
            'Hello John, This is your shopping list:\n'
            'This is item: A\n'
            'This is item: B\n'
            'This is item: C\n'
        ),
    )

  async def test_render_dataclass_without_dataclass(self):
    p = jinja_template.RenderDataClass(
        template_str='Hello {{ data.first_name }}!',
        data_class=ExampleDataClass,
    )
    self.assertEqual(await p('not a dataclass').text(), 'not a dataclass')


class RenderJsonTest(unittest.IsolatedAsyncioTestCase):

  async def test_render_basic_json(self):
    p = jinja_template.RenderJson(
        template_str=(
            'Hello {{ data.first_name }} {{ data.last_name }}, address: {{'
            ' data.address.street }}!'
        ),
    )
    self.assertEqual(
        await p(
            content_api.ProcessorPart(
                '{"first_name": "John", "last_name": "Doe", "address":'
                ' {"street": "123 Main St", "city": "Anytown", "state": "CA"}}',
                mimetype='application/json',
            ),
        ).text(),
        'Hello John Doe, address: 123 Main St!',
    )

  async def test_render_json_with_additional_variables(self):
    p = jinja_template.RenderJson(
        template_str='Hello {{ data.name }}! {{ other_var }}',
        other_var='Welcome',
    )
    self.assertEqual(
        await p(
            content_api.ProcessorPart(
                '{"name": "John"}',
                mimetype='application/json',
            ),
        ).text(),
        'Hello John! Welcome',
    )

  async def test_render_json_without_json(self):
    p = jinja_template.RenderJson(
        template_str='Hello {{ data.name }}!',
    )
    self.assertEqual(await p('not a json').text(), 'not a json')


class RenderProtoMessageTest(unittest.IsolatedAsyncioTestCase):

  async def test_render_basic_proto_message(self):
    p = jinja_template.RenderProtoMessage(
        proto_message=struct_pb2.Struct,
        template_str=(
            'Name: {{ data.name }}, age: {{'
            ' data.age }}!'
        ),
    )
    # number_value are floats.
    self.assertEqual(
        await p(
            content_api.ProcessorPart.from_proto_message(
                proto_message=struct_pb2.Struct(
                    fields={
                        'name': struct_pb2.Value(string_value='John'),
                        'age': struct_pb2.Value(number_value=25),
                    }
                )
            ),
        ).text(),
        'Name: John, age: 25.0!',
    )

  async def test_render_proto_message_with_additional_variables(self):
    p = jinja_template.RenderProtoMessage(
        proto_message=struct_pb2.Struct,
        template_str=(
            'Name: {{ data.name }}! {{ other_var }}'
        ),
        other_var='Welcome',
    )
    self.assertEqual(
        await p(
            content_api.ProcessorPart.from_proto_message(
                proto_message=struct_pb2.Struct(
                    fields={
                        'name': struct_pb2.Value(string_value='John'),
                    }
                )
            ),
        ).text(),
        'Name: John! Welcome',
    )

  async def test_render_proto_message_without_proto_message(self):
    p = jinja_template.RenderProtoMessage(
        proto_message=struct_pb2.Struct,
        template_str='Name: {{ data.name }}!',
    )
    self.assertEqual(await p('not a proto').text(), 'not a proto')


if __name__ == '__main__':
  absltest.main()
