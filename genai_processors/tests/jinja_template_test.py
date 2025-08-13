import dataclasses
import unittest

from absl.testing import absltest
import dataclasses_json
from genai_processors import content_api
from genai_processors import processor
from genai_processors.core import jinja_template


class JinjaTemplateTest(unittest.IsolatedAsyncioTestCase):

  def test_empty_template(self):
    p = jinja_template.JinjaTemplate('')
    output = processor.apply_sync(p, [])
    self.assertEqual(content_api.as_text(output), '')

  def test_empty_template_with_processor_content(self):
    p = jinja_template.JinjaTemplate('')
    output = processor.apply_sync(p, [content_api.ProcessorPart('Hello World')])
    self.assertEqual(content_api.as_text(output), '')

  def test_template_without_content_variable(self):
    p = jinja_template.JinjaTemplate(
        'Hello {{ name }}',
        content_varname='content',
        name='World',
    )
    output = processor.apply_sync(p, [])
    self.assertEqual(content_api.as_text(output), 'Hello World')

  def test_template_with_content_variable_only(self):
    p = jinja_template.JinjaTemplate(
        '{{ content }}',
        content_varname='content',
    )
    output = processor.apply_sync(
        p,
        [
            content_api.ProcessorPart('Hello '),
            content_api.ProcessorPart('World'),
        ],
    )
    self.assertEqual(content_api.as_text(output), 'Hello World')

  def test_empty_content_value(self):
    p = jinja_template.JinjaTemplate(
        '{{ content }}',
        content_varname='content',
    )
    output = processor.apply_sync(p, [])
    self.assertEqual(content_api.as_text(output), '')

  def test_template_starting_with_content(self):
    p = jinja_template.JinjaTemplate(
        '{{ content }} is amazing',
        content_varname='content',
    )
    output = processor.apply_sync(
        p,
        [
            content_api.ProcessorPart('The '),
            content_api.ProcessorPart('world'),
        ],
    )
    self.assertEqual(content_api.as_text(output), 'The world is amazing')

  def test_template_ending_with_content(self):
    p = jinja_template.JinjaTemplate(
        'Amazing is {{ content }}',
        content_varname='content',
    )
    output = processor.apply_sync(
        p,
        [
            content_api.ProcessorPart('the '),
            content_api.ProcessorPart('world'),
        ],
    )
    self.assertEqual(content_api.as_text(output), 'Amazing is the world')

  def test_template_with_multiple_content_variables(self):
    p = jinja_template.JinjaTemplate(
        '{{ content }} = {{ content }} = {{ content }}',
        content_varname='content',
    )
    output = processor.apply_sync(
        p,
        [
            content_api.ProcessorPart('4'),
            content_api.ProcessorPart('2'),
        ],
    )
    self.assertEqual(content_api.as_text(output), '42 = 42 = 42')

  def test_template_with_consecutive_content_variables(self):
    p = jinja_template.JinjaTemplate(
        '{{ content }}{{ content }}{{ content }}',
        content_varname='content',
    )
    output = processor.apply_sync(
        p,
        [
            content_api.ProcessorPart('4'),
            content_api.ProcessorPart('2'),
        ],
    )
    self.assertEqual(content_api.as_text(output), '424242')

  def test_template_with_content_and_custom_variables(self):
    p = jinja_template.JinjaTemplate(
        'Hello {{ name }}, answer this question: {{ content }}',
        content_varname='content',
        name='World',
    )
    output = processor.apply_sync(
        p,
        [content_api.ProcessorPart('What is this landmark?')],
    )
    self.assertEqual(
        content_api.as_text(output),
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


class RenderDataClassTest(unittest.TestCase):

  def test_render_basic_dataclass(self):

    p = jinja_template.RenderDataClass(
        template_str='Hello {{ data.first_name }} {{ data.last_name }}!',
        data_class=ExampleDataClass,
    )
    output = processor.apply_sync(
        p,
        [
            content_api.ProcessorPart.from_dataclass(
                dataclass=ExampleDataClass(first_name='John', last_name='Doe')
            )
        ],
    )
    self.assertEqual(content_api.as_text(output), 'Hello John Doe!')

  def test_render_dataclass_with_additional_variables(self):

    shopping_list = ['A', 'B', 'C']
    p = jinja_template.RenderDataClass(
        template_str=(
            'Hello {{ data.first_name }},\n This is your shopping list:\n{%'
            ' for item in your_list %}This is item: {{ item }}\n{% endfor %}'
        ),
        data_class=ExampleDataClass,
        your_list=shopping_list,
    )
    output = processor.apply_sync(
        p,
        [
            content_api.ProcessorPart.from_dataclass(
                dataclass=ExampleDataClass(first_name='John', last_name='Doe')
            )
        ],
    )
    expected_output = (
        'Hello John,\n This is your shopping list:\n'
        'This is item: A\n'
        'This is item: B\n'
        'This is item: C\n'
    )
    self.assertEqual(content_api.as_text(output), expected_output)

  def test_render_dataclass_without_dataclass(self):
    p = jinja_template.RenderDataClass(
        template_str='Hello {{ data.first_name }}!',
        data_class=ExampleDataClass,
    )
    output = processor.apply_sync(
        p,
        [
            content_api.ProcessorPart(
                'not a dataclass',
                mimetype='text/plain',
            )
        ],
    )
    self.assertEqual(content_api.as_text(output), 'not a dataclass')


if __name__ == '__main__':
  absltest.main()
