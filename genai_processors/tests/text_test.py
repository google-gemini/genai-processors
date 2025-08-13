import asyncio
import dataclasses
import re
from typing import Sequence
import unittest

from absl.testing import absltest
from absl.testing import parameterized
import dataclasses_json
from genai_processors import content_api
from genai_processors import processor
from genai_processors.core import text


class MatchProcessorTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'small_parts',
          content_api.ProcessorContent(
              ['before ', '[ev', 'ent: ', 'query', ']', ' after']
          ),
      ),
      (
          'medium_parts',
          (
              content_api.ProcessorContent(
                  ['before ', '[event: query]', ' after']
              )
          ),
      ),
      (
          'large_parts',
          (content_api.ProcessorContent(['before [event: query] after'])),
      ),
  )
  def test_extract_parts_ok(self, input_content):
    extractor = text.MatchProcessor(
        word_start='[event:',
        pattern=r'\[event:(.*)\]',
        substream_output='event',
    )
    output = processor.apply_sync(extractor, input_content)
    actual = content_api.ProcessorContent(output)
    self.assertEqual(actual.as_text(substream_name='event'), '[event: query]')
    self.assertEqual(actual.as_text(substream_name=''), 'before  after')

  def test_extract_with_compiled_pattern_and_flags(self):
    compiled_pattern = re.compile(r'\[event:(.*?)\]', re.IGNORECASE)
    extractor = text.MatchProcessor(
        pattern=compiled_pattern,
        substream_output='event',
    )
    # Input has mixed case that would fail without re.IGNORECASE.
    input_content = ['before ', '[EVENT: query]', ' after']
    output = processor.apply_sync(extractor, input_content)
    actual = content_api.ProcessorContent(output)
    self.assertEqual(actual.as_text(substream_name='event'), '[EVENT: query]')
    self.assertEqual(actual.as_text(substream_name=''), 'before  after')

  @parameterized.named_parameters(
      (
          'small_parts',
          content_api.ProcessorContent([
              'before ',
              '[ev',
              'ent: ',
              'query1',
              ']',
              ' after',
              '[ev',
              'ent: qu',
              'ery2]',
          ]),
      ),
      (
          'medium_parts',
          (
              content_api.ProcessorContent([
                  'before ',
                  '[event: query1]',
                  ' after',
                  '[event: query2]',
              ])
          ),
      ),
      (
          'large_parts',
          (
              content_api.ProcessorContent(
                  ['before [event: query1] after [event: query2]']
              )
          ),
      ),
  )
  def test_extract_two_parts_ok(self, input_content):
    extractor = text.MatchProcessor(
        word_start='[event:',
        pattern=r'\[event:([^\].]*)\]',
        substream_output='event',
    )
    output = processor.apply_sync(extractor, input_content)
    self.assertLen(output, 4)
    actual = content_api.ProcessorContent(output)
    self.assertEqual(
        actual.as_text(substream_name='event'), '[event: query1][event: query2]'
    )
    self.assertEqual(actual.as_text(substream_name='').strip(), 'before  after')

  @parameterized.named_parameters(
      (
          'small_parts',
          content_api.ProcessorContent([
              'before ',
              '[ev',
              content_api.ProcessorPart(b'a', mimetype='image/png'),
              content_api.ProcessorPart(b'b', mimetype='image/png'),
              'ent: ',
              'query1',
              ']',
              ' after',
              '[ev',
              'ent: qu',
              content_api.ProcessorPart(b'c', mimetype='image/png'),
              'ery2]',
          ]),
      ),
      (
          'medium_parts',
          (
              content_api.ProcessorContent([
                  'before ',
                  '[event: query1]',
                  content_api.ProcessorPart(b'a', mimetype='image/png'),
                  content_api.ProcessorPart(b'b', mimetype='image/png'),
                  ' after',
                  '[event: query2]',
                  content_api.ProcessorPart(b'c', mimetype='image/png'),
              ])
          ),
      ),
      (
          'large_parts',
          (
              content_api.ProcessorContent([
                  'before [event: query1]',
                  content_api.ProcessorPart(b'a', mimetype='image/png'),
                  content_api.ProcessorPart(b'b', mimetype='image/png'),
                  ' after [event: query2]',
                  content_api.ProcessorPart(b'c', mimetype='image/png'),
              ])
          ),
      ),
  )
  def test_extract_mm_parts_ok(self, input_content):
    extractor = text.MatchProcessor(
        word_start='[event:',
        pattern=r'\[event:([^\].]*)\]',
        substream_output='event',
    )
    output = processor.apply_sync(extractor, input_content)
    self.assertLen(output, 7)
    actual = content_api.ProcessorContent(output)
    self.assertEqual(
        actual.as_text(substream_name='event'), '[event: query1][event: query2]'
    )
    self.assertEqual(actual.as_text(substream_name='').strip(), 'before  after')

  def test_extract_many_parts_single_ok(self):
    extractor = text.MatchProcessor(
        word_start='[event:',
        pattern=r'\[event:([^\].]*)\]',
        substream_output='event',
    )
    input_content = content_api.ProcessorContent(
        ['[event: query1][event: query2][event: query3]']
    )
    output = processor.apply_sync(extractor, input_content)
    self.assertLen(output, 3)
    for i in range(3):
      self.assertEqual(
          output[i],
          content_api.ProcessorPart(
              f'[event: query{i+1}]',
              mimetype='text/plain',
              substream_name='event',
          ),
      )
    input_content = content_api.ProcessorContent(
        ['before\n[event: query1]\n[event: query2]\n[event: query3]\nafter']
    )
    output = processor.apply_sync(extractor, input_content)
    actual = content_api.ProcessorContent(output)
    self.assertEqual(actual.as_text(substream_name=''), 'before\n\n\n\nafter')

  def test_long_prefix_ok(self):
    extractor = text.MatchProcessor(
        word_start='[event:',
        pattern=r'\[event:([^\].]*)\]',
        substream_output='event',
    )
    input_content = content_api.ProcessorContent([
        'looooooonnnnnnnnngggggg prefix',
        'another lllllooooonnnggg text',
        '[event: query1]',
        'text post regex',
    ])
    output = processor.apply_sync(extractor, input_content)
    self.assertLen(output, 4)
    self.assertEqual(
        output[2],
        content_api.ProcessorPart(
            '[event: query1]',
            mimetype='text/plain',
            substream_name='event',
        ),
    )

  def test_split_parts_ok(self):
    extractor = text.MatchProcessor(
        word_start=r'[event query:',
        pattern=r'\[event query: ([^\].]*)\]',
        substream_output='event',
    )
    input_content = content_api.ProcessorContent([
        '\n',
        content_api.ProcessorPart(b'a', mimetype='image/png'),
        content_api.ProcessorPart(b'a', mimetype='image/png'),
        content_api.ProcessorPart(b'a', mimetype='image/png'),
        content_api.ProcessorPart(b'a', mimetype='image/png'),
        '[',
        'event',
        ' query',
        ':',
        ' Is',
        ' a',
        ' person',
        ' waving',
        ' their',
        ' hand',
        ' at',
        ' you',
        '?]',
        ' ',
        '\n',
    ])
    output = processor.apply_sync(extractor, input_content)
    actual = content_api.ProcessorContent(output)
    self.assertEqual(actual.as_text(substream_name=''), '\n \n')
    self.assertLen(output, 8)
    self.assertEqual(
        output[5],
        content_api.ProcessorPart(
            '[event query: Is a person waving their hand at you?]',
            mimetype='text/plain',
            substream_name='event',
        ),
    )

  def test_extract_special_characters_ok(self):
    extractor = text.MatchProcessor(
        word_start='```start\n',
        pattern='```start\n(.*?)```',
        substream_output='block',
    )
    input_content = content_api.ProcessorContent([
        """\n```start\nagent_1()\nagent_2()```\n""",
        """```start\nagent_3()\nagent_4()```\n""",
    ])
    output = processor.apply_sync(extractor, input_content)
    self.assertLen(output, 5)
    actual = content_api.ProcessorContent(output)
    self.assertEqual(
        actual.as_text(substream_name='block'),
        '```start\nagent_1()\nagent_2()``````start\nagent_3()\nagent_4()```',
    )

  def test_empty_text_buffer_ok(self):
    extractor = text.MatchProcessor(
        word_start='[event:',
        pattern=r'\[event:([^\].]*)\]',
        substream_input='input',
        substream_output='block',
    )
    input_content = [
        content_api.ProcessorPart(b'a', mimetype='image/png'),
        content_api.ProcessorPart('[event: text_a]'),
        content_api.ProcessorPart(b'b', mimetype='image/png'),
        content_api.ProcessorPart('[event: text_b]'),
        content_api.ProcessorPart(b'c', mimetype='image/png'),
    ]
    output = processor.apply_sync(extractor, input_content)
    self.assertEqual(output, input_content)
    img_content = [
        content_api.ProcessorPart(b'a', mimetype='image/png'),
        content_api.ProcessorPart(b'b', mimetype='image/png'),
        content_api.ProcessorPart(b'c', mimetype='image/png'),
    ]
    output = processor.apply_sync(extractor, img_content)
    self.assertEqual(output, img_content)

  def test_mixed_substream_names_ok(self):
    stream_name_output = 'block'
    extractor = text.MatchProcessor(
        word_start='[event:',
        pattern=r'\[event:([^\].]*)\]',
        substream_input='input',
        substream_output=stream_name_output,
    )
    input_content = [
        content_api.ProcessorPart(b'a', mimetype='image/png'),
        content_api.ProcessorPart('[event: ', substream_name='input'),
        content_api.ProcessorPart(b'b', mimetype='image/png'),
        content_api.ProcessorPart('[event: text_b]', substream_name='other'),
        content_api.ProcessorPart('text_a]', substream_name='input'),
        content_api.ProcessorPart(b'c', mimetype='image/png'),
    ]
    output = processor.apply_sync(extractor, input_content)
    self.assertLen(output, 5)
    self.assertEqual(
        output,
        [
            content_api.ProcessorPart(b'a', mimetype='image/png'),
            content_api.ProcessorPart(
                '[event: text_a]', substream_name=stream_name_output
            ),
            content_api.ProcessorPart(b'b', mimetype='image/png'),
            content_api.ProcessorPart(
                '[event: text_b]', substream_name='other'
            ),
            content_api.ProcessorPart(b'c', mimetype='image/png'),
        ],
    )

  def test_flush_fn_remove_ok(self):
    extractor = text.MatchProcessor(
        word_start='[',
        pattern=r'\[event:([^\].]*)\]',
        substream_input='input',
        substream_output='block',
        flush_fn=lambda x: x.get_metadata('flush'),
    )
    input_content = [
        content_api.ProcessorPart(b'a', mimetype='image/png'),
        content_api.ProcessorPart('[event: ', substream_name='input'),
        content_api.ProcessorPart('text_a', substream_name='other'),
        content_api.ProcessorPart(
            ' text_b',
            substream_name='other',
            metadata={'flush': True},
        ),
        content_api.ProcessorPart('[event: ', substream_name='input'),
        content_api.ProcessorPart(b'c', mimetype='image/png'),
        content_api.ProcessorPart('text_c]', substream_name='input'),
    ]
    output = processor.apply_sync(extractor, input_content)
    self.assertLen(output, 6)
    self.assertEqual(
        output,
        [
            content_api.ProcessorPart(b'a', mimetype='image/png'),
            content_api.ProcessorPart('[event: ', substream_name='input'),
            content_api.ProcessorPart('text_a', substream_name='other'),
            content_api.ProcessorPart(
                ' text_b',
                substream_name='other',
                metadata={'flush': True},
            ),
            content_api.ProcessorPart(
                '[event: text_c]', substream_name='block'
            ),
            content_api.ProcessorPart(b'c', mimetype='image/png'),
        ],
    )

  def test_flush_fn_do_not_remove_ok(self):
    extractor = text.MatchProcessor(
        word_start='[',
        pattern=r'\[event:([^\].]*)\]',
        substream_input='input',
        substream_output='block',
        flush_fn=lambda x: x.get_metadata('flush'),
        remove_from_input_stream=False,
    )
    input_content = [
        content_api.ProcessorPart(b'a', mimetype='image/png'),
        content_api.ProcessorPart('[event: text_a', substream_name='input'),
        content_api.ProcessorPart(
            '', substream_name='other', metadata={'flush': True}
        ),
        content_api.ProcessorPart(']', substream_name='input'),
        content_api.ProcessorPart('[event: ', substream_name='input'),
        content_api.ProcessorPart('text_c]', substream_name='input'),
    ]
    output = processor.apply_sync(extractor, input_content)
    self.assertLen(output, len(input_content) + 1)
    self.assertEqual(
        output,
        input_content
        + [
            content_api.ProcessorPart('[event: text_c]', substream_name='block')
        ],
    )

  @parameterized.named_parameters(
      (
          'small_parts',
          content_api.ProcessorContent(
              ['before ', '[ev', 'ent: ', 'query', ']', ' after']
          ),
      ),
      (
          'medium_parts',
          (
              content_api.ProcessorContent(
                  ['before ', '[event: query]', ' after']
              )
          ),
      ),
      (
          'large_parts',
          (content_api.ProcessorContent(['before [event: query] after'])),
      ),
  )
  def test_find_parts_ok(self, input_content):
    extractor = text.MatchProcessor(
        word_start='[event:',
        pattern=r'\[event:(.*)\]',
        substream_output='event',
        remove_from_input_stream=False,
    )
    output = processor.apply_sync(extractor, input_content)
    actual = content_api.ProcessorContent(output)
    self.assertEqual(actual.as_text(substream_name='event'), '[event: query]')
    self.assertEqual(
        actual.as_text(substream_name=''), 'before [event: query] after'
    )

  def test_order_when_not_removing_parts_ok(self):
    extractor = text.MatchProcessor(
        word_start='[event:',
        pattern=r'\[event:([^\].]*)\]',
        substream_output='event',
        remove_from_input_stream=False,
    )
    input_content = content_api.ProcessorContent([
        'prefix',
        'another text',
        '[event:',
        content_api.ProcessorPart(b'a', mimetype='image/png'),
        'query1]',
        'postfix',
    ])
    output = processor.apply_sync(extractor, input_content)
    self.assertLen(output, 7)
    self.assertEqual(
        output[2],
        content_api.ProcessorPart(
            '[event:',
            mimetype='text/plain',
        ),
    )
    self.assertEqual(
        output[5],
        content_api.ProcessorPart(
            '[event:query1]',
            mimetype='text/plain',
            substream_name='event',
        ),
    )


@dataclasses_json.dataclass_json
@dataclasses.dataclass(frozen=True)
class YouTubeUrl:
  url: str


@dataclasses_json.dataclass_json
@dataclasses.dataclass(frozen=True)
class GithubUrl:
  url: str


class UrlExtractorTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  def test_extracts_url(self):
    extractor = text.UrlExtractor(
        {'https://youtube.': YouTubeUrl, 'https://github.com': GithubUrl}
    )
    input_content = content_api.ProcessorContent([
        'What is at',
        'https://git',
        'hub.com/google-gemini/genai-processors ?',
    ])
    output = processor.apply_sync(extractor, input_content)
    self.assertEqual(
        output,
        [
            content_api.ProcessorPart('What is at'),
            content_api.ProcessorPart.from_dataclass(
                dataclass=GithubUrl(
                    'https://github.com/google-gemini/genai-processors'
                )
            ),
            content_api.ProcessorPart(' ?'),
        ],
    )

  def test_mismatching_schemes(self):
    with self.assertRaises(ValueError):
      text.UrlExtractor(
          {'https://youtube.': YouTubeUrl, 'ftp://domain.com': GithubUrl}
      )

  def test_http_and_https_mix(self):
    # No exception is raised.
    text.UrlExtractor(
        {'https://youtube.': YouTubeUrl, 'http://github.com': GithubUrl}
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='single_url_with_surrounding_text',
          input_parts=[
              content_api.ProcessorPart(
                  "Go to https://example.com, it's great."
              )
          ],
          expected_parts=[
              content_api.ProcessorPart('Go to '),
              content_api.ProcessorPart.from_dataclass(
                  dataclass=text.FetchRequest(url='https://example.com')
              ),
              content_api.ProcessorPart(", it's great."),
          ],
      ),
      dict(
          testcase_name='multiple_urls_in_one_part',
          input_parts=[
              content_api.ProcessorPart(
                  'See http://a.com and https://b.org today.'
              )
          ],
          expected_parts=[
              content_api.ProcessorPart('See '),
              content_api.ProcessorPart.from_dataclass(
                  dataclass=text.FetchRequest(url='http://a.com')
              ),
              content_api.ProcessorPart(' and '),
              content_api.ProcessorPart.from_dataclass(
                  dataclass=text.FetchRequest(url='https://b.org')
              ),
              content_api.ProcessorPart(' today.'),
          ],
      ),
      dict(
          testcase_name='no_urls',
          input_parts=[content_api.ProcessorPart('This is just plain text.')],
          expected_parts=[
              content_api.ProcessorPart('This is just plain text.')
          ],
      ),
      dict(
          testcase_name='non_text_parts_are_passed_through',
          input_parts=[
              content_api.ProcessorPart('Text before, '),
              content_api.ProcessorPart(b'bytes', mimetype='image/png'),
              content_api.ProcessorPart(' text after.'),
          ],
          expected_parts=[
              content_api.ProcessorPart('Text before, '),
              content_api.ProcessorPart(b'bytes', mimetype='image/png'),
              content_api.ProcessorPart(' text after.'),
          ],
      ),
      dict(
          testcase_name='url_is_entire_part',
          input_parts=[content_api.ProcessorPart('https://example.com')],
          expected_parts=[
              content_api.ProcessorPart.from_dataclass(
                  dataclass=text.FetchRequest(url='https://example.com')
              )
          ],
      ),
      dict(
          testcase_name='url_at_start_of_part',
          input_parts=[
              content_api.ProcessorPart('https://example.com is a site.')
          ],
          expected_parts=[
              content_api.ProcessorPart.from_dataclass(
                  dataclass=text.FetchRequest(url='https://example.com')
              ),
              content_api.ProcessorPart(' is a site.'),
          ],
      ),
      dict(
          testcase_name='url_with_parentheses',
          input_parts=[
              content_api.ProcessorPart(
                  'See https://en.wikipedia.org/wiki/Text_(disambiguation) for'
                  ' details.'
              )
          ],
          expected_parts=[
              content_api.ProcessorPart('See '),
              content_api.ProcessorPart.from_dataclass(
                  dataclass=text.FetchRequest(
                      url='https://en.wikipedia.org/wiki/Text_(disambiguation)'
                  )
              ),
              content_api.ProcessorPart(' for details.'),
          ],
      ),
  )
  async def test_url_extraction(
      self,
      input_parts: Sequence[content_api.ProcessorPart],
      expected_parts: Sequence[content_api.ProcessorPart],
  ):
    extractor = text.UrlExtractor()
    input_stream = processor.stream_content(input_parts)

    results = []
    async for output_part in extractor(input_stream):
      results.append(output_part)

    self.assertLen(results, len(expected_parts))

    for i, (actual, expected) in enumerate(zip(results, expected_parts)):
      with self.subTest(f'part_{i}'):
        self.assertEqual(actual.mimetype, expected.mimetype)
        if content_api.is_dataclass(actual.mimetype, text.FetchRequest):
          self.assertEqual(
              actual.get_dataclass(text.FetchRequest),
              expected.get_dataclass(text.FetchRequest),
          )
        elif content_api.is_text(actual.mimetype):
          self.assertEqual(actual.text, expected.text)
        else:
          self.assertEqual(actual.bytes, expected.bytes)

  async def test_streaming(self):
    extractor = text.UrlExtractor()

    async def one_hello():
      yield content_api.ProcessorPart('Hello', metadata={'turn_complete': True})
      await asyncio.sleep(300)

    async for part in extractor(one_hello()):
      # Test that we got one part despite the input stream not being complete.
      self.assertEqual(part.text, 'Hello')
      break


if __name__ == '__main__':
  absltest.main()
