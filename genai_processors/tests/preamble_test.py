from absl.testing import absltest
from absl.testing import parameterized
from genai_processors import content_api
from genai_processors import processor
from genai_processors.core import preamble as preamble_lib


class PreambleTest(parameterized.TestCase):

  @parameterized.parameters([
      ('a', 'abc'),
      (
          [content_api.ProcessorPart('a'), content_api.ProcessorPart('b')],
          'abbc',
      ),
      (content_api.ProcessorContent('a'), 'abc'),
  ])
  def test_adds_preamble(self, preamble, expected_text):
    pr = preamble_lib.Preamble(content=preamble)
    output = processor.apply_sync(
        pr, [content_api.ProcessorPart('b'), content_api.ProcessorPart('c')]
    )
    self.assertEqual(content_api.as_text(output), expected_text)

  def test_adds_preamble_from_factory(self):
    global_state = 'a'

    def preamble_factory():
      return global_state

    # global_state = 'a' at time of creating the processor.
    pr = preamble_lib.Preamble(content_factory=preamble_factory)

    # global state = 'a' at time of applying the processor, so it should
    # prepend 'a'.
    output = processor.apply_sync(pr, ['c'])
    self.assertEqual(content_api.as_text(output), 'ac')

    # global state = 'b' at time of applying the processor, so it should
    # prepend 'b'.
    global_state = 'b'
    output = processor.apply_sync(pr, ['c'])

    self.assertEqual(content_api.as_text(output), 'bc')


class SuffixTest(parameterized.TestCase):

  @parameterized.parameters([
      ('a', 'bca'),
      (
          [content_api.ProcessorPart('a'), content_api.ProcessorPart('b')],
          'bcab',
      ),
      (content_api.ProcessorContent('a'), 'bca'),
  ])
  def test_adds_suffix(self, suffix, expected_text):
    pr = preamble_lib.Suffix(content=suffix)
    output = processor.apply_sync(
        pr, [content_api.ProcessorPart('b'), content_api.ProcessorPart('c')]
    )
    self.assertEqual(content_api.as_text(output), expected_text)

  def test_adds_suffix_from_factory(self):
    global_state = 'a'

    def suffix_factory():
      return global_state

    # global_state = 'a' at time of creating the processor.
    pr = preamble_lib.Suffix(content_factory=suffix_factory)

    # global state = 'a' at time of applying the processor, so it should
    # prepend 'a'.
    output = processor.apply_sync(pr, ['c'])
    self.assertEqual(content_api.as_text(output), 'ca')

    # global state = 'b' at time of applying the processor, so it should
    # prepend 'b'.
    global_state = 'b'
    output = processor.apply_sync(pr, ['c'])

    self.assertEqual(content_api.as_text(output), 'cb')


if __name__ == '__main__':
  absltest.main()
