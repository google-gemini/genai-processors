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
"""Processors operating on text and regular expressions."""

import asyncio
from collections.abc import AsyncIterable, Callable
import dataclasses
import re
from typing import Mapping, Type

import dataclasses_json
from genai_processors import content_api
from genai_processors import processor
import termcolor


_MAX_LOOP_COUNT = 1000


class MatchProcessor(processor.Processor):
  r"""Processor finding text patterns and extracting them from the input stream.

  There are two modes controlled by the `remove_from_input_stream` parameter.

  1. [True] When the pattern is found, the matched text is removed from the
     input stream and returned as a single part with the substream name
     `substream_output` and the mimetype text/plain.
  2. [False] When the pattern is found, it is not removed from the input stream.
     It is still returned as a single part with the substream name
     `substream_output` and the mimetype text/plain.

  Example [mode 1]:
  ```python
    p = MatchProcessor(
          word_start='[',
          pattern=r'\[.*\]',
          substream_output='from_regex'
        )
    output = processor.apply_sync(p, ['a', 'b', 'c[d', 'e]f','g'])
    ```
    output will contain 6 parts:

    -  content_api.ProcessorPart('a', mimetype='text/plain')
    -  content_api.ProcessorPart('b', mimetype='text/plain')
    -  content_api.ProcessorPart('c', mimetype='text/plain')
    -  content_api.ProcessorPart('[de]',
         mimetype='text/plain',
         substream_name='from_regex',
       )
    -  content_api.ProcessorPart('f', mimetype='text/plain')
    -  content_api.ProcessorPart('g', mimetype='text/plain')

    The text part of `output` will be `abcfg`.

  In [mode 2], the output will be:

    -  content_api.ProcessorPart('a', mimetype='text/plain')
    -  content_api.ProcessorPart('b', mimetype='text/plain')
    -  content_api.ProcessorPart('c[d', mimetype='text/plain')
    -  content_api.ProcessorPart('e]f', mimetype='text/plain')
    -  content_api.ProcessorPart('[de]',
         mimetype='text/plain',
         substream_name='from_regex',
       )
    -  content_api.ProcessorPart('g', mimetype='text/plain')

  When using this processor in a real-time setting, it is recommended to set
  `remove_from_input_stream` to False. This will allow the processor to output
  the parts as soon as possible and not block the stream. If you still need
  to remove the matched text from the input stream, it is advised to define a
  short `word_start` and/or flush conditions (via `flush_fn`) that are met often
  to reset the buffer and to keep producing parts in the output stream.
  """

  def __init__(
      self,
      *,
      pattern: str | re.Pattern[str],
      word_start: str | None = None,
      substream_input: str = '',
      substream_output: str = '',
      flush_fn: Callable[[content_api.ProcessorPart], bool] | None = (None),
      remove_from_input_stream: bool = True,
      transform: (
          Callable[[content_api.ProcessorPart], content_api.ProcessorPart]
          | None
      ) = None,
  ):
    """Extracts text parts from the input stream that match the pattern.

    Only considers text parts from the input stream `substream_input`.

    See class docstring for more details.

    This processor buffers input parts until one of the following happens:

    1. the `pattern` is found, then it outputs all parts in the buffer up to the
      end of the match.
    2. when the `flush_fn` returns True, then it outputs all parts in the
      buffer.

    When `word_start` is set, the following happens:

    3. the word_start is found, then it outputs all the parts before the
      `word_start`.
    4. `word_start` is not found in buffer_text[-len(word_start):], where
       buffer_text is the concatenation of all the text parts in the buffer. All
       parts whose text is in buffer_text[-len(word_start):] are returned.

    While in the buffer, the parts are not output which means they can lead to
    delays in the output stream.

    To avoid such delays, set the `remove_from_input_stream` to False and/or
    define a `flush_fn` that returns True often to discard the parts in the
    buffer frequently.

    Args:
      pattern: pattern to match a text to extract into a part. When
        `remove_from_input_stream` is True, the matched text will be removed
        from the stream and will be replaced by a single extracted part. The
        parts before and after this match will be returned as is. Note that by
        default, re.DOTALL is used to match newlines. To override this behavior,
        pass a re.Pattern object instead.
      word_start: text to match the start of the text that needs to be captured.
        `word_start` is not a regular expression but a plain string that will be
        matched exactly. `word_start` should be a substring of the pattern and
        should indicate that the pattern is about to be matched. Whenever
        `word_start` is found, the parts after it will be buffered (not
        returned) until the pattern is found, the `flush_fn` returns True, or
        the input stream is exhausted. When set to None (default), this logic is
        not applied.
      substream_input: name of the substream to use for the input part.
      substream_output: name of the substream to use for the extracted part.
      flush_fn: function to check when to reset the buffer and yield all the
        parts in the buffer. The part where `flush_fn` returns True will be
        returned as is and will not be matched against the pattern.
      remove_from_input_stream: if True, the processor will remove the matched
        parts from the input stream. If False, the input stream will be
        preserved and the parts will be returned as is quickly. The processor
        will output into its `substream_output` substream once a match is found.
      transform: A transformation to be applied to the matched Parts.
    """
    self._word_start = word_start
    if isinstance(pattern, str):
      self._pattern = re.compile(pattern, re.DOTALL)
    else:
      self._pattern = pattern
    self._substream_input = substream_input
    self._substream_output = substream_output
    self._flush_fn = flush_fn or content_api.is_end_of_turn
    self._remove_from_input_stream = remove_from_input_stream
    if transform:
      self._transform = transform
    else:
      self._transform = lambda part: part

  def _extract_part(
      self, text_buffer: str, part_buffer: list[content_api.ProcessorPart]
  ) -> tuple[list[content_api.ProcessorPart], list[content_api.ProcessorPart]]:
    """Returns the list of parts to yield and the remaining parts to process."""
    to_yield = []
    to_process = part_buffer
    left_over = []
    if (match := self._pattern.search(text_buffer)) is None:
      return to_yield, to_process
    # We have found the pattern, we can yield all the parts up to the
    # beginning of the match and then yield the parts after the match.
    offset = 0
    part_idx = -1
    for c in part_buffer:
      part_idx += 1
      if (
          not content_api.is_text(c.mimetype)
          or c.substream_name != self._substream_input
      ):
        if self._remove_from_input_stream:
          to_yield.append(c)
        continue
      # Find the start of the part to extract. Yields all parts until
      # this start.
      if (offset + len(c.text)) <= match.start():
        part_end = len(c.text)
      else:
        part_end = match.start() - offset
      if part_end > 0 and self._remove_from_input_stream:
        to_yield.append(
            content_api.ProcessorPart(
                c.text[:part_end],
                metadata=c.metadata,
                substream_name=c.substream_name,
                mimetype=c.mimetype,
            )
        )
      if match.start() < offset + len(c.text) and match.start() >= offset:
        to_yield.append(
            self._transform(
                content_api.ProcessorPart(
                    match.group(0),
                    metadata=c.metadata,
                    substream_name=self._substream_output,
                    mimetype=c.mimetype,
                )
            )
        )

      # Find the start of the parts after the match.
      part_start = match.end() - offset
      if part_start < len(c.text) and self._remove_from_input_stream:
        # We have reached the end of the match, there can be another one later
        # in the buffer. We keep the part after the match and stop.
        left_over = [
            content_api.ProcessorPart(
                c.text[part_start:],
                metadata=c.metadata,
                substream_name=c.substream_name,
                mimetype=c.mimetype,
            )
        ]
        break
      offset += len(c.text)
    to_process = left_over + part_buffer[part_idx + 1 :]
    return to_yield, to_process

  async def call(
      self, content: AsyncIterable[content_api.ProcessorPart]
  ) -> AsyncIterable[content_api.ProcessorPartTypes]:
    part_buffer = []
    async for part in content:
      if not self._remove_from_input_stream:
        yield part
      part_buffer.append(part)
      if self._flush_fn(part):
        if self._remove_from_input_stream:
          for part_b in part_buffer:
            yield part_b
        part_buffer = []
      text_buffer = content_api.as_text(
          part_buffer, substream_name=self._substream_input
      )
      # If the word_start is not in the buffer, we can already yield all the
      # parts up to the last ones that could contain word_start.
      # This is a quick check to avoid buffering more than necessary.
      # Only applies when word_start is set.
      if self._word_start is not None and self._word_start not in text_buffer:
        offset = 0
        idx = 0  # index of first part not yielded.
        for c in part_buffer:
          if (
              not content_api.is_text(c.mimetype)
              and self._remove_from_input_stream
          ):
            yield c
            idx += 1
            continue
          offset += len(content_api.as_text(c))
          if (
              offset < len(text_buffer) - len(self._word_start)
              and self._remove_from_input_stream
          ):
            yield c
            idx += 1
          else:
            break
        part_buffer = part_buffer[idx:]
      else:
        to_yield, part_buffer = self._extract_part(text_buffer, part_buffer)
        for part in to_yield:
          yield part

    # Process the last part which can contain the pattern many times.
    loop_count = 0
    while part_buffer and loop_count < _MAX_LOOP_COUNT:
      text_buffer = content_api.as_text(
          part_buffer, substream_name=self._substream_input
      )
      to_yield, part_buffer = self._extract_part(text_buffer, part_buffer)
      if not to_yield:
        # No match found, we can yield all the parts.
        if self._remove_from_input_stream:
          for part in part_buffer:
            yield part
        break
      else:
        for part in to_yield:
          yield part
    if loop_count >= _MAX_LOOP_COUNT:
      raise RuntimeError(
          'Max loop count reached, the pattern or the input stream is probably'
          ' malformed.'
      )


@dataclasses_json.dataclass_json
@dataclasses.dataclass(frozen=True)
class FetchRequest:
  """Dataclass to represent a request to fetch a document by a URL."""

  url: str


class UrlExtractor(MatchProcessor):
  """Replaces encountered text URLs with strongly typed Parts.

  Also see core/web.py for the infrastructure to fetch the extracted URLs.
  Alternatively see core/github.py for a processor that provides a tailored
  handling for github URLs.

  In some scenarios it is useful to replace URLs mentioned in the prompt with
  the content they point to. In many cases it can be handled by tool calls, but
  if we want to avoid additional roundtrip or the underlying model does not
  support tools, hardwired logic might be preferrable.

  We recommend splitting detecting the URLs and fetching them into separate
  processors. The processor that does the fetching should act on Parts with a
  special MIME type to avoid fetching them unintentionally. And a separate
  processor should decide which URLs should be processed.

  Usage:
    In the basic case use UrlExtractor() without arguments and handle
    FetchRequest parts produced. Alternatively you can define separate
    dataclasses for specific URL prefixes:

      @dataclasses_json.dataclass_json
      @dataclasses.dataclass(frozen=True)
      class YouTubeUrl:
        url: str

    And then tell UrlExtractor to extract them:

      UrlExtractor({
          'https://youtube.': YouTubeUrl,
          'https://github.com': GithubUrl
      })

    Note that all URLs must have the same scheme to allow efficient matching.
  """

  def __init__(
      self,
      urls: Mapping[str, Type] | None = None,  # pylint: disable=g-bare-generic
      *,
      substream_input: str = '',
      substream_output: str = '',
  ):
    """Initiallizes the extractor.

    Args:
      urls: A map from URL prefix (e.g. 'https://github.com') to a Dataclass)
      substream_input: name of the substream to use for the input part.
      substream_output: name of the substream to use for the extracted part.
    """
    if urls is None:
      urls = {'http://': FetchRequest, 'https://': FetchRequest}

    schemes = set()
    for prefix in urls.keys():
      scheme = prefix.split(':')[0]
      if scheme == 'https':
        # Allow mixing http and https URLs.
        # This works because http is a prefix of https.
        scheme = 'http'
      schemes.add(scheme)

    if len(schemes) != 1:
      raise ValueError(
          'All URL prefixes must have the same scheme e.g. https. Got'
          f' {schemes!r}'
      )

    def transform(part: content_api.ProcessorPart):
      for prefix, dataclass in urls.items():
        if part.text.startswith(prefix):
          return content_api.ProcessorPart.from_dataclass(
              dataclass=dataclass(part.text),
              metadata=part.metadata,
              substream_name=part.substream_name,
          )

    super().__init__(
        pattern=re.compile(
            '('
            + '|'.join(urls.keys())
            + r')([^\s<>"\'\u200B]*[^\s<>"\'\u200B\.\,])?',
            re.IGNORECASE,
        ),
        word_start=next(iter(schemes)),
        substream_input=substream_input,
        substream_output=substream_output,
        remove_from_input_stream=True,
        transform=transform,
    )


@processor.source()
async def terminal_input(
    prompt: str = '',
) -> AsyncIterable[content_api.ProcessorPartTypes]:
  """Yields lines from the terminal, exits on ctrl+D."""
  while True:
    try:
      yield await asyncio.to_thread(input, prompt)
      yield content_api.ProcessorPart.end_of_turn()
    except EOFError:
      # Exit on ctrl+D.
      return


async def terminal_output(
    content: AsyncIterable[content_api.ProcessorPartTypes],
    prompt: str = '',
) -> None:
  """Prints the part to the terminal.

  Consumes all the content and prints it to the terminal. Prints the prompt
  when an `end_of_turn` part is encountered.

  The parts are printed with their role in green, red or yellow. The text parts
  are printed in bold.

  Args:
    content: The content to print.
    prompt: The prompt to print when the model is done.
  """
  old_part_role = None
  async for part in content:
    match part.role:
      case 'user':
        color = 'green'
      case 'model':
        color = 'red'
      case _:
        color = 'yellow'
    part_role = part.role or 'default'
    if content_api.is_text(part.mimetype):
      content_text = part.text
    else:
      content_text = f'<{part.mimetype}>'
    if part_role != old_part_role:
      old_part_role = part_role
      print(
          termcolor.colored(
              f'\n{part_role}: {content_text}', color, attrs=['bold']
          ),
          end='',
          flush=True,
      )
    else:
      print(
          termcolor.colored(f'{content_text}', color, attrs=['bold']),
          end='',
          flush=True,
      )
    # Reprint the prompt when the model is done.
    if content_api.is_end_of_turn(part):
      print('\n' + prompt, end='', flush=True)
