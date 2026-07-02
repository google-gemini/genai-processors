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

"""Processor for converting text to lowercase."""

from collections.abc import AsyncIterable

from genai_processors import content_api
from genai_processors import processor


class LowercaseTextProcessor(processor.PartProcessor):
  """PartProcessor that converts all text content to lowercase.

  This processor takes text input and converts it to lowercase while preserving
  all other attributes of the ProcessorPart (metadata, substream_name, mimetype,
  etc.). Non-text parts are passed through unchanged.

  Example usage:

  ```python
  from genai_processors import content_api
  from genai_processors import processor
  from genai_processors.contrib import lowercase_text_processor

  p = lowercase_text_processor.LowercaseTextProcessor()
  output = processor.apply_sync(
      p,
      [
          content_api.ProcessorPart(
              'Hello World! This is A TEST.',
              mimetype='text/plain',
          ),
          content_api.ProcessorPart(
              'ANOTHER Example String.',
              mimetype='text/plain',
          ),
      ],
  )
  print(content_api.as_text(output))
  # Output: "hello world! this is a test.another example string."
  ```
  """

  def match(self, part: content_api.ProcessorPart) -> bool:
    """Returns True if the part should be processed (i.e., if it's text).

    Args:
      part: The ProcessorPart to check.

    Returns:
      True if the part is text and should be processed, False otherwise.
    """
    return content_api.is_text(part.mimetype)

  async def call(
      self, part: content_api.ProcessorPart
  ) -> AsyncIterable[content_api.ProcessorPartTypes]:
    """Processes a single part and converts text to lowercase.

    Args:
      part: The ProcessorPart to process.

    Yields:
      ProcessorPart with text converted to lowercase, preserving all metadata.
    """
    # Convert text to lowercase while preserving all other attributes
    yield content_api.ProcessorPart(
        part.text.lower(),
        metadata=part.metadata,
        substream_name=part.substream_name,
        mimetype=part.mimetype,
        role=part.role,
    )
