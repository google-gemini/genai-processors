# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
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
"""Processors for local filesystem operations."""

import asyncio
import glob
import mimetypes
import os
import re
from typing import AsyncIterable

import aiofiles
from genai_processors import processor
from google.genai import types as genai_types


def _natural_sort_key(s: str) -> list[str | int]:
  """Key for natural sorting based on filename."""
  _, filename = os.path.split(s)
  return [
      int(text) if text.isdigit() else text.lower()
      for text in re.split("([0-9]+)", filename)
  ]


@processor.source(stop_on_first=False)
async def GlobSource(  # pylint: disable=invalid-name
    pattern: str,
    base_dir: str = ".",
    inline_file_data: bool = True,
) -> AsyncIterable[processor.ProcessorPartTypes]:
  """A source that yields files matching a glob pattern.

  Can be used to load context from files or to process a stream of them. In
  particular can be applied to process photos from a camera as they need to be
  treated as a stream to detect duplicates and select the best shots.

  Args:
    pattern: The glob pattern to match. This can be a path with a pattern.
    base_dir: The base directory to search from.
    inline_file_data: If True, the file content is loaded into the part.
      Otherwise, the part will contain a FileData object pointing to the file.
  """
  glob_pattern = os.path.join(base_dir, pattern)
  filenames = await asyncio.to_thread(glob.glob, glob_pattern, recursive=True)
  filenames.sort(key=_natural_sort_key)
  for filename in filenames:
    if os.path.isfile(filename):
      original_file_name = os.path.relpath(filename, base_dir)
      mimetype, _ = mimetypes.guess_type(filename)
      if not mimetype:
        raise ValueError(f"Could not guess MIME type of {filename!r}.")
      if inline_file_data:
        async with aiofiles.open(filename, "rb") as f:
          yield processor.ProcessorPart(
              await f.read(),
              mimetype=mimetype,
              metadata={"original_file_name": original_file_name},
          )
      else:
        yield processor.ProcessorPart(
            genai_types.Part(
                file_data=genai_types.FileData(
                    display_name=original_file_name,
                    file_uri=filename,
                    mime_type=mimetype,
                )
            ),
            mimetype=mimetype,
            metadata={"original_file_name": original_file_name},
        )
