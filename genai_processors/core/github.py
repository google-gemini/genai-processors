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
"""Processor for extracting file content from GitHub."""

from collections.abc import AsyncIterable
import dataclasses
from urllib.parse import urlparse

import dataclasses_json
from genai_processors import content_api
from genai_processors import processor
import httpx

GITHUB_URL_MIMETYPE = 'application/json; type=GithubUrl'


@dataclasses_json.dataclass_json
@dataclasses.dataclass(frozen=True)
class GithubUrl(dataclasses_json.DataClassJsonMixin):
  url: str


def parse_github_url(url: str) -> dict[str, str] | None:
  """Parses a GitHub file URL to extract owner, repo, ref (branch/tag/commit),

    and file path.
    Example URL: https://github.com/owner/repo/blob/main/path/to/file.txt

  Args:
    url: URL of file on GitHub.

  Returns:
    dict of URL component to value, if the URL can be parsed; otherwise None.
  """
  url_object = urlparse(url)
  path_parts = [part for part in url_object.path.split('/') if part]
  if (
      url_object.hostname != 'github.com'
      or len(path_parts) < 5
      or path_parts[2] != 'blob'
  ):
    return None
  [owner, repo, _, ref] = path_parts[:4]
  file_path = '/'.join(path_parts[4:])
  return {'owner': owner, 'repo': repo, 'ref': ref, 'path': file_path}


async def fetch_url(api_key: str, parsed_url: dict[str, str]) -> str | None:
  """Fetches GitHub content from a URL."""
  # See https://docs.github.com/en/rest/repos/contents#get-repository-content
  api_url = (
      f'https://api.github.com/repos/{parsed_url["owner"]}/{parsed_url["repo"]}'
      f'/contents/{parsed_url["path"]}'
  )
  headers = {
      # Get raw content directly, not JSON with base64 encoded content.
      'Accept': 'application/vnd.github.raw',
      'Authorization': f'Bearer {api_key}',
      'X-GitHub-Api-Version': '2022-11-28',
  }
  params = {'ref': parsed_url['ref']}

  async with httpx.AsyncClient() as client:
    response = await client.get(api_url, headers=headers, params=params)
    response.raise_for_status()
    return response.text


class GithubProcessor(processor.PartProcessor):
  """Processor for extracting file content from GitHub.

  Example usage:

  ```python
  from genai_processors import content_api
  from genai_processors.core import github
  from genai_processors.core import text

  processor = (
      text.UrlExtractor({'https://github.com': GithubUrl}) +
      github.GithubProcessor(api_key="YOUR_GITHUB_API_KEY")
  )
  url = content_api.Content(
      'https://github.com/owner/repo/blob/main/path/to/file.txt'
  )
  async for part in processor(url):
    # Will print the contents of file.txt, followed by "Some other text".
    print(part.text)
  ```
  """

  def __init__(self, api_key: str):
    self._api_key = api_key

  def match(self, part: content_api.ProcessorPart) -> bool:
    return part.mimetype == GITHUB_URL_MIMETYPE

  async def call(
      self,
      part: content_api.ProcessorPart,
  ) -> AsyncIterable[content_api.ProcessorPartTypes]:
    """Fetches content from a Github URL."""
    url = GithubUrl.from_json(part.text)
    parsed_url = parse_github_url(url.url)
    if not parsed_url:
      return
    content = await fetch_url(self._api_key, parsed_url)
    yield content_api.ProcessorPart(content)
