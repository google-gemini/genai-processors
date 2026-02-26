import http
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from genai_processors import content_api
from genai_processors.core import github
import httpx


class GithubProcessorTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  async def test_ignores_non_url_parts(self):
    p = github.GithubProcessor(api_key='unused')
    self.assertEqual(await p('normal text').text(), 'normal text')

  async def test_fetches_github_content(self):
    url = content_api.ProcessorPart.from_dataclass(
        dataclass=github.GithubUrl(
            url='https://github.com/owner/repo/blob/main/path/to/file.txt'
        )
    )
    p = github.GithubProcessor(api_key='user_api_key')

    def request_handler(request: httpx.Request):
      # Test that the request to Github is the correctly parsed URL.
      expected_request_url = (
          'https://api.github.com/repos/owner/repo/contents/path/to/file.txt'
          '?ref=main'
      )
      self.assertEqual(str(request.url), expected_request_url)
      # Test that the request correctly applies the user's API key.
      self.assertEqual(request.headers['Authorization'], 'Bearer user_api_key')

      # Return a successful response.
      return httpx.Response(
          http.HTTPStatus.OK, content='File content'.encode('utf-8')
      )

    mock_client = httpx.AsyncClient(
        transport=httpx.MockTransport(request_handler)
    )
    with mock.patch.object(httpx, 'AsyncClient', return_value=mock_client):
      self.assertEqual(await p(url).text(), 'File content')


if __name__ == '__main__':
  absltest.main()
