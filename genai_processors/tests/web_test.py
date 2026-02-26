import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from genai_processors import content_api
from genai_processors import mime_types
from genai_processors import processor
from genai_processors.core import text
from genai_processors.core import web
import httpx


class WebTest(unittest.IsolatedAsyncioTestCase, parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name='html',
          cleaning_mode='html',
          expected_output=(
              '<html>\n <body>\n  <h1>\n   Test\n  </h1>\n </body>\n</html>'
          ),
      ),
      dict(
          testcase_name='text',
          cleaning_mode='plain',
          expected_output='Test',
      ),
  ])
  async def test_url_fetch_html(self, cleaning_mode, expected_output):
    url = 'http://example.com'

    def request_handler(request: httpx.Request):
      self.assertEqual(request.url, url)
      return httpx.Response(
          status_code=200,
          content='<html><body><h1>Test</h1></body></html>',
      )

    mock_client = httpx.AsyncClient(
        transport=httpx.MockTransport(request_handler)
    )

    with mock.patch.object(httpx, 'AsyncClient', return_value=mock_client):

      fetch_request = text.FetchRequest(url=url)
      p = web.UrlFetch() + text.HtmlCleaner(cleaning_mode=cleaning_mode)
      output = await p(
          content_api.ProcessorPart.from_dataclass(fetch_request)
      ).gather()
      expected_output_parts = [
          content_api.ProcessorPart(
              f'Fetch result for URL: {url}\n',
              mimetype='text/plain',
          ),
          content_api.ProcessorPart(
              expected_output,
              mimetype=f'text/{cleaning_mode}',
          ),
      ]
      self.assertEqual(
          output,
          expected_output_parts,
      )

  async def test_url_fetch_error(self):
    url = 'http://example.com'

    def request_handler(request: httpx.Request):
      self.assertEqual(request.url, url)
      return httpx.Response(
          status_code=404,
          content='404: Not Found',
      )

    mock_client = httpx.AsyncClient(
        transport=httpx.MockTransport(request_handler)
    )
    with mock.patch.object(httpx, 'AsyncClient', return_value=mock_client):
      fetch_request = text.FetchRequest(url=url)
      p = web.UrlFetch()
      output = await p(
          content_api.ProcessorPart.from_dataclass(fetch_request)
      ).gather()
      self.assertLen(output, 2)
      self.assertEqual(
          output[0],
          content_api.ProcessorPart(
              f'Fetch result for URL: {url}\n',
              mimetype='text/plain',
          ),
      )
      self.assertTrue(mime_types.is_exception(output[1].mimetype))
      self.assertStartsWith(
          output[1].text, 'An unexpected error occurred: HTTPStatusError'
      )
      self.assertStartsWith(output[1].substream_name, processor.STATUS_STREAM)


if __name__ == '__main__':
  absltest.main()
