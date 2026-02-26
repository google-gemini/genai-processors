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
"""Tests for Google Drive processors."""
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from genai_processors import content_api
from genai_processors.core import drive

FAKE_PDF_BYTES = b'%PDF-1.4 fake pdf content'
FAKE_DOC_ID = 'fake-doc-id'
FAKE_SPREADSHEET_ID = 'fake-spreadsheet-id'
FAKE_PRESENTATION_ID = 'fake-presentation-id'


class GoogleDriveProcessorTestBase(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):
  """Base class for Google Drive processor tests with shared mock setup."""

  def setUp(self):
    super().setUp()
    self.mock_build_patcher = mock.patch(
        'googleapiclient.discovery.build', autospec=True
    )
    self.mock_build = self.mock_build_patcher.start()
    self.addCleanup(self.mock_build_patcher.stop)

    self.mock_service = self.mock_build.return_value
    self.mock_creds = mock.Mock()


class DocsProcessorTest(GoogleDriveProcessorTestBase):

  def setUp(self):
    super().setUp()
    self.mock_files = self.mock_service.files.return_value
    self.mock_export = self.mock_files.export.return_value
    self.mock_export.execute.return_value = FAKE_PDF_BYTES

  async def test_ignores_non_matching_part(self):
    """Tests that the processor ignores parts with non-matching mimetypes."""
    p = drive.Docs(creds=self.mock_creds)
    non_matching_part = content_api.ProcessorPart('some text')

    self.assertEqual(
        await p(non_matching_part).gather(),
        [non_matching_part],
    )
    self.mock_build.assert_not_called()

  async def test_fetches_doc_as_pdf(self):
    """Tests that Docs processor fetches a doc and returns it as a PDF."""
    p = drive.Docs(creds=self.mock_creds)
    req = drive.DocsRequest(doc_id=FAKE_DOC_ID)
    req_part = content_api.ProcessorPart.from_dataclass(req)

    output = await p(req_part).gather()

    self.mock_build.assert_called_once_with(
        'drive', 'v3', credentials=self.mock_creds
    )
    self.mock_files.export.assert_called_once_with(
        fileId=FAKE_DOC_ID, mimeType='application/pdf'
    )
    self.assertLen(output, 2)
    self.assertEqual(output[0].text, 'Document:\n\n')
    self.assertEqual(output[1].mimetype, 'application/pdf')
    self.assertEqual(output[1].bytes, FAKE_PDF_BYTES)


class SheetsProcessorTest(GoogleDriveProcessorTestBase):

  def setUp(self):
    super().setUp()
    self.mock_spreadsheets = self.mock_service.spreadsheets.return_value
    self.mock_get = self.mock_spreadsheets.get.return_value

  @parameterized.named_parameters(
      (
          'all_sheets',
          drive.SheetsRequest(spreadsheet_id=FAKE_SPREADSHEET_ID),
          [
              ('Sheet Sheet1:\n\n', 'text/plain'),
              ('Name,Age\nAlice,30\nBob,\nCharlie,25\n', 'text/csv'),
              ('Sheet Sheet2:\n\n', 'text/plain'),
              ('City,Country\nParis,France\n', 'text/csv'),
          ],
      ),
      (
          'filtered_sheets',
          drive.SheetsRequest(
              spreadsheet_id=FAKE_SPREADSHEET_ID, worksheet_names=['Sheet2']
          ),
          [
              ('Sheet Sheet2:\n\n', 'text/plain'),
              ('City,Country\nParis,France\n', 'text/csv'),
          ],
      ),
  )
  async def test_fetches_sheet_as_csv(self, request, expected_outputs):
    """Tests that Sheets processor fetches sheet data and returns it as CSV."""
    self.mock_build.reset_mock()
    self.mock_get.reset_mock()

    self.mock_get.execute.return_value = {
        'sheets': [
            {
                'properties': {'title': 'Sheet1'},
                'data': [{
                    'rowData': [
                        {
                            'values': [
                                {'formattedValue': 'Name'},
                                {'formattedValue': 'Age'},
                            ]
                        },
                        {
                            'values': [
                                {'formattedValue': 'Alice'},
                                {'formattedValue': '30'},
                            ]
                        },
                        {'values': [{'formattedValue': 'Bob'}]},
                        {
                            'values': [
                                {'formattedValue': 'Charlie'},
                                {'formattedValue': '25'},
                                {'formattedValue': 'Extra'},
                            ]
                        },
                    ]
                }],
            },
            {
                'properties': {'title': 'Sheet2'},
                'data': [{
                    'rowData': [
                        {
                            'values': [
                                {'formattedValue': 'City'},
                                {'formattedValue': 'Country'},
                            ]
                        },
                        {
                            'values': [
                                {'formattedValue': 'Paris'},
                                {'formattedValue': 'France'},
                            ]
                        },
                    ]
                }],
            },
        ]
    }
    p = drive.Sheets(creds=self.mock_creds)
    req_part = content_api.ProcessorPart.from_dataclass(request)

    output = await p(req_part).gather()

    self.mock_build.assert_called_once_with(
        'sheets', 'v4', credentials=self.mock_creds
    )
    self.mock_spreadsheets.get.assert_called_once_with(
        spreadsheetId=request.spreadsheet_id, includeGridData=True, ranges=None
    )
    self.assertLen(output, len(expected_outputs))
    for part, (expected_text, expected_mimetype) in zip(
        output, expected_outputs
    ):
      self.assertEqual(part.text, expected_text)
      self.assertEqual(part.mimetype, expected_mimetype)

  async def test_handles_parse_failure(self):
    """Tests that a failure to parse sheet data is handled gracefully."""
    # Return data that will cause an IndexError
    self.mock_get.execute.return_value = {
        'sheets': [{'properties': {'title': 'BadSheet'}, 'data': []}]
    }
    p = drive.Sheets(creds=self.mock_creds)
    req = drive.SheetsRequest(spreadsheet_id=FAKE_SPREADSHEET_ID)
    req_part = content_api.ProcessorPart.from_dataclass(req)

    output = await p(req_part).gather()

    self.assertLen(output, 1)
    self.assertEqual(output[0].text, 'Failed to parse sheet data.')


class SlidesProcessorTest(GoogleDriveProcessorTestBase):

  def setUp(self):
    super().setUp()
    self.mock_files = self.mock_service.files.return_value
    self.mock_export = self.mock_files.export

    self.mock_pdf_reader_patcher = mock.patch('pdfrw.PdfReader', autospec=True)
    self.mock_pdf_reader_cls = self.mock_pdf_reader_patcher.start()
    self.addCleanup(self.mock_pdf_reader_patcher.stop)

    self.mock_pdf_writer_patcher = mock.patch('pdfrw.PdfWriter', autospec=True)
    self.mock_pdf_writer_cls = self.mock_pdf_writer_patcher.start()
    self.addCleanup(self.mock_pdf_writer_patcher.stop)

    self.mock_pdf_reader = self.mock_pdf_reader_cls.return_value
    self.mock_pdf_reader.numPages = 3
    self.mock_pdf_reader.getPage.return_value = mock.Mock()
    self.mock_pdf_writer = self.mock_pdf_writer_cls.return_value

  @parameterized.named_parameters(
      (
          'all_slides',
          drive.SlidesRequest(presentation_id=FAKE_PRESENTATION_ID),
          [mock.call(0), mock.call(1), mock.call(2)],
          [
              ('Slide 1:\n\n', b'processed_page_0'),
              ('Slide 2:\n\n', b'processed_page_1'),
              ('Slide 3:\n\n', b'processed_page_2'),
          ],
      ),
      (
          'filtered_slides',
          drive.SlidesRequest(
              presentation_id=FAKE_PRESENTATION_ID, slide_numbers=[1, 3]
          ),
          [mock.call(0), mock.call(2)],
          [
              ('Slide 1:\n\n', b'processed_page_0'),
              ('Slide 3:\n\n', b'processed_page_1'),
          ],
      ),
  )
  async def test_fetches_slides_as_pdfs(
      self, request, expected_getpage_calls, expected_outputs
  ):
    """Tests that Slides processor fetches slides and returns them as PDFs."""
    self.mock_build.reset_mock()
    self.mock_export.reset_mock()
    self.mock_pdf_reader_cls.reset_mock()

    self.mock_export.return_value.execute.return_value = FAKE_PDF_BYTES
    self.mock_pdf_reader_cls.return_value = self.mock_pdf_reader

    write_call_counter = 0

    def write_side_effect(stream):
      nonlocal write_call_counter
      stream.write(f'processed_page_{write_call_counter}'.encode())
      write_call_counter += 1

    self.mock_pdf_writer.write.side_effect = write_side_effect
    p = drive.Slides(creds=self.mock_creds)
    req_part = content_api.ProcessorPart.from_dataclass(request)

    output = await p(req_part).gather()

    self.mock_build.assert_called_once_with(
        'drive', 'v3', credentials=self.mock_creds
    )
    self.mock_export.assert_called_once_with(
        fileId=FAKE_PRESENTATION_ID, mimeType='application/pdf'
    )
    self.mock_pdf_reader_cls.assert_called_once()
    self.assertEqual(
        self.mock_pdf_reader_cls.call_args[0][0].read(), FAKE_PDF_BYTES
    )
    self.mock_pdf_reader.getPage.assert_has_calls(expected_getpage_calls)
    self.assertLen(
        expected_getpage_calls, self.mock_pdf_reader.getPage.call_count
    )

    self.assertLen(output, len(expected_outputs) * 2)
    for i, (expected_text, expected_bytes) in enumerate(expected_outputs):
      self.assertEqual(output[i * 2].text, expected_text)
      self.assertEqual(output[i * 2 + 1].mimetype, 'application/pdf')
      self.assertEqual(output[i * 2 + 1].bytes, expected_bytes)


if __name__ == '__main__':
  absltest.main()
