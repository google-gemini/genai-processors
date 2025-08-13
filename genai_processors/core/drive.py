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
"""Processors for extracting file content from Google Drive content.

## Example usage

### Authentication

Auth is handled either by "pre-authorizing" in Colab, e.g.:

```py
from google.colab import auth as colab_auth
colab_auth.authenticate_user()
```

Or, you will need to provide credentials directly to the processor in the
constructor, e.g.:

```py
from google.auth import credentials as google_credentials

creds = google_credentials.Credentials(
    token=YOUR_ACCESS_TOKEN,
    scopes=['https://www.googleapis.com/auth/drive']
)
p_slides = drive.Slides(creds=creds)
```

### PDF Handling

In the examples below, we pass the PDF bytes directly to the `GenaiModel`,
since the `GenaiModel` will handle the PDF conversion.

If you require more control over the PDF conversion, you could make use of
the `PDFExtract` processor.

## Docs Example

```py
from genai_processors.core import drive

USER_PROMPT = 'Describe the document in detail.'
p_docs = drive.Docs()

# Or, if you are using credentials:
# p_docs = drive.Docs(creds=creds)

p_preamble = preamble.Preamble(
    content=f'''You are an expert in reviewing Google Docs documents.

You have been provided with a document, and must use it to answer the question.

User question: {USER_PROMPT}'''
)

p_genai = genai_model.GenaiModel(
    model_name='gemini-2.5-flash',
    api_key=GOOGLE_API_KEY
)

pipeline = (
    p_docs + p_preamble + p_genai
)

req = drive.DocsRequest(doc_id='YOUR-DOC-ID')

req_part = processor.ProcessorPart.from_dataclass(dataclass=req)

input_stream = processor.stream_content([req_part])

async for content_part in pipeline(input_stream):
  print(content_part.text)
```

## Sheets Example

```py
from genai_processors.core import drive

USER_PROMPT = "Describe the spreadsheet in detail."

p_sheets = drive.Sheets()

# Or, if you are using credentials:
# p_sheets = drive.Sheets(creds=creds)

p_preamble = preamble.Preamble(
    content=f'''You are an expert in reviewing Google Sheets spreadsheets.

You have been provided with a spreadsheet, and must use it to answer the
question.

User question: {USER_PROMPT}'''
)

p_genai = genai_model.GenaiModel(
    model_name='gemini-2.5-flash',
    api_key=GOOGLE_API_KEY
)

pipeline = (
    p_sheets + p_preamble + p_genai
)

req = drive.SheetsRequest(spreadsheet_id='YOUR-SPREADSHEET-ID')

req_part = processor.ProcessorPart.from_dataclass(dataclass=req)

input_stream = processor.stream_content([req_part])

async for content_part in pipeline(input_stream):
  print(content_part.text)
```

## Slides Example

```py
from genai_processors.core import drive

USER_PROMPT = "Describe the presentation in detail."

p_slides = drive.Slides()

# Or, if you are using credentials:
# p_slides = drive.Slides(creds=creds)

p_preamble = preamble.Preamble(
    content=f'''You are an expert in reviewing Google Slides presentations.

You have been provided with slides, and must use them to answer the question.

User question: {USER_PROMPT}'''
)

p_genai = genai_model.GenaiModel(
    model_name='gemini-2.5-flash',
    api_key=GOOGLE_API_KEY
)

pipeline = (
    p_slides + p_preamble + p_genai
)

req = drive.SlidesRequest(presentation_id='YOUR-PRESENTATION-ID')

req_part = processor.ProcessorPart.from_dataclass(dataclass=req)

input_stream = processor.stream_content([req_part])

async for content_part in pipeline(input_stream):
  print(content_part.text)
```
"""
import csv
import dataclasses
import io
from typing import Any, AsyncIterable
import dataclasses_json
from genai_processors import content_api
from genai_processors import processor
from google.auth import credentials as google_credentials
from googleapiclient import discovery
import pdfrw

# Shared helper functions


def get_drive_pdf(
    file_id: str,
    creds: google_credentials.Credentials | None = None,
) -> bytes:
  """Fetches the requested file as PDF bytes.

  Args:
    file_id: The ID of the file to fetch.
    creds: The credentials to use for the request. If not provided, the
      application default credentials will be used.

  Returns:
    The requested file as PDF bytes.
  """
  service = discovery.build('drive', 'v3', credentials=creds)
  return (
      service.files()
      .export(fileId=file_id, mimeType='application/pdf')
      .execute()
  )


# Google Docs
@dataclasses_json.dataclass_json
@dataclasses.dataclass
class DocsRequest:
  """Request for fetching Google Docs data.

  Attributes:
    doc_id: The ID of the document to fetch. (e.g. for a doc with URL
      https://docs.google.com/document/d/foo, the ID is "foo")
  """

  doc_id: str


class Docs(processor.PartProcessor):
  """Processor for extracting PDF content from Google Docs."""

  def __init__(
      self,
      creds: google_credentials.Credentials | None = None,
  ) -> None:
    self._creds = creds

  def match(self, part: content_api.ProcessorPart) -> bool:
    return part.mimetype == 'application/json; type=DocsRequest'

  async def call(
      self, part: content_api.ProcessorPart
  ) -> AsyncIterable[content_api.ProcessorPart]:
    google_docs_request = part.get_dataclass(DocsRequest)
    doc_id = google_docs_request.doc_id
    doc_pdf = get_drive_pdf(
        file_id=doc_id,
        creds=self._creds,
    )
    yield content_api.ProcessorPart('Document:\n\n')
    yield content_api.ProcessorPart(doc_pdf, mimetype='application/pdf')


# Google Sheets
@dataclasses_json.dataclass_json
@dataclasses.dataclass
class SheetsRequest:
  """Request for fetching Google Sheets data.

  Attributes:
    spreadsheet_id: The ID of the spreadsheet to fetch. (e.g. for a spreadsheet
      with URL https://docs.google.com/spreadsheets/d/foo, the ID is "foo")
    ranges: Cell ranges to fetch. If not specified, the entire worksheet is
      fetched. For additional details on how to specify the ranges, see here the
      `ranges` paramter in
      https://developers.google.com/workspace/sheets/api/reference/rest/v4/spreadsheets/get
    worksheet_names: The names of the worksheets to fetch. If not provided, all
      worksheets will be fetched.
  """

  spreadsheet_id: str
  ranges: list[str] | None = None
  worksheet_names: list[str] | None = None


class Sheets(processor.PartProcessor):
  """Processor for extracting PDF content from Google Sheets."""

  def __init__(
      self,
      creds: google_credentials.Credentials | None = None,
  ) -> None:
    self._creds = creds

  def _fetch_sheet_data(
      self, sheets_request: SheetsRequest
  ) -> list[dict[str, Any]]:
    """Returns a list of dicts representing sheet data, for the given spreadsheet or range.

    The response dicts contain sheets data with the fields described here:
    https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/sheets

    Args:
      sheets_request: The SheetsRequest to fetch.

    Returns:
      A list of response dicts for the given spreadsheet or range.
    """
    service = discovery.build('sheets', 'v4', credentials=self._creds)
    spreadsheet_id = sheets_request.spreadsheet_id
    ranges = sheets_request.ranges
    return [
        service.spreadsheets()
        .get(spreadsheetId=spreadsheet_id, includeGridData=True, ranges=r)
        .execute()
        for r in ranges or [None]
    ]

  def match(self, part: content_api.ProcessorPart) -> bool:
    return part.mimetype == 'application/json; type=SheetsRequest'

  async def call(
      self, part: content_api.ProcessorPart
  ) -> AsyncIterable[content_api.ProcessorPart]:
    google_sheets_request = part.get_dataclass(SheetsRequest)
    spreadsheet_data = self._fetch_sheet_data(google_sheets_request)
    sheet_ranges = google_sheets_request.ranges
    worksheet_names = google_sheets_request.worksheet_names
    for i, res in enumerate(spreadsheet_data):
      for sheet in res['sheets']:
        try:
          title = sheet['properties']['title']
          if worksheet_names is not None and title not in worksheet_names:
            continue
          data = []
          for row in sheet['data'][0]['rowData']:
            if 'values' in row.keys():
              values = []
              for cell in row['values']:
                values.append(cell.get('formattedValue', ''))
              data.append(values)
          columns = data[0]
          num_columns = len(columns)
          rows = data[1:]
          for index in range(len(rows)):
            if len(rows[index]) > len(columns):
              rows[index] = rows[index][:num_columns]
            if len(rows[index]) < len(columns):
              rows[index] = rows[index] + [''] * (
                  num_columns - len(rows[index])
              )
          output = io.StringIO()
          writer = csv.writer(output, lineterminator='\n')
          writer.writerow(columns)
          writer.writerows(rows)
          range_or_title = sheet_ranges[i] if sheet_ranges else title
          yield content_api.ProcessorPart(f'Sheet {range_or_title}:\n\n')
          yield content_api.ProcessorPart(
              output.getvalue(), mimetype='text/csv'
          )
        except (ValueError, TypeError, IndexError):
          yield content_api.ProcessorPart('Failed to parse sheet data.')


# Google Slides
@dataclasses_json.dataclass_json
@dataclasses.dataclass
class SlidesRequest:
  """Request for fetching Google Slides data.

  Attributes:
    presentation_id: The ID of the presentation to fetch. (e.g. for a
      presentation with URL https://docs.google.com/presentation/d/foo, the ID
      is "foo")
    slide_numbers: The slide numbers to fetch, indexed from 1. If not provided,
      all slides will be fetched.
  """

  presentation_id: str
  slide_numbers: list[int] | None = None


class Slides(processor.PartProcessor):
  """Processor for extracting PDF content from Google Slides."""

  def __init__(
      self,
      creds: google_credentials.Credentials | None = None,
  ) -> None:
    self._creds = creds

  def _get_presentation_pdf_by_slide(
      self,
      presentation_id: str,
      slide_numbers: list[int] | None = None,
  ) -> dict[int, bytes]:
    """Fetches the requested presentation as a dictionary of slide numbers to PDF bytes.

    Args:
      presentation_id: The ID of the presentation to fetch.
      slide_numbers: (Optional) A list of slide numbers to fetch (indexed from
        1). If not provided, all slides will be fetched.

    Returns:
      A dictionary of slide numbers to PDF bytes.
    """
    pdf_bytes = get_drive_pdf(
        file_id=presentation_id,
        creds=self._creds,
    )
    read_pdf = pdfrw.PdfReader(io.BytesIO(pdf_bytes))
    num_slides = read_pdf.numPages
    slide_dict = {}
    for i in range(num_slides):
      if slide_numbers and i + 1 not in slide_numbers:
        continue
      slide_writer = pdfrw.PdfWriter()
      slide_writer.addPage(read_pdf.getPage(i))
      slide_bytes_stream = io.BytesIO()
      slide_writer.write(slide_bytes_stream)
      slide_bytes_stream.seek(0)
      slide_bytes = slide_bytes_stream.read()
      slide_dict[i + 1] = slide_bytes
    return slide_dict

  def match(self, part: content_api.ProcessorPart) -> bool:
    return part.mimetype == 'application/json; type=SlidesRequest'

  async def call(
      self, part: content_api.ProcessorPart
  ) -> AsyncIterable[content_api.ProcessorPart]:
    google_slides_request = part.get_dataclass(SlidesRequest)
    presentation_id = google_slides_request.presentation_id
    slide_numbers = google_slides_request.slide_numbers
    presentation_pdfs = self._get_presentation_pdf_by_slide(
        presentation_id=presentation_id,
        slide_numbers=slide_numbers,
    )
    for slide_num, pdf_bytes in presentation_pdfs.items():
      yield content_api.ProcessorPart(f"""Slide {slide_num}:\n\n""")
      yield content_api.ProcessorPart(pdf_bytes, mimetype='application/pdf')
