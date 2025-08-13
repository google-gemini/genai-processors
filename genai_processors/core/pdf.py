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

"""Processor extracting parts from PDF documents (using pypdfium2).

Requires pypdfium2 to be installed:

```
pip install pypdfium2
```

See `pdf_cli.py` for an example of usage and to test this processor with a local
PDF file.
"""

import asyncio
from collections.abc import AsyncIterable
import io
import threading
from genai_processors import content_api
from genai_processors import processor
from PIL import Image
import pypdfium2 as pdfium


PDF_MIMETYPE = 'application/pdf'


def is_pdf(part: content_api.ProcessorPart) -> bool:
  return part.mimetype == PDF_MIMETYPE


def _png_from_pil(img: Image.Image) -> bytes:
  """Converts PIL Image to PNG bytes."""
  img_byte_arr = io.BytesIO()
  img.convert('RGB').save(img_byte_arr, format='png')
  return img_byte_arr.getvalue()


# PDFium is not thread-safe.
_PDFIUM_RENDERER_LOCK = threading.Lock()


class PDFExtract(processor.PartProcessor):
  """Extracts PDFs into content parts.

  Takes the PDF file bytes and extracts the text and images from it.
  The images are rendered as PNGs.
  The text is rendered as UTF-8.
  The PDF is rendered as a list of ProcessorParts.
  The PDF is rendered with pypdfium2.

  Note that the input parts with the PDF file bytes should also have a metadata
  field with the `original_file_name` key. This is used to generate the status
  messages.
  """

  def _format(
      self, file_name: str, pdf_doc: pdfium.PdfDocument
  ) -> tuple[content_api.ProcessorContent, tuple[str, int, int]]:
    total_pages_with_images = 0

    content = content_api.ProcessorContent()
    content += f'--- START OF PDF {file_name} ---\n\n'

    for i, page in enumerate(pdf_doc):
      # We first find out if there are any images on this page.
      # FPDF_PAGEOBJ_IMAGE is for images,
      # FPDF_PAGEOBJ_FORM is for images embedded as PDFs.
      images = page.get_objects(
          filter=(
              pdfium.raw.FPDF_PAGEOBJ_IMAGE,
              pdfium.raw.FPDF_PAGEOBJ_FORM,
              pdfium.raw.FPDF_PAGEOBJ_PATH,
          ),
          max_depth=0,
      )
      images = list(images)
      # If there are images on this page, we pass in a screenshot / render of
      # the page along with the raw text.
      if images:
        total_pages_with_images += 1
        screenshot_pil = page.render().to_pil()
        content += f'---- Screenshot for PAGE {i + 1} ----\n\n'
        content += content_api.ProcessorPart(screenshot_pil)
      content += f'--- PAGE {i + 1} ----\n\n'
      all_text_on_page = page.get_textpage().get_text_bounded()
      content += all_text_on_page
      page.close()

    return content, (file_name, len(pdf_doc), total_pages_with_images)

  def _process(
      self,
      part: content_api.ProcessorPart,
  ) -> tuple[content_api.ProcessorContent, tuple[str, int, int]]:
    # pdfium is not thread safe.
    with _PDFIUM_RENDERER_LOCK:
      file_name = part.get_metadata('original_file_name') or 'unknown.pdf'
      pdf_doc = pdfium.PdfDocument(part.bytes)
      try:
        return self._format(file_name, pdf_doc)
      finally:
        pdf_doc.close()

  def match(self, part: content_api.ProcessorPart) -> bool:
    return is_pdf(part)

  async def call(
      self, part: content_api.ProcessorPart
  ) -> AsyncIterable[content_api.ProcessorPart]:

    content, stats = await asyncio.to_thread(self._process, part)

    filename, pages, pages_with_images = stats
    yield processor.status(
        'Parsed PDF'
        f' {filename} ({pages} pages,'
        f' {pages_with_images} pages with images)'
    )

    for part in content.all_parts:
      yield part
