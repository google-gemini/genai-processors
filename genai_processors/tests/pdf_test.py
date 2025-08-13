import concurrent.futures
import time
from unittest import mock

from absl.testing import absltest
from genai_processors import content_api
from genai_processors import processor
from genai_processors.core import pdf
from PIL import Image


class PDFExtractTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.pdf_processor = pdf.PDFExtract()

  def test_apply_non_pdf_part(self):
    """Test that non-PDF parts are passed through unchanged."""
    part = content_api.ProcessorPart(b'test data', mimetype='text/plain')

    processed_parts = processor.apply_sync(
        self.pdf_processor.to_processor(), [part]
    )

    self.assertLen(processed_parts, 1)
    self.assertEqual(processed_parts[0], part)

  @mock.patch('pypdfium2.PdfDocument')
  def test_apply_pdf_part_no_images(self, mock_pdf_document):
    """Test that PDF parts with no images are processed correctly."""
    mock_page = mock.Mock()
    mock_page.get_objects.return_value = []
    mock_page.get_textpage.return_value.get_text_bounded.return_value = (
        'page text'
    )
    mock_pdf_document.return_value.__iter__.return_value = [mock_page]
    mock_pdf_document.return_value.__len__.return_value = 1

    metadata = {'original_file_name': 'test.pdf'}
    part = content_api.ProcessorPart(
        b'pdf data', mimetype=pdf.PDF_MIMETYPE, metadata=metadata
    )

    processed_parts = processor.apply_sync(
        self.pdf_processor.to_processor(), [part]
    )

    # Assert start and end markers, page header, page text
    content_parts = [c for c in processed_parts if not c.substream_name]
    status_parts = [c for c in processed_parts if c.substream_name == 'status']
    self.assertEqual(content_parts[0].text, '--- START OF PDF test.pdf ---\n\n')
    self.assertEqual(content_parts[1].text, '--- PAGE 1 ----\n\n')
    self.assertEqual(content_parts[2].text, 'page text')
    self.assertEqual(
        status_parts[0].text,
        'Parsed PDF test.pdf (1 pages, 0 pages with images)',
    )

  @mock.patch('pypdfium2.PdfDocument')
  def test_apply_pdf_part_with_images(self, mock_pdf_document):
    """Test that PDF parts with images are processed correctly."""
    mock_page = mock.Mock()
    mock_page.get_objects.return_value = [mock.Mock()]
    mock_page.get_textpage.return_value.get_text_bounded.return_value = (
        'page text'
    )
    mock_page.render.return_value.to_pil.return_value = Image.new(
        'RGB', (100, 100)
    )
    mock_pdf_document.return_value.__iter__.return_value = [mock_page]
    mock_pdf_document.return_value.__len__.return_value = 1

    metadata = {'original_file_name': 'test.pdf'}
    part = content_api.ProcessorPart(
        b'pdf data', mimetype=pdf.PDF_MIMETYPE, metadata=metadata
    )

    processed_parts = processor.apply_sync(
        self.pdf_processor.to_processor(), [part]
    )

    # Assert start and end markers, page header, screenshot messages, screenshot
    # parts, page text
    content_parts = [c for c in processed_parts if not c.substream_name]
    self.assertEqual(content_parts[0].text, '--- START OF PDF test.pdf ---\n\n')
    self.assertEqual(
        content_parts[1].text, '---- Screenshot for PAGE 1 ----\n\n'
    )
    self.assertEqual(content_parts[2].mimetype, 'image/webp')
    self.assertEqual(content_parts[3].text, '--- PAGE 1 ----\n\n')
    self.assertEqual(content_parts[4].text, 'page text')

    status_parts = [c for c in processed_parts if c.substream_name == 'status']
    self.assertEqual(
        status_parts[0].text,
        'Parsed PDF test.pdf (1 pages, 1 pages with images)',
    )

  @mock.patch('pypdfium2.PdfDocument')
  def test_apply_pdf_part_parallel(self, mock_pdf_document):
    """Test that multiple PDF parts processed correctly in parallel."""
    mock_page = mock.Mock()
    mock_page.get_objects.return_value = []
    mock_page.get_textpage.return_value.get_text_bounded.return_value = (
        'page text'
    )

    def doc_iter(*args, **kwargs):
      del args, kwargs
      for _ in range(3):
        yield mock_page
        time.sleep(1)

    mock_pdf_document.return_value.__iter__.side_effect = doc_iter
    metadata = {'original_file_name': 'test.pdf'}
    part = content_api.ProcessorPart(
        b'pdf data', mimetype=pdf.PDF_MIMETYPE, metadata=metadata
    )

    pdf_processor = self.pdf_processor.to_processor()
    processed_parts = processor.apply_sync(pdf_processor, [part])

    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
      processed_parts_threaded = []
      for parts in executor.map(
          processor.apply_sync,
          [pdf_processor, pdf_processor],
          [[part], [part]],
      ):
        processed_parts_threaded.extend(parts)
    end_time = time.time()

    time_elapsed = end_time - start_time

    self.assertEqual(
        processed_parts_threaded, processed_parts + processed_parts
    )

    # Ensure that lock was used and total elapsed time is 6-7s.
    self.assertGreater(time_elapsed, 6)
    self.assertLess(time_elapsed, 7)


if __name__ == '__main__':
  absltest.main()
