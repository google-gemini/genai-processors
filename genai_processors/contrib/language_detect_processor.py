"""Language detection processor for text parts."""

from typing import AsyncIterator, Any, Dict, Optional

try:
    from langdetect import detect, DetectorFactory, LangDetectException
    # Set seed for consistent results
    DetectorFactory.seed = 0
except ImportError as e:
    raise ImportError(
        "langdetect is required for LanguageDetectProcessor. "
        "Install it with: pip install 'genai-processors[contrib]' or pip install langdetect"
    ) from e


# Define minimal classes if they don't exist
try:
    from genai_processors.processor import Processor, ProcessorPart
except ImportError:
    class ProcessorPart:
        """Minimal ProcessorPart for language detection."""
        def __init__(self, text: Optional[str] = None, mimetype: str = "text/plain",
                     metadata: Optional[Dict[str, Any]] = None, **kwargs):
            self.text = text
            self.mimetype = mimetype
            self.metadata = metadata or {}


def is_text(mimetype: str) -> bool:
    """Check if mimetype is text-based."""
    return mimetype and (mimetype.startswith("text/") or "text" in mimetype.lower())


class LanguageDetectProcessor:
    """Detects the language of text parts and adds it to metadata.

    This processor automatically detects the language of text parts using
    the langdetect library and adds the detected language code (e.g., "en", "fr", "zh")
    to the part's metadata.

    Args:
        metadata_key: The metadata key to store the detected language.
            Defaults to "language".
        unknown_label: The label to use when language detection fails.
            Defaults to "unknown".
        min_text_length: Minimum text length required for detection.
            Shorter texts will be labeled as unknown. Defaults to 3.

    Example:
        ```python
        from genai_processors.contrib import LanguageDetectProcessor

        processor = LanguageDetectProcessor()
        async for part in processor(part_stream):
            print(part.metadata["language"])  # e.g., "en", "fr", "bn"
        ```
    """

    def __init__(
        self,
        metadata_key: str = "language",
        unknown_label: str = "unknown",
        min_text_length: int = 3,
    ):
        """Initialize the LanguageDetectProcessor.

        Args:
            metadata_key: Metadata key for storing detected language.
            unknown_label: Label for unknown/undetectable languages.
            min_text_length: Minimum text length for detection.
        """
        self.metadata_key = metadata_key
        self.unknown_label = unknown_label
        self.min_text_length = min_text_length

    def _detect_language(self, text: str) -> str:
        """Detect the language of the given text.

        Args:
            text: The text to detect language for.

        Returns:
            ISO 639-1 language code (e.g., "en", "fr", "zh") or unknown_label.
        """
        if not text or len(text.strip()) < self.min_text_length:
            return self.unknown_label

        try:
            return detect(text)
        except (LangDetectException, Exception):
            return self.unknown_label

    async def process(
        self, part_stream: AsyncIterator[ProcessorPart]
    ) -> AsyncIterator[ProcessorPart]:
        """Process parts and add language detection to text parts.

        Args:
            part_stream: Stream of ProcessorPart objects.

        Yields:
            ProcessorPart objects with language metadata added to text parts.
        """
        async for part in part_stream:
            if is_text(part.mimetype) and part.text is not None:
                # Detect language and add to metadata
                detected_language = self._detect_language(part.text)
                part.metadata[self.metadata_key] = detected_language

            yield part

    async def __call__(
        self, part_stream: AsyncIterator[ProcessorPart]
    ) -> AsyncIterator[ProcessorPart]:
        """Allow calling the processor directly."""
        async for part in self.process(part_stream):
            yield part
