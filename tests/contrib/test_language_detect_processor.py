"""Tests for LanguageDetectProcessor."""

import pytest
import pytest_asyncio
from genai_processors.contrib.language_detect_processor import (
    LanguageDetectProcessor,
    ProcessorPart,
)

# Configure pytest-asyncio
pytestmark = pytest.mark.asyncio


async def async_part_generator(parts):
    """Helper to convert list to async generator."""
    for part in parts:
        yield part


async def test_detect_english():
    """Test detection of English text."""
    processor = LanguageDetectProcessor()
    part = ProcessorPart(
        text="This is a test in English language.",
        mimetype="text/plain",
        metadata={}
    )

    result = [p async for p in processor(async_part_generator([part]))]

    assert len(result) == 1
    assert result[0].metadata["language"] == "en"


async def test_detect_french():
    """Test detection of French text."""
    processor = LanguageDetectProcessor()
    part = ProcessorPart(
        text="Bonjour, comment allez-vous? Ceci est un texte en français.",
        mimetype="text/plain",
        metadata={}
    )

    result = [p async for p in processor(async_part_generator([part]))]

    assert len(result) == 1
    assert result[0].metadata["language"] == "fr"


async def test_detect_bengali():
    """Test detection of Bengali text."""
    processor = LanguageDetectProcessor()
    part = ProcessorPart(
        text="আমি বাংলায় কথা বলি। এটি একটি বাংলা পাঠ্য।",
        mimetype="text/plain",
        metadata={}
    )

    result = [p async for p in processor(async_part_generator([part]))]

    assert len(result) == 1
    assert result[0].metadata["language"] == "bn"


async def test_short_text_unknown():
    """Test that very short text returns unknown."""
    processor = LanguageDetectProcessor(min_text_length=5)
    part = ProcessorPart(
        text="Hi",
        mimetype="text/plain",
        metadata={}
    )

    result = [p async for p in processor(async_part_generator([part]))]

    assert len(result) == 1
    assert result[0].metadata["language"] == "unknown"


async def test_empty_text_unknown():
    """Test that empty text returns unknown."""
    processor = LanguageDetectProcessor()
    part = ProcessorPart(
        text="",
        mimetype="text/plain",
        metadata={}
    )

    result = [p async for p in processor(async_part_generator([part]))]

    assert len(result) == 1
    assert result[0].metadata["language"] == "unknown"


async def test_non_text_part_unchanged():
    """Test that non-text parts are passed through unchanged."""
    processor = LanguageDetectProcessor()
    part = ProcessorPart(
        text=None,
        mimetype="image/jpeg",
        metadata={"some": "data"}
    )

    result = [p async for p in processor(async_part_generator([part]))]

    assert len(result) == 1
    assert "language" not in result[0].metadata
    assert result[0].metadata["some"] == "data"


async def test_preserves_existing_metadata():
    """Test that existing metadata is preserved."""
    processor = LanguageDetectProcessor()
    part = ProcessorPart(
        text="This is English text.",
        mimetype="text/plain",
        metadata={"author": "John", "date": "2024-01-01"}
    )

    result = [p async for p in processor(async_part_generator([part]))]

    assert len(result) == 1
    assert result[0].metadata["language"] == "en"
    assert result[0].metadata["author"] == "John"
    assert result[0].metadata["date"] == "2024-01-01"


async def test_multiple_parts():
    """Test processing multiple parts in a stream."""
    processor = LanguageDetectProcessor()
    parts = [
        ProcessorPart(text="This is English.", mimetype="text/plain", metadata={}),
        ProcessorPart(text="Bonjour le monde.", mimetype="text/plain", metadata={}),
        ProcessorPart(text=None, mimetype="image/png", metadata={}),
        ProcessorPart(text="这是中文。", mimetype="text/plain", metadata={}),
    ]

    result = [p async for p in processor(async_part_generator(parts))]

    assert len(result) == 4
    assert result[0].metadata["language"] == "en"
    assert result[1].metadata["language"] == "fr"
    assert "language" not in result[2].metadata  # image, no language
    assert result[3].metadata["language"] in ["zh-cn", "zh-tw"]
