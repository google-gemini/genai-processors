# Copyright 2026 Google LLC. All Rights Reserved.
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
"""Tests for semantic_cache module."""

import asyncio
import unittest
from unittest import mock

from absl.testing import parameterized
from genai_processors import content_api
from genai_processors import processor
from genai_processors import streams
from genai_processors.contrib import semantic_cache


# =============================================================================
# Test Utilities
# =============================================================================


def _create_mock_embedding_client(embeddings_map: dict[str, list[float]]):
  """Create a mock embedding client that returns predefined embeddings."""
  client = mock.AsyncMock(spec=semantic_cache.GeminiEmbeddingClient)

  async def mock_embed(content):
    # Extract text from content
    text = content_api.as_text(content)
    # Return predefined embedding or a default
    for key, embedding in embeddings_map.items():
      if key.lower() in text.lower():
        return embedding
    # Default embedding with same dimension as first entry in map
    if embeddings_map:
      dim = len(next(iter(embeddings_map.values())))
      return [0.5] * dim
    return [0.5] * 768

  client.embed = mock_embed
  return client


def _create_mock_processor(response_text: str):
  """Create a mock processor that returns predefined response."""

  class MockProcessor(processor.Processor):

    def __init__(self):
      self.call_count = 0

    async def call(self, content):
      self.call_count += 1
      async for _ in content:
        pass  # Consume input
      yield content_api.ProcessorPart(
          response_text,
          role='model',
          mimetype='text/plain',
      )

  return MockProcessor()


# =============================================================================
# Cosine Similarity Tests
# =============================================================================


class CosineSimilarityTest(unittest.TestCase):
  """Tests for cosine_similarity function."""

  def test_identical_vectors(self):
    """Identical vectors should have similarity of 1.0."""
    vec = [1.0, 0.0, 0.0]
    self.assertAlmostEqual(
        semantic_cache.cosine_similarity(vec, vec), 1.0, places=5
    )

  def test_orthogonal_vectors(self):
    """Orthogonal vectors should have similarity of 0.0."""
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [0.0, 1.0, 0.0]
    self.assertAlmostEqual(
        semantic_cache.cosine_similarity(vec1, vec2), 0.0, places=5
    )

  def test_opposite_vectors(self):
    """Opposite vectors should have similarity of -1.0."""
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [-1.0, 0.0, 0.0]
    self.assertAlmostEqual(
        semantic_cache.cosine_similarity(vec1, vec2), -1.0, places=5
    )

  def test_zero_vector(self):
    """Zero vector should return similarity of 0.0."""
    vec1 = [1.0, 2.0, 3.0]
    vec2 = [0.0, 0.0, 0.0]
    self.assertEqual(semantic_cache.cosine_similarity(vec1, vec2), 0.0)

  def test_similar_vectors(self):
    """Similar vectors should have high similarity."""
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [0.9, 0.1, 0.0]
    similarity = semantic_cache.cosine_similarity(vec1, vec2)
    self.assertGreater(similarity, 0.9)

  def test_different_vectors(self):
    """Different vectors should have lower similarity."""
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [0.0, 0.0, 1.0]
    similarity = semantic_cache.cosine_similarity(vec1, vec2)
    self.assertAlmostEqual(similarity, 0.0, places=5)

  def test_normalized_vectors(self):
    """Normalized vectors should work correctly."""
    import math

    vec1 = [1.0 / math.sqrt(2), 1.0 / math.sqrt(2), 0.0]
    vec2 = [1.0, 0.0, 0.0]
    similarity = semantic_cache.cosine_similarity(vec1, vec2)
    self.assertAlmostEqual(similarity, 1.0 / math.sqrt(2), places=5)


# =============================================================================
# InMemoryVectorCache Tests
# =============================================================================


class InMemoryVectorCacheTest(unittest.IsolatedAsyncioTestCase):
  """Tests for InMemoryVectorCache."""

  async def test_store_and_find_exact(self):
    """Store an entry and find it with exact embedding."""
    cache = semantic_cache.InMemoryVectorCache()
    embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

    await cache.store(
        embedding=embedding,
        query_text='test query',
        response_parts=[content_api.ProcessorPart('test response')],
    )

    results = await cache.find_similar(
        embedding=embedding,
        threshold=0.99,
    )

    self.assertEqual(len(results), 1)
    self.assertGreater(results[0].similarity_score, 0.99)
    self.assertEqual(results[0].entry.query_text, 'test query')

  async def test_find_similar_above_threshold(self):
    """Find entry when similarity is above threshold."""
    cache = semantic_cache.InMemoryVectorCache()

    await cache.store(
        embedding=[1.0, 0.0, 0.0],
        query_text='test',
        response_parts=[content_api.ProcessorPart('response')],
    )

    # Similar embedding should match
    results = await cache.find_similar(
        embedding=[0.95, 0.05, 0.0],
        threshold=0.9,
    )

    self.assertEqual(len(results), 1)
    self.assertGreater(results[0].similarity_score, 0.9)

  async def test_find_similar_below_threshold(self):
    """Return empty list when similarity is below threshold."""
    cache = semantic_cache.InMemoryVectorCache()

    await cache.store(
        embedding=[1.0, 0.0, 0.0],
        query_text='test',
        response_parts=[content_api.ProcessorPart('response')],
    )

    # Different embedding should not match with high threshold
    results = await cache.find_similar(
        embedding=[0.0, 1.0, 0.0],
        threshold=0.9,
    )

    self.assertEqual(results, [])

  async def test_ttl_expiration(self):
    """Entries should expire after TTL."""
    cache = semantic_cache.InMemoryVectorCache(ttl_seconds=0.1)

    await cache.store(
        embedding=[1.0, 0.0, 0.0],
        query_text='test',
        response_parts=[content_api.ProcessorPart('response')],
    )

    # Entry should be found immediately
    results = await cache.find_similar(
        embedding=[1.0, 0.0, 0.0],
        threshold=0.9,
    )
    self.assertEqual(len(results), 1)

    # Wait for expiration
    await asyncio.sleep(0.2)

    # Entry should be expired
    results = await cache.find_similar(
        embedding=[1.0, 0.0, 0.0],
        threshold=0.9,
    )
    self.assertEqual(results, [])

  async def test_max_entries_eviction(self):
    """Cache should evict entries when max_entries is reached."""
    cache = semantic_cache.InMemoryVectorCache(max_entries=2)

    await cache.store([1.0, 0, 0], 'q1', [content_api.ProcessorPart('r1')])
    await cache.store([0, 1.0, 0], 'q2', [content_api.ProcessorPart('r2')])
    await cache.store([0, 0, 1.0], 'q3', [content_api.ProcessorPart('r3')])

    stats = await cache.stats()
    self.assertEqual(stats['total_entries'], 2)
    self.assertEqual(stats['evictions'], 1)

  async def test_lru_eviction_order(self):
    """LRU eviction should remove least used entries first."""
    cache = semantic_cache.InMemoryVectorCache(max_entries=2)

    # Store two entries
    await cache.store([1.0, 0, 0], 'q1', [content_api.ProcessorPart('r1')])
    await cache.store([0, 1.0, 0], 'q2', [content_api.ProcessorPart('r2')])

    # Access first entry to increase hit count
    await cache.find_similar([1.0, 0, 0], threshold=0.9)
    await cache.find_similar([1.0, 0, 0], threshold=0.9)

    # Add third entry - should evict q2 (fewer hits)
    await cache.store([0, 0, 1.0], 'q3', [content_api.ProcessorPart('r3')])

    # q1 should still be there (more hits)
    results = await cache.find_similar([1.0, 0, 0], threshold=0.9)
    self.assertEqual(len(results), 1)
    self.assertEqual(results[0].entry.query_text, 'q1')

    # q2 should be evicted
    results = await cache.find_similar([0, 1.0, 0], threshold=0.9)
    self.assertEqual(results, [])

  async def test_remove_entry(self):
    """Remove should delete specific entry."""
    cache = semantic_cache.InMemoryVectorCache()

    entry_id = await cache.store(
        embedding=[1.0, 0.0, 0.0],
        query_text='test',
        response_parts=[content_api.ProcessorPart('response')],
    )

    # Entry should exist
    results = await cache.find_similar([1.0, 0.0, 0.0], threshold=0.9)
    self.assertEqual(len(results), 1)

    # Remove entry
    removed = await cache.remove(entry_id)
    self.assertTrue(removed)

    # Entry should not exist
    results = await cache.find_similar([1.0, 0.0, 0.0], threshold=0.9)
    self.assertEqual(results, [])

  async def test_remove_nonexistent_entry(self):
    """Remove should return False for nonexistent entry."""
    cache = semantic_cache.InMemoryVectorCache()
    removed = await cache.remove('nonexistent-id')
    self.assertFalse(removed)

  async def test_clear_cache(self):
    """Clear should remove all entries."""
    cache = semantic_cache.InMemoryVectorCache()

    await cache.store([1.0, 0, 0], 'q1', [content_api.ProcessorPart('r1')])
    await cache.store([0, 1.0, 0], 'q2', [content_api.ProcessorPart('r2')])

    stats = await cache.stats()
    self.assertEqual(stats['total_entries'], 2)

    await cache.clear()

    stats = await cache.stats()
    self.assertEqual(stats['total_entries'], 0)
    self.assertEqual(stats['hits'], 0)
    self.assertEqual(stats['misses'], 0)

  async def test_stats_tracking(self):
    """Stats should track hits, misses, and stores."""
    cache = semantic_cache.InMemoryVectorCache()

    # Store entry
    await cache.store([1.0, 0, 0], 'q1', [content_api.ProcessorPart('r1')])

    # Cache hit
    await cache.find_similar([1.0, 0, 0], threshold=0.9)

    # Cache miss
    await cache.find_similar([0, 1.0, 0], threshold=0.9)

    stats = await cache.stats()
    self.assertEqual(stats['stores'], 1)
    self.assertEqual(stats['hits'], 1)
    self.assertEqual(stats['misses'], 1)
    self.assertAlmostEqual(stats['hit_rate'], 0.5, places=2)

  async def test_multiple_entries_best_match(self):
    """Should return the best matching entry when multiple exist."""
    cache = semantic_cache.InMemoryVectorCache()

    # Store multiple entries
    await cache.store([1.0, 0, 0], 'exact', [content_api.ProcessorPart('r1')])
    await cache.store(
        [0.8, 0.2, 0], 'similar', [content_api.ProcessorPart('r2')]
    )
    await cache.store(
        [0.5, 0.5, 0], 'different', [content_api.ProcessorPart('r3')]
    )

    # Query should match the most similar entry
    results = await cache.find_similar([0.95, 0.05, 0], threshold=0.8)

    self.assertEqual(len(results), 1)
    self.assertEqual(results[0].entry.query_text, 'exact')

  async def test_response_parts_serialization(self):
    """Response parts should be serialized and deserialized correctly."""
    cache = semantic_cache.InMemoryVectorCache()

    original_parts = [
        content_api.ProcessorPart(
            'Hello',
            role='model',
            mimetype='text/plain',
            metadata={'key': 'value'},
        ),
        content_api.ProcessorPart(
            'World',
            role='model',
            mimetype='text/plain',
        ),
    ]

    await cache.store(
        embedding=[1.0, 0, 0],
        query_text='test',
        response_parts=original_parts,
    )

    results = await cache.find_similar([1.0, 0, 0], threshold=0.9)

    self.assertEqual(len(results), 1)
    restored_parts = results[0].entry.get_response_parts()

    self.assertEqual(len(restored_parts), 2)
    self.assertEqual(restored_parts[0].text, 'Hello')
    self.assertEqual(restored_parts[0].role, 'model')
    self.assertEqual(restored_parts[1].text, 'World')

  async def test_find_similar_with_limit(self):
    """find_similar should return multiple results when limit > 1."""
    cache = semantic_cache.InMemoryVectorCache()

    await cache.store([1.0, 0, 0], 'q1', [content_api.ProcessorPart('r1')])
    await cache.store(
        [0.9, 0.1, 0], 'q2', [content_api.ProcessorPart('r2')]
    )
    await cache.store(
        [0.8, 0.2, 0], 'q3', [content_api.ProcessorPart('r3')]
    )

    # Request up to 3 results
    results = await cache.find_similar(
        [0.95, 0.05, 0], threshold=0.7, limit=3
    )

    self.assertEqual(len(results), 3)
    # Results should be sorted by similarity descending
    self.assertGreaterEqual(
        results[0].similarity_score, results[1].similarity_score
    )
    self.assertGreaterEqual(
        results[1].similarity_score, results[2].similarity_score
    )

  async def test_find_similar_limit_caps_results(self):
    """find_similar should respect the limit parameter."""
    cache = semantic_cache.InMemoryVectorCache()

    await cache.store([1.0, 0, 0], 'q1', [content_api.ProcessorPart('r1')])
    await cache.store(
        [0.9, 0.1, 0], 'q2', [content_api.ProcessorPart('r2')]
    )
    await cache.store(
        [0.8, 0.2, 0], 'q3', [content_api.ProcessorPart('r3')]
    )

    # Request only 1 result
    results = await cache.find_similar(
        [0.95, 0.05, 0], threshold=0.7, limit=1
    )

    self.assertEqual(len(results), 1)


# =============================================================================
# SemanticCacheProcessor Tests
# =============================================================================


class SemanticCacheProcessorTest(
    unittest.IsolatedAsyncioTestCase, parameterized.TestCase
):
  """Tests for SemanticCacheProcessor."""

  async def test_cache_miss_calls_wrapped_processor(self):
    """Cache miss should call the wrapped processor."""
    mock_cache = semantic_cache.InMemoryVectorCache()
    mock_model = _create_mock_processor('Paris is the capital.')

    proc = semantic_cache.SemanticCacheProcessor(
        wrapped_processor=mock_model,
        api_key='fake-key',
        similarity_threshold=0.9,
        cache=mock_cache,
    )

    # Replace embedding client with mock
    proc._embedding_client = _create_mock_embedding_client({
        'france': [1.0, 0, 0],
    })

    result = await processor.apply_async(
        proc, ['What is the capital of France?']
    )

    # Filter out status messages
    content_parts = [
        p for p in result if p.substream_name != processor.STATUS_STREAM
    ]

    self.assertEqual(mock_model.call_count, 1)
    self.assertEqual(len(content_parts), 1)
    self.assertIn('Paris', content_parts[0].text)

  async def test_cache_hit_skips_wrapped_processor(self):
    """Cache hit should return cached response without calling wrapped."""
    mock_cache = semantic_cache.InMemoryVectorCache()
    mock_model = _create_mock_processor('Paris is the capital.')

    proc = semantic_cache.SemanticCacheProcessor(
        wrapped_processor=mock_model,
        api_key='fake-key',
        similarity_threshold=0.85,
        cache=mock_cache,
    )

    # Mock embedding client returns similar embeddings for similar queries
    proc._embedding_client = _create_mock_embedding_client({
        'capital': [1.0, 0, 0],
        'france': [1.0, 0, 0],
    })

    # First call - cache miss
    await processor.apply_async(proc, ['What is the capital of France?'])
    self.assertEqual(mock_model.call_count, 1)

    # Second call with similar query - cache hit
    await processor.apply_async(proc, ["Tell me France's capital"])
    self.assertEqual(mock_model.call_count, 1)  # No additional call

  async def test_different_queries_both_miss(self):
    """Different queries should both result in cache misses."""
    mock_cache = semantic_cache.InMemoryVectorCache()
    mock_model = _create_mock_processor('Response')

    proc = semantic_cache.SemanticCacheProcessor(
        wrapped_processor=mock_model,
        api_key='fake-key',
        similarity_threshold=0.9,
        cache=mock_cache,
    )

    # Different embeddings for different queries
    proc._embedding_client = _create_mock_embedding_client({
        'weather': [1.0, 0, 0],
        'python': [0, 1.0, 0],
    })

    await processor.apply_async(proc, ['What is the weather?'])
    await processor.apply_async(proc, ['What is Python?'])

    self.assertEqual(mock_model.call_count, 2)

  async def test_cache_metadata_included(self):
    """Cache metadata should be included in output parts."""
    mock_cache = semantic_cache.InMemoryVectorCache()
    mock_model = _create_mock_processor('Response')

    proc = semantic_cache.SemanticCacheProcessor(
        wrapped_processor=mock_model,
        api_key='fake-key',
        similarity_threshold=0.85,
        cache=mock_cache,
        include_cache_metadata=True,
    )

    proc._embedding_client = _create_mock_embedding_client({
        'test': [1.0, 0, 0],
    })

    # First call - cache miss
    result1 = await processor.apply_async(proc, ['test query'])
    content_parts1 = [
        p for p in result1 if p.substream_name != processor.STATUS_STREAM
    ]
    self.assertFalse(content_parts1[0].metadata.get('semantic_cache_hit'))

    # Second call - cache hit
    result2 = await processor.apply_async(proc, ['test query again'])
    content_parts2 = [
        p for p in result2 if p.substream_name != processor.STATUS_STREAM
    ]
    self.assertTrue(content_parts2[0].metadata.get('semantic_cache_hit'))

  async def test_cache_metadata_disabled(self):
    """Cache metadata should not be included when disabled."""
    mock_cache = semantic_cache.InMemoryVectorCache()
    mock_model = _create_mock_processor('Response')

    proc = semantic_cache.SemanticCacheProcessor(
        wrapped_processor=mock_model,
        api_key='fake-key',
        similarity_threshold=0.9,
        cache=mock_cache,
        include_cache_metadata=False,
    )

    proc._embedding_client = _create_mock_embedding_client({
        'test': [1.0, 0, 0],
    })

    result = await processor.apply_async(proc, ['test query'])
    content_parts = [
        p for p in result if p.substream_name != processor.STATUS_STREAM
    ]

    self.assertNotIn('semantic_cache_hit', content_parts[0].metadata)

  async def test_embedding_failure_bypasses_cache(self):
    """Embedding failure should bypass cache and call wrapped processor."""
    mock_cache = semantic_cache.InMemoryVectorCache()
    mock_model = _create_mock_processor('Response')

    proc = semantic_cache.SemanticCacheProcessor(
        wrapped_processor=mock_model,
        api_key='fake-key',
        similarity_threshold=0.9,
        cache=mock_cache,
    )

    # Mock embedding client that raises exception
    mock_client = mock.AsyncMock()
    mock_client.embed.side_effect = Exception('Embedding API error')
    proc._embedding_client = mock_client

    result = await processor.apply_async(proc, ['test query'])

    # Should still get response from wrapped processor
    content_parts = [
        p for p in result if p.substream_name != processor.STATUS_STREAM
    ]
    self.assertEqual(len(content_parts), 1)
    self.assertEqual(mock_model.call_count, 1)

  async def test_empty_input_bypasses_cache(self):
    """Empty input should bypass cache via ValueError from embed."""
    mock_cache = semantic_cache.InMemoryVectorCache()
    mock_model = _create_mock_processor('Response')

    proc = semantic_cache.SemanticCacheProcessor(
        wrapped_processor=mock_model,
        api_key='fake-key',
        cache=mock_cache,
    )

    # Mock embed to raise ValueError on empty content
    mock_client = mock.AsyncMock()
    mock_client.embed.side_effect = ValueError(
        'Empty content, cannot generate embedding.'
    )
    proc._embedding_client = mock_client

    result = await processor.apply_async(proc, [])

    # Empty input triggers ValueError in embed, bypasses cache
    stats = await mock_cache.stats()
    self.assertEqual(stats['stores'], 0)

  async def test_non_text_content_bypasses_cache(self):
    """Non-text content should trigger cache miss via ValueError."""
    mock_cache = semantic_cache.InMemoryVectorCache()
    mock_model = _create_mock_processor('Response')

    proc = semantic_cache.SemanticCacheProcessor(
        wrapped_processor=mock_model,
        api_key='fake-key',
        cache=mock_cache,
    )

    # Mock embed to raise ValueError on non-text content
    mock_client = mock.AsyncMock()
    mock_client.embed.side_effect = ValueError(
        'Non-text content detected. Semantic cache only supports text.'
    )
    proc._embedding_client = mock_client

    result = await processor.apply_async(
        proc, [content_api.ProcessorPart(b'image', mimetype='image/png')]
    )

    # Non-text triggers ValueError, falls through to wrapped processor
    content_parts = [
        p for p in result if p.substream_name != processor.STATUS_STREAM
    ]
    self.assertEqual(len(content_parts), 1)
    self.assertEqual(mock_model.call_count, 1)
    stats = await mock_cache.stats()
    self.assertEqual(stats['stores'], 0)

  async def test_get_cache_stats(self):
    """get_cache_stats should return cache statistics."""
    mock_cache = semantic_cache.InMemoryVectorCache()
    mock_model = _create_mock_processor('Response')

    proc = semantic_cache.SemanticCacheProcessor(
        wrapped_processor=mock_model,
        api_key='fake-key',
        cache=mock_cache,
    )

    proc._embedding_client = _create_mock_embedding_client({
        'test': [1.0, 0, 0],
    })

    await processor.apply_async(proc, ['test'])

    stats = await proc.get_cache_stats()

    self.assertIn('total_entries', stats)
    self.assertIn('hits', stats)
    self.assertIn('misses', stats)
    self.assertEqual(stats['total_entries'], 1)

  async def test_clear_cache(self):
    """clear_cache should clear all entries."""
    mock_cache = semantic_cache.InMemoryVectorCache()
    mock_model = _create_mock_processor('Response')

    proc = semantic_cache.SemanticCacheProcessor(
        wrapped_processor=mock_model,
        api_key='fake-key',
        cache=mock_cache,
    )

    proc._embedding_client = _create_mock_embedding_client({
        'test': [1.0, 0, 0],
    })

    await processor.apply_async(proc, ['test'])

    stats = await proc.get_cache_stats()
    self.assertEqual(stats['total_entries'], 1)

    await proc.clear_cache()

    stats = await proc.get_cache_stats()
    self.assertEqual(stats['total_entries'], 0)

  @parameterized.parameters(
      (0.99, False),  # Very high threshold - should miss
      (0.50, True),   # Low threshold - should hit
  )
  async def test_threshold_behavior(self, threshold, should_hit):
    """Test that similarity threshold affects cache behavior."""
    mock_cache = semantic_cache.InMemoryVectorCache()
    mock_model = _create_mock_processor('Response')

    proc = semantic_cache.SemanticCacheProcessor(
        wrapped_processor=mock_model,
        api_key='fake-key',
        similarity_threshold=threshold,
        cache=mock_cache,
    )

    # First query gets embedding [1, 0, 0]
    # Second query gets embedding [0.8, 0.2, 0] (similarity ~0.97)
    embeddings = {
        'first': [1.0, 0, 0],
        'second': [0.8, 0.2, 0],
    }
    proc._embedding_client = _create_mock_embedding_client(embeddings)

    # First call - always miss
    await processor.apply_async(proc, ['first query'])
    self.assertEqual(mock_model.call_count, 1)

    # Second call - depends on threshold
    await processor.apply_async(proc, ['second query'])

    if should_hit:
      self.assertEqual(mock_model.call_count, 1)  # Cache hit
    else:
      self.assertEqual(mock_model.call_count, 2)  # Cache miss


# =============================================================================
# SemanticCachePartProcessor Tests
# =============================================================================


class SemanticCachePartProcessorTest(unittest.IsolatedAsyncioTestCase):
  """Tests for SemanticCachePartProcessor."""

  async def test_cache_miss_calls_wrapped_processor(self):
    """Cache miss should call the wrapped part processor."""

    class MockPartProcessor(processor.PartProcessor):
      def __init__(self):
        self.call_count = 0

      def match(self, part):
        return True

      async def call(self, part):
        self.call_count += 1
        yield content_api.ProcessorPart(f'Processed: {part.text}')

    mock_cache = semantic_cache.InMemoryVectorCache()
    mock_part_proc = MockPartProcessor()

    proc = semantic_cache.SemanticCachePartProcessor(
        wrapped_part_processor=mock_part_proc,
        api_key='fake-key',
        similarity_threshold=0.9,
        cache=mock_cache,
    )

    proc._embedding_client = _create_mock_embedding_client({
        'test': [1.0, 0, 0],
    })

    input_part = content_api.ProcessorPart('test input')
    result = []
    async for p in proc(input_part):
      result.append(p)

    self.assertEqual(mock_part_proc.call_count, 1)
    self.assertEqual(len(result), 1)

  async def test_cache_hit_skips_wrapped_processor(self):
    """Cache hit should return cached response without calling wrapped."""

    class MockPartProcessor(processor.PartProcessor):
      def __init__(self):
        self.call_count = 0

      def match(self, part):
        return True

      async def call(self, part):
        self.call_count += 1
        yield content_api.ProcessorPart(f'Processed: {part.text}')

    mock_cache = semantic_cache.InMemoryVectorCache()
    mock_part_proc = MockPartProcessor()

    proc = semantic_cache.SemanticCachePartProcessor(
        wrapped_part_processor=mock_part_proc,
        api_key='fake-key',
        similarity_threshold=0.85,
        cache=mock_cache,
    )

    proc._embedding_client = _create_mock_embedding_client({
        'test': [1.0, 0, 0],
    })

    # First call - cache miss
    input_part1 = content_api.ProcessorPart('test input')
    async for _ in proc(input_part1):
      pass

    self.assertEqual(mock_part_proc.call_count, 1)

    # Second call - cache hit
    input_part2 = content_api.ProcessorPart('test query')
    async for _ in proc(input_part2):
      pass

    self.assertEqual(mock_part_proc.call_count, 1)  # No additional call

  async def test_non_text_part_bypasses_cache(self):
    """Non-text parts should bypass cache."""

    class MockPartProcessor(processor.PartProcessor):
      def __init__(self):
        self.call_count = 0

      def match(self, part):
        return True

      async def call(self, part):
        self.call_count += 1
        yield content_api.ProcessorPart('Processed')

    mock_cache = semantic_cache.InMemoryVectorCache()
    mock_part_proc = MockPartProcessor()

    proc = semantic_cache.SemanticCachePartProcessor(
        wrapped_part_processor=mock_part_proc,
        api_key='fake-key',
        cache=mock_cache,
    )

    # Non-text part
    input_part = content_api.ProcessorPart(
        b'binary data',
        mimetype='application/octet-stream',
    )
    async for _ in proc(input_part):
      pass

    self.assertEqual(mock_part_proc.call_count, 1)

    # Cache should be empty (non-text not cached)
    stats = await mock_cache.stats()
    self.assertEqual(stats['stores'], 0)


# =============================================================================
# SemanticCacheEntry Tests
# =============================================================================


class SemanticCacheEntryTest(unittest.TestCase):
  """Tests for SemanticCacheEntry dataclass."""

  def test_get_response_parts(self):
    """get_response_parts should deserialize parts via from_dict."""
    # Use ProcessorPart.to_dict() format for stored response parts
    original = content_api.ProcessorPart(
        'Hello',
        role='model',
        mimetype='text/plain',
        metadata={'key': 'value'},
    )
    entry = semantic_cache.SemanticCacheEntry(
        entry_id='test-id',
        query_embedding=[0.1, 0.2],
        query_text='test query',
        response_parts=[original.to_dict()],
        created_at=0.0,
    )

    parts = entry.get_response_parts()

    self.assertEqual(len(parts), 1)
    self.assertEqual(parts[0].text, 'Hello')
    self.assertEqual(parts[0].role, 'model')

  def test_hit_count_default(self):
    """hit_count should default to 0."""
    entry = semantic_cache.SemanticCacheEntry(
        entry_id='test-id',
        query_embedding=[0.1],
        query_text='test',
        response_parts=[],
        created_at=0.0,
    )

    self.assertEqual(entry.hit_count, 0)

  def test_metadata_default(self):
    """metadata should default to empty dict."""
    entry = semantic_cache.SemanticCacheEntry(
        entry_id='test-id',
        query_embedding=[0.1],
        query_text='test',
        response_parts=[],
        created_at=0.0,
    )

    self.assertEqual(entry.metadata, {})


# =============================================================================
# Integration Tests
# =============================================================================


class IntegrationTest(unittest.IsolatedAsyncioTestCase):
  """Integration tests for semantic cache with real-ish scenarios."""

  async def test_chatbot_conversation_caching(self):
    """Simulate chatbot with repeated similar questions."""
    mock_model = _create_mock_processor('Paris is the capital of France.')
    cache = semantic_cache.InMemoryVectorCache(max_entries=100)

    proc = semantic_cache.SemanticCacheProcessor(
        wrapped_processor=mock_model,
        api_key='fake-key',
        similarity_threshold=0.88,
        cache=cache,
    )

    # Simulate embeddings for similar questions (case-insensitive matching)
    proc._embedding_client = _create_mock_embedding_client({
        'capital of france': [0.9, 0.1, 0.0],
        'france': [0.88, 0.12, 0.0],  # Matches "France's capital?"
        'french': [0.85, 0.15, 0.0],
        'weather': [0.1, 0.9, 0.0],
    })

    # First question - cache miss
    await processor.apply_async(proc, ['What is the capital of France?'])

    # Similar questions - should be cache hits
    await processor.apply_async(proc, ["France's capital?"])
    await processor.apply_async(proc, ['Tell me the French capital'])

    # Different question - cache miss
    await processor.apply_async(proc, ['What is the weather?'])

    # Verify call count
    self.assertEqual(mock_model.call_count, 2)  # Only 2 misses

    # Verify stats
    stats = await cache.stats()
    self.assertEqual(stats['hits'], 2)
    self.assertEqual(stats['misses'], 2)

  async def test_high_concurrency(self):
    """Test cache under concurrent access."""
    mock_model = _create_mock_processor('Response')
    cache = semantic_cache.InMemoryVectorCache()

    proc = semantic_cache.SemanticCacheProcessor(
        wrapped_processor=mock_model,
        api_key='fake-key',
        similarity_threshold=0.9,
        cache=cache,
    )

    proc._embedding_client = _create_mock_embedding_client({
        'query': [1.0, 0, 0],
    })

    # Run many concurrent requests with same query
    tasks = [
        processor.apply_async(proc, [f'query {i}']) for i in range(10)
    ]

    await asyncio.gather(*tasks)

    # With concurrent requests, all may start before any finishes storing,
    # so hits are not guaranteed. Verify total operations equal 10.
    stats = await cache.stats()
    self.assertEqual(stats['hits'] + stats['misses'], 10)


if __name__ == '__main__':
  unittest.main()
