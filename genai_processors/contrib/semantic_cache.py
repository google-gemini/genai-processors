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
"""Semantic caching processor using vector embeddings.

This module adds a semantic caching layer for GenAI processors. The cache
stores LLM responses and matches new queries by semantic similarity instead of
exact string matching. When a query arrives, the processor computes its
embedding and searches the cache for similar previous queries. If a match
scores above the similarity threshold, the cached response is returned.

Note: This cache only works for turn-based processors, not realtime ones.

Key benefits:
- 50-70% reduction in API costs from semantic cache hits
- Sub-50ms latency for cache hits (vs 500ms-2s for API calls)
- Configurable similarity thresholds for different use cases

Example usage:

```python
from genai_processors.contrib import semantic_cache
from genai_processors.core import genai_model
from genai_processors import processor

# Create model with semantic caching
model = genai_model.GenaiModel(
    api_key=API_KEY,
    model_name="gemini-3-flash-preview",
)

cached_model = semantic_cache.SemanticCacheProcessor(
    wrapped_processor=model,
    api_key=API_KEY,
    similarity_threshold=0.90,
)

# First call - cache miss, calls API
result1 = processor.apply_sync(cached_model, ["What is the capital of France?"])

# Second call - semantically similar, cache hit!
result2 = processor.apply_sync(cached_model, ["Tell me France's capital city"])
```

See semantic_cache.md for detailed documentation.
"""

from __future__ import annotations

import abc
import asyncio
import dataclasses
import functools
import time
import uuid
from collections.abc import AsyncIterable
from typing import Any

import numpy as np

from genai_processors import content_api
from genai_processors import mime_types
from genai_processors import processor
from genai_processors import streams
from google import genai
from google.genai import types as genai_types


# =============================================================================
# Data Structures
# =============================================================================


@dataclasses.dataclass
class SemanticCacheEntry:
  """Represents a cached query-response pair with embedding.

  Attributes:
    entry_id: Unique identifier for this cache entry.
    query_embedding: The embedding vector of the query.
    query_text: Original query text (for debugging/inspection).
    response_parts: Cached response parts as serializable dictionaries.
    created_at: Timestamp when entry was created.
    hit_count: Number of times this entry was retrieved.
    metadata: Additional metadata for the entry.
  """

  entry_id: str
  query_embedding: list[float]
  query_text: str
  response_parts: list[dict[str, Any]]
  created_at: float
  hit_count: int = 0
  metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

  def get_response_parts(self) -> list[content_api.ProcessorPart]:
    """Deserialize response parts back to ProcessorPart objects."""
    return [
        content_api.ProcessorPart.from_dict(data=part_dict)
        for part_dict in self.response_parts
    ]


@dataclasses.dataclass
class SimilaritySearchResult:
  """Result of a similarity search in the vector cache.

  Attributes:
    entry: The matched cache entry.
    similarity_score: Cosine similarity score between query and cached entry.
  """

  entry: SemanticCacheEntry
  similarity_score: float


# =============================================================================
# Utilities
# =============================================================================


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
  """Compute cosine similarity between two vectors.

  Args:
    vec1: First vector.
    vec2: Second vector.

  Returns:
    Cosine similarity score between -1 and 1.
  """
  a = np.array(vec1, dtype=np.float32)
  b = np.array(vec2, dtype=np.float32)

  norm_a = np.linalg.norm(a)
  norm_b = np.linalg.norm(b)

  if norm_a == 0 or norm_b == 0:
    return 0.0

  return float(np.dot(a, b) / (norm_a * norm_b))


# =============================================================================
# Embedding Client
# =============================================================================


class EmbeddingClientBase(abc.ABC):
  """Abstract base class for embedding clients."""

  @abc.abstractmethod
  async def embed(
      self, content: content_api.ProcessorContent
  ) -> list[float]:
    """Generate an embedding for the given content.

    Args:
      content: The content to embed.

    Returns:
      Embedding vector as a list of floats.
    """
    ...


class GeminiEmbeddingClient(EmbeddingClientBase):
  """Client for generating embeddings using Gemini API.

  Uses the Gemini embedding models to generate semantic embeddings
  for text content.
  """

  def __init__(
      self,
      api_key: str,
      model_name: str = 'text-embedding-004',
      task_type: str = 'SEMANTIC_SIMILARITY',
  ):
    """Initialize the Gemini embedding client.

    Args:
      api_key: Gemini API key.
      model_name: Name of the embedding model to use.
      task_type: Type of embedding task (SEMANTIC_SIMILARITY, RETRIEVAL_QUERY,
        RETRIEVAL_DOCUMENT, CLASSIFICATION, CLUSTERING).
    """
    self._client = genai.Client(api_key=api_key)
    self._model_name = model_name
    self._task_type = task_type

  async def embed(
      self, content: content_api.ProcessorContent
  ) -> list[float]:
    """Generate embedding for processor content.

    Extracts text from all parts and generates a combined embedding.

    Args:
      content: The content to embed.

    Returns:
      Embedding vector as a list of floats.

    Raises:
      ValueError: If content is empty or contains non-text parts.
    """
    if not content.all_parts:
      raise ValueError('Empty content, cannot generate embedding.')

    # Check for non-text content and raise to trigger cache miss
    text_parts = []
    for part in content.all_parts:
      if not content_api.is_text(part.mimetype):
        raise ValueError(
            f'Non-text content detected (mimetype={part.mimetype}). '
            'Semantic cache only supports text content.'
        )
      text_parts.append(part.text)

    if not text_parts:
      raise ValueError('No text content found for embedding generation.')

    combined_text = ' '.join(text_parts)

    # Truncate if too long (embedding models have limits)
    max_chars = 8000
    if len(combined_text) > max_chars:
      combined_text = combined_text[:max_chars]

    # Call Gemini embedding API
    response = await asyncio.to_thread(
        self._client.models.embed_content,
        model=self._model_name,
        contents=combined_text,
        config=genai_types.EmbedContentConfig(task_type=self._task_type),
    )

    return list(response.embeddings[0].values)


# =============================================================================
# Vector Cache Base & Implementations
# =============================================================================


class VectorCacheBase(abc.ABC):
  """Abstract base class for semantic vector caches.

  Defines the interface for storing and retrieving cache entries based on
  semantic similarity of query embeddings.
  """

  @abc.abstractmethod
  async def find_similar(
      self,
      embedding: list[float],
      threshold: float,
      limit: int = 1,
  ) -> list[SimilaritySearchResult]:
    """Find the most similar cached entries above threshold.

    Args:
      embedding: Query embedding vector.
      threshold: Minimum similarity score to consider a match.
      limit: Maximum number of results to return.

    Returns:
      List of SimilaritySearchResult sorted by similarity (descending).
      Empty list if no matches found.
    """
    ...

  @abc.abstractmethod
  async def store(
      self,
      embedding: list[float],
      query_text: str,
      response_parts: list[content_api.ProcessorPart],
      metadata: dict[str, Any] | None = None,
  ) -> str:
    """Store a new entry in the cache.

    Args:
      embedding: Query embedding vector.
      query_text: Original query text.
      response_parts: Response parts to cache.
      metadata: Optional metadata for the entry.

    Returns:
      Entry ID of the stored entry.
    """
    ...

  @abc.abstractmethod
  async def remove(self, entry_id: str) -> bool:
    """Remove an entry from the cache.

    Args:
      entry_id: ID of the entry to remove.

    Returns:
      True if entry was removed, False if not found.
    """
    ...

  @abc.abstractmethod
  async def clear(self) -> None:
    """Clear all entries from the cache."""
    ...

  @abc.abstractmethod
  async def stats(self) -> dict[str, Any]:
    """Return cache statistics.

    Returns:
      Dictionary containing cache statistics (hits, misses, size, etc.).
    """
    ...


class InMemoryVectorCache(VectorCacheBase):
  """In-memory vector cache with TTL and size limits.

  Uses linear scan for similarity search. Works well for small to medium
  cache sizes (up to ~10,000 entries). For larger caches, use a
  FAISS-based implementation.

  Example:
    ```python
    cache = InMemoryVectorCache(
        max_entries=1000,
        ttl_seconds=3600,  # 1 hour
    )
    ```
  """

  def __init__(
      self,
      max_entries: int = 1000,
      ttl_seconds: float = 3600,
  ):
    """Initialize the in-memory vector cache.

    Args:
      max_entries: Maximum number of entries to store. When exceeded,
        least recently used entries are evicted.
      ttl_seconds: Time-to-live for cache entries in seconds.
    """
    self._entries: dict[str, SemanticCacheEntry] = {}
    self._max_entries = max_entries
    self._ttl_seconds = ttl_seconds
    self._lock = asyncio.Lock()
    self._stats = {'hits': 0, 'misses': 0, 'stores': 0, 'evictions': 0}

  async def find_similar(
      self,
      embedding: list[float],
      threshold: float,
      limit: int = 1,
  ) -> list[SimilaritySearchResult]:
    """Find the most similar cached entries above threshold.

    Performs linear scan over all entries computing cosine similarity.

    Args:
      embedding: Query embedding vector.
      threshold: Minimum similarity score to consider a match.
      limit: Maximum number of results to return.

    Returns:
      List of SimilaritySearchResult sorted by similarity (descending).
      Empty list if no matches found.
    """
    async with self._lock:
      await asyncio.to_thread(self._evict_expired)

      matches: list[SimilaritySearchResult] = []
      current_time = time.time()

      for entry in self._entries.values():
        # Check TTL
        if current_time - entry.created_at > self._ttl_seconds:
          continue

        score = cosine_similarity(embedding, entry.query_embedding)

        if score > threshold:
          matches.append(SimilaritySearchResult(
              entry=entry,
              similarity_score=score,
          ))

      # Sort by similarity descending and limit results
      matches.sort(key=lambda r: r.similarity_score, reverse=True)
      matches = matches[:limit]

      if matches:
        self._stats['hits'] += 1
        for match in matches:
          match.entry.hit_count += 1
      else:
        self._stats['misses'] += 1

      return matches

  async def store(
      self,
      embedding: list[float],
      query_text: str,
      response_parts: list[content_api.ProcessorPart],
      metadata: dict[str, Any] | None = None,
  ) -> str:
    """Store a new cache entry.

    Args:
      embedding: Query embedding vector.
      query_text: Original query text.
      response_parts: Response parts to cache.
      metadata: Optional metadata for the entry.

    Returns:
      Entry ID of the stored entry.
    """
    async with self._lock:
      # Evict if at capacity (LRU based on hit_count and age)
      while len(self._entries) >= self._max_entries:
        self._evict_lru()
        self._stats['evictions'] += 1

      entry_id = str(uuid.uuid4())

      # Serialize response parts for storage
      serialized_parts = [part.to_dict() for part in response_parts]

      entry = SemanticCacheEntry(
          entry_id=entry_id,
          query_embedding=embedding,
          query_text=query_text,
          response_parts=serialized_parts,
          created_at=time.time(),
          metadata=metadata or {},
      )

      self._entries[entry_id] = entry
      self._stats['stores'] += 1

      return entry_id

  async def remove(self, entry_id: str) -> bool:
    """Remove an entry from the cache."""
    async with self._lock:
      if entry_id in self._entries:
        del self._entries[entry_id]
        return True
      return False

  async def clear(self) -> None:
    """Clear all entries from the cache."""
    async with self._lock:
      self._entries.clear()
      self._stats = {'hits': 0, 'misses': 0, 'stores': 0, 'evictions': 0}

  async def stats(self) -> dict[str, Any]:
    """Return cache statistics."""
    async with self._lock:
      total_entries = len(self._entries)
      hit_rate = (
          self._stats['hits'] / (self._stats['hits'] + self._stats['misses'])
          if (self._stats['hits'] + self._stats['misses']) > 0
          else 0.0
      )
      return {
          'total_entries': total_entries,
          'max_entries': self._max_entries,
          'hits': self._stats['hits'],
          'misses': self._stats['misses'],
          'stores': self._stats['stores'],
          'evictions': self._stats['evictions'],
          'hit_rate': hit_rate,
          'ttl_seconds': self._ttl_seconds,
      }

  def _evict_expired(self) -> None:
    """Remove expired entries."""
    current_time = time.time()
    expired = [
        eid
        for eid, entry in self._entries.items()
        if current_time - entry.created_at > self._ttl_seconds
    ]
    for eid in expired:
      del self._entries[eid]

  def _evict_lru(self) -> None:
    """Evict least recently used entry."""
    if not self._entries:
      return

    # Sort by hit_count (ascending) then by created_at (ascending)
    # This evicts entries that are rarely hit and oldest
    lru_entry = min(
        self._entries.values(),
        key=lambda e: (e.hit_count, e.created_at),
    )
    del self._entries[lru_entry.entry_id]


# =============================================================================
# Main Processors
# =============================================================================


class SemanticCacheProcessor(processor.Processor):
  """A Processor that caches responses based on semantic similarity.

  Uses vector embeddings to find semantically similar previous queries.
  If a match is found above the similarity threshold, returns the cached
  response instead of calling the wrapped processor.

  Reduces API costs and latency when similar queries are frequently repeated.

  Note: This processor only works for turn-based processors, not for
  realtime ones.

  Example usage:

  ```python
  from genai_processors.contrib import semantic_cache
  from genai_processors.core import genai_model

  # Create the wrapped model
  model = genai_model.GenaiModel(
      api_key=API_KEY,
      model_name="gemini-3-flash-preview",
  )

  # Wrap with semantic cache
  cached_model = semantic_cache.SemanticCacheProcessor(
      wrapped_processor=model,
      api_key=API_KEY,
      similarity_threshold=0.90,
      cache=semantic_cache.InMemoryVectorCache(
          max_entries=1000,
          ttl_seconds=3600,
      ),
  )

  # Use like any other processor
  result = processor.apply_sync(cached_model, ["What is AI?"])
  ```

  Attributes:
    wrapped_processor: The underlying processor being cached.
    similarity_threshold: Minimum similarity for cache hits.
  """

  def __init__(
      self,
      wrapped_processor: processor.Processor,
      api_key: str,
      similarity_threshold: float = 0.90,
      embedding_model: str = 'text-embedding-004',
      cache: VectorCacheBase | None = None,
      include_cache_metadata: bool = True,
      skip_cache_for_errors: bool = True,
  ):
    """Initialize the SemanticCacheProcessor.

    Args:
      wrapped_processor: The processor to wrap (typically a GenaiModel).
      api_key: API key for the embedding model.
      similarity_threshold: Minimum similarity score to consider a cache
        hit. Range: 0.0 to 1.0. Higher values mean stricter matching.
        Recommended: 0.85-0.95.
      embedding_model: Name of the embedding model to use.
      cache: Vector cache instance. If None, creates InMemoryVectorCache
        with default settings.
      include_cache_metadata: If True, adds metadata to output parts
        indicating cache hit/miss status.
      skip_cache_for_errors: If True, don't cache responses containing
        error parts.
    """
    self._wrapped_processor = wrapped_processor
    self._embedding_client = GeminiEmbeddingClient(
        api_key=api_key,
        model_name=embedding_model,
    )
    self._similarity_threshold = similarity_threshold
    self._cache = cache or InMemoryVectorCache()
    self._include_cache_metadata = include_cache_metadata
    self._skip_cache_for_errors = skip_cache_for_errors

  @functools.cached_property
  def key_prefix(self) -> str:
    """Cache key prefix for this processor."""
    return f'SemanticCache[{self._wrapped_processor.key_prefix}]'

  async def call(
      self,
      content: AsyncIterable[content_api.ProcessorPart],
  ) -> AsyncIterable[content_api.ProcessorPartTypes]:
    """Process content with semantic caching.

    Args:
      content: Input stream of processor parts.

    Yields:
      Output parts, either from cache or from wrapped processor.
    """
    # 1. Buffer input content (needed for embedding)
    input_parts = await streams.gather_stream(content)
    input_content = content_api.ProcessorContent(input_parts)

    # 2. Generate embedding for the query
    # Handles empty input and non-text content by raising ValueError,
    # which triggers a cache miss and falls through to wrapped processor.
    try:
      query_embedding = await self._embedding_client.embed(input_content)
    except Exception as e:
      # If embedding fails (empty input, non-text content, API error),
      # fall through to wrapped processor
      yield processor.status(f'Embedding failed: {e}, bypassing cache')
      async for part in self._wrapped_processor(
          streams.stream_content(input_parts)
      ):
        yield part
      return

    # 3. Search cache for similar queries
    cache_results = await self._cache.find_similar(
        embedding=query_embedding,
        threshold=self._similarity_threshold,
    )

    # 4. Cache hit: return cached response
    if cache_results:
      cache_result = cache_results[0]
      yield processor.status(
          f'Semantic cache hit (similarity: {cache_result.similarity_score:.3f})'
      )

      for part in cache_result.entry.get_response_parts():
        if self._include_cache_metadata:
          # Add cache metadata to the part
          updated_metadata = dict(part.metadata)
          updated_metadata.update({
              'semantic_cache_hit': True,
              'similarity_score': cache_result.similarity_score,
              'cached_query': cache_result.entry.query_text[:100],
          })
          yield content_api.ProcessorPart(
              part,
              metadata=updated_metadata,
          )
        else:
          yield part
      return

    # 5. Cache miss: forward to wrapped processor
    yield processor.status('Semantic cache miss, calling model...')

    response_parts: list[content_api.ProcessorPart] = []
    has_error = False

    async for part in self._wrapped_processor(
        streams.stream_content(input_parts)
    ):
      response_parts.append(part)

      if mime_types.is_exception(part.mimetype):
        has_error = True

      if self._include_cache_metadata:
        updated_metadata = dict(part.metadata)
        updated_metadata['semantic_cache_hit'] = False
        yield content_api.ProcessorPart(
            part,
            metadata=updated_metadata,
        )
      else:
        yield part

    # 6. Store in cache (unless error occurred)
    if not (has_error and self._skip_cache_for_errors) and response_parts:
      query_text = content_api.as_text(input_content)
      await self._cache.store(
          embedding=query_embedding,
          query_text=query_text,
          response_parts=response_parts,
      )

  async def get_cache_stats(self) -> dict[str, Any]:
    """Return cache statistics.

    Returns:
      Dictionary containing cache statistics.
    """
    return await self._cache.stats()

  async def clear_cache(self) -> None:
    """Clear all cached entries."""
    await self._cache.clear()


class SemanticCachePartProcessor(processor.PartProcessor):
  """A PartProcessor variant of SemanticCacheProcessor.

  Caches responses for individual parts based on semantic similarity.
  Useful when processing parts independently with high concurrency.

  Example usage:

  ```python
  from genai_processors.contrib import semantic_cache

  cached_part_proc = semantic_cache.SemanticCachePartProcessor(
      wrapped_part_processor=my_part_processor,
      api_key=API_KEY,
      similarity_threshold=0.90,
  )
  ```
  """

  def __init__(
      self,
      wrapped_part_processor: processor.PartProcessor,
      api_key: str,
      similarity_threshold: float = 0.90,
      embedding_model: str = 'text-embedding-004',
      cache: VectorCacheBase | None = None,
  ):
    """Initialize the SemanticCachePartProcessor.

    Args:
      wrapped_part_processor: The part processor to wrap.
      api_key: API key for the embedding model.
      similarity_threshold: Minimum similarity score for cache hits.
      embedding_model: Name of the embedding model to use.
      cache: Vector cache instance. If None, creates InMemoryVectorCache.
    """
    self._wrapped_processor = wrapped_part_processor
    self._embedding_client = GeminiEmbeddingClient(
        api_key=api_key,
        model_name=embedding_model,
    )
    self._similarity_threshold = similarity_threshold
    self._cache = cache or InMemoryVectorCache()

  @functools.cached_property
  def key_prefix(self) -> str:
    """Cache key prefix for this processor."""
    return f'SemanticCachePart[{self._wrapped_processor.key_prefix}]'

  def match(self, part: content_api.ProcessorPart) -> bool:
    """Match if the underlying processor matches."""
    return self._wrapped_processor.match(part)

  async def call(
      self,
      part: content_api.ProcessorPart,
  ) -> AsyncIterable[content_api.ProcessorPartTypes]:
    """Process a single part with semantic caching.

    Args:
      part: Input part to process.

    Yields:
      Output parts, either from cache or from wrapped processor.
    """
    # Skip non-text parts (can't embed them easily)
    if not content_api.is_text(part.mimetype):
      async for p in self._wrapped_processor(part):
        yield p
      return

    # Generate embedding for the part
    try:
      embedding = await self._embedding_client.embed(
          content_api.ProcessorContent([part])
      )
    except Exception:
      # Fallback to wrapped processor
      async for p in self._wrapped_processor(part):
        yield p
      return

    # Check cache
    cache_results = await self._cache.find_similar(
        embedding=embedding,
        threshold=self._similarity_threshold,
    )

    if cache_results:
      for p in cache_results[0].entry.get_response_parts():
        yield p
      return

    # Cache miss
    response_parts: list[content_api.ProcessorPart] = []
    async for p in self._wrapped_processor(part):
      response_parts.append(p)
      yield p

    # Store in cache
    if response_parts:
      await self._cache.store(
          embedding=embedding,
          query_text=part.text,
          response_parts=response_parts,
      )

  async def get_cache_stats(self) -> dict[str, Any]:
    """Return cache statistics."""
    return await self._cache.stats()

  async def clear_cache(self) -> None:
    """Clear all cached entries."""
    await self._cache.clear()
