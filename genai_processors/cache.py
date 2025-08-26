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
"""Low-level `ProcessorContent` caching implementation for processors.

Please don't use Caches directly, but instead wrap your processors with
processor.CachedPartProcessor.
"""

import asyncio
from collections.abc import Callable
import json
from typing import Optional, overload

import cachetools
from genai_processors import cache_base
from genai_processors import content_api
from typing_extensions import override
import xxhash


CacheMiss = cache_base.CacheMiss
CacheMissT = cache_base.CacheMissT
ProcessorContentTypes = content_api.ProcessorContentTypes
ProcessorContent = content_api.ProcessorContent


def default_processor_content_hash(
    processor_content_query: ProcessorContentTypes,
) -> str:
  """Creates a deterministic hash key from a ProcessorContent-like query.

  Serializes its parts to JSON and then hashes the result. The hash is
  order-sensitive with respect to the parts.

  Args:
    processor_content_query: The ProcessorContent-like object to hash.

  Returns:
    A string hash key.
  """
  content_obj = ProcessorContent(processor_content_query)
  raw_part_dicts = [part.to_dict() for part in content_obj.all_parts]
  for part in raw_part_dicts:
    part['metadata'].pop('capture_time', None)
  canonical_representation_str = json.dumps(raw_part_dicts, sort_keys=True)
  hasher = xxhash.xxh128()
  hasher.update(canonical_representation_str.encode('utf-8'))
  return hasher.hexdigest()


def _serialize_content(value: ProcessorContent) -> bytes:
  """Serializes ProcessorContent to bytes (via JSON)."""
  list_of_part_dicts_val = [part.to_dict() for part in value.all_parts]
  json_string_val = json.dumps(list_of_part_dicts_val)
  return json_string_val.encode('utf-8')


def _deserialize_content(data_bytes: bytes) -> ProcessorContent:
  """Deserializer for ProcessorContent from bytes (via JSON)."""
  json_string_val = data_bytes.decode('utf-8')
  list_of_part_dicts_val = json.loads(json_string_val)
  return ProcessorContent([
      content_api.ProcessorPart.from_dict(data=pd)
      for pd in list_of_part_dicts_val
  ])


class InMemoryCache(cache_base.CacheBase):
  """An in-memory cache with TTL and size limits, specifically for caching `ProcessorContent`."""

  @overload
  def __init__(
      self,
      *,
      ttl_hours: float = 12,
      max_items: int = 1000,
      hash_fn: Callable[[ProcessorContentTypes], str | None] | None = None,
  ):
    ...

  @overload
  def __init__(
      self,
      *,
      base: 'InMemoryCache',
      hash_fn: Callable[[ProcessorContentTypes], str | None] | None = None,
  ):
    ...

  def __init__(
      self,
      *,
      ttl_hours: float = 12,
      max_items: int = 1000,
      base: Optional['InMemoryCache'] = None,
      hash_fn: Callable[[ProcessorContentTypes], str | None] | None = None,
  ):
    """Initializes the InMemoryCache for ProcessorContent.

    Args:
      ttl_hours: Time-to-live for cache items in hours.
      max_items: Maximum number of items in the cache. Must be positive.
      base: An instance of InMemoryCachewith which this instance should share
        the cached items. Used to create a view of the cache `with_key_prefix`.
      hash_fn: Function to convert a ProcessorContentTypes query into a string
        key. If None, `default_processor_content_hash` is used. If it returns
        None, the item is considered not cacheable.
    """
    if max_items <= 0:
      raise ValueError('max_items must be positive, got: %d' % max_items)

    self._hash_fn = (
        hash_fn if hash_fn is not None else default_processor_content_hash
    )
    self._serialize_fn = _serialize_content
    self._deserialize_fn = _deserialize_content

    if base:
      self._cache = base._cache
    else:
      ttl_seconds = ttl_hours * 3600
      self._cache = cachetools.TTLCache(
          maxsize=max_items,
          ttl=ttl_seconds if ttl_seconds > 0 else float('inf'),
      )

  @override
  def with_key_prefix(self, prefix: str) -> 'InMemoryCache':
    """Creates a new InMemoryCache instance with its hash function wrapped to prepend the given prefix to generated string keys.

    The new instance uses the same TTL/max_items configuration but operates
    on a *new* underlying `cachetools.TTLCache`.

    Args:
      prefix: String to prepend to generated string keys.

    Returns:
      A new InMemoryCache instance with the given prefix.
    """
    original_hash_fn = self._hash_fn

    def prefixed_hash_fn(query: ProcessorContentTypes) -> str | None:
      key = original_hash_fn(query)
      return f'{prefix}{key}' if key is not None else None

    return InMemoryCache(
        base=self,
        hash_fn=prefixed_hash_fn,
    )

  @override
  async def lookup(
      self, query: ProcessorContentTypes
  ) -> ProcessorContent | CacheMissT:
    query_key = self._hash_fn(query)
    if query_key is None:
      return CacheMiss

    cached_bytes = self._cache.get(query_key, CacheMiss)

    if cached_bytes is CacheMiss:
      return CacheMiss

    if not isinstance(cached_bytes, bytes):
      await self._remove_by_string_key(query_key)
      return CacheMiss

    try:
      value = self._deserialize_fn(cached_bytes)
      return value
    except Exception:  # pylint: disable=broad-exception-caught
      await self._remove_by_string_key(query_key)
      return CacheMiss

  @override
  async def put(
      self, query: ProcessorContentTypes, value: ProcessorContentTypes
  ) -> None:
    if self._cache.maxsize == 0:
      return

    query_key = self._hash_fn(query)
    if query_key is None:
      return

    data_to_cache_bytes = await asyncio.to_thread(
        self._serialize_fn, content_api.ProcessorContent(value)
    )
    self._cache[query_key] = data_to_cache_bytes

  async def _remove_by_string_key(self, string_key: str) -> None:
    """Internal helper to remove by the actual string key."""
    if string_key in self._cache:
      del self._cache[string_key]

  @override
  async def remove(self, query: ProcessorContentTypes) -> None:
    query_key = self._hash_fn(query)
    if query_key is None:
      return
    await self._remove_by_string_key(query_key)
