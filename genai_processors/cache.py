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


def prefixed_hash_fn(
    original_hash_fn: Callable[[content_api.ProcessorContentTypes], str | None],
    prefix: str,
) -> Callable[[content_api.ProcessorContentTypes], str | None]:
  """Returns a wrapped hash function that prepends a prefix to the key."""

  def _wrapper(query: content_api.ProcessorContentTypes) -> str | None:
    key = original_hash_fn(query)
    return f'{prefix}{key}' if key is not None else None

  return _wrapper


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

    if base:
      self._cache = base._cache
    else:
      ttl_seconds = ttl_hours * 3600
      self._cache = cachetools.TTLCache(
          maxsize=max_items,
          ttl=ttl_seconds if ttl_seconds > 0 else float('inf'),
      )

  @property
  @override
  def hash_fn(self) -> Callable[[ProcessorContentTypes], str | None]:
    """Returns the function used to convert a query into a lookup key."""
    return self._hash_fn

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
    return InMemoryCache(
        base=self,
        hash_fn=prefixed_hash_fn(self._hash_fn, prefix),
    )

  @override
  async def lookup(
      self,
      query: ProcessorContentTypes | None = None,
      *,
      key: str | None = None,
  ) -> ProcessorContent | CacheMissT:
    query_key = key if key is not None else self._hash_fn(query)
    if query_key is None:
      return CacheMiss

    cached_value = self._cache.get(query_key, CacheMiss)

    if cached_value is CacheMiss:
      return CacheMiss

    if not isinstance(cached_value, ProcessorContent):
      await self._remove_by_string_key(query_key)
      return CacheMiss

    return cached_value

  @override
  async def put(
      self,
      *,
      query: ProcessorContentTypes | None = None,
      key: str | None = None,
      value: ProcessorContentTypes,
  ) -> None:
    if self._cache.maxsize == 0:
      return

    query_key = key if key is not None else self._hash_fn(query)
    if query_key is None:
      return

    self._cache[query_key] = content_api.ProcessorContent(value)

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
