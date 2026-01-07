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
"""SQLite-based cache for processors using SQLAlchemy.

This is a persistent cache suitable for small Parts, for example metadata
extracted using constrained decoding. It might not scale for Parts containing
large amounts of data e.g. video frames.

Cache is very handy for writing long-running agents or during development as it
provides a lightweight way to resume execution from the point of failure. By
wrapping all LLM calls and other heavy logic in a cache, while keeping the rest
of the code idempotent, restarting the agent from the beginning will promptly
catch up to the place where it has previously failed. During development one can
force the changed code to be rerun by altering key_prefix e.g. by appending code
version to it.
"""

import asyncio
from collections.abc import Callable
import contextlib
import datetime
import json
from typing import AsyncIterator

from genai_processors import cache
from genai_processors import cache_base
from genai_processors import content_api
import sqlalchemy
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine
import sqlalchemy.orm
from typing_extensions import override


ProcessorContent = content_api.ProcessorContent

_Base = sqlalchemy.orm.declarative_base()


class _ContentCacheEntry(_Base):
  """SQLAlchemy model for the cache table."""

  __tablename__ = 'content_cache'

  key = sqlalchemy.Column(sqlalchemy.String, primary_key=True)
  value = sqlalchemy.Column(sqlalchemy.LargeBinary)
  expires_at = sqlalchemy.Column(sqlalchemy.DateTime(timezone=True), index=True)


@contextlib.asynccontextmanager
async def sql_cache(
    db_url: str,
    ttl_hours: float | None = 12,
    hash_fn: (
        Callable[[content_api.ProcessorContentTypes], str | None] | None
    ) = None,
) -> AsyncIterator['SqlCache']:
  """Context manager that creates an SqlCache instance.

  Args:
    db_url: SQLAlchemy database URL.
    ttl_hours: Time-to-live for cache items in hours. If None, the cache items
      never expire.
    hash_fn: Function to convert a content_api.ProcessorContentTypes query into
      a string key. If None, `cache.default_processor_content_hash` is used.

  Yields:
    A SqlCache instance.
  """
  engine = create_async_engine(db_url, poolclass=sqlalchemy.pool.NullPool)
  try:
    async with engine.begin() as conn:
      await conn.run_sync(_Base.metadata.create_all)

    async with AsyncSession(engine) as session:
      yield SqlCache(
          session=session,
          ttl_hours=ttl_hours,
          hash_fn=hash_fn or cache.default_processor_content_hash,
          lock=asyncio.Lock(),
      )
  finally:
    await engine.dispose()


class SqlCache(cache_base.CacheBase):
  """An SQLAlchemy based persistent content cache."""

  def __init__(
      self,
      *,
      session: AsyncSession,
      ttl_hours: float | None,
      hash_fn: Callable[[content_api.ProcessorContentTypes], str | None],
      lock: asyncio.Lock,
  ):
    """Prefer using sql_cache() factory to construct the cache."""
    self._hash_fn = hash_fn
    self._ttl = (
        datetime.timedelta(hours=ttl_hours) if ttl_hours is not None else None
    )
    self._session = session
    self._lock = lock

  @property
  @override
  def hash_fn(
      self,
  ) -> Callable[[content_api.ProcessorContentTypes], str | None]:
    return self._hash_fn

  @override
  def with_key_prefix(self, prefix: str) -> 'SqlCache':
    """Creates a new SqlCache instance with a key prefix.

    Args:
      prefix: String to prepend to generated string keys.

    Returns:
      A new SqlCache instance with the given prefix.
    """
    # This creates a new instance but shares the same session.
    return SqlCache(
        session=self._session,
        ttl_hours=self._ttl.total_seconds() / 3600
        if self._ttl is not None
        else None,
        hash_fn=cache.prefixed_hash_fn(self._hash_fn, prefix),
        lock=self._lock,
    )

  @override
  async def lookup(
      self,
      query: content_api.ProcessorContentTypes | None = None,
      *,
      key: str | None = None,
  ) -> content_api.ProcessorContent | cache_base.CacheMissT:
    query_key = key if key is not None else self._hash_fn(query)

    async with self._lock:
      item = await self._session.get(_ContentCacheEntry, query_key)
      if item is None:
        return cache_base.CacheMiss

      if self._ttl is not None:
        # Ensure item.expires_at is offset-aware for comparison
        # Assuming stored times are UTC, add UTC timezone info if none.
        expires_at = item.expires_at
        if expires_at.tzinfo is None:
          expires_at = expires_at.replace(tzinfo=datetime.timezone.utc)

        if expires_at < datetime.datetime.now(datetime.timezone.utc):
          await self._remove_by_string_key(query_key)
          return cache_base.CacheMiss

      try:
        return _deserialize_content(item.value)
      except Exception:  # pylint: disable=broad-exception-caught
        await self._remove_by_string_key(query_key)
        return cache_base.CacheMiss

  @override
  async def put(
      self,
      *,
      query: content_api.ProcessorContentTypes | None = None,
      key: str | None = None,
      value: content_api.ProcessorContentTypes,
  ) -> None:
    query_key = key if key is not None else self._hash_fn(query)

    data_to_cache_bytes = _serialize_content(
        content_api.ProcessorContent(value)
    )

    expires_at = None
    if self._ttl is not None:
      expires_at = datetime.datetime.now(datetime.timezone.utc) + self._ttl

    item = _ContentCacheEntry()
    item.key = query_key
    item.value = data_to_cache_bytes
    item.expires_at = expires_at
    async with self._lock:
      self._session.add(item)
      await self._cleanup_expired()
      await self._session.commit()

  async def _remove_by_string_key(self, string_key: str) -> None:
    """Internal helper to remove by the actual string key."""
    item = await self._session.get(_ContentCacheEntry, string_key)
    if item:
      await self._session.delete(item)
      await self._session.commit()

  @override
  async def remove(self, query: content_api.ProcessorContentTypes) -> None:
    query_key = self._hash_fn(query)
    if query_key is None:
      return
    async with self._lock:
      await self._remove_by_string_key(query_key)

  async def _cleanup_expired(self) -> None:
    """Removes expired items from the cache."""
    if self._ttl is None:
      return
    now = datetime.datetime.now(datetime.timezone.utc)
    await self._session.execute(
        sqlalchemy.delete(_ContentCacheEntry).where(
            _ContentCacheEntry.expires_at < now
        )
    )


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
