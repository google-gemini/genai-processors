import asyncio
import datetime
import os
import tempfile
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from genai_processors import cache_base
from genai_processors import content_api
from genai_processors import sql_cache

# Test Content
TEST_QUERY = content_api.ProcessorContent('test query')
TEST_VALUE = content_api.ProcessorContent('test value')
TEST_QUERY_2 = content_api.ProcessorContent('test query 2')
TEST_VALUE_2 = content_api.ProcessorContent('test value 2')


class SqlCacheTest(parameterized.TestCase, unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    super().setUp()
    with tempfile.NamedTemporaryFile(suffix='.sqlite', delete=False) as tmp:
      self.db_url = f'sqlite+aiosqlite:///{tmp.name}'
    self.addCleanup(os.remove, tmp.name)

  async def asyncTearDown(self):
    # Sanity check that there are no tasks left in the event loop.
    tasks = {t for t in asyncio.all_tasks() if t is not asyncio.current_task()}
    if tasks:
      for task in tasks:
        task.cancel()
      await asyncio.gather(*tasks, return_exceptions=True)
    await super().asyncTearDown()

  async def test_cache_put_and_lookup(self):
    """Tests basic put and lookup functionality."""
    async with sql_cache.sql_cache(self.db_url) as cache:
      await cache.put(query=TEST_QUERY, value=TEST_VALUE)
      result = await cache.lookup(TEST_QUERY)
      self.assertEqual(result, TEST_VALUE)

  async def test_cache_miss(self):
    """Tests cache miss for a non-existent key."""
    async with sql_cache.sql_cache(self.db_url) as cache:
      result = await cache.lookup(TEST_QUERY)
      self.assertIs(result, cache_base.CacheMiss)

  async def test_cache_ttl(self):
    """Tests that cache entries expire after TTL."""
    async with sql_cache.sql_cache(
        self.db_url, ttl_hours=0.0001 / 60 / 60
    ) as cache:  # Very short TTL
      await cache.put(query=TEST_QUERY, value=TEST_VALUE)
      await asyncio.sleep(0.0002)  # Wait for TTL to expire
      result = await cache.lookup(TEST_QUERY)
      self.assertIs(result, cache_base.CacheMiss)

  async def test_cache_remove(self):
    """Tests removing an item from the cache."""
    async with sql_cache.sql_cache(self.db_url) as cache:
      await cache.put(query=TEST_QUERY, value=TEST_VALUE)
      await cache.remove(TEST_QUERY)
      result = await cache.lookup(TEST_QUERY)
      self.assertIs(result, cache_base.CacheMiss)

  async def test_with_key_prefix(self):
    """Tests that with_key_prefix creates a namespace."""
    async with sql_cache.sql_cache(self.db_url) as cache1:
      cache2 = cache1.with_key_prefix('prefix:')

      await cache1.put(query=TEST_QUERY, value=TEST_VALUE)
      await cache2.put(query=TEST_QUERY, value=TEST_VALUE_2)

      result1 = await cache1.lookup(TEST_QUERY)
      result2 = await cache2.lookup(TEST_QUERY)

      self.assertEqual(result1, TEST_VALUE)
      self.assertEqual(result2, TEST_VALUE_2)
      self.assertNotEqual(result1, result2)

  async def test_different_content_types(self):
    """Tests caching with different content types."""
    async with sql_cache.sql_cache(self.db_url) as cache:
      query1 = content_api.ProcessorContent('text query')
      value1 = content_api.ProcessorContent(['list ', 'of ', 'strings'])
      query2 = content_api.ProcessorContent(
          [content_api.ProcessorPart(b'imagedata', mimetype='image/png')]
      )

      value2 = content_api.ProcessorContent({'a': 1, 'b': True})

      await cache.put(query=query1, value=value1)
      await cache.put(query=query2, value=value2)

      self.assertEqual(await cache.lookup(query1), value1)
      self.assertEqual(await cache.lookup(query2), value2)

  async def test_cleanup_expired(self):
    """Tests that the _cleanup_expired method removes old entries."""
    async with sql_cache.sql_cache(self.db_url, ttl_hours=0.0001) as cache:
      await cache.put(query=TEST_QUERY, value=TEST_VALUE)
      # Mock datetime to control time
      with mock.patch('genai_processors.sql_cache.datetime') as mock_datetime:
        mock_datetime.datetime.now.return_value = datetime.datetime.now(
            datetime.timezone.utc
        ) + datetime.timedelta(hours=1)
        mock_datetime.timedelta.side_effect = datetime.timedelta
        mock_datetime.timezone.utc = datetime.timezone.utc
        await cache._cleanup_expired()
        await cache._session.commit()

      # Check that the entry is gone
      async def _lookup():
        key = cache._hash_fn(TEST_QUERY)
        return await cache._session.get(sql_cache._ContentCacheEntry, key)

      result = await _lookup()
      self.assertIsNone(result)


if __name__ == '__main__':
  absltest.main()
