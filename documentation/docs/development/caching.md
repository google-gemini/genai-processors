# Caching

GenAI Processors includes a caching mechanism to store and retrieve the results
of processor executions. This can significantly improve performance for
expensive operations (like LLM calls or embeddings generation), reduce API
costs, and provide resilience for long-running tasks by allowing faster restarts
or resuming partial results.

## Core Concepts

Caching is enabled by wrapping processor instances in one of two caching
wrappers: `CachedPartProcessor` or `CachedProcessor`. These wrappers intercept
calls, check for cached results based on the input, and either return the cached
result or execute the wrapped processor and store its output before returning
it.

We recommend applying `CachedProcessor` directly to your most expensive
operations (like LLM calls) while keeping the surrounding logic outside the
cache. By minimizing the amount of code wrapped in the cache, you ensure that
any changes to your preprocessing or post-processing logic are picked up
immediately, rather than being stuck behind a stale cached result.

### `CachedPartProcessor`

`processor.CachedPartProcessor` wraps a `PartProcessor`. It checks the cache for
each incoming `ProcessorPart` individually.

-   If a part's result is found in the cache, the stored result is yielded
    immediately.
-   If not found (a cache miss), the wrapped `PartProcessor` is called for that
    part. Its output parts are yielded as they are produced and simultaneously
    collected. Once the wrapped processor finishes processing the part, the
    collected output is stored in the cache for future use.

This per-part caching approach preserves streaming behavior and is ideal for
stateless transformations on individual items in a stream.

### `CachedProcessor`

`processor.CachedProcessor` wraps a `Processor`. Because a `Processor`'s output
may depend on the entire input stream, this wrapper must consume and buffer the
*entire* input stream to compute a single cache key representing the whole
input.

-   If the buffered input is found in the cache, the stored result stream is
    yielded.
-   If not found, the wrapped `Processor` is called with the buffered input. Its
    output parts are yielded as they are produced and simultaneously collected.
    Once the wrapped processor finishes, the collected output is stored in the
    cache.

**Note:** `CachedProcessor` buffers the *input* stream completely, meaning
processing only begins once the input stream ends. Use this wrapper only when
the processor's logic depends on seeing the full stream or when input streaming
is not required.

## Enabling Caching

To enable caching, wrap an existing processor instance with either
`processor.CachedPartProcessor` or `processor.CachedProcessor`.

```python
from genai_processors import cache
from genai_processors import content_api
from genai_processors import processor
from genai_processors.core import genai_model

# GenaiModel makes expensive LLM calls.
llm_processor = genai_model.GenaiModel(
    api_key='API_KEY',
    model_name='...',
)

# Wrap with caching using CachedProcessor
cached_llm_processor = processor.CachedProcessor(
    llm_processor,
    key_prefix='llm_v1',  # Optional: for key namespacing
)
```

### Setting the Cache Instance

The actual cache instance used by these wrappers is managed via a Python
`contextvar`, allowing different cache instances to be used in different
execution contexts (e.g., per-request in a server). To specify which cache to
use, call `set_cache` within the desired context:

```python
my_cache = cache.InMemoryCache(ttl_hours=1)

# Set cache for the current context.
# CachedProcessor.set_cache and CachedPartProcessor.set_cache
# access the same context variable.
processor.CachedProcessor.set_cache(my_cache)

# Now, any CachedProcessor or CachedPartProcessor instances
# called within this context will use my_cache.
await cached_llm_processor(input_stream)
```

Alternatively, a `default_cache` can be passed during initialization, which will
be used if no cache is set in the context:

```python
cached_processor_with_default = processor.CachedProcessor(
    llm_processor,
    default_cache=cache.InMemoryCache(ttl_hours=1)
)
```

## Cache Backends

Caching behavior is defined by classes inheriting from `cache_base.CacheBase`.
GenAI Processors provides two built-in implementations:

### `InMemoryCache`

`cache.InMemoryCache` provides a volatile, in-memory cache using
`cachetools.TTLCache`. It is useful for speeding up repeated operations within a
single process lifetime. It supports time-to-live (TTL) and maximum item limits.

```python
from genai_processors import cache

in_mem_cache = cache.InMemoryCache(ttl_hours=12, max_items=1000)
```

### `SqlCache`

`sql_cache.SqlCache` provides a persistent cache backed by a SQL database (via
SQLAlchemy). This is useful for caching results across different process runs,
which is essential for developing and debugging long-running agents, as it
allows execution to be resumed quickly by reusing results from a previous failed
or interrupted run. It supports TTL and is best used via its async context
manager.

```python
from genai_processors import processor, sql_cache
from genai_processors import sql_cache


async def run_with_persistent_cache():
    # Requires SQLite backend: pip install aiosqlite
    async with sql_cache.sql_cache(
        'sqlite+aiosqlite:///my_cache.db', ttl_hours=24
    ) as persistent_cache:
      processor.CachedProcessor.set_cache(persistent_cache)
      # Run processors that use caching here...
```

## Cache Keys and Hashing

A cache key is required to uniquely identify an input for lookup.

-   **Hashing**: By default (`cache.default_processor_content_hash`), the cache
    key is generated by serializing the input `ProcessorPart` or
    `ProcessorContent` to a canonical JSON string and hashing this string using
    `xxhash`. The serialization process ignores volatile metadata, like
    `capture_time`, to ensure deterministic hashing.
-   **Exceptions**: If any part in the content to be cached has a mimetype
    indicating an exception (e.g., `text/x-exception`), hashing returns `None`,
    and the input/output is deemed uncacheable. Consequently, results of failed
    operations are never cached.
-   **Key Prefix**: Each `CachedPartProcessor` or `CachedProcessor` instance
    uses a `key_prefix` (either user-provided or defaulting to the wrapped
    processor's `key_prefix`), which is prepended to the generated hash. This
    acts as a namespace, preventing key collisions between different processors
    or different versions of the same processor logic using the same cache
    backend. If you change a processor's logic, updating its `key_prefix` is
    recommended to invalidate old cache entries.

## Custom Caches

You can implement custom caching logic (e.g., Redis, Google Cloud Storage) by
creating a class that inherits from `cache_base.CacheBase` and implements its
abstract methods: `hash_fn`, `lookup`, `put`, `remove`, and `with_key_prefix`.

## Best Practices

-   **Streaming**: Be aware that `CachedProcessor` buffers the entire input
    stream and breaks streaming from input to processing start, whereas
    `CachedPartProcessor` does not.
-   **Development**: Use `SqlCache` during development of complex agents to
    avoid re-running expensive steps after code changes or interruptions. Change
    `key_prefix` to force re-computation when logic is updated.
-   **Idempotency**: Wrapping a processor in a cache makes it idempotent. This
    may break patterns where LLM is sampled multiple times.
