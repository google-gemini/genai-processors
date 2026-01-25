# SemanticCacheProcessor

A processor that caches LLM responses based on semantic similarity using vector embeddings. This approach outperforms exact string matching.

## Overview

SemanticCacheProcessor wraps any GenAI processor and caches responses. When a new query arrives, the processor:

1. Generates an embedding for the query using Gemini's embedding model
2. Searches the cache for semantically similar previous queries
3. If a match is found (above the similarity threshold), returns the cached response
4. Otherwise, forwards to the wrapped processor and caches the result

### Key Benefits

| Metric | Without Cache | With SemanticCache |
|--------|---------------|-------------------|
| **API Calls** | 100% | ~30-50% reduction |
| **Latency** | 500ms-2s | <50ms (cache hit) |
| **Cost** | Full | 50-70% savings |

## Installation

SemanticCacheProcessor requires `numpy` for vector operations:

```bash
pip install genai-processors numpy
```

## Quick Start

```python
from genai_processors.contrib import semantic_cache
from genai_processors.core import genai_model
from genai_processors import processor

API_KEY = "your-gemini-api-key"

# Create the base model
model = genai_model.GenaiModel(
    api_key=API_KEY,
    model_name="gemini-2.0-flash",
)

# Wrap with semantic cache
cached_model = semantic_cache.SemanticCacheProcessor(
    wrapped_processor=model,
    api_key=API_KEY,
    similarity_threshold=0.90,
)

# First call - cache miss, calls API
result1 = processor.apply_sync(
    cached_model, 
    ["What is the capital of France?"]
)
print(result1[0].text)  # "Paris is the capital of France..."

# Second call - semantically similar query, cache hit!
result2 = processor.apply_sync(
    cached_model, 
    ["Tell me France's capital city"]
)
print(result2[0].text)  # Same response, from cache!
```

## How It Works

```
┌──────────────────────────────────────────────────────────────┐
│                    SemanticCacheProcessor                    │
│                                                              │
│  Input ──▶ Embedding ──▶ Similarity Search                   │
│              │                  │                            │
│              │           ┌──────┴──────┐                     │
│              │           │             │                     │
│              │      Cache Hit     Cache Miss                 │
│              │      (≥ 0.90)      (< 0.90)                   │
│              │           │             │                     │
│              │           ▼             ▼                     │
│              │      Return        Call Model                 │
│              │      Cached        & Cache                    │
│              │      Response      Response                   │
│              │           │             │                     │
│              └───────────┴─────────────┘                     │
│                          │                                   │
│                          ▼                                   │
│                       Output                                 │
└──────────────────────────────────────────────────────────────┘
```

## Configuration Options

### SemanticCacheProcessor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `wrapped_processor` | `Processor` | Required | The processor to cache (e.g., GenaiModel) |
| `api_key` | `str` | Required | Gemini API key for embedding generation |
| `similarity_threshold` | `float` | `0.90` | Minimum cosine similarity for cache hit (0.0-1.0) |
| `embedding_model` | `str` | `"text-embedding-004"` | Gemini embedding model to use |
| `cache` | `VectorCacheBase` | `InMemoryVectorCache()` | Cache backend instance |
| `include_cache_metadata` | `bool` | `True` | Add cache hit/miss info to output metadata |
| `skip_cache_for_errors` | `bool` | `True` | Don't cache responses with errors |

### InMemoryVectorCache

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_entries` | `int` | `1000` | Maximum cache entries before LRU eviction |
| `ttl_seconds` | `float` | `3600` | Time-to-live for cache entries (1 hour) |

## Choosing Similarity Threshold

The `similarity_threshold` parameter controls how strictly queries must match:

| Threshold | Behavior | Use Case |
|-----------|----------|----------|
| **0.95+** | Strict. Near-identical queries only. | High-precision applications |
| **0.90-0.95** | Balanced. Recommended default. | Chatbots and assistants |
| **0.85-0.90** | Lenient. Higher hit rate. | FAQ systems, search |
| **< 0.85** | Too lenient. Risk of wrong matches. | Not recommended |

### Examples of Similarity Scores

```
Query 1: "What is the capital of France?"
Query 2: "Tell me the capital of France"     → ~0.95 similarity
Query 3: "France capital city?"              → ~0.92 similarity  
Query 4: "What's the weather in Paris?"      → ~0.70 similarity (different topic!)
```

## Cache Backends

### InMemoryVectorCache (Default)

Fast in-memory cache for single-process applications:

```python
from genai_processors.contrib import semantic_cache

cache = semantic_cache.InMemoryVectorCache(
    max_entries=5000,      # Store up to 5000 entries
    ttl_seconds=7200,      # Entries expire after 2 hours
)

cached_model = semantic_cache.SemanticCacheProcessor(
    wrapped_processor=model,
    api_key=API_KEY,
    cache=cache,
)
```

### Custom Cache Backend

Implement `VectorCacheBase` for custom backends like Redis, SQLite, or FAISS:

```python
from genai_processors.contrib.semantic_cache import VectorCacheBase

class MyCustomCache(VectorCacheBase):
    async def find_similar(self, embedding, threshold, limit=1):
        # Your implementation
        ...
    
    async def store(self, embedding, query_text, response_parts, metadata=None):
        # Your implementation
        ...
    
    async def remove(self, entry_id):
        ...
    
    async def clear(self):
        ...
    
    async def stats(self):
        ...
```

## Advanced Usage

### In a Pipeline

```python
from genai_processors.core import preamble

# Build a pipeline with semantic caching
pipeline = (
    preamble.Preamble("You are a helpful assistant.")
    + semantic_cache.SemanticCacheProcessor(
        wrapped_processor=genai_model.GenaiModel(
            api_key=API_KEY,
            model_name="gemini-2.0-flash",
        ),
        api_key=API_KEY,
        similarity_threshold=0.88,
    )
)

result = processor.apply_sync(pipeline, ["Hello!"])
```

### Monitoring Cache Performance

```python
# Get cache statistics
stats = await cached_model.get_cache_stats()
print(f"Hit rate: {stats['hit_rate']:.2%}")
print(f"Total entries: {stats['total_entries']}")
print(f"Hits: {stats['hits']}, Misses: {stats['misses']}")

# Clear cache when needed
await cached_model.clear_cache()
```

### Async Usage

```python
import asyncio
from genai_processors import processor

async def main():
    cached_model = semantic_cache.SemanticCacheProcessor(
        wrapped_processor=model,
        api_key=API_KEY,
    )
    
    # Process multiple queries concurrently
    queries = [
        ["What is Python?"],
        ["Explain JavaScript"],
        ["What is Python programming?"],  # Will hit cache from query 1
    ]
    
    results = await asyncio.gather(*[
        processor.apply_async(cached_model, q) for q in queries
    ])
    
    # Check cache stats
    stats = await cached_model.get_cache_stats()
    print(f"Cache hit rate: {stats['hit_rate']:.2%}")

asyncio.run(main())
```

### PartProcessor Variant

For caching individual parts with high concurrency:

```python
cached_part_proc = semantic_cache.SemanticCachePartProcessor(
    wrapped_part_processor=my_part_processor,
    api_key=API_KEY,
    similarity_threshold=0.90,
)

# Use in a pipeline
pipeline = cached_part_proc.to_processor()
```

## Best Practices

1. Start with threshold 0.90. Good balance between hit rate and accuracy.
2. Monitor hit rates. Adjust threshold based on your use case.
3. Set appropriate TTL. Shorter for dynamic content, longer for static.
4. Size cache appropriately. More entries means more memory but higher hit rate.
5. Handle errors gracefully. Cache misses on embedding failures are normal.

## Limitations

- Text-only embedding. Only embeds text content.
- Linear search. InMemoryVectorCache uses O(n) search. For large caches (>10K), use FAISS.
- Single-process. InMemoryVectorCache does not share across processes.
- Embedding costs. Each cache miss requires an embedding API call.

## API Reference

### Classes

- `SemanticCacheProcessor` - Main processor wrapper
- `SemanticCachePartProcessor` - PartProcessor variant
- `InMemoryVectorCache` - In-memory cache backend
- `VectorCacheBase` - Abstract base for cache backends
- `SemanticCacheEntry` - Cache entry data structure
- `SimilaritySearchResult` - Search result data structure

### Functions

- `cosine_similarity(vec1, vec2)` - Compute cosine similarity between vectors

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines on contributing to GenAI Processors.

## License

Apache License 2.0 - See [LICENSE](../../LICENSE) for details.
