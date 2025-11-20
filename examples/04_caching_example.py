"""
Caching Strategies Example
===========================

This example demonstrates the advanced caching utilities for improving
performance and reducing redundant computations.

Features demonstrated:
- LRU cache with TTL (time-to-live)
- Disk-based persistent caching
- Function result caching with decorators
- Cache statistics and hit rates
- Cache persistence and loading
"""

import time
import numpy as np
from pathlib import Path

from llm_efficiency.utils import (
    LRUCacheWithTTL,
    DiskCache,
    cached_with_ttl,
)


# Example 1: LRU Cache with TTL
def example_lru_cache():
    """Demonstrate LRU cache with time-to-live."""

    print("=" * 70)
    print("Example 1: LRU Cache with TTL")
    print("=" * 70)

    # Create cache with max 5 items and 10-second TTL
    cache = LRUCacheWithTTL(maxsize=5, ttl=10.0)

    # Store items
    print("\nStoring 5 items...")
    for i in range(5):
        cache.set(f"key_{i}", f"value_{i}")
        print(f"  Stored: key_{i} -> value_{i}")

    # Access items (hits)
    print("\nAccessing items (should be cache hits)...")
    for i in range(3):
        value = cache.get(f"key_{i}")
        print(f"  Retrieved: key_{i} -> {value}")

    # Add new item (triggers LRU eviction)
    print("\nAdding 6th item (exceeds maxsize=5, will evict LRU)...")
    cache.set("key_5", "value_5")

    # Try accessing evicted item
    print("\nTrying to access evicted item...")
    value = cache.get("key_3", default="NOT_FOUND")
    print(f"  key_3: {value} (evicted due to LRU)")

    # Check cache statistics
    print("\n--- Cache Statistics ---")
    print(f"Size: {len(cache.cache)}/{cache.maxsize}")
    print(f"Hits: {cache.hits}")
    print(f"Misses: {cache.misses}")
    print(f"Hit rate: {cache.hit_rate():.1%}")

    # Demonstrate TTL expiration
    print("\n--- TTL Expiration Test ---")
    cache_short = LRUCacheWithTTL(maxsize=10, ttl=2.0)  # 2-second TTL
    cache_short.set("temp_key", "temp_value")

    print("Stored item with 2-second TTL")
    print(f"  Immediate access: {cache_short.get('temp_key')}")

    print("Waiting 3 seconds...")
    time.sleep(3)

    value = cache_short.get("temp_key", default="EXPIRED")
    print(f"  After 3 seconds: {value}")

    return cache


# Example 2: Disk Cache for Persistence
def example_disk_cache():
    """Demonstrate persistent disk-based caching."""

    print("\n" + "=" * 70)
    print("Example 2: Disk Cache (Persistent)")
    print("=" * 70)

    cache_dir = Path("./cache_example")
    cache = DiskCache(cache_dir=cache_dir, ttl=3600)  # 1-hour TTL

    # Store various data types
    print("\nStoring different data types...")
    cache.set("string_data", "Hello, caching!")
    cache.set("numeric_data", 42)
    cache.set("list_data", [1, 2, 3, 4, 5])
    cache.set("dict_data", {"model": "gpt2", "precision": "float16"})
    cache.set("numpy_data", np.array([1.0, 2.0, 3.0]))

    print("  ✓ Stored 5 items to disk")

    # Retrieve items
    print("\nRetrieving cached items...")
    print(f"  String: {cache.get('string_data')}")
    print(f"  Numeric: {cache.get('numeric_data')}")
    print(f"  List: {cache.get('list_data')}")
    print(f"  Dict: {cache.get('dict_data')}")
    print(f"  NumPy: {cache.get('numpy_data')}")

    # Check cache statistics
    print("\n--- Cache Statistics ---")
    print(f"Hits: {cache.hits}")
    print(f"Misses: {cache.misses}")
    print(f"Hit rate: {cache.hit_rate():.1%}")

    # Demonstrate persistence
    print("\n--- Persistence Test ---")
    print("Creating new cache instance from same directory...")
    cache2 = DiskCache(cache_dir=cache_dir, ttl=3600)

    # Data should still be available
    value = cache2.get("dict_data")
    print(f"  Retrieved from disk: {value}")
    print("  ✓ Cache persists across instances!")

    # Cleanup
    print(f"\nCache stored in: {cache_dir}")
    print("(Remove directory to clear cache)")

    return cache


# Example 3: Function Caching Decorator
@cached_with_ttl(ttl=60, maxsize=100)
def expensive_computation(n: int) -> float:
    """Simulate expensive computation (cached automatically)."""
    print(f"  Computing for n={n} (slow operation)...")
    time.sleep(0.5)  # Simulate slow computation
    return sum(i ** 2 for i in range(n))


def example_cached_decorator():
    """Demonstrate automatic function result caching."""

    print("\n" + "=" * 70)
    print("Example 3: Cached Function Decorator")
    print("=" * 70)

    print("\nFirst call (cache miss, slow):")
    start = time.time()
    result1 = expensive_computation(1000)
    elapsed1 = time.time() - start
    print(f"  Result: {result1}")
    print(f"  Time: {elapsed1:.3f}s")

    print("\nSecond call with same argument (cache hit, fast):")
    start = time.time()
    result2 = expensive_computation(1000)
    elapsed2 = time.time() - start
    print(f"  Result: {result2}")
    print(f"  Time: {elapsed2:.3f}s")

    print(f"\nSpeedup: {elapsed1/elapsed2:.1f}x faster!")

    print("\nThird call with different argument (cache miss):")
    start = time.time()
    result3 = expensive_computation(2000)
    elapsed3 = time.time() - start
    print(f"  Result: {result3}")
    print(f"  Time: {elapsed3:.3f}s")


# Example 4: Caching Model Outputs
def example_model_caching():
    """Demonstrate caching for model inference results."""

    print("\n" + "=" * 70)
    print("Example 4: Caching Model Outputs")
    print("=" * 70)

    # Create cache for model outputs
    model_cache = DiskCache(cache_dir=Path("./model_cache"), ttl=86400)  # 24h TTL

    def get_model_output(prompt: str, model: str = "gpt2"):
        """Get model output with caching."""
        cache_key = f"{model}:{hash(prompt)}"

        # Check cache first
        cached_result = model_cache.get(cache_key)
        if cached_result is not None:
            print(f"  ✓ Cache hit for prompt: '{prompt[:50]}...'")
            return cached_result

        # Simulate model inference (slow)
        print(f"  ⚠ Cache miss - running inference for: '{prompt[:50]}...'")
        time.sleep(1.0)  # Simulate inference time

        # Generate dummy result
        result = {
            "prompt": prompt,
            "model": model,
            "output": f"Generated response for: {prompt}",
            "tokens": len(prompt.split()) * 2,
        }

        # Cache the result
        model_cache.set(cache_key, result)
        return result

    # Test prompts
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "What is the capital of France?",  # Duplicate - should hit cache
    ]

    print("\nProcessing prompts...")
    for i, prompt in enumerate(prompts, 1):
        print(f"\nPrompt {i}:")
        result = get_model_output(prompt)
        print(f"  Tokens: {result['tokens']}")

    # Show statistics
    print("\n--- Cache Statistics ---")
    print(f"Total hits: {model_cache.hits}")
    print(f"Total misses: {model_cache.misses}")
    print(f"Hit rate: {model_cache.hit_rate():.1%}")
    print(f"Inference calls saved: {model_cache.hits}")


# Example 5: Best Practices
def example_best_practices():
    """Demonstrate caching best practices."""

    print("\n" + "=" * 70)
    print("Example 5: Caching Best Practices")
    print("=" * 70)

    print("\n1. Use LRUCacheWithTTL for:")
    print("   - Small, frequently accessed data")
    print("   - Data that expires (API responses, temporary results)")
    print("   - In-memory speed is critical")

    print("\n2. Use DiskCache for:")
    print("   - Large data (model weights, datasets)")
    print("   - Data that persists between runs")
    print("   - Sharing cache across processes")

    print("\n3. Choose appropriate TTL:")
    print("   - Short TTL (minutes): Real-time data, API calls")
    print("   - Medium TTL (hours): Model outputs, processed data")
    print("   - Long TTL (days): Static data, configurations")
    print("   - No TTL (0): Permanent until manually cleared")

    print("\n4. Monitor cache performance:")

    cache = LRUCacheWithTTL(maxsize=100, ttl=3600)

    # Simulate usage
    for i in range(150):
        cache.set(f"key_{i}", f"value_{i}")

    for i in range(0, 150, 3):  # Access every 3rd item
        cache.get(f"key_{i}")

    print(f"   Hit rate: {cache.hit_rate():.1%}")
    print(f"   Total operations: {cache.hits + cache.misses}")

    if cache.hit_rate() < 0.5:
        print("   ⚠ Consider increasing cache size or TTL")
    else:
        print("   ✓ Good cache performance!")


def main():
    """Run all caching examples."""

    print("\n" + "=" * 70)
    print("CACHING STRATEGIES EXAMPLES")
    print("=" * 70)

    # Run examples
    example_lru_cache()
    example_disk_cache()
    example_cached_decorator()
    example_model_caching()
    example_best_practices()

    print("\n" + "=" * 70)
    print("CACHING EXAMPLES COMPLETE!")
    print("=" * 70)
    print("\nKey takeaways:")
    print("  1. Use caching to avoid redundant computations")
    print("  2. LRU + TTL for fast, temporary data")
    print("  3. Disk cache for large, persistent data")
    print("  4. Decorators for automatic function caching")
    print("  5. Monitor hit rates to tune cache settings")
    print("\nFor LLM experiments:")
    print("  - Cache tokenized inputs (fast retrieval)")
    print("  - Cache model outputs (avoid re-inference)")
    print("  - Cache FLOPs calculations (expensive to compute)")
    print("  - Use disk cache for model weights")
    print("=" * 70)


if __name__ == "__main__":
    main()
