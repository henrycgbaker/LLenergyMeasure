"""
Unit tests for caching utilities.

Tests LRU cache with TTL, disk cache, and caching decorators.
"""

import pytest
import time
from pathlib import Path

from llm_efficiency.utils.cache import (
    LRUCacheWithTTL,
    CacheEntry,
    DiskCache,
    cached_with_ttl,
)


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_creation(self):
        """Test creating a cache entry."""
        entry = CacheEntry(value="test_value", timestamp=time.time())
        
        assert entry.value == "test_value"
        assert entry.hits == 0

    def test_expiration(self):
        """Test entry expiration check."""
        old_entry = CacheEntry(value="old", timestamp=time.time() - 100)
        
        assert old_entry.is_expired(ttl=50) is True
        assert old_entry.is_expired(ttl=200) is False
        assert old_entry.is_expired(ttl=0) is False  # No expiration


class TestLRUCacheWithTTL:
    """Tests for LRU cache with TTL."""

    def test_initialization(self):
        """Test cache initialization."""
        cache = LRUCacheWithTTL(maxsize=10, ttl=60)
        
        assert cache.maxsize == 10
        assert cache.ttl == 60
        assert len(cache.cache) == 0

    def test_set_and_get(self):
        """Test basic set and get operations."""
        cache = LRUCacheWithTTL()
        
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_missing_key(self):
        """Test getting a missing key."""
        cache = LRUCacheWithTTL()
        
        assert cache.get("missing") is None
        assert cache.get("missing", default="default") == "default"

    def test_ttl_expiration(self):
        """Test TTL-based expiration."""
        cache = LRUCacheWithTTL(maxsize=10, ttl=0.1)
        
        cache.set("key", "value")
        assert cache.get("key") == "value"
        
        time.sleep(0.15)
        assert cache.get("key") is None  # Expired

    def test_lru_eviction(self):
        """Test LRU eviction when size limit reached."""
        cache = LRUCacheWithTTL(maxsize=3)
        
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        cache.set("d", 4)  # Should evict 'a'
        
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3
        assert cache.get("d") == 4

    def test_lru_ordering(self):
        """Test LRU ordering on access."""
        cache = LRUCacheWithTTL(maxsize=3)
        
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        
        # Access 'a' to make it recently used
        cache.get("a")
        
        # Add 'd', should evict 'b' (least recently used)
        cache.set("d", 4)
        
        assert cache.get("a") == 1
        assert cache.get("b") is None  # Evicted
        assert cache.get("c") == 3
        assert cache.get("d") == 4

    def test_update_existing_key(self):
        """Test updating an existing key."""
        cache = LRUCacheWithTTL()
        
        cache.set("key", "value1")
        cache.set("key", "value2")
        
        assert cache.get("key") == "value2"
        assert len(cache.cache) == 1  # Should not add duplicate

    def test_delete(self):
        """Test deleting a key."""
        cache = LRUCacheWithTTL()
        
        cache.set("key", "value")
        assert cache.get("key") == "value"
        
        cache.delete("key")
        assert cache.get("key") is None

    def test_clear(self):
        """Test clearing cache."""
        cache = LRUCacheWithTTL()
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        cache.clear()
        
        assert len(cache.cache) == 0
        assert cache.get("key1") is None

    def test_hit_rate_tracking(self):
        """Test cache hit rate tracking."""
        cache = LRUCacheWithTTL()
        
        cache.set("key", "value")
        
        cache.get("key")  # Hit
        cache.get("key")  # Hit
        cache.get("missing")  # Miss
        
        stats = cache.get_stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == pytest.approx(66.67, rel=0.1)

    def test_get_stats(self):
        """Test getting cache statistics."""
        cache = LRUCacheWithTTL(maxsize=100, ttl=3600)
        
        cache.set("key", "value")
        stats = cache.get_stats()
        
        assert stats["size"] == 1
        assert stats["maxsize"] == 100
        assert stats["ttl"] == 3600

    def test_save_and_load(self, tmp_path):
        """Test saving and loading cache from disk."""
        cache1 = LRUCacheWithTTL(maxsize=10, ttl=60)
        cache1.set("key1", "value1")
        cache1.set("key2", "value2")
        
        filepath = tmp_path / "cache.pkl"
        cache1.save_to_disk(filepath)
        
        cache2 = LRUCacheWithTTL.load_from_disk(filepath)
        
        assert cache2.get("key1") == "value1"
        assert cache2.get("key2") == "value2"
        assert cache2.maxsize == 10
        assert cache2.ttl == 60


class TestDiskCache:
    """Tests for disk-based cache."""

    def test_initialization(self, tmp_path):
        """Test disk cache initialization."""
        cache_dir = tmp_path / "cache"
        cache = DiskCache(cache_dir=cache_dir)
        
        assert cache.cache_dir.exists()

    def test_set_and_get(self, tmp_path):
        """Test basic disk cache operations."""
        cache = DiskCache(cache_dir=tmp_path / "cache")
        
        cache.set("key", "value")
        assert cache.get("key") == "value"

    def test_get_missing(self, tmp_path):
        """Test getting missing key."""
        cache = DiskCache(cache_dir=tmp_path / "cache")
        
        assert cache.get("missing") is None
        assert cache.get("missing", "default") == "default"

    def test_ttl_expiration(self, tmp_path):
        """Test TTL-based expiration in disk cache."""
        cache = DiskCache(cache_dir=tmp_path / "cache", ttl=0.1)
        
        cache.set("key", "value")
        assert cache.get("key") == "value"
        
        time.sleep(0.15)
        assert cache.get("key") is None  # Expired

    def test_delete(self, tmp_path):
        """Test deleting from disk cache."""
        cache = DiskCache(cache_dir=tmp_path / "cache")
        
        cache.set("key", "value")
        assert cache.get("key") == "value"
        
        cache.delete("key")
        assert cache.get("key") is None

    def test_clear(self, tmp_path):
        """Test clearing disk cache."""
        cache = DiskCache(cache_dir=tmp_path / "cache")
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        assert cache.get_size() == 2
        
        cache.clear()
        
        assert cache.get_size() == 0

    def test_get_size(self, tmp_path):
        """Test getting cache size."""
        cache = DiskCache(cache_dir=tmp_path / "cache")
        
        assert cache.get_size() == 0
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        assert cache.get_size() == 2


class TestCachedWithTTL:
    """Tests for cached_with_ttl decorator."""

    def test_decorator_basic(self):
        """Test basic caching decorator."""
        call_count = 0
        
        @cached_with_ttl(ttl=60, maxsize=10)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # First call - cache miss
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1
        
        # Second call - cache hit
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Not called again

    def test_decorator_different_args(self):
        """Test caching with different arguments."""
        call_count = 0
        
        @cached_with_ttl()
        def add(a, b):
            nonlocal call_count
            call_count += 1
            return a + b
        
        assert add(2, 3) == 5
        assert add(2, 3) == 5  # Cache hit
        assert add(3, 4) == 7  # Cache miss (different args)
        
        assert call_count == 2  # Called twice for different args

    def test_cache_clear(self):
        """Test cache clearing."""
        call_count = 0
        
        @cached_with_ttl()
        def func(x):
            nonlocal call_count
            call_count += 1
            return x
        
        func(1)
        func(1)  # Cache hit
        assert call_count == 1
        
        func.cache_clear()
        
        func(1)  # Cache miss after clear
        assert call_count == 2

    def test_cache_stats(self):
        """Test getting cache stats."""
        @cached_with_ttl()
        def func(x):
            return x
        
        func(1)
        func(1)
        func(2)
        
        stats = func.cache_stats()
        assert stats["size"] == 2
        assert stats["hits"] == 1
        assert stats["misses"] == 2
