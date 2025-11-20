"""
Caching utilities with TTL and size limits.

Provides LRU cache with time-to-live support, persistent disk cache,
and cache management utilities.
"""

import json
import logging
import pickle
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Dict
import functools
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with value and metadata."""
    value: Any
    timestamp: float
    hits: int = 0
    
    def is_expired(self, ttl: float) -> bool:
        """Check if entry has expired."""
        if ttl <= 0:
            return False
        return (time.time() - self.timestamp) > ttl


class LRUCacheWithTTL:
    """
    LRU Cache with time-to-live and size limits.
    
    Features:
    - LRU eviction when size limit reached
    - TTL-based expiration
    - Hit rate tracking
    - Serializable to disk
    
    Example:
        cache = LRUCacheWithTTL(maxsize=100, ttl=3600)
        cache.set("key", value)
        result = cache.get("key")
    """
    
    def __init__(self, maxsize: int = 128, ttl: float = 0):
        """
        Initialize cache.
        
        Args:
            maxsize: Maximum number of entries (0 for unlimited)
            ttl: Time-to-live in seconds (0 for no expiration)
        """
        self.maxsize = maxsize
        self.ttl = ttl
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            default: Default value if key not found
            
        Returns:
            Cached value or default
        """
        if key not in self.cache:
            self.misses += 1
            return default
        
        entry = self.cache[key]
        
        # Check if expired
        if entry.is_expired(self.ttl):
            del self.cache[key]
            self.misses += 1
            return default
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        entry.hits += 1
        self.hits += 1
        
        return entry.value
    
    def set(self, key: str, value: Any):
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        # If key exists, update and move to end
        if key in self.cache:
            self.cache[key].value = value
            self.cache[key].timestamp = time.time()
            self.cache.move_to_end(key)
            return
        
        # Add new entry
        self.cache[key] = CacheEntry(
            value=value,
            timestamp=time.time(),
        )
        
        # Evict oldest if over size limit
        if self.maxsize > 0 and len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)  # Remove oldest (FIFO for LRU)
    
    def delete(self, key: str):
        """Delete entry from cache."""
        if key in self.cache:
            del self.cache[key]
    
    def clear(self):
        """Clear all entries."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "maxsize": self.maxsize,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "ttl": self.ttl,
        }
    
    def save_to_disk(self, filepath: Path):
        """Save cache to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "maxsize": self.maxsize,
            "ttl": self.ttl,
            "hits": self.hits,
            "misses": self.misses,
            "entries": [
                {
                    "key": k,
                    "value": e.value,
                    "timestamp": e.timestamp,
                    "hits": e.hits,
                }
                for k, e in self.cache.items()
            ],
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved cache to {filepath} ({len(self.cache)} entries)")
    
    @classmethod
    def load_from_disk(cls, filepath: Path) -> "LRUCacheWithTTL":
        """Load cache from disk."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Cache file not found: {filepath}")
        
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        
        cache = cls(maxsize=data["maxsize"], ttl=data["ttl"])
        cache.hits = data["hits"]
        cache.misses = data["misses"]
        
        # Restore entries
        for entry_data in data["entries"]:
            cache.cache[entry_data["key"]] = CacheEntry(
                value=entry_data["value"],
                timestamp=entry_data["timestamp"],
                hits=entry_data["hits"],
            )
        
        logger.info(f"Loaded cache from {filepath} ({len(cache.cache)} entries)")
        return cache


class DiskCache:
    """
    Persistent disk-based cache.
    
    Stores cached values as individual files in a directory.
    Useful for large objects that don't fit well in memory.
    
    Example:
        cache = DiskCache(cache_dir="~/.cache/llm-efficiency")
        cache.set("model_flops_gpt2", 12345678)
        flops = cache.get("model_flops_gpt2")
    """
    
    def __init__(self, cache_dir: Path, ttl: float = 0):
        """
        Initialize disk cache.
        
        Args:
            cache_dir: Directory for cache files
            ttl: Time-to-live in seconds (0 for no expiration)
        """
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl
    
    def _get_cache_path(self, key: str) -> Path:
        """Get file path for cache key."""
        # Hash key to create safe filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.pkl"
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache."""
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return default
        
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            
            # Check TTL
            if self.ttl > 0:
                age = time.time() - cache_path.stat().st_mtime
                if age > self.ttl:
                    cache_path.unlink()
                    return default
            
            return data
            
        except Exception as e:
            logger.warning(f"Failed to load cache for key '{key}': {e}")
            return default
    
    def set(self, key: str, value: Any):
        """Set value in cache."""
        cache_path = self._get_cache_path(key)
        
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.warning(f"Failed to save cache for key '{key}': {e}")
    
    def delete(self, key: str):
        """Delete entry from cache."""
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            cache_path.unlink()
    
    def clear(self):
        """Clear all cache entries."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        logger.info(f"Cleared disk cache at {self.cache_dir}")
    
    def get_size(self) -> int:
        """Get number of cached entries."""
        return len(list(self.cache_dir.glob("*.pkl")))
    
    def get_disk_usage(self) -> int:
        """Get total disk usage in bytes."""
        return sum(f.stat().st_size for f in self.cache_dir.glob("*.pkl"))


def cached_with_ttl(ttl: float = 3600, maxsize: int = 128):
    """
    Decorator for caching function results with TTL.
    
    Args:
        ttl: Time-to-live in seconds
        maxsize: Maximum cache size
    
    Example:
        @cached_with_ttl(ttl=3600, maxsize=100)
        def expensive_computation(x, y):
            return x ** y
    """
    cache = LRUCacheWithTTL(maxsize=maxsize, ttl=ttl)
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key_parts = [func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            key = ":".join(key_parts)
            
            # Try to get from cache
            result = cache.get(key)
            if result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return result
            
            # Compute and cache
            result = func(*args, **kwargs)
            cache.set(key, result)
            logger.debug(f"Cache miss for {func.__name__}, cached result")
            
            return result
        
        # Attach cache management methods
        wrapper.cache = cache
        wrapper.cache_clear = cache.clear
        wrapper.cache_stats = cache.get_stats
        
        return wrapper
    return decorator
