"""
Enterprise Caching Layer

Provides a flexible caching system with multiple backends,
TTL support, and cache invalidation strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic
from functools import wraps
import hashlib
import json
import threading
import time


T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """A cache entry with metadata."""

    key: str
    value: T
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    hits: int = 0
    tags: List[str] = field(default_factory=list)

    @property
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "value": self.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "hits": self.hits,
            "tags": self.tags,
        }


class CacheBackend(ABC):
    """Abstract cache backend interface."""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass

    @abstractmethod
    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        tags: Optional[List[str]] = None,
    ) -> bool:
        """Set value in cache."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass

    @abstractmethod
    def clear(self) -> int:
        """Clear all entries. Returns count of deleted entries."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass


class MemoryCache(CacheBackend):
    """
    Thread-safe in-memory cache implementation.

    Example:
        cache = MemoryCache(max_size=1000, default_ttl=300)
        cache.set("key", "value", ttl_seconds=60)
        value = cache.get("key")
    """

    def __init__(
        self,
        max_size: int = 10000,
        default_ttl: Optional[int] = None,
        eviction_policy: str = "lru",  # lru, lfu, fifo
    ):
        """
        Initialize memory cache.

        Args:
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds
            eviction_policy: Eviction policy (lru, lfu, fifo)
        """
        self._entries: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._eviction_policy = eviction_policy
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "sets": 0,
        }

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            entry = self._entries.get(key)

            if entry is None:
                self._stats["misses"] += 1
                return None

            if entry.is_expired:
                del self._entries[key]
                self._stats["misses"] += 1
                return None

            entry.hits += 1
            self._stats["hits"] += 1
            return entry.value

    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        tags: Optional[List[str]] = None,
    ) -> bool:
        """Set value in cache."""
        with self._lock:
            # Evict if at capacity
            if len(self._entries) >= self._max_size and key not in self._entries:
                self._evict()

            ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
            expires_at = None
            if ttl is not None:
                expires_at = datetime.utcnow() + timedelta(seconds=ttl)

            self._entries[key] = CacheEntry(
                key=key,
                value=value,
                expires_at=expires_at,
                tags=tags or [],
            )
            self._stats["sets"] += 1
            return True

    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        with self._lock:
            if key in self._entries:
                del self._entries[key]
                return True
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return False
            if entry.is_expired:
                del self._entries[key]
                return False
            return True

    def clear(self) -> int:
        """Clear all entries."""
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            return count

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total if total > 0 else 0.0

            return {
                **self._stats,
                "size": len(self._entries),
                "max_size": self._max_size,
                "hit_rate": hit_rate,
            }

    def invalidate_by_tag(self, tag: str) -> int:
        """Invalidate all entries with given tag."""
        with self._lock:
            keys_to_delete = [
                key for key, entry in self._entries.items()
                if tag in entry.tags
            ]
            for key in keys_to_delete:
                del self._entries[key]
            return len(keys_to_delete)

    def invalidate_by_prefix(self, prefix: str) -> int:
        """Invalidate all entries with key prefix."""
        with self._lock:
            keys_to_delete = [
                key for key in self._entries.keys()
                if key.startswith(prefix)
            ]
            for key in keys_to_delete:
                del self._entries[key]
            return len(keys_to_delete)

    def _evict(self) -> None:
        """Evict entries based on policy."""
        if not self._entries:
            return

        if self._eviction_policy == "lru":
            # Least recently used (by hits)
            key = min(self._entries.keys(), key=lambda k: self._entries[k].hits)
        elif self._eviction_policy == "lfu":
            # Least frequently used
            key = min(self._entries.keys(), key=lambda k: self._entries[k].hits)
        else:  # fifo
            # First in first out
            key = min(self._entries.keys(), key=lambda k: self._entries[k].created_at)

        del self._entries[key]
        self._stats["evictions"] += 1

    def cleanup_expired(self) -> int:
        """Remove expired entries."""
        with self._lock:
            expired_keys = [
                key for key, entry in self._entries.items()
                if entry.is_expired
            ]
            for key in expired_keys:
                del self._entries[key]
            return len(expired_keys)


class CacheManager:
    """
    Enterprise Cache Manager.

    Manages multiple cache instances with namespacing and
    coordinated invalidation.

    Example:
        manager = CacheManager()

        # Get or create cache
        user_cache = manager.get_cache("users", max_size=1000)
        user_cache.set("user:123", user_data)

        # Use decorator
        @manager.cached("predictions", ttl_seconds=300)
        def predict(model_id, input_data):
            return model.predict(input_data)
    """

    def __init__(self, default_backend: str = "memory"):
        """
        Initialize cache manager.

        Args:
            default_backend: Default backend type
        """
        self._caches: Dict[str, CacheBackend] = {}
        self._default_backend = default_backend
        self._lock = threading.RLock()

    def get_cache(
        self,
        namespace: str,
        max_size: int = 10000,
        default_ttl: Optional[int] = None,
        eviction_policy: str = "lru",
    ) -> CacheBackend:
        """
        Get or create a cache for a namespace.

        Args:
            namespace: Cache namespace
            max_size: Maximum entries
            default_ttl: Default TTL
            eviction_policy: Eviction policy

        Returns:
            Cache backend instance
        """
        with self._lock:
            if namespace not in self._caches:
                if self._default_backend == "memory":
                    self._caches[namespace] = MemoryCache(
                        max_size=max_size,
                        default_ttl=default_ttl,
                        eviction_policy=eviction_policy,
                    )
                else:
                    raise ValueError(f"Unknown backend: {self._default_backend}")

            return self._caches[namespace]

    def cached(
        self,
        namespace: str,
        ttl_seconds: Optional[int] = None,
        key_prefix: str = "",
        tags: Optional[List[str]] = None,
    ) -> Callable:
        """
        Decorator for caching function results.

        Args:
            namespace: Cache namespace
            ttl_seconds: TTL for cached values
            key_prefix: Prefix for cache keys
            tags: Tags for cache entries

        Returns:
            Decorated function
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                # Generate cache key
                key_parts = [key_prefix, func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = hashlib.sha256(":".join(key_parts).encode()).hexdigest()[:32]

                cache = self.get_cache(namespace)

                # Try to get from cache
                cached_value = cache.get(cache_key)
                if cached_value is not None:
                    return cached_value

                # Execute function
                result = func(*args, **kwargs)

                # Store in cache
                cache.set(cache_key, result, ttl_seconds=ttl_seconds, tags=tags)

                return result

            return wrapper
        return decorator

    def invalidate_namespace(self, namespace: str) -> int:
        """Invalidate all entries in a namespace."""
        with self._lock:
            cache = self._caches.get(namespace)
            if cache:
                return cache.clear()
            return 0

    def invalidate_by_tag(self, tag: str) -> int:
        """Invalidate all entries with tag across all namespaces."""
        total = 0
        with self._lock:
            for cache in self._caches.values():
                if hasattr(cache, "invalidate_by_tag"):
                    total += cache.invalidate_by_tag(tag)
        return total

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get stats for all caches."""
        with self._lock:
            return {
                namespace: cache.get_stats()
                for namespace, cache in self._caches.items()
            }

    def cleanup_all(self) -> Dict[str, int]:
        """Cleanup expired entries in all caches."""
        results = {}
        with self._lock:
            for namespace, cache in self._caches.items():
                if hasattr(cache, "cleanup_expired"):
                    results[namespace] = cache.cleanup_expired()
        return results

    def clear_all(self) -> int:
        """Clear all caches."""
        total = 0
        with self._lock:
            for cache in self._caches.values():
                total += cache.clear()
        return total


def cached(
    namespace: str = "default",
    ttl_seconds: Optional[int] = None,
    key_prefix: str = "",
    tags: Optional[List[str]] = None,
) -> Callable:
    """
    Convenience decorator using global cache manager.

    Example:
        @cached("predictions", ttl_seconds=300)
        def predict(model_id: str, data: dict):
            return model.predict(data)
    """
    return get_cache_manager().cached(
        namespace=namespace,
        ttl_seconds=ttl_seconds,
        key_prefix=key_prefix,
        tags=tags,
    )


# Global singleton
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager
