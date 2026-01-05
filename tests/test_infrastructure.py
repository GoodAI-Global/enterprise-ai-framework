"""
Tests for Infrastructure Module

Tests for caching, async utilities, and data validation.
"""

import asyncio
import pytest
import time
from datetime import datetime, timedelta

from goodai.infrastructure.cache import (
    CacheEntry,
    MemoryCache,
    CacheManager,
    cached,
    get_cache_manager,
)
from goodai.infrastructure.async_utils import (
    AsyncExecutor,
    ExecutionResult,
    run_async,
    run_in_thread,
    gather_with_concurrency,
    retry_async,
    timeout_async,
    get_async_executor,
)
from goodai.infrastructure.validation import (
    SchemaVersion,
    Schema,
    FieldSchema,
    ValidationResult,
    ValidationError,
    ValidationLevel,
    SchemaValidator,
    SchemaRegistry,
    validate,
    get_schema_registry,
)


# ============== Cache Tests ==============

class TestCacheEntry:
    """Test CacheEntry dataclass."""

    def test_not_expired(self):
        """Test non-expired entry."""
        entry = CacheEntry(
            key="test",
            value="value",
            expires_at=datetime.utcnow() + timedelta(hours=1)
        )
        assert not entry.is_expired

    def test_expired(self):
        """Test expired entry."""
        entry = CacheEntry(
            key="test",
            value="value",
            expires_at=datetime.utcnow() - timedelta(seconds=1)
        )
        assert entry.is_expired

    def test_no_expiry(self):
        """Test entry with no expiry."""
        entry = CacheEntry(key="test", value="value")
        assert not entry.is_expired


class TestMemoryCache:
    """Test MemoryCache class."""

    def test_set_and_get(self):
        """Test basic set and get."""
        cache = MemoryCache()
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_nonexistent(self):
        """Test getting nonexistent key."""
        cache = MemoryCache()
        assert cache.get("nonexistent") is None

    def test_ttl_expiry(self):
        """Test TTL expiry."""
        cache = MemoryCache()
        cache.set("key1", "value1", ttl_seconds=0)  # Expired immediately
        time.sleep(0.01)
        assert cache.get("key1") is None

    def test_delete(self):
        """Test delete."""
        cache = MemoryCache()
        cache.set("key1", "value1")
        assert cache.delete("key1")
        assert cache.get("key1") is None
        assert not cache.delete("nonexistent")

    def test_exists(self):
        """Test exists."""
        cache = MemoryCache()
        cache.set("key1", "value1")
        assert cache.exists("key1")
        assert not cache.exists("nonexistent")

    def test_clear(self):
        """Test clear."""
        cache = MemoryCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        count = cache.clear()
        assert count == 2
        assert cache.get("key1") is None

    def test_eviction_at_max_size(self):
        """Test eviction when at max size."""
        cache = MemoryCache(max_size=3, eviction_policy="fifo")
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        cache.set("key4", "value4")  # Should evict key1

        assert cache.get("key1") is None
        assert cache.get("key4") == "value4"

    def test_invalidate_by_tag(self):
        """Test invalidation by tag."""
        cache = MemoryCache()
        cache.set("key1", "value1", tags=["user:123"])
        cache.set("key2", "value2", tags=["user:123"])
        cache.set("key3", "value3", tags=["user:456"])

        count = cache.invalidate_by_tag("user:123")
        assert count == 2
        assert cache.get("key1") is None
        assert cache.get("key3") == "value3"

    def test_invalidate_by_prefix(self):
        """Test invalidation by prefix."""
        cache = MemoryCache()
        cache.set("user:1", "value1")
        cache.set("user:2", "value2")
        cache.set("product:1", "value3")

        count = cache.invalidate_by_prefix("user:")
        assert count == 2
        assert cache.get("user:1") is None
        assert cache.get("product:1") == "value3"

    def test_stats(self):
        """Test statistics."""
        cache = MemoryCache()
        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key1")  # Hit
        cache.get("nonexistent")  # Miss

        stats = cache.get_stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["size"] == 1

    def test_cleanup_expired(self):
        """Test cleanup of expired entries."""
        cache = MemoryCache()
        cache.set("key1", "value1", ttl_seconds=0)
        cache.set("key2", "value2", ttl_seconds=3600)
        time.sleep(0.01)

        count = cache.cleanup_expired()
        assert count == 1
        assert cache.get("key2") == "value2"


class TestCacheManager:
    """Test CacheManager class."""

    def test_get_cache(self):
        """Test getting/creating cache."""
        manager = CacheManager()
        cache1 = manager.get_cache("test")
        cache2 = manager.get_cache("test")
        assert cache1 is cache2

    def test_cached_decorator(self):
        """Test cached decorator."""
        manager = CacheManager()
        call_count = 0

        @manager.cached("test", ttl_seconds=60)
        def expensive_func(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        result1 = expensive_func(5)
        result2 = expensive_func(5)
        result3 = expensive_func(10)

        assert result1 == 10
        assert result2 == 10
        assert result3 == 20
        assert call_count == 2  # Only 2 unique calls

    def test_invalidate_namespace(self):
        """Test namespace invalidation."""
        manager = CacheManager()
        cache = manager.get_cache("test")
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        count = manager.invalidate_namespace("test")
        assert count == 2

    def test_get_all_stats(self):
        """Test getting all stats."""
        manager = CacheManager()
        manager.get_cache("ns1").set("k1", "v1")
        manager.get_cache("ns2").set("k2", "v2")

        stats = manager.get_all_stats()
        assert "ns1" in stats
        assert "ns2" in stats


# ============== Async Utils Tests ==============

class TestAsyncExecutor:
    """Test AsyncExecutor class."""

    def test_run_coroutine(self):
        """Test running a coroutine."""
        executor = AsyncExecutor()

        async def async_func():
            await asyncio.sleep(0.01)
            return "result"

        async def test():
            return await executor.run(async_func())

        result = run_async(test())
        assert result == "result"

    def test_run_with_timeout(self):
        """Test running with timeout."""
        executor = AsyncExecutor()

        async def slow_func():
            await asyncio.sleep(10)
            return "result"

        async def test():
            await executor.run(slow_func(), timeout=0.1)

        with pytest.raises(asyncio.TimeoutError):
            run_async(test())

    def test_run_in_thread(self):
        """Test running sync function in thread."""
        executor = AsyncExecutor()

        def sync_func(x):
            return x * 2

        async def test():
            return await executor.run_in_thread(sync_func, 5)

        result = run_async(test())
        assert result == 10

    def test_gather(self):
        """Test gathering multiple coroutines."""
        executor = AsyncExecutor()

        async def async_func(x):
            await asyncio.sleep(0.01)
            return x * 2

        async def test():
            coros = [async_func(i) for i in range(5)]
            return await executor.gather(coros)

        results = run_async(test())
        assert results == [0, 2, 4, 6, 8]

    def test_gather_with_concurrency(self):
        """Test gathering with concurrency limit."""
        executor = AsyncExecutor()

        async def test():
            max_concurrent = 0
            current_concurrent = 0

            async def track_concurrent(x):
                nonlocal max_concurrent, current_concurrent
                current_concurrent += 1
                max_concurrent = max(max_concurrent, current_concurrent)
                await asyncio.sleep(0.02)
                current_concurrent -= 1
                return x

            coros = [track_concurrent(i) for i in range(6)]
            await executor.gather(coros, max_concurrency=3)
            return max_concurrent

        max_concurrent = run_async(test())
        assert max_concurrent <= 3

    def test_retry_success_on_retry(self):
        """Test retry with eventual success."""
        executor = AsyncExecutor()

        async def test():
            attempts = 0

            async def flaky_func():
                nonlocal attempts
                attempts += 1
                if attempts < 3:
                    raise ValueError("Temporary error")
                return "success"

            result = await executor.retry(
                lambda: flaky_func(),
                max_retries=3,
                delay=0.01,
            )
            return result, attempts

        result, attempts = run_async(test())
        assert result.success
        assert result.value == "success"
        # retries tracks the attempt index of the last failure (0-indexed)
        assert result.retries >= 1

    def test_retry_all_failures(self):
        """Test retry with all failures."""
        executor = AsyncExecutor()

        async def always_fail():
            raise ValueError("Always fails")

        async def test():
            return await executor.retry(
                lambda: always_fail(),
                max_retries=2,
                delay=0.01,
            )

        result = run_async(test())
        assert not result.success
        assert isinstance(result.error, ValueError)

    def test_stats(self):
        """Test execution statistics."""
        executor = AsyncExecutor()

        async def success_func():
            return "ok"

        run_async(executor.run(success_func()))

        stats = executor.get_stats()
        assert stats["successful"] == 1
        assert stats["total_executions"] == 1


class TestAsyncDecorators:
    """Test async decorator functions."""

    def test_retry_async_decorator(self):
        """Test retry_async decorator."""
        attempts = 0

        @retry_async(max_retries=3, delay=0.01)
        async def flaky():
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                raise ValueError("Fail")
            return "success"

        result = run_async(flaky())
        assert result == "success"
        assert attempts == 2

    def test_timeout_async_decorator(self):
        """Test timeout_async decorator."""
        @timeout_async(0.1)
        async def slow():
            await asyncio.sleep(10)
            return "done"

        with pytest.raises(asyncio.TimeoutError):
            run_async(slow())


class TestExecutionResult:
    """Test ExecutionResult dataclass."""

    def test_success_result(self):
        """Test successful result."""
        result = ExecutionResult(success=True, value="data")
        assert result.success
        assert result.value == "data"

    def test_failure_result(self):
        """Test failed result."""
        error = ValueError("test error")
        result = ExecutionResult(success=False, error=error)
        assert not result.success
        assert result.error is error


# ============== Validation Tests ==============

class TestSchemaVersion:
    """Test SchemaVersion dataclass."""

    def test_from_string(self):
        """Test parsing from string."""
        version = SchemaVersion.from_string("1.2.3")
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3

    def test_comparison(self):
        """Test version comparison."""
        v1 = SchemaVersion(1, 0, 0)
        v2 = SchemaVersion(1, 1, 0)
        v3 = SchemaVersion(2, 0, 0)

        assert v1 < v2
        assert v2 < v3
        assert not v3 < v1

    def test_compatibility(self):
        """Test version compatibility."""
        v1 = SchemaVersion(1, 0, 0)
        v2 = SchemaVersion(1, 5, 0)
        v3 = SchemaVersion(2, 0, 0)

        assert v1.is_compatible_with(v2)
        assert not v1.is_compatible_with(v3)


class TestFieldSchema:
    """Test FieldSchema dataclass."""

    def test_to_dict(self):
        """Test to_dict method."""
        field = FieldSchema(
            name="email",
            field_type="string",
            required=True,
            pattern=r".*@.*"
        )
        result = field.to_dict()
        assert result["name"] == "email"
        assert result["pattern"] == r".*@.*"


class TestSchemaValidator:
    """Test SchemaValidator class."""

    def test_validate_required_field(self):
        """Test required field validation."""
        schema = Schema(
            name="test",
            version=SchemaVersion(1, 0, 0),
            fields={
                "name": FieldSchema("name", "string", required=True),
            }
        )
        validator = SchemaValidator()

        result = validator.validate({}, schema)
        assert not result.valid
        assert any(e.error_code == "required_field" for e in result.errors)

    def test_validate_type(self):
        """Test type validation."""
        schema = Schema(
            name="test",
            version=SchemaVersion(1, 0, 0),
            fields={
                "age": FieldSchema("age", "integer", required=True),
            }
        )
        validator = SchemaValidator()

        result = validator.validate({"age": "not an int"}, schema)
        assert not result.valid
        assert any(e.error_code == "invalid_type" for e in result.errors)

    def test_validate_range(self):
        """Test range validation."""
        schema = Schema(
            name="test",
            version=SchemaVersion(1, 0, 0),
            fields={
                "score": FieldSchema("score", "float", min_value=0, max_value=100),
            }
        )
        validator = SchemaValidator()

        result = validator.validate({"score": 150}, schema)
        assert not result.valid

        result = validator.validate({"score": 50}, schema)
        assert result.valid

    def test_validate_pattern(self):
        """Test pattern validation."""
        schema = Schema(
            name="test",
            version=SchemaVersion(1, 0, 0),
            fields={
                "email": FieldSchema("email", "string", pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$"),
            }
        )
        validator = SchemaValidator()

        result = validator.validate({"email": "invalid"}, schema)
        assert not result.valid

        result = validator.validate({"email": "test@example.com"}, schema)
        assert result.valid

    def test_validate_enum(self):
        """Test enum validation."""
        schema = Schema(
            name="test",
            version=SchemaVersion(1, 0, 0),
            fields={
                "status": FieldSchema("status", "string", enum_values=["active", "inactive"]),
            }
        )
        validator = SchemaValidator()

        result = validator.validate({"status": "unknown"}, schema)
        assert not result.valid

        result = validator.validate({"status": "active"}, schema)
        assert result.valid

    def test_validate_default_value(self):
        """Test default value application."""
        schema = Schema(
            name="test",
            version=SchemaVersion(1, 0, 0),
            fields={
                "count": FieldSchema("count", "integer", required=True, default=0),
            }
        )
        validator = SchemaValidator()

        result = validator.validate({}, schema)
        assert result.valid
        assert result.validated_data["count"] == 0

    def test_strict_mode(self):
        """Test strict mode (disallow extra fields)."""
        schema = Schema(
            name="test",
            version=SchemaVersion(1, 0, 0),
            fields={
                "name": FieldSchema("name", "string"),
            },
            strict_mode=True,
        )
        validator = SchemaValidator()

        result = validator.validate({"name": "test", "extra": "field"}, schema)
        assert not result.valid
        assert any(e.error_code == "unknown_field" for e in result.errors)

    def test_deprecated_field_warning(self):
        """Test deprecated field warning."""
        schema = Schema(
            name="test",
            version=SchemaVersion(1, 0, 0),
            fields={
                "old_field": FieldSchema("old_field", "string", deprecated=True, required=False),
            }
        )
        validator = SchemaValidator()

        result = validator.validate({"old_field": "value"}, schema)
        assert result.valid  # Deprecated doesn't make it invalid
        assert any(w.error_code == "deprecated_field" for w in result.warnings)


class TestSchemaRegistry:
    """Test SchemaRegistry class."""

    def test_register_and_get(self):
        """Test schema registration and retrieval."""
        registry = SchemaRegistry()
        schema = Schema(
            name="user",
            version=SchemaVersion(1, 0, 0),
            fields={
                "id": FieldSchema("id", "string"),
            }
        )
        registry.register(schema)

        retrieved = registry.get_schema("user")
        assert retrieved is not None
        assert retrieved.name == "user"

    def test_get_specific_version(self):
        """Test getting specific version."""
        registry = SchemaRegistry()
        v1 = Schema(name="user", version=SchemaVersion(1, 0, 0))
        v2 = Schema(name="user", version=SchemaVersion(2, 0, 0))
        registry.register(v1)
        registry.register(v2)

        result = registry.get_schema("user", SchemaVersion(1, 0, 0))
        assert result.version == SchemaVersion(1, 0, 0)

    def test_get_latest_version(self):
        """Test getting latest version."""
        registry = SchemaRegistry()
        v1 = Schema(name="user", version=SchemaVersion(1, 0, 0))
        v2 = Schema(name="user", version=SchemaVersion(2, 0, 0))
        registry.register(v1)
        registry.register(v2)

        result = registry.get_schema("user")
        assert result.version == SchemaVersion(2, 0, 0)

    def test_list_versions(self):
        """Test listing versions."""
        registry = SchemaRegistry()
        registry.register(Schema(name="user", version=SchemaVersion(1, 0, 0)))
        registry.register(Schema(name="user", version=SchemaVersion(2, 0, 0)))
        registry.register(Schema(name="user", version=SchemaVersion(1, 5, 0)))

        versions = registry.list_versions("user")
        assert len(versions) == 3
        assert versions[0] == SchemaVersion(1, 0, 0)
        assert versions[-1] == SchemaVersion(2, 0, 0)

    def test_migration(self):
        """Test schema migration."""
        registry = SchemaRegistry()

        def migrate_1_to_2(data):
            # Rename field
            data["full_name"] = data.pop("name", "")
            return data

        registry.register_migration(
            "user",
            SchemaVersion(1, 0, 0),
            SchemaVersion(2, 0, 0),
            migrate_1_to_2,
        )

        data = {"name": "John Doe", "email": "john@example.com"}
        migrated = registry.migrate(
            "user",
            data,
            SchemaVersion(1, 0, 0),
            SchemaVersion(2, 0, 0),
        )

        assert "full_name" in migrated
        assert migrated["full_name"] == "John Doe"
        assert "name" not in migrated

    def test_migration_path(self):
        """Test getting migration path."""
        registry = SchemaRegistry()

        registry.register_migration("user", SchemaVersion(1, 0, 0), SchemaVersion(1, 1, 0), lambda d: d)
        registry.register_migration("user", SchemaVersion(1, 1, 0), SchemaVersion(2, 0, 0), lambda d: d)

        path = registry.get_migration_path(
            "user",
            SchemaVersion(1, 0, 0),
            SchemaVersion(2, 0, 0),
        )

        assert len(path) == 2
        assert path[0] == (SchemaVersion(1, 0, 0), SchemaVersion(1, 1, 0))
        assert path[1] == (SchemaVersion(1, 1, 0), SchemaVersion(2, 0, 0))

    def test_validate(self):
        """Test validation through registry."""
        registry = SchemaRegistry()
        registry.register(Schema(
            name="user",
            version=SchemaVersion(1, 0, 0),
            fields={
                "email": FieldSchema("email", "string", required=True),
            }
        ))

        result = registry.validate("user", {"email": "test@example.com"})
        assert result.valid

        result = registry.validate("user", {})
        assert not result.valid


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_to_dict(self):
        """Test to_dict method."""
        result = ValidationResult(
            valid=False,
            errors=[ValidationError(path="field", message="error")],
        )
        d = result.to_dict()
        assert d["valid"] is False
        assert len(d["errors"]) == 1


# ============== Global Singleton Tests ==============

class TestGlobalSingletons:
    """Test global singleton instances."""

    def test_get_cache_manager(self):
        """Test cache manager singleton."""
        m1 = get_cache_manager()
        m2 = get_cache_manager()
        assert m1 is m2

    def test_get_async_executor(self):
        """Test async executor singleton."""
        e1 = get_async_executor()
        e2 = get_async_executor()
        assert e1 is e2

    def test_get_schema_registry(self):
        """Test schema registry singleton."""
        r1 = get_schema_registry()
        r2 = get_schema_registry()
        assert r1 is r2
