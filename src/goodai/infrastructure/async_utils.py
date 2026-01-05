"""
Async Utilities for Enterprise AI

Provides async/await utilities for I/O-bound operations,
concurrent execution, and error handling.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Coroutine, Dict, Generic, List, Optional, TypeVar, Union
import threading
import time


T = TypeVar("T")


class AsyncExecutionError(Exception):
    """Error during async execution."""

    def __init__(self, message: str, errors: Optional[List[Exception]] = None):
        super().__init__(message)
        self.errors = errors or []


@dataclass
class ExecutionResult(Generic[T]):
    """Result of an async execution."""

    success: bool
    value: Optional[T] = None
    error: Optional[Exception] = None
    duration_ms: float = 0.0
    retries: int = 0


class AsyncExecutor:
    """
    Enterprise Async Executor.

    Manages async operations with thread pool backing,
    error handling, and metrics.

    Example:
        executor = AsyncExecutor(max_workers=10)

        # Run async function
        result = await executor.run(async_fetch_data, url)

        # Run sync function in thread
        result = await executor.run_in_thread(sync_db_query, query)

        # Run multiple with concurrency limit
        results = await executor.gather(
            tasks,
            max_concurrency=5,
            return_exceptions=True
        )
    """

    def __init__(
        self,
        max_workers: int = 10,
        default_timeout: Optional[float] = None,
    ):
        """
        Initialize async executor.

        Args:
            max_workers: Maximum thread pool workers
            default_timeout: Default timeout in seconds
        """
        self._max_workers = max_workers
        self._default_timeout = default_timeout
        self._thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self._stats = {
            "total_executions": 0,
            "successful": 0,
            "failed": 0,
            "timeouts": 0,
            "total_duration_ms": 0.0,
        }
        self._lock = threading.Lock()

    async def run(
        self,
        coro: Coroutine[Any, Any, T],
        timeout: Optional[float] = None,
    ) -> T:
        """
        Run a coroutine with optional timeout.

        Args:
            coro: Coroutine to run
            timeout: Timeout in seconds

        Returns:
            Coroutine result

        Raises:
            asyncio.TimeoutError: If timeout exceeded
            Exception: Any exception from coroutine
        """
        effective_timeout = timeout or self._default_timeout
        start = time.time()

        try:
            if effective_timeout:
                result = await asyncio.wait_for(coro, timeout=effective_timeout)
            else:
                result = await coro

            self._record_success(time.time() - start)
            return result

        except asyncio.TimeoutError:
            self._record_timeout(time.time() - start)
            raise

        except Exception:
            self._record_failure(time.time() - start)
            raise

    async def run_in_thread(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Run a sync function in thread pool.

        Args:
            func: Sync function
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result
        """
        loop = asyncio.get_event_loop()
        start = time.time()

        try:
            # Run in thread pool
            if kwargs:
                result = await loop.run_in_executor(
                    self._thread_pool,
                    lambda: func(*args, **kwargs)
                )
            else:
                result = await loop.run_in_executor(
                    self._thread_pool,
                    func,
                    *args
                )

            self._record_success(time.time() - start)
            return result

        except Exception:
            self._record_failure(time.time() - start)
            raise

    async def gather(
        self,
        coros: List[Coroutine[Any, Any, T]],
        max_concurrency: Optional[int] = None,
        return_exceptions: bool = False,
    ) -> List[Union[T, Exception]]:
        """
        Run multiple coroutines with concurrency limit.

        Args:
            coros: List of coroutines
            max_concurrency: Maximum concurrent executions
            return_exceptions: Return exceptions instead of raising

        Returns:
            List of results (or exceptions if return_exceptions)
        """
        if not coros:
            return []

        if max_concurrency:
            return await self._gather_with_semaphore(
                coros, max_concurrency, return_exceptions
            )
        else:
            return await asyncio.gather(
                *coros, return_exceptions=return_exceptions
            )

    async def _gather_with_semaphore(
        self,
        coros: List[Coroutine[Any, Any, T]],
        max_concurrency: int,
        return_exceptions: bool,
    ) -> List[Union[T, Exception]]:
        """Gather with semaphore-based concurrency limiting."""
        semaphore = asyncio.Semaphore(max_concurrency)

        async def limited(coro: Coroutine[Any, Any, T]) -> T:
            async with semaphore:
                return await coro

        limited_coros = [limited(c) for c in coros]
        return await asyncio.gather(*limited_coros, return_exceptions=return_exceptions)

    async def retry(
        self,
        coro_factory: Callable[[], Coroutine[Any, Any, T]],
        max_retries: int = 3,
        delay: float = 1.0,
        exponential_backoff: bool = True,
        retry_on: Optional[tuple] = None,
    ) -> ExecutionResult[T]:
        """
        Retry a coroutine on failure.

        Args:
            coro_factory: Factory function that creates the coroutine
            max_retries: Maximum retry attempts
            delay: Initial delay between retries
            exponential_backoff: Use exponential backoff
            retry_on: Tuple of exception types to retry on

        Returns:
            ExecutionResult with success status and value/error
        """
        start = time.time()
        last_error: Optional[Exception] = None
        retries = 0

        for attempt in range(max_retries + 1):
            try:
                coro = coro_factory()
                value = await coro
                return ExecutionResult(
                    success=True,
                    value=value,
                    duration_ms=(time.time() - start) * 1000,
                    retries=retries,
                )

            except Exception as e:
                last_error = e
                retries = attempt

                # Check if we should retry on this exception
                if retry_on and not isinstance(e, retry_on):
                    break

                if attempt < max_retries:
                    wait_time = delay * (2 ** attempt if exponential_backoff else 1)
                    await asyncio.sleep(wait_time)

        return ExecutionResult(
            success=False,
            error=last_error,
            duration_ms=(time.time() - start) * 1000,
            retries=retries,
        )

    def _record_success(self, duration: float) -> None:
        """Record successful execution."""
        with self._lock:
            self._stats["total_executions"] += 1
            self._stats["successful"] += 1
            self._stats["total_duration_ms"] += duration * 1000

    def _record_failure(self, duration: float) -> None:
        """Record failed execution."""
        with self._lock:
            self._stats["total_executions"] += 1
            self._stats["failed"] += 1
            self._stats["total_duration_ms"] += duration * 1000

    def _record_timeout(self, duration: float) -> None:
        """Record timeout."""
        with self._lock:
            self._stats["total_executions"] += 1
            self._stats["timeouts"] += 1
            self._stats["total_duration_ms"] += duration * 1000

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        with self._lock:
            stats = self._stats.copy()
            if stats["total_executions"] > 0:
                stats["avg_duration_ms"] = (
                    stats["total_duration_ms"] / stats["total_executions"]
                )
                stats["success_rate"] = (
                    stats["successful"] / stats["total_executions"]
                )
            else:
                stats["avg_duration_ms"] = 0.0
                stats["success_rate"] = 0.0
            return stats

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor."""
        self._thread_pool.shutdown(wait=wait)


# Convenience functions

def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """
    Run an async function from sync context.

    Args:
        coro: Coroutine to run

    Returns:
        Coroutine result
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(coro)


async def run_in_thread(
    func: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> T:
    """
    Run a sync function in thread pool.

    Args:
        func: Sync function
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        Function result
    """
    return await get_async_executor().run_in_thread(func, *args, **kwargs)


async def gather_with_concurrency(
    coros: List[Coroutine[Any, Any, T]],
    max_concurrency: int,
    return_exceptions: bool = False,
) -> List[Union[T, Exception]]:
    """
    Run coroutines with concurrency limit.

    Args:
        coros: List of coroutines
        max_concurrency: Maximum concurrent executions
        return_exceptions: Return exceptions instead of raising

    Returns:
        List of results
    """
    return await get_async_executor().gather(
        coros, max_concurrency, return_exceptions
    )


def retry_async(
    max_retries: int = 3,
    delay: float = 1.0,
    exponential_backoff: bool = True,
    retry_on: Optional[tuple] = None,
) -> Callable:
    """
    Decorator for async retry.

    Example:
        @retry_async(max_retries=3, delay=1.0)
        async def fetch_data():
            return await client.get_data()
    """
    def decorator(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            result = await get_async_executor().retry(
                lambda: func(*args, **kwargs),
                max_retries=max_retries,
                delay=delay,
                exponential_backoff=exponential_backoff,
                retry_on=retry_on,
            )
            if result.success:
                return result.value
            raise result.error

        return wrapper
    return decorator


def timeout_async(seconds: float) -> Callable:
    """
    Decorator for async timeout.

    Example:
        @timeout_async(5.0)
        async def slow_operation():
            await asyncio.sleep(10)
    """
    def decorator(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await get_async_executor().run(
                func(*args, **kwargs),
                timeout=seconds,
            )

        return wrapper
    return decorator


# Global singleton
_async_executor: Optional[AsyncExecutor] = None


def get_async_executor() -> AsyncExecutor:
    """Get the global async executor instance."""
    global _async_executor
    if _async_executor is None:
        _async_executor = AsyncExecutor()
    return _async_executor
