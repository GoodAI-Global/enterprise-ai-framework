"""
Health Check Module

Enterprise-grade health checks for Kubernetes probes and load balancers.
Supports liveness, readiness, and startup probes.

Good AI Philosophy: Non-invasive by default - health checks don't affect core logic.
"""

import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from functools import wraps


class HealthStatus(Enum):
    """Health check status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status of a single component."""

    name: str
    status: HealthStatus
    message: Optional[str] = None
    latency_ms: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    last_check: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "latency_ms": self.latency_ms,
            "details": self.details,
            "last_check": self.last_check,
        }


@dataclass
class HealthReport:
    """Overall health report."""

    status: HealthStatus
    version: str
    uptime_seconds: float
    components: List[ComponentHealth]
    timestamp: str

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON response."""
        return {
            "status": self.status.value,
            "version": self.version,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "timestamp": self.timestamp,
            "components": [c.to_dict() for c in self.components],
        }

    @property
    def is_healthy(self) -> bool:
        """Check if overall status is healthy."""
        return self.status == HealthStatus.HEALTHY

    @property
    def is_ready(self) -> bool:
        """Check if system is ready to serve traffic."""
        return self.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]


class HealthCheck:
    """
    Health check manager for enterprise deployments.

    Supports:
    - Multiple health check components
    - Liveness/readiness/startup probe patterns
    - Timeout handling for slow checks
    - Caching to prevent check storms
    - Thread-safe operation

    Example:
        >>> health = HealthCheck(version="1.0.0")
        >>>
        >>> @health.register("database")
        ... def check_database():
        ...     # Returns True/False or HealthStatus
        ...     return db.ping()
        >>>
        >>> @health.register("redis", critical=False)
        ... def check_redis():
        ...     return redis.ping()
        >>>
        >>> report = health.check()
        >>> print(report.status)  # HealthStatus.HEALTHY
    """

    def __init__(
        self,
        version: str = "0.0.0",
        cache_ttl_seconds: float = 5.0,
        default_timeout_seconds: float = 10.0
    ):
        self.version = version
        self.cache_ttl_seconds = cache_ttl_seconds
        self.default_timeout_seconds = default_timeout_seconds

        self._checks: Dict[str, Dict[str, Any]] = {}
        self._cache: Optional[HealthReport] = None
        self._cache_time: float = 0
        self._start_time: float = time.time()
        self._lock = threading.Lock()

    def register(
        self,
        name: str,
        critical: bool = True,
        timeout_seconds: Optional[float] = None,
        description: Optional[str] = None
    ) -> Callable:
        """
        Register a health check function.

        Args:
            name: Unique name for this health check
            critical: If True, failure makes overall status UNHEALTHY
                     If False, failure only makes status DEGRADED
            timeout_seconds: Timeout for this check (uses default if not set)
            description: Human-readable description

        Returns:
            Decorator function

        Example:
            >>> @health.register("api_connection", critical=True)
            ... def check_api():
            ...     response = requests.get(API_URL, timeout=5)
            ...     return response.status_code == 200
        """
        def decorator(func: Callable) -> Callable:
            self._checks[name] = {
                "func": func,
                "critical": critical,
                "timeout": timeout_seconds or self.default_timeout_seconds,
                "description": description,
            }

            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        return decorator

    def add_check(
        self,
        name: str,
        check_func: Callable[[], Union[bool, HealthStatus]],
        critical: bool = True,
        timeout_seconds: Optional[float] = None,
        description: Optional[str] = None
    ) -> None:
        """
        Add a health check function directly (non-decorator).

        Args:
            name: Unique name for this health check
            check_func: Function that returns True/False or HealthStatus
            critical: If True, failure makes overall status UNHEALTHY
            timeout_seconds: Timeout for this check
            description: Human-readable description
        """
        self._checks[name] = {
            "func": check_func,
            "critical": critical,
            "timeout": timeout_seconds or self.default_timeout_seconds,
            "description": description,
        }

    def remove_check(self, name: str) -> bool:
        """Remove a health check by name."""
        with self._lock:
            if name in self._checks:
                del self._checks[name]
                self._invalidate_cache()
                return True
            return False

    def check(self, use_cache: bool = True) -> HealthReport:
        """
        Run all health checks and return report.

        Args:
            use_cache: If True, return cached result if still valid

        Returns:
            HealthReport with overall and component statuses
        """
        with self._lock:
            # Check cache
            if use_cache and self._cache is not None:
                if time.time() - self._cache_time < self.cache_ttl_seconds:
                    return self._cache

            # Run all checks
            components = []
            has_critical_failure = False
            has_any_failure = False

            for name, check_config in self._checks.items():
                component = self._run_single_check(name, check_config)
                components.append(component)

                if component.status == HealthStatus.UNHEALTHY:
                    has_any_failure = True
                    if check_config["critical"]:
                        has_critical_failure = True
                elif component.status == HealthStatus.DEGRADED:
                    has_any_failure = True

            # Determine overall status
            if has_critical_failure:
                overall_status = HealthStatus.UNHEALTHY
            elif has_any_failure:
                overall_status = HealthStatus.DEGRADED
            elif not components:
                overall_status = HealthStatus.HEALTHY  # No checks = healthy
            else:
                overall_status = HealthStatus.HEALTHY

            # Build report
            report = HealthReport(
                status=overall_status,
                version=self.version,
                uptime_seconds=time.time() - self._start_time,
                components=components,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            # Cache result
            self._cache = report
            self._cache_time = time.time()

            return report

    def _run_single_check(
        self,
        name: str,
        config: Dict[str, Any]
    ) -> ComponentHealth:
        """Run a single health check with timeout."""
        start_time = time.time()
        check_func = config["func"]
        timeout = config["timeout"]

        result = {"status": HealthStatus.UNKNOWN, "message": None, "details": {}}

        def run_check():
            try:
                check_result = check_func()

                if isinstance(check_result, HealthStatus):
                    result["status"] = check_result
                elif isinstance(check_result, bool):
                    result["status"] = HealthStatus.HEALTHY if check_result else HealthStatus.UNHEALTHY
                elif isinstance(check_result, dict):
                    # Allow returning dict with status and details
                    if "status" in check_result:
                        status = check_result["status"]
                        if isinstance(status, HealthStatus):
                            result["status"] = status
                        elif isinstance(status, bool):
                            result["status"] = HealthStatus.HEALTHY if status else HealthStatus.UNHEALTHY
                        else:
                            result["status"] = HealthStatus.HEALTHY
                    else:
                        result["status"] = HealthStatus.HEALTHY
                    result["message"] = check_result.get("message")
                    result["details"] = check_result.get("details", {})
                else:
                    # Truthy = healthy
                    result["status"] = HealthStatus.HEALTHY if check_result else HealthStatus.UNHEALTHY

            except Exception as e:
                result["status"] = HealthStatus.UNHEALTHY
                result["message"] = f"Check failed: {type(e).__name__}: {str(e)}"

        # Run with timeout using thread
        thread = threading.Thread(target=run_check)
        thread.daemon = True
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            result["status"] = HealthStatus.UNHEALTHY
            result["message"] = f"Check timed out after {timeout}s"

        latency_ms = (time.time() - start_time) * 1000

        return ComponentHealth(
            name=name,
            status=result["status"],
            message=result["message"],
            latency_ms=round(latency_ms, 2),
            details=result["details"],
            last_check=datetime.now(timezone.utc).isoformat(),
        )

    def _invalidate_cache(self):
        """Invalidate the health check cache."""
        self._cache = None

    def liveness(self) -> Dict[str, Any]:
        """
        Liveness probe endpoint.

        Returns simple alive status without running full checks.
        Use for Kubernetes liveness probes.

        Returns:
            Simple status dict
        """
        return {
            "status": "alive",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def readiness(self) -> Dict[str, Any]:
        """
        Readiness probe endpoint.

        Runs full health check and returns readiness status.
        Use for Kubernetes readiness probes.

        Returns:
            Readiness status dict with component details
        """
        report = self.check()
        return {
            "ready": report.is_ready,
            "status": report.status.value,
            "timestamp": report.timestamp,
        }

    def startup(self) -> Dict[str, Any]:
        """
        Startup probe endpoint.

        Checks if all critical components are ready.
        Use for Kubernetes startup probes.

        Returns:
            Startup status dict
        """
        report = self.check(use_cache=False)  # Always fresh for startup
        return {
            "started": report.is_healthy,
            "status": report.status.value,
            "uptime_seconds": report.uptime_seconds,
            "timestamp": report.timestamp,
        }


# Global health check instance
_health_check: Optional[HealthCheck] = None


def health_check(version: str = "0.0.0") -> HealthCheck:
    """
    Get or create the global health check instance.

    Args:
        version: Application version (only used on first call)

    Returns:
        Global HealthCheck instance
    """
    global _health_check
    if _health_check is None:
        _health_check = HealthCheck(version=version)
    return _health_check


def reset_health_check() -> None:
    """Reset the global health check instance (for testing)."""
    global _health_check
    _health_check = None
