"""
Tests for Monitoring Module

Tests for structured logging and health checks.
"""

import json
import logging
import time
import pytest

from goodai.monitoring.logging import (
    get_logger,
    CorrelationContext,
    StructuredLogger,
    StructuredFormatter,
    HumanReadableFormatter,
    get_correlation_id,
    get_tenant_id,
    get_user_id,
    log_execution,
    _is_sensitive_key,
    _redact_sensitive,
    REDACTED_VALUE,
)
from goodai.monitoring.health import (
    HealthCheck,
    HealthStatus,
    ComponentHealth,
    HealthReport,
    health_check,
    reset_health_check,
)


class TestStructuredLogging:
    """Test structured logging functionality."""

    def test_get_logger_returns_structured_logger(self):
        """Test get_logger returns StructuredLogger."""
        logger = get_logger("test.module")
        assert isinstance(logger, StructuredLogger)
        assert logger.name == "test.module"

    def test_get_logger_caches_instances(self):
        """Test that get_logger returns same instance."""
        logger1 = get_logger("test.cached")
        logger2 = get_logger("test.cached")
        assert logger1 is logger2

    def test_correlation_context_generates_id(self):
        """Test CorrelationContext generates correlation ID."""
        with CorrelationContext() as ctx:
            assert ctx.correlation_id is not None
            assert len(ctx.correlation_id) == 36  # UUID format

    def test_correlation_context_uses_provided_id(self):
        """Test CorrelationContext uses provided ID."""
        with CorrelationContext(correlation_id="custom-123") as ctx:
            assert ctx.correlation_id == "custom-123"
            assert get_correlation_id() == "custom-123"

    def test_correlation_context_restores_previous(self):
        """Test CorrelationContext restores previous values."""
        with CorrelationContext(correlation_id="outer") as outer:
            assert get_correlation_id() == "outer"
            with CorrelationContext(correlation_id="inner") as inner:
                assert get_correlation_id() == "inner"
            assert get_correlation_id() == "outer"
        assert get_correlation_id() is None

    def test_correlation_context_with_tenant_and_user(self):
        """Test CorrelationContext with tenant and user IDs."""
        with CorrelationContext(
            correlation_id="req-123",
            tenant_id="tenant-abc",
            user_id="user-xyz"
        ):
            assert get_correlation_id() == "req-123"
            assert get_tenant_id() == "tenant-abc"
            assert get_user_id() == "user-xyz"

    def test_structured_formatter_produces_json(self):
        """Test StructuredFormatter produces valid JSON."""
        formatter = StructuredFormatter(
            service_name="test-service",
            environment="test"
        )

        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["level"] == "INFO"
        assert parsed["message"] == "Test message"
        assert parsed["service"] == "test-service"
        assert parsed["environment"] == "test"
        assert parsed["logger"] == "test.logger"
        assert "timestamp" in parsed
        assert "source" in parsed

    def test_structured_formatter_includes_correlation_id(self):
        """Test StructuredFormatter includes correlation ID."""
        formatter = StructuredFormatter()

        with CorrelationContext(correlation_id="test-corr-id"):
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="/test.py",
                lineno=1,
                msg="Test",
                args=(),
                exc_info=None
            )
            output = formatter.format(record)
            parsed = json.loads(output)

            assert parsed["correlation_id"] == "test-corr-id"

    def test_human_readable_formatter(self):
        """Test HumanReadableFormatter produces readable output."""
        formatter = HumanReadableFormatter(use_colors=False)

        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )

        output = formatter.format(record)

        assert "INFO" in output
        assert "test.logger" in output
        assert "Test message" in output

    def test_structured_logger_methods(self):
        """Test StructuredLogger has all expected methods."""
        logger = StructuredLogger("test")

        assert hasattr(logger, "debug")
        assert hasattr(logger, "info")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "error")
        assert hasattr(logger, "critical")
        assert hasattr(logger, "exception")
        assert hasattr(logger, "bind")

    def test_bound_logger(self):
        """Test BoundLogger maintains context."""
        logger = get_logger("test.bound")
        bound = logger.bind(request_id="123", user="alice")

        # Bound logger should have bind method for chaining
        bound2 = bound.bind(extra_field="value")
        assert bound2 is not bound

    def test_log_execution_decorator(self):
        """Test log_execution decorator."""
        call_count = 0

        @log_execution(include_args=True, include_result=True)
        def sample_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y

        result = sample_function(1, 2)
        assert result == 3
        assert call_count == 1

    def test_log_execution_decorator_with_exception(self):
        """Test log_execution decorator handles exceptions."""
        @log_execution()
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            failing_function()


class TestSensitiveDataRedaction:
    """Test sensitive data redaction for security."""

    def test_is_sensitive_key_detects_password(self):
        """Test detection of password fields."""
        assert _is_sensitive_key("password") is True
        assert _is_sensitive_key("user_password") is True
        assert _is_sensitive_key("PASSWORD") is True
        assert _is_sensitive_key("db_passwd") is True

    def test_is_sensitive_key_detects_tokens(self):
        """Test detection of token fields."""
        assert _is_sensitive_key("token") is True
        assert _is_sensitive_key("access_token") is True
        assert _is_sensitive_key("refresh_token") is True
        assert _is_sensitive_key("api_key") is True
        assert _is_sensitive_key("apikey") is True

    def test_is_sensitive_key_detects_secrets(self):
        """Test detection of secret fields."""
        assert _is_sensitive_key("secret") is True
        assert _is_sensitive_key("secret_key") is True
        assert _is_sensitive_key("private_key") is True
        assert _is_sensitive_key("ssh_key") is True

    def test_is_sensitive_key_allows_safe_fields(self):
        """Test that safe fields are not marked sensitive."""
        assert _is_sensitive_key("username") is False
        assert _is_sensitive_key("email") is False
        assert _is_sensitive_key("request_id") is False
        assert _is_sensitive_key("data") is False

    def test_redact_sensitive_dict(self):
        """Test redaction in dictionaries."""
        data = {
            "username": "alice",
            "password": "secret123",
            "email": "alice@example.com",
        }
        result = _redact_sensitive(data)

        assert result["username"] == "alice"
        assert result["password"] == REDACTED_VALUE
        assert result["email"] == "alice@example.com"

    def test_redact_sensitive_nested_dict(self):
        """Test redaction in nested dictionaries."""
        data = {
            "user": {
                "name": "alice",
                "settings": {
                    "api_key": "sk-12345",
                    "theme": "dark"
                }
            }
        }
        result = _redact_sensitive(data)

        assert result["user"]["name"] == "alice"
        assert result["user"]["settings"]["api_key"] == REDACTED_VALUE
        assert result["user"]["settings"]["theme"] == "dark"

    def test_redact_sensitive_credential_field(self):
        """Test that 'credentials' field is fully redacted (contains 'credential')."""
        data = {
            "user": "alice",
            "credentials": {"api_key": "sk-12345", "level": "admin"}
        }
        result = _redact_sensitive(data)

        assert result["user"] == "alice"
        # "credentials" contains "credential" pattern, so entire value is redacted
        assert result["credentials"] == REDACTED_VALUE

    def test_redact_sensitive_list(self):
        """Test redaction in lists."""
        data = [
            {"name": "alice", "token": "abc"},
            {"name": "bob", "token": "xyz"},
        ]
        result = _redact_sensitive(data)

        assert result[0]["name"] == "alice"
        assert result[0]["token"] == REDACTED_VALUE
        assert result[1]["name"] == "bob"
        assert result[1]["token"] == REDACTED_VALUE

    def test_redact_sensitive_max_depth(self):
        """Test that max depth prevents infinite recursion."""
        # Deeply nested structure
        data = {"a": {"b": {"c": {"d": {"secret": "value"}}}}}
        result = _redact_sensitive(data, max_depth=2)

        # Should stop recursing before reaching the secret
        assert result["a"]["b"] == {"c": {"d": {"secret": "value"}}}

    def test_structured_formatter_redacts_sensitive_fields(self):
        """Test that StructuredFormatter redacts sensitive extra fields."""
        formatter = StructuredFormatter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="/test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None
        )
        record.password = "secret123"
        record.user_id = "12345"

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["extra"]["password"] == REDACTED_VALUE
        assert parsed["extra"]["user_id"] == "12345"

    def test_structured_formatter_redacts_nested_sensitive(self):
        """Test that StructuredFormatter redacts nested sensitive data."""
        formatter = StructuredFormatter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="/test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None
        )
        record.request_data = {
            "user": "alice",
            "auth_token": "bearer xyz123"
        }

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["extra"]["request_data"]["user"] == "alice"
        assert parsed["extra"]["request_data"]["auth_token"] == REDACTED_VALUE


class TestHealthCheck:
    """Test health check functionality."""

    def setup_method(self):
        """Reset global health check before each test."""
        reset_health_check()

    def test_health_check_init(self):
        """Test HealthCheck initialization."""
        hc = HealthCheck(version="1.0.0")
        assert hc.version == "1.0.0"

    def test_register_decorator(self):
        """Test health check registration via decorator."""
        hc = HealthCheck()

        @hc.register("test_component")
        def check_test():
            return True

        assert "test_component" in hc._checks
        assert check_test() is True

    def test_add_check_method(self):
        """Test add_check method."""
        hc = HealthCheck()
        hc.add_check("manual_check", lambda: True)

        assert "manual_check" in hc._checks

    def test_remove_check(self):
        """Test remove_check method."""
        hc = HealthCheck()
        hc.add_check("removable", lambda: True)
        assert hc.remove_check("removable") is True
        assert "removable" not in hc._checks
        assert hc.remove_check("nonexistent") is False

    def test_check_returns_health_report(self):
        """Test check() returns HealthReport."""
        hc = HealthCheck(version="2.0.0")
        hc.add_check("component1", lambda: True)

        report = hc.check()

        assert isinstance(report, HealthReport)
        assert report.version == "2.0.0"
        assert report.status == HealthStatus.HEALTHY
        assert len(report.components) == 1

    def test_check_healthy_status(self):
        """Test healthy status when all checks pass."""
        hc = HealthCheck()
        hc.add_check("db", lambda: True)
        hc.add_check("cache", lambda: True)

        report = hc.check()

        assert report.status == HealthStatus.HEALTHY
        assert report.is_healthy is True
        assert report.is_ready is True

    def test_check_unhealthy_critical_failure(self):
        """Test unhealthy status when critical check fails."""
        hc = HealthCheck()
        hc.add_check("critical_db", lambda: False, critical=True)
        hc.add_check("cache", lambda: True)

        report = hc.check()

        assert report.status == HealthStatus.UNHEALTHY
        assert report.is_healthy is False

    def test_check_degraded_non_critical_failure(self):
        """Test degraded status when non-critical check fails."""
        hc = HealthCheck()
        hc.add_check("db", lambda: True, critical=True)
        hc.add_check("cache", lambda: False, critical=False)

        report = hc.check()

        assert report.status == HealthStatus.DEGRADED
        assert report.is_ready is True  # Degraded is still ready

    def test_check_handles_exceptions(self):
        """Test check handles exceptions in health checks."""
        hc = HealthCheck()

        def failing_check():
            raise RuntimeError("Connection failed")

        hc.add_check("failing", failing_check, critical=True)

        report = hc.check()

        assert report.status == HealthStatus.UNHEALTHY
        assert "Connection failed" in report.components[0].message

    def test_check_timeout(self):
        """Test check timeout handling."""
        hc = HealthCheck(default_timeout_seconds=0.1)

        def slow_check():
            time.sleep(1)
            return True

        hc.add_check("slow", slow_check, timeout_seconds=0.1)

        report = hc.check()

        assert report.components[0].status == HealthStatus.UNHEALTHY
        assert "timed out" in report.components[0].message.lower()

    def test_check_caching(self):
        """Test health check caching."""
        hc = HealthCheck(cache_ttl_seconds=1.0)
        call_count = 0

        def counting_check():
            nonlocal call_count
            call_count += 1
            return True

        hc.add_check("counting", counting_check)

        # First call
        hc.check()
        assert call_count == 1

        # Second call (should use cache)
        hc.check()
        assert call_count == 1

        # Third call with cache disabled
        hc.check(use_cache=False)
        assert call_count == 2

    def test_check_dict_result(self):
        """Test check with dict result."""
        hc = HealthCheck()

        def detailed_check():
            return {
                "status": True,
                "message": "All good",
                "details": {"connections": 5}
            }

        hc.add_check("detailed", detailed_check)
        report = hc.check()

        assert report.status == HealthStatus.HEALTHY
        assert report.components[0].message == "All good"
        assert report.components[0].details == {"connections": 5}

    def test_liveness_probe(self):
        """Test liveness probe endpoint."""
        hc = HealthCheck()
        result = hc.liveness()

        assert result["status"] == "alive"
        assert "timestamp" in result

    def test_readiness_probe(self):
        """Test readiness probe endpoint."""
        hc = HealthCheck()
        hc.add_check("db", lambda: True)

        result = hc.readiness()

        assert result["ready"] is True
        assert result["status"] == "healthy"

    def test_startup_probe(self):
        """Test startup probe endpoint."""
        hc = HealthCheck()
        hc.add_check("db", lambda: True)

        result = hc.startup()

        assert result["started"] is True
        assert "uptime_seconds" in result

    def test_health_report_to_dict(self):
        """Test HealthReport to_dict method."""
        report = HealthReport(
            status=HealthStatus.HEALTHY,
            version="1.0.0",
            uptime_seconds=100.5,
            components=[
                ComponentHealth(
                    name="db",
                    status=HealthStatus.HEALTHY,
                    latency_ms=5.2
                )
            ],
            timestamp="2025-01-01T00:00:00Z"
        )

        d = report.to_dict()

        assert d["status"] == "healthy"
        assert d["version"] == "1.0.0"
        assert d["uptime_seconds"] == 100.5
        assert len(d["components"]) == 1

    def test_global_health_check_singleton(self):
        """Test global health_check function returns singleton."""
        hc1 = health_check(version="1.0.0")
        hc2 = health_check(version="2.0.0")  # Version ignored

        assert hc1 is hc2
        assert hc1.version == "1.0.0"

    def test_component_health_to_dict(self):
        """Test ComponentHealth to_dict method."""
        component = ComponentHealth(
            name="database",
            status=HealthStatus.HEALTHY,
            message="Connected",
            latency_ms=12.5,
            details={"pool_size": 10},
            last_check="2025-01-01T00:00:00Z"
        )

        d = component.to_dict()

        assert d["name"] == "database"
        assert d["status"] == "healthy"
        assert d["message"] == "Connected"
        assert d["latency_ms"] == 12.5
        assert d["details"]["pool_size"] == 10


class TestLoggingIntegration:
    """Integration tests for logging with correlation context."""

    def test_full_request_flow(self):
        """Test logging across a simulated request flow."""
        logger = get_logger("test.integration")

        with CorrelationContext(
            correlation_id="req-abc-123",
            tenant_id="tenant-1",
            user_id="user-42"
        ):
            # Simulate request handling
            logger.info("Request received", endpoint="/api/test")
            logger.debug("Processing data")
            logger.info("Request completed", status=200)

            # Verify context is available
            assert get_correlation_id() == "req-abc-123"
            assert get_tenant_id() == "tenant-1"
            assert get_user_id() == "user-42"

        # Context should be cleared
        assert get_correlation_id() is None

    def test_nested_correlation_contexts(self):
        """Test nested correlation contexts."""
        with CorrelationContext(correlation_id="parent"):
            assert get_correlation_id() == "parent"

            with CorrelationContext(correlation_id="child"):
                assert get_correlation_id() == "child"

            assert get_correlation_id() == "parent"

        assert get_correlation_id() is None
