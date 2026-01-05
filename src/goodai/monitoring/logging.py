"""
Structured Logging Module

Enterprise-grade structured logging with correlation IDs for distributed tracing.
Supports JSON output for log aggregation systems (ELK, Datadog, Splunk).

Good AI Philosophy: Evidence over opinions - comprehensive logging enables this.
"""

import json
import logging
import sys
import threading
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, Optional, Union


# Context variable for correlation ID (thread-safe, async-safe)
_correlation_id: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)
_tenant_id: ContextVar[Optional[str]] = ContextVar("tenant_id", default=None)
_user_id: ContextVar[Optional[str]] = ContextVar("user_id", default=None)

# Sensitive field patterns for redaction (security)
SENSITIVE_PATTERNS = frozenset({
    "password", "passwd", "pwd", "secret", "token", "api_key", "apikey",
    "access_token", "refresh_token", "authorization", "auth", "credential",
    "private_key", "privatekey", "ssh_key", "sshkey", "secret_key", "secretkey",
    "bearer", "jwt", "session", "cookie", "credit_card", "creditcard", "ssn",
    "social_security", "bank_account", "account_number"
})

REDACTED_VALUE = "[REDACTED]"


def _is_sensitive_key(key: str) -> bool:
    """Check if a key name suggests sensitive data."""
    key_lower = key.lower()
    return any(pattern in key_lower for pattern in SENSITIVE_PATTERNS)


def _redact_sensitive(data: Any, max_depth: int = 10) -> Any:
    """
    Recursively redact sensitive fields from data.

    Args:
        data: Data to redact
        max_depth: Maximum recursion depth to prevent infinite loops

    Returns:
        Data with sensitive fields redacted
    """
    if max_depth <= 0:
        return data

    if isinstance(data, dict):
        return {
            k: REDACTED_VALUE if _is_sensitive_key(k) else _redact_sensitive(v, max_depth - 1)
            for k, v in data.items()
        }
    elif isinstance(data, list):
        return [_redact_sensitive(item, max_depth - 1) for item in data]
    elif isinstance(data, tuple):
        return tuple(_redact_sensitive(item, max_depth - 1) for item in data)
    return data


class LogLevel(Enum):
    """Log level enumeration."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogContext:
    """Additional context for log entries."""

    correlation_id: Optional[str] = None
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class CorrelationContext:
    """
    Context manager for correlation ID propagation.

    Automatically generates or uses provided correlation ID for request tracing.
    Thread-safe and async-safe using contextvars.

    Example:
        >>> with CorrelationContext() as ctx:
        ...     logger.info("Processing request")  # Includes correlation_id
        ...     result = process_data()
        ...     logger.info("Request complete", extra={"result": result})

        >>> # With existing correlation ID (from HTTP header)
        >>> with CorrelationContext(correlation_id="abc-123"):
        ...     logger.info("Continuing trace")
    """

    def __init__(
        self,
        correlation_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.tenant_id = tenant_id
        self.user_id = user_id
        self._previous_correlation: Optional[str] = None
        self._previous_tenant: Optional[str] = None
        self._previous_user: Optional[str] = None

    def __enter__(self) -> "CorrelationContext":
        # Save previous values
        self._previous_correlation = _correlation_id.get()
        self._previous_tenant = _tenant_id.get()
        self._previous_user = _user_id.get()

        # Set new values
        _correlation_id.set(self.correlation_id)
        if self.tenant_id:
            _tenant_id.set(self.tenant_id)
        if self.user_id:
            _user_id.set(self.user_id)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous values
        _correlation_id.set(self._previous_correlation)
        _tenant_id.set(self._previous_tenant)
        _user_id.set(self._previous_user)
        return False


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID from context."""
    return _correlation_id.get()


def get_tenant_id() -> Optional[str]:
    """Get current tenant ID from context."""
    return _tenant_id.get()


def get_user_id() -> Optional[str]:
    """Get current user ID from context."""
    return _user_id.get()


class StructuredFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.

    Produces log entries compatible with ELK, Datadog, Splunk, etc.
    """

    def __init__(
        self,
        service_name: str = "goodai",
        environment: str = "development",
        include_stack_info: bool = True
    ):
        super().__init__()
        self.service_name = service_name
        self.environment = environment
        self.include_stack_info = include_stack_info

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Base log entry
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.service_name,
            "environment": self.environment,
        }

        # Add correlation context
        correlation_id = get_correlation_id()
        if correlation_id:
            log_entry["correlation_id"] = correlation_id

        tenant_id = get_tenant_id()
        if tenant_id:
            log_entry["tenant_id"] = tenant_id

        user_id = get_user_id()
        if user_id:
            log_entry["user_id"] = user_id

        # Add source location
        log_entry["source"] = {
            "file": record.pathname,
            "line": record.lineno,
            "function": record.funcName,
        }

        # Add exception info if present
        if record.exc_info and self.include_stack_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "stacktrace": self.formatException(record.exc_info),
            }

        # Add extra fields (with sensitive data redaction)
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "exc_info", "exc_text", "thread", "threadName",
                "message", "asctime"
            }:
                # Redact sensitive fields at top level
                if _is_sensitive_key(key):
                    extra_fields[key] = REDACTED_VALUE
                    continue

                try:
                    # Ensure value is JSON serializable
                    json.dumps(value)
                    # Recursively redact sensitive data in nested structures
                    extra_fields[key] = _redact_sensitive(value)
                except (TypeError, ValueError):
                    extra_fields[key] = str(value)

        if extra_fields:
            log_entry["extra"] = extra_fields

        return json.dumps(log_entry, default=str)


class HumanReadableFormatter(logging.Formatter):
    """Human-readable formatter for development."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        """Format log record for human readability."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level = record.levelname

        # Add color if enabled
        if self.use_colors:
            color = self.COLORS.get(level, "")
            level_str = f"{color}{level:8}{self.RESET}"
        else:
            level_str = f"{level:8}"

        # Build message
        correlation_id = get_correlation_id()
        correlation_str = f"[{correlation_id[:8]}] " if correlation_id else ""

        message = f"{timestamp} {level_str} {correlation_str}{record.name}: {record.getMessage()}"

        # Add exception if present
        if record.exc_info:
            message += f"\n{self.formatException(record.exc_info)}"

        return message


class StructuredLogger:
    """
    Structured logger wrapper with convenience methods.

    Provides structured logging with automatic context propagation.

    Example:
        >>> logger = StructuredLogger("myservice.module")
        >>> logger.info("User logged in", user_id="123", ip="192.168.1.1")
        >>> logger.error("Failed to process", error=e, request_id="abc")
    """

    def __init__(self, name: str):
        self._logger = logging.getLogger(name)
        self.name = name

    def _log(
        self,
        level: int,
        message: str,
        exc_info: Optional[Exception] = None,
        **kwargs
    ):
        """Internal log method with extra fields."""
        self._logger.log(level, message, exc_info=exc_info, extra=kwargs)

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, exc_info: Optional[Exception] = None, **kwargs):
        """Log error message with optional exception."""
        self._log(logging.ERROR, message, exc_info=exc_info, **kwargs)

    def critical(self, message: str, exc_info: Optional[Exception] = None, **kwargs):
        """Log critical message with optional exception."""
        self._log(logging.CRITICAL, message, exc_info=exc_info, **kwargs)

    def exception(self, message: str, **kwargs):
        """Log exception with traceback."""
        self._logger.exception(message, extra=kwargs)

    def bind(self, **kwargs) -> "BoundLogger":
        """Create a bound logger with persistent extra fields."""
        return BoundLogger(self, kwargs)


class BoundLogger:
    """Logger with persistent bound context."""

    def __init__(self, logger: StructuredLogger, context: Dict[str, Any]):
        self._logger = logger
        self._context = context

    def _merge_context(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge bound context with call-time kwargs."""
        merged = self._context.copy()
        merged.update(kwargs)
        return merged

    def debug(self, message: str, **kwargs):
        self._logger.debug(message, **self._merge_context(kwargs))

    def info(self, message: str, **kwargs):
        self._logger.info(message, **self._merge_context(kwargs))

    def warning(self, message: str, **kwargs):
        self._logger.warning(message, **self._merge_context(kwargs))

    def error(self, message: str, **kwargs):
        self._logger.error(message, **self._merge_context(kwargs))

    def critical(self, message: str, **kwargs):
        self._logger.critical(message, **self._merge_context(kwargs))

    def bind(self, **kwargs) -> "BoundLogger":
        """Create new bound logger with additional context."""
        merged = self._context.copy()
        merged.update(kwargs)
        return BoundLogger(self._logger, merged)


# Global logger cache
_loggers: Dict[str, StructuredLogger] = {}
_lock = threading.Lock()


def get_logger(name: str) -> StructuredLogger:
    """
    Get or create a structured logger.

    Args:
        name: Logger name (typically __name__ of the module)

    Returns:
        StructuredLogger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Starting process", job_id="123")
    """
    with _lock:
        if name not in _loggers:
            _loggers[name] = StructuredLogger(name)
        return _loggers[name]


def configure_logging(
    level: Union[str, LogLevel] = LogLevel.INFO,
    format: str = "json",
    service_name: str = "goodai",
    environment: str = "development",
    output: str = "stderr"
) -> None:
    """
    Configure global logging settings.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Output format ("json" for structured, "human" for readable)
        service_name: Service name for log entries
        environment: Environment name (development, staging, production)
        output: Output destination ("stderr", "stdout", or file path)

    Example:
        >>> configure_logging(
        ...     level=LogLevel.INFO,
        ...     format="json",
        ...     service_name="manufacturing-api",
        ...     environment="production"
        ... )
    """
    # Convert level if needed
    if isinstance(level, LogLevel):
        level_value = getattr(logging, level.value)
    else:
        level_value = getattr(logging, level.upper())

    # Create formatter
    if format == "json":
        formatter = StructuredFormatter(
            service_name=service_name,
            environment=environment
        )
    else:
        formatter = HumanReadableFormatter(use_colors=output == "stderr")

    # Create handler
    if output == "stderr":
        handler = logging.StreamHandler(sys.stderr)
    elif output == "stdout":
        handler = logging.StreamHandler(sys.stdout)
    else:
        handler = logging.FileHandler(output)

    handler.setFormatter(formatter)
    handler.setLevel(level_value)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level_value)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Also configure goodai logger
    goodai_logger = logging.getLogger("goodai")
    goodai_logger.setLevel(level_value)


def log_execution(
    logger: Optional[StructuredLogger] = None,
    level: LogLevel = LogLevel.INFO,
    include_args: bool = False,
    include_result: bool = False
) -> Callable:
    """
    Decorator to log function execution.

    Args:
        logger: Logger to use (defaults to function's module logger)
        level: Log level for execution logs
        include_args: Whether to include function arguments in log
        include_result: Whether to include return value in log

    Example:
        >>> @log_execution(include_args=True)
        ... def process_data(data_id: str) -> dict:
        ...     return {"processed": True}
    """
    def decorator(func: Callable) -> Callable:
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)

        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__qualname__

            # Build log context
            log_kwargs = {"function": func_name}
            if include_args:
                log_kwargs["args"] = str(args)[:200]  # Truncate for safety
                log_kwargs["kwargs"] = str(kwargs)[:200]

            # Log start
            logger.info(f"Executing {func_name}", **log_kwargs)

            try:
                result = func(*args, **kwargs)

                # Log success
                success_kwargs = {"function": func_name, "status": "success"}
                if include_result:
                    success_kwargs["result"] = str(result)[:200]
                logger.info(f"Completed {func_name}", **success_kwargs)

                return result

            except Exception as e:
                # Log failure
                logger.error(
                    f"Failed {func_name}",
                    exc_info=e,
                    function=func_name,
                    status="failed",
                    error_type=type(e).__name__
                )
                raise

        return wrapper
    return decorator
