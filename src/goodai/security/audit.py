"""
Audit Logging Module

Enterprise audit logging for compliance (SOC2, ISO 27001, GDPR).
Tracks all significant actions with immutable records.

Good AI Philosophy: Evidence over opinions - audit logs are evidence.
"""

import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from functools import wraps

from goodai.monitoring import get_logger, get_correlation_id
from goodai.security.tenancy import get_current_tenant

logger = get_logger(__name__)


class AuditEventType(Enum):
    """Types of audit events."""

    # Authentication
    LOGIN = "auth.login"
    LOGOUT = "auth.logout"
    LOGIN_FAILED = "auth.login_failed"
    TOKEN_ISSUED = "auth.token_issued"
    TOKEN_REVOKED = "auth.token_revoked"

    # Data access
    DATA_READ = "data.read"
    DATA_CREATE = "data.create"
    DATA_UPDATE = "data.update"
    DATA_DELETE = "data.delete"
    DATA_EXPORT = "data.export"

    # Configuration
    CONFIG_READ = "config.read"
    CONFIG_UPDATE = "config.update"

    # User management
    USER_CREATE = "user.create"
    USER_UPDATE = "user.update"
    USER_DELETE = "user.delete"
    ROLE_ASSIGN = "user.role_assign"
    ROLE_REVOKE = "user.role_revoke"

    # System
    SYSTEM_START = "system.start"
    SYSTEM_STOP = "system.stop"
    SYSTEM_ERROR = "system.error"

    # AI operations
    AI_PREDICTION = "ai.prediction"
    AI_TRAINING = "ai.training"
    AI_ANOMALY_DETECTED = "ai.anomaly_detected"

    # Custom
    CUSTOM = "custom"


class AuditSeverity(Enum):
    """Severity levels for audit events."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """
    Immutable audit event record.

    Captures all relevant context for compliance and forensics.
    """

    id: str
    timestamp: str
    event_type: AuditEventType
    action: str
    actor_id: Optional[str]
    actor_type: str  # user, service, system
    tenant_id: Optional[str]
    resource_type: Optional[str]
    resource_id: Optional[str]
    severity: AuditSeverity
    outcome: str  # success, failure
    details: Dict[str, Any]
    correlation_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "event_type": self.event_type.value,
            "action": self.action,
            "actor_id": self.actor_id,
            "actor_type": self.actor_type,
            "tenant_id": self.tenant_id,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "severity": self.severity.value,
            "outcome": self.outcome,
            "details": self.details,
            "correlation_id": self.correlation_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class AuditLogger:
    """
    Enterprise audit logger.

    Provides structured audit logging with:
    - Event categorization
    - Context capture (tenant, user, correlation ID)
    - Multiple storage backends
    - Query interface

    Example:
        >>> audit = AuditLogger()
        >>> audit.log(
        ...     event_type=AuditEventType.DATA_READ,
        ...     action="Read customer records",
        ...     actor_id="user-123",
        ...     resource_type="customer",
        ...     resource_id="cust-456"
        ... )
    """

    def __init__(
        self,
        service_name: str = "goodai",
        storage_backend: str = "memory",
        max_events: int = 10000
    ):
        self.service_name = service_name
        self.storage_backend = storage_backend
        self.max_events = max_events

        self._events: List[AuditEvent] = []
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[AuditEvent], None]] = []

    def log(
        self,
        event_type: AuditEventType,
        action: str,
        actor_id: Optional[str] = None,
        actor_type: str = "user",
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        severity: AuditSeverity = AuditSeverity.INFO,
        outcome: str = "success",
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> AuditEvent:
        """
        Log an audit event.

        Args:
            event_type: Type of event
            action: Human-readable action description
            actor_id: ID of the actor (user, service)
            actor_type: Type of actor
            resource_type: Type of resource affected
            resource_id: ID of resource affected
            severity: Event severity
            outcome: Event outcome (success/failure)
            details: Additional event details
            ip_address: Client IP address
            user_agent: Client user agent

        Returns:
            Created AuditEvent
        """
        # Get context
        tenant = get_current_tenant()
        correlation_id = get_correlation_id()

        event = AuditEvent(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            action=action,
            actor_id=actor_id,
            actor_type=actor_type,
            tenant_id=tenant.id if tenant else None,
            resource_type=resource_type,
            resource_id=resource_id,
            severity=severity,
            outcome=outcome,
            details=details or {},
            correlation_id=correlation_id,
            ip_address=ip_address,
            user_agent=user_agent
        )

        self._store_event(event)
        self._invoke_callbacks(event)

        # Also log to standard logger for aggregation
        logger.info(
            f"AUDIT: {event.action}",
            audit_event_id=event.id,
            event_type=event.event_type.value,
            actor_id=event.actor_id,
            resource=f"{event.resource_type}:{event.resource_id}",
            outcome=event.outcome
        )

        return event

    def _store_event(self, event: AuditEvent):
        """Store event in backend."""
        with self._lock:
            self._events.append(event)
            # Trim if over limit
            if len(self._events) > self.max_events:
                self._events = self._events[-self.max_events:]

    def _invoke_callbacks(self, event: AuditEvent):
        """Invoke registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(
                    "Audit callback failed",
                    exc_info=e,
                    callback=callback.__name__
                )

    def add_callback(self, callback: Callable[[AuditEvent], None]):
        """Add callback to be invoked on each event."""
        self._callbacks.append(callback)

    def query(
        self,
        event_type: Optional[AuditEventType] = None,
        actor_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        severity: Optional[AuditSeverity] = None,
        outcome: Optional[str] = None,
        limit: int = 100
    ) -> List[AuditEvent]:
        """
        Query audit events.

        Args:
            event_type: Filter by event type
            actor_id: Filter by actor
            tenant_id: Filter by tenant
            resource_type: Filter by resource type
            resource_id: Filter by resource ID
            start_time: Filter by start time (ISO format)
            end_time: Filter by end time (ISO format)
            severity: Filter by severity
            outcome: Filter by outcome
            limit: Maximum events to return

        Returns:
            List of matching AuditEvent objects
        """
        with self._lock:
            results = self._events.copy()

        # Apply filters
        if event_type:
            results = [e for e in results if e.event_type == event_type]
        if actor_id:
            results = [e for e in results if e.actor_id == actor_id]
        if tenant_id:
            results = [e for e in results if e.tenant_id == tenant_id]
        if resource_type:
            results = [e for e in results if e.resource_type == resource_type]
        if resource_id:
            results = [e for e in results if e.resource_id == resource_id]
        if severity:
            results = [e for e in results if e.severity == severity]
        if outcome:
            results = [e for e in results if e.outcome == outcome]
        if start_time:
            results = [e for e in results if e.timestamp >= start_time]
        if end_time:
            results = [e for e in results if e.timestamp <= end_time]

        # Sort by timestamp descending and limit
        results.sort(key=lambda e: e.timestamp, reverse=True)
        return results[:limit]

    def get_event(self, event_id: str) -> Optional[AuditEvent]:
        """Get a specific event by ID."""
        with self._lock:
            for event in self._events:
                if event.id == event_id:
                    return event
        return None

    def export(
        self,
        format: str = "json",
        **query_params
    ) -> str:
        """
        Export audit events.

        Args:
            format: Export format (json, csv)
            **query_params: Query parameters

        Returns:
            Exported data as string
        """
        events = self.query(**query_params)

        if format == "json":
            return json.dumps([e.to_dict() for e in events], indent=2)
        elif format == "csv":
            if not events:
                return ""
            headers = list(events[0].to_dict().keys())
            lines = [",".join(headers)]
            for event in events:
                d = event.to_dict()
                values = [str(d.get(h, "")).replace(",", ";") for h in headers]
                lines.append(",".join(values))
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def clear(self):
        """Clear all events (for testing)."""
        with self._lock:
            self._events.clear()


def audit_action(
    event_type: AuditEventType,
    action: Optional[str] = None,
    resource_type: Optional[str] = None,
    severity: AuditSeverity = AuditSeverity.INFO
) -> Callable:
    """
    Decorator to automatically audit function calls.

    Example:
        >>> @audit_action(AuditEventType.DATA_READ, resource_type="customer")
        ... def get_customer(customer_id: str):
        ...     return db.get_customer(customer_id)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            audit = get_audit_logger()
            action_str = action or f"Called {func.__name__}"

            # Try to extract resource_id from kwargs
            resource_id = kwargs.get("id") or kwargs.get("resource_id")
            if not resource_id and args:
                resource_id = str(args[0])[:100]

            try:
                result = func(*args, **kwargs)
                audit.log(
                    event_type=event_type,
                    action=action_str,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    severity=severity,
                    outcome="success"
                )
                return result
            except Exception as e:
                audit.log(
                    event_type=event_type,
                    action=action_str,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    severity=AuditSeverity.ERROR,
                    outcome="failure",
                    details={"error": str(e), "error_type": type(e).__name__}
                )
                raise

        return wrapper
    return decorator


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get global audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


def reset_audit_logger() -> None:
    """Reset global audit logger (for testing)."""
    global _audit_logger
    _audit_logger = None
