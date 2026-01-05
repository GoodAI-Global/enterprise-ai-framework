"""
Multi-Tenancy Module

Enterprise multi-tenancy support for:
- Tenant isolation
- Per-tenant configuration
- Tenant context propagation

Good AI Philosophy: Non-invasive by default - tenancy is optional.
"""

import threading
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

from goodai.monitoring import get_logger

logger = get_logger(__name__)

# Context variable for current tenant
_current_tenant: ContextVar[Optional["Tenant"]] = ContextVar(
    "current_tenant", default=None
)


@dataclass
class TenantConfig:
    """Tenant-specific configuration."""

    # Feature flags
    features: Dict[str, bool] = field(default_factory=dict)

    # Limits
    max_requests_per_minute: int = 100
    max_concurrent_jobs: int = 10
    max_data_retention_days: int = 365

    # AI settings
    anomaly_threshold: float = 2.5
    confidence_threshold: float = 0.7

    # Custom settings
    custom: Dict[str, Any] = field(default_factory=dict)

    def get_feature(self, name: str, default: bool = False) -> bool:
        """Get feature flag value."""
        return self.features.get(name, default)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "features": self.features,
            "max_requests_per_minute": self.max_requests_per_minute,
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "max_data_retention_days": self.max_data_retention_days,
            "anomaly_threshold": self.anomaly_threshold,
            "confidence_threshold": self.confidence_threshold,
            "custom": self.custom,
        }


@dataclass
class Tenant:
    """Tenant entity."""

    id: str
    name: str
    config: TenantConfig = field(default_factory=TenantConfig)
    active: bool = True
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.config, dict):
            self.config = TenantConfig(**self.config)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "config": self.config.to_dict(),
            "active": self.active,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


class TenantContext:
    """
    Context manager for tenant scope.

    Sets the current tenant for the duration of the context.
    Thread-safe and async-safe using contextvars.

    Example:
        >>> tenant = Tenant(id="acme", name="ACME Corp")
        >>> with TenantContext(tenant):
        ...     current = get_current_tenant()
        ...     print(current.name)  # ACME Corp
    """

    def __init__(self, tenant: Tenant):
        self.tenant = tenant
        self._previous_tenant: Optional[Tenant] = None

    def __enter__(self) -> "TenantContext":
        self._previous_tenant = _current_tenant.get()
        _current_tenant.set(self.tenant)
        logger.debug(
            "Tenant context entered",
            tenant_id=self.tenant.id,
            tenant_name=self.tenant.name
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _current_tenant.set(self._previous_tenant)
        return False


def get_current_tenant() -> Optional[Tenant]:
    """Get the current tenant from context."""
    return _current_tenant.get()


def require_tenant(func: Callable) -> Callable:
    """
    Decorator to require tenant context.

    Raises TenantRequiredError if no tenant is set.

    Example:
        >>> @require_tenant
        ... def process_data(data):
        ...     tenant = get_current_tenant()
        ...     # Process with tenant context
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        tenant = get_current_tenant()
        if tenant is None:
            raise TenantRequiredError(
                f"Function {func.__name__} requires tenant context"
            )
        return func(*args, **kwargs)
    return wrapper


class TenantRequiredError(Exception):
    """Raised when tenant context is required but not set."""
    pass


class TenantManager:
    """
    Manage tenant lifecycle and lookup.

    Provides in-memory tenant storage with optional persistence hooks.

    Example:
        >>> manager = TenantManager()
        >>> tenant = manager.create_tenant("acme", "ACME Corp")
        >>> manager.get_tenant("acme")
    """

    def __init__(self):
        self._tenants: Dict[str, Tenant] = {}
        self._lock = threading.Lock()

    def create_tenant(
        self,
        tenant_id: str,
        name: str,
        config: Optional[TenantConfig] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tenant:
        """
        Create a new tenant.

        Args:
            tenant_id: Unique tenant identifier
            name: Human-readable name
            config: Tenant configuration
            metadata: Additional metadata

        Returns:
            Created Tenant

        Raises:
            ValueError: If tenant already exists
        """
        with self._lock:
            if tenant_id in self._tenants:
                raise ValueError(f"Tenant {tenant_id} already exists")

            tenant = Tenant(
                id=tenant_id,
                name=name,
                config=config or TenantConfig(),
                metadata=metadata or {}
            )
            self._tenants[tenant_id] = tenant

            logger.info(
                "Tenant created",
                tenant_id=tenant_id,
                tenant_name=name
            )

            return tenant

    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get tenant by ID."""
        return self._tenants.get(tenant_id)

    def get_tenant_or_raise(self, tenant_id: str) -> Tenant:
        """Get tenant by ID or raise error."""
        tenant = self.get_tenant(tenant_id)
        if tenant is None:
            raise ValueError(f"Tenant {tenant_id} not found")
        return tenant

    def update_tenant(
        self,
        tenant_id: str,
        name: Optional[str] = None,
        config: Optional[TenantConfig] = None,
        metadata: Optional[Dict[str, Any]] = None,
        active: Optional[bool] = None
    ) -> Tenant:
        """
        Update tenant properties.

        Args:
            tenant_id: Tenant to update
            name: New name (optional)
            config: New config (optional)
            metadata: New metadata (optional)
            active: Active status (optional)

        Returns:
            Updated Tenant
        """
        with self._lock:
            tenant = self.get_tenant_or_raise(tenant_id)

            if name is not None:
                tenant.name = name
            if config is not None:
                tenant.config = config
            if metadata is not None:
                tenant.metadata = metadata
            if active is not None:
                tenant.active = active

            logger.info("Tenant updated", tenant_id=tenant_id)

            return tenant

    def delete_tenant(self, tenant_id: str) -> bool:
        """
        Delete a tenant.

        Args:
            tenant_id: Tenant to delete

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if tenant_id in self._tenants:
                del self._tenants[tenant_id]
                logger.info("Tenant deleted", tenant_id=tenant_id)
                return True
            return False

    def list_tenants(self, active_only: bool = False) -> List[Tenant]:
        """
        List all tenants.

        Args:
            active_only: Only return active tenants

        Returns:
            List of Tenant objects
        """
        tenants = list(self._tenants.values())
        if active_only:
            tenants = [t for t in tenants if t.active]
        return tenants

    def tenant_exists(self, tenant_id: str) -> bool:
        """Check if tenant exists."""
        return tenant_id in self._tenants


# Global tenant manager instance
_tenant_manager: Optional[TenantManager] = None


def get_tenant_manager() -> TenantManager:
    """Get global tenant manager instance."""
    global _tenant_manager
    if _tenant_manager is None:
        _tenant_manager = TenantManager()
    return _tenant_manager


def reset_tenant_manager() -> None:
    """Reset global tenant manager (for testing)."""
    global _tenant_manager
    _tenant_manager = None
