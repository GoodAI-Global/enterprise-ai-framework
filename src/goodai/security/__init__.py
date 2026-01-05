"""Security module for enterprise deployments."""

from goodai.security.tenancy import (
    TenantContext,
    Tenant,
    TenantManager,
    get_current_tenant,
    require_tenant,
)
from goodai.security.audit import (
    AuditLogger,
    AuditEvent,
    AuditEventType,
    get_audit_logger,
)
from goodai.security.rbac import (
    Permission,
    Role,
    RBACManager,
    require_permission,
    require_role,
)

__all__ = [
    # Tenancy
    "TenantContext",
    "Tenant",
    "TenantManager",
    "get_current_tenant",
    "require_tenant",
    # Audit
    "AuditLogger",
    "AuditEvent",
    "AuditEventType",
    "get_audit_logger",
    # RBAC
    "Permission",
    "Role",
    "RBACManager",
    "require_permission",
    "require_role",
]
