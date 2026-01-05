"""
RBAC (Role-Based Access Control) Module

Enterprise access control with:
- Permission definitions
- Role management
- Permission checking

Good AI Philosophy: Non-invasive by default - RBAC is optional.
"""

import threading
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Callable, Dict, List, Optional, Set

from goodai.monitoring import get_logger

logger = get_logger(__name__)

# Context variable for current user's roles
_current_roles: ContextVar[Optional[Set[str]]] = ContextVar("current_roles", default=None)
_current_user_id: ContextVar[Optional[str]] = ContextVar("current_user_id", default=None)


class Permission(Enum):
    """Standard permissions."""

    # Data permissions
    DATA_READ = "data:read"
    DATA_WRITE = "data:write"
    DATA_DELETE = "data:delete"
    DATA_EXPORT = "data:export"

    # Analysis permissions
    ANOMALY_DETECT = "anomaly:detect"
    ASSESSMENT_RUN = "assessment:run"
    ROI_CALCULATE = "roi:calculate"

    # Admin permissions
    USER_MANAGE = "user:manage"
    ROLE_MANAGE = "role:manage"
    TENANT_MANAGE = "tenant:manage"
    CONFIG_MANAGE = "config:manage"

    # System permissions
    SYSTEM_MONITOR = "system:monitor"
    SYSTEM_ADMIN = "system:admin"
    AUDIT_READ = "audit:read"

    # Special
    SUPERUSER = "*"


@dataclass
class Role:
    """Role with assigned permissions."""

    name: str
    description: str
    permissions: Set[Permission] = field(default_factory=set)
    inherits: List[str] = field(default_factory=list)  # Inherited roles

    def has_permission(self, permission: Permission) -> bool:
        """Check if role has permission."""
        if Permission.SUPERUSER in self.permissions:
            return True
        return permission in self.permissions

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "permissions": [p.value for p in self.permissions],
            "inherits": self.inherits,
        }


class UserContext:
    """
    Context manager for user/role scope.

    Sets the current user and roles for access control checks.

    Example:
        >>> with UserContext(user_id="user-123", roles={"analyst", "viewer"}):
        ...     if has_permission(Permission.DATA_READ):
        ...         data = fetch_data()
    """

    def __init__(self, user_id: str, roles: Set[str]):
        self.user_id = user_id
        self.roles = roles
        self._previous_roles: Set[str] = set()
        self._previous_user_id: Optional[str] = None

    def __enter__(self) -> "UserContext":
        self._previous_roles = _current_roles.get()
        self._previous_user_id = _current_user_id.get()
        _current_roles.set(self.roles)
        _current_user_id.set(self.user_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _current_roles.set(self._previous_roles)
        _current_user_id.set(self._previous_user_id)
        return False


def get_current_roles() -> Set[str]:
    """Get current user's roles from context."""
    return _current_roles.get() or set()


def get_current_user_id() -> Optional[str]:
    """Get current user ID from context."""
    return _current_user_id.get()


class RBACManager:
    """
    Manage roles and permissions.

    Provides role definition, assignment, and checking.

    Example:
        >>> rbac = RBACManager()
        >>>
        >>> # Define roles
        >>> rbac.create_role(
        ...     "analyst",
        ...     "Data Analyst",
        ...     {Permission.DATA_READ, Permission.ANOMALY_DETECT}
        ... )
        >>>
        >>> # Check permissions
        >>> if rbac.has_permission("analyst", Permission.DATA_READ):
        ...     print("Can read data")
    """

    # Built-in roles
    BUILT_IN_ROLES = {
        "viewer": Role(
            name="viewer",
            description="Read-only access",
            permissions={Permission.DATA_READ, Permission.SYSTEM_MONITOR}
        ),
        "analyst": Role(
            name="analyst",
            description="Data analysis access",
            permissions={
                Permission.DATA_READ,
                Permission.ANOMALY_DETECT,
                Permission.ASSESSMENT_RUN,
                Permission.ROI_CALCULATE,
            },
            inherits=["viewer"]
        ),
        "operator": Role(
            name="operator",
            description="Operational access",
            permissions={
                Permission.DATA_READ,
                Permission.DATA_WRITE,
                Permission.ANOMALY_DETECT,
            },
            inherits=["viewer"]
        ),
        "admin": Role(
            name="admin",
            description="Administrative access",
            permissions={
                Permission.USER_MANAGE,
                Permission.ROLE_MANAGE,
                Permission.CONFIG_MANAGE,
                Permission.AUDIT_READ,
            },
            inherits=["analyst", "operator"]
        ),
        "superuser": Role(
            name="superuser",
            description="Full access",
            permissions={Permission.SUPERUSER}
        ),
    }

    def __init__(self, include_builtin: bool = True):
        self._roles: Dict[str, Role] = {}
        self._lock = threading.Lock()

        if include_builtin:
            for name, role in self.BUILT_IN_ROLES.items():
                self._roles[name] = role

    def create_role(
        self,
        name: str,
        description: str,
        permissions: Set[Permission],
        inherits: Optional[List[str]] = None
    ) -> Role:
        """
        Create a new role.

        Args:
            name: Role name
            description: Human-readable description
            permissions: Set of permissions
            inherits: List of role names to inherit from

        Returns:
            Created Role

        Raises:
            ValueError: If role already exists
        """
        with self._lock:
            if name in self._roles:
                raise ValueError(f"Role {name} already exists")

            role = Role(
                name=name,
                description=description,
                permissions=permissions,
                inherits=inherits or []
            )
            self._roles[name] = role

            logger.info("Role created", role_name=name)
            return role

    def get_role(self, name: str) -> Optional[Role]:
        """Get role by name."""
        return self._roles.get(name)

    def update_role(
        self,
        name: str,
        description: Optional[str] = None,
        permissions: Optional[Set[Permission]] = None,
        inherits: Optional[List[str]] = None
    ) -> Role:
        """Update role properties."""
        with self._lock:
            role = self._roles.get(name)
            if role is None:
                raise ValueError(f"Role {name} not found")

            if description is not None:
                role.description = description
            if permissions is not None:
                role.permissions = permissions
            if inherits is not None:
                role.inherits = inherits

            logger.info("Role updated", role_name=name)
            return role

    def delete_role(self, name: str) -> bool:
        """Delete a role."""
        with self._lock:
            if name in self._roles:
                # Don't allow deleting built-in roles
                if name in self.BUILT_IN_ROLES:
                    raise ValueError(f"Cannot delete built-in role: {name}")
                del self._roles[name]
                logger.info("Role deleted", role_name=name)
                return True
            return False

    def list_roles(self) -> List[Role]:
        """List all roles."""
        return list(self._roles.values())

    def get_effective_permissions(self, role_name: str) -> Set[Permission]:
        """
        Get all permissions for a role, including inherited.

        Args:
            role_name: Role to check

        Returns:
            Set of all effective permissions
        """
        role = self._roles.get(role_name)
        if role is None:
            return set()

        permissions = role.permissions.copy()

        # Add inherited permissions
        for parent_name in role.inherits:
            permissions.update(self.get_effective_permissions(parent_name))

        return permissions

    def has_permission(
        self,
        role_name: str,
        permission: Permission
    ) -> bool:
        """
        Check if role has permission.

        Args:
            role_name: Role to check
            permission: Permission to check

        Returns:
            True if role has permission
        """
        permissions = self.get_effective_permissions(role_name)
        return Permission.SUPERUSER in permissions or permission in permissions

    def check_permission(
        self,
        permission: Permission,
        roles: Optional[Set[str]] = None
    ) -> bool:
        """
        Check if any of the given roles has permission.

        Args:
            permission: Permission to check
            roles: Roles to check (uses context if not provided)

        Returns:
            True if any role has permission
        """
        if roles is None:
            roles = get_current_roles()

        for role_name in roles:
            if self.has_permission(role_name, permission):
                return True
        return False


class PermissionDeniedError(Exception):
    """Raised when permission check fails."""

    def __init__(self, permission: Permission, message: Optional[str] = None):
        self.permission = permission
        self.message = message or f"Permission denied: {permission.value}"
        super().__init__(self.message)


def require_permission(permission: Permission) -> Callable:
    """
    Decorator to require a permission.

    Raises PermissionDeniedError if current user lacks permission.

    Example:
        >>> @require_permission(Permission.DATA_READ)
        ... def get_data():
        ...     return fetch_sensitive_data()
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            rbac = get_rbac_manager()
            if not rbac.check_permission(permission):
                logger.warning(
                    "Permission denied",
                    permission=permission.value,
                    user_id=get_current_user_id(),
                    roles=list(get_current_roles())
                )
                raise PermissionDeniedError(permission)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def require_role(role_name: str) -> Callable:
    """
    Decorator to require a specific role.

    Raises PermissionDeniedError if current user lacks role.

    Example:
        >>> @require_role("admin")
        ... def admin_action():
        ...     return perform_admin_task()
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            roles = get_current_roles()
            if role_name not in roles:
                logger.warning(
                    "Role required",
                    required_role=role_name,
                    user_id=get_current_user_id(),
                    roles=list(roles)
                )
                raise PermissionDeniedError(
                    Permission.SUPERUSER,
                    f"Role required: {role_name}"
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Global RBAC manager instance
_rbac_manager: Optional[RBACManager] = None


def get_rbac_manager() -> RBACManager:
    """Get global RBAC manager instance."""
    global _rbac_manager
    if _rbac_manager is None:
        _rbac_manager = RBACManager()
    return _rbac_manager


def reset_rbac_manager() -> None:
    """Reset global RBAC manager (for testing)."""
    global _rbac_manager
    _rbac_manager = None
