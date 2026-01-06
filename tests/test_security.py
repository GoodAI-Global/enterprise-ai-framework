"""
Tests for Security Module

Tests for multi-tenancy, audit logging, and RBAC.
"""

import json
import pytest

from goodai.security.tenancy import (
    Tenant,
    TenantConfig,
    TenantContext,
    TenantManager,
    TenantRequiredError,
    get_current_tenant,
    require_tenant,
    reset_tenant_manager,
)
from goodai.security.audit import (
    AuditLogger,
    AuditEvent,
    AuditEventType,
    AuditSeverity,
    audit_action,
    get_audit_logger,
    reset_audit_logger,
)
from goodai.security.rbac import (
    Permission,
    Role,
    RBACManager,
    UserContext,
    PermissionDeniedError,
    require_permission,
    require_role,
    get_current_roles,
    get_current_user_id,
    reset_rbac_manager,
)


class TestTenantConfig:
    """Test TenantConfig."""

    def test_default_values(self):
        """Test default config values."""
        config = TenantConfig()
        assert config.max_requests_per_minute == 100
        assert config.anomaly_threshold == 2.5

    def test_get_feature(self):
        """Test feature flag access."""
        config = TenantConfig(features={"beta": True, "legacy": False})
        assert config.get_feature("beta") is True
        assert config.get_feature("legacy") is False
        assert config.get_feature("unknown") is False
        assert config.get_feature("unknown", default=True) is True

    def test_to_dict(self):
        """Test config serialization."""
        config = TenantConfig(max_requests_per_minute=200)
        d = config.to_dict()
        assert d["max_requests_per_minute"] == 200
        assert "features" in d


class TestTenant:
    """Test Tenant."""

    def test_create_tenant(self):
        """Test tenant creation."""
        tenant = Tenant(id="acme", name="ACME Corp")
        assert tenant.id == "acme"
        assert tenant.name == "ACME Corp"
        assert tenant.active is True
        assert isinstance(tenant.config, TenantConfig)

    def test_tenant_with_config(self):
        """Test tenant with custom config."""
        config = TenantConfig(anomaly_threshold=3.0)
        tenant = Tenant(id="acme", name="ACME", config=config)
        assert tenant.config.anomaly_threshold == 3.0

    def test_tenant_to_dict(self):
        """Test tenant serialization."""
        tenant = Tenant(id="acme", name="ACME Corp")
        d = tenant.to_dict()
        assert d["id"] == "acme"
        assert d["name"] == "ACME Corp"
        assert "config" in d


class TestTenantContext:
    """Test TenantContext."""

    def setup_method(self):
        """Reset before each test."""
        reset_tenant_manager()

    def test_context_sets_tenant(self):
        """Test context sets current tenant."""
        tenant = Tenant(id="test", name="Test")
        with TenantContext(tenant):
            current = get_current_tenant()
            assert current is not None
            assert current.id == "test"

    def test_context_restores_previous(self):
        """Test context restores previous tenant."""
        tenant1 = Tenant(id="tenant1", name="Tenant 1")
        tenant2 = Tenant(id="tenant2", name="Tenant 2")

        with TenantContext(tenant1):
            assert get_current_tenant().id == "tenant1"
            with TenantContext(tenant2):
                assert get_current_tenant().id == "tenant2"
            assert get_current_tenant().id == "tenant1"

        assert get_current_tenant() is None

    def test_require_tenant_decorator(self):
        """Test require_tenant decorator."""
        @require_tenant
        def tenant_required_func():
            return get_current_tenant().id

        tenant = Tenant(id="required", name="Required")

        # Should fail without tenant
        with pytest.raises(TenantRequiredError):
            tenant_required_func()

        # Should succeed with tenant
        with TenantContext(tenant):
            result = tenant_required_func()
            assert result == "required"


class TestTenantManager:
    """Test TenantManager."""

    def setup_method(self):
        """Reset before each test."""
        reset_tenant_manager()

    def test_create_tenant(self):
        """Test tenant creation."""
        manager = TenantManager()
        tenant = manager.create_tenant("new-tenant", "New Tenant")

        assert tenant.id == "new-tenant"
        assert tenant.name == "New Tenant"
        assert manager.tenant_exists("new-tenant")

    def test_create_duplicate_raises(self):
        """Test duplicate tenant raises error."""
        manager = TenantManager()
        manager.create_tenant("dup", "First")

        with pytest.raises(ValueError, match="already exists"):
            manager.create_tenant("dup", "Second")

    def test_get_tenant(self):
        """Test tenant retrieval."""
        manager = TenantManager()
        manager.create_tenant("get-test", "Get Test")

        tenant = manager.get_tenant("get-test")
        assert tenant is not None
        assert tenant.name == "Get Test"

        assert manager.get_tenant("nonexistent") is None

    def test_update_tenant(self):
        """Test tenant update."""
        manager = TenantManager()
        manager.create_tenant("update-test", "Original")

        updated = manager.update_tenant("update-test", name="Updated")
        assert updated.name == "Updated"

    def test_delete_tenant(self):
        """Test tenant deletion."""
        manager = TenantManager()
        manager.create_tenant("delete-test", "Delete Me")

        assert manager.delete_tenant("delete-test") is True
        assert manager.get_tenant("delete-test") is None
        assert manager.delete_tenant("nonexistent") is False

    def test_list_tenants(self):
        """Test tenant listing."""
        manager = TenantManager()
        manager.create_tenant("t1", "Tenant 1")
        manager.create_tenant("t2", "Tenant 2", metadata={"status": "active"})

        tenants = manager.list_tenants()
        assert len(tenants) == 2

    def test_list_active_only(self):
        """Test listing active tenants only."""
        manager = TenantManager()
        manager.create_tenant("active", "Active")
        manager.create_tenant("inactive", "Inactive")
        manager.update_tenant("inactive", active=False)

        active = manager.list_tenants(active_only=True)
        assert len(active) == 1
        assert active[0].id == "active"


class TestAuditLogger:
    """Test AuditLogger."""

    def setup_method(self):
        """Reset before each test."""
        reset_audit_logger()

    def test_log_event(self):
        """Test logging an audit event."""
        audit = AuditLogger()
        event = audit.log(
            event_type=AuditEventType.DATA_READ,
            action="Read customer data",
            actor_id="user-123",
            resource_type="customer",
            resource_id="cust-456"
        )

        assert event.id is not None
        assert event.event_type == AuditEventType.DATA_READ
        assert event.action == "Read customer data"
        assert event.actor_id == "user-123"
        assert event.outcome == "success"

    def test_log_with_tenant_context(self):
        """Test logging captures tenant context."""
        audit = AuditLogger()
        tenant = Tenant(id="audit-test", name="Audit Test")

        with TenantContext(tenant):
            event = audit.log(
                event_type=AuditEventType.DATA_READ,
                action="Read in tenant"
            )

        assert event.tenant_id == "audit-test"

    def test_query_events(self):
        """Test querying audit events."""
        audit = AuditLogger()
        audit.log(AuditEventType.DATA_READ, "Read 1", actor_id="user-1")
        audit.log(AuditEventType.DATA_CREATE, "Write 1", actor_id="user-1")
        audit.log(AuditEventType.DATA_READ, "Read 2", actor_id="user-2")

        # Query by event type
        reads = audit.query(event_type=AuditEventType.DATA_READ)
        assert len(reads) == 2

        # Query by actor
        user1_events = audit.query(actor_id="user-1")
        assert len(user1_events) == 2

    def test_query_by_resource(self):
        """Test querying by resource."""
        audit = AuditLogger()
        audit.log(
            AuditEventType.DATA_READ,
            "Read customer",
            resource_type="customer",
            resource_id="c1"
        )
        audit.log(
            AuditEventType.DATA_READ,
            "Read order",
            resource_type="order",
            resource_id="o1"
        )

        customers = audit.query(resource_type="customer")
        assert len(customers) == 1
        assert customers[0].resource_id == "c1"

    def test_get_event(self):
        """Test getting specific event by ID."""
        audit = AuditLogger()
        event = audit.log(AuditEventType.DATA_READ, "Test event")

        retrieved = audit.get_event(event.id)
        assert retrieved is not None
        assert retrieved.id == event.id

        assert audit.get_event("nonexistent") is None

    def test_export_json(self):
        """Test JSON export."""
        audit = AuditLogger()
        audit.log(AuditEventType.DATA_READ, "Event 1")
        audit.log(AuditEventType.DATA_CREATE, "Event 2")

        exported = audit.export(format="json")
        data = json.loads(exported)
        assert len(data) == 2

    def test_export_csv(self):
        """Test CSV export."""
        audit = AuditLogger()
        audit.log(AuditEventType.DATA_READ, "Event 1")

        exported = audit.export(format="csv")
        assert "id" in exported
        assert "event_type" in exported
        assert "data.read" in exported

    def test_callback_invoked(self):
        """Test callbacks are invoked."""
        audit = AuditLogger()
        received_events = []

        def callback(event):
            received_events.append(event)

        audit.add_callback(callback)
        audit.log(AuditEventType.DATA_READ, "Callback test")

        assert len(received_events) == 1
        assert received_events[0].action == "Callback test"

    def test_audit_action_decorator(self):
        """Test audit_action decorator."""
        audit = get_audit_logger()
        audit.clear()

        @audit_action(AuditEventType.DATA_READ, resource_type="customer")
        def get_customer(customer_id: str):
            return {"id": customer_id}

        result = get_customer("c123")
        assert result["id"] == "c123"

        events = audit.query()
        assert len(events) == 1
        assert events[0].outcome == "success"

    def test_audit_action_captures_failure(self):
        """Test audit_action captures failures."""
        audit = get_audit_logger()
        audit.clear()

        @audit_action(AuditEventType.DATA_READ)
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_function()

        events = audit.query()
        assert len(events) == 1
        assert events[0].outcome == "failure"
        assert "Test error" in events[0].details.get("error", "")


class TestAuditEvent:
    """Test AuditEvent."""

    def test_to_dict(self):
        """Test event serialization."""
        event = AuditEvent(
            id="test-id",
            timestamp="2025-01-01T00:00:00Z",
            event_type=AuditEventType.DATA_READ,
            action="Test action",
            actor_id="user-1",
            actor_type="user",
            tenant_id="tenant-1",
            resource_type="customer",
            resource_id="c1",
            severity=AuditSeverity.INFO,
            outcome="success",
            details={"key": "value"},
            correlation_id="corr-123",
            ip_address="192.168.1.1",
            user_agent="Test Agent"
        )

        d = event.to_dict()
        assert d["id"] == "test-id"
        assert d["event_type"] == "data.read"
        assert d["severity"] == "info"

    def test_to_json(self):
        """Test JSON serialization."""
        event = AuditEvent(
            id="test-id",
            timestamp="2025-01-01T00:00:00Z",
            event_type=AuditEventType.DATA_READ,
            action="Test",
            actor_id=None,
            actor_type="system",
            tenant_id=None,
            resource_type=None,
            resource_id=None,
            severity=AuditSeverity.INFO,
            outcome="success",
            details={},
            correlation_id=None,
            ip_address=None,
            user_agent=None
        )

        json_str = event.to_json()
        data = json.loads(json_str)
        assert data["id"] == "test-id"


class TestRBACManager:
    """Test RBACManager."""

    def setup_method(self):
        """Reset before each test."""
        reset_rbac_manager()

    def test_builtin_roles_exist(self):
        """Test built-in roles are created."""
        rbac = RBACManager()
        assert rbac.get_role("viewer") is not None
        assert rbac.get_role("analyst") is not None
        assert rbac.get_role("admin") is not None
        assert rbac.get_role("superuser") is not None

    def test_create_role(self):
        """Test role creation."""
        rbac = RBACManager()
        role = rbac.create_role(
            "custom",
            "Custom Role",
            {Permission.DATA_READ}
        )

        assert role.name == "custom"
        assert Permission.DATA_READ in role.permissions

    def test_create_duplicate_raises(self):
        """Test duplicate role raises error."""
        rbac = RBACManager()
        rbac.create_role("dup", "First", set())

        with pytest.raises(ValueError, match="already exists"):
            rbac.create_role("dup", "Second", set())

    def test_has_permission(self):
        """Test permission checking."""
        rbac = RBACManager()

        assert rbac.has_permission("viewer", Permission.DATA_READ) is True
        assert rbac.has_permission("viewer", Permission.DATA_WRITE) is False

    def test_superuser_has_all_permissions(self):
        """Test superuser has all permissions."""
        rbac = RBACManager()

        assert rbac.has_permission("superuser", Permission.DATA_READ) is True
        assert rbac.has_permission("superuser", Permission.SYSTEM_ADMIN) is True
        assert rbac.has_permission("superuser", Permission.TENANT_MANAGE) is True

    def test_inherited_permissions(self):
        """Test role inheritance."""
        rbac = RBACManager()

        # Analyst inherits from viewer
        assert rbac.has_permission("analyst", Permission.DATA_READ) is True
        # Analyst also has own permissions
        assert rbac.has_permission("analyst", Permission.ANOMALY_DETECT) is True

    def test_get_effective_permissions(self):
        """Test getting all effective permissions."""
        rbac = RBACManager()

        analyst_perms = rbac.get_effective_permissions("analyst")
        assert Permission.DATA_READ in analyst_perms  # Inherited
        assert Permission.ANOMALY_DETECT in analyst_perms  # Own

    def test_check_permission_with_context(self):
        """Test permission check with user context."""
        rbac = RBACManager()

        with UserContext(user_id="user-1", roles={"analyst"}):
            assert rbac.check_permission(Permission.DATA_READ) is True
            assert rbac.check_permission(Permission.USER_MANAGE) is False

    def test_delete_role(self):
        """Test role deletion."""
        rbac = RBACManager()
        rbac.create_role("deletable", "Can delete", set())

        assert rbac.delete_role("deletable") is True
        assert rbac.get_role("deletable") is None

    def test_delete_builtin_raises(self):
        """Test cannot delete built-in roles."""
        rbac = RBACManager()

        with pytest.raises(ValueError, match="Cannot delete built-in"):
            rbac.delete_role("admin")

    def test_list_roles(self):
        """Test listing roles."""
        rbac = RBACManager()
        roles = rbac.list_roles()

        assert len(roles) >= 5  # At least built-in roles
        names = [r.name for r in roles]
        assert "viewer" in names


class TestUserContext:
    """Test UserContext."""

    def setup_method(self):
        """Reset before each test."""
        reset_rbac_manager()

    def test_context_sets_roles(self):
        """Test context sets current roles."""
        with UserContext(user_id="user-1", roles={"analyst", "viewer"}):
            roles = get_current_roles()
            assert "analyst" in roles
            assert "viewer" in roles
            assert get_current_user_id() == "user-1"

    def test_context_restores_previous(self):
        """Test context restores previous roles."""
        with UserContext(user_id="outer", roles={"admin"}):
            assert "admin" in get_current_roles()
            with UserContext(user_id="inner", roles={"viewer"}):
                assert "viewer" in get_current_roles()
                assert "admin" not in get_current_roles()
            assert "admin" in get_current_roles()

        assert len(get_current_roles()) == 0


class TestPermissionDecorators:
    """Test permission decorators."""

    def setup_method(self):
        """Reset before each test."""
        reset_rbac_manager()

    def test_require_permission_success(self):
        """Test require_permission allows access."""
        @require_permission(Permission.DATA_READ)
        def read_data():
            return "data"

        with UserContext(user_id="user-1", roles={"viewer"}):
            result = read_data()
            assert result == "data"

    def test_require_permission_denied(self):
        """Test require_permission denies access."""
        @require_permission(Permission.DATA_WRITE)
        def write_data():
            return "written"

        with UserContext(user_id="user-1", roles={"viewer"}):
            with pytest.raises(PermissionDeniedError):
                write_data()

    def test_require_role_success(self):
        """Test require_role allows access."""
        @require_role("admin")
        def admin_action():
            return "admin stuff"

        with UserContext(user_id="admin-user", roles={"admin"}):
            result = admin_action()
            assert result == "admin stuff"

    def test_require_role_denied(self):
        """Test require_role denies access."""
        @require_role("admin")
        def admin_only():
            return "secret"

        with UserContext(user_id="user-1", roles={"viewer"}):
            with pytest.raises(PermissionDeniedError):
                admin_only()


class TestRole:
    """Test Role dataclass."""

    def test_has_permission(self):
        """Test role has_permission method."""
        role = Role(
            name="test",
            description="Test role",
            permissions={Permission.DATA_READ}
        )

        assert role.has_permission(Permission.DATA_READ) is True
        assert role.has_permission(Permission.DATA_WRITE) is False

    def test_superuser_permission(self):
        """Test superuser permission grants all."""
        role = Role(
            name="super",
            description="Super role",
            permissions={Permission.SUPERUSER}
        )

        assert role.has_permission(Permission.DATA_READ) is True
        assert role.has_permission(Permission.SYSTEM_ADMIN) is True

    def test_to_dict(self):
        """Test role serialization."""
        role = Role(
            name="test",
            description="Test",
            permissions={Permission.DATA_READ},
            inherits=["viewer"]
        )

        d = role.to_dict()
        assert d["name"] == "test"
        assert "data:read" in d["permissions"]
        assert "viewer" in d["inherits"]
