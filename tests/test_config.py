"""
Tests for Configuration Module

Tests for settings management and environment configuration.
"""

import os
import pytest

from goodai.config.settings import (
    Settings,
    get_settings,
    reset_settings,
    create_settings,
    Environment,
    DatabaseConfig,
    CacheConfig,
    LoggingConfig,
    SecurityConfig,
    AIConfig,
    _get_env,
)


class TestEnvironment:
    """Test Environment enum."""

    def test_environment_values(self):
        """Test environment enum values."""
        assert Environment.DEVELOPMENT.value == "development"
        assert Environment.STAGING.value == "staging"
        assert Environment.PRODUCTION.value == "production"
        assert Environment.TESTING.value == "testing"


class TestDatabaseConfig:
    """Test DatabaseConfig."""

    def test_default_values(self):
        """Test default database config values."""
        config = DatabaseConfig()
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.name == "goodai"
        assert config.pool_size == 10

    def test_url_property(self):
        """Test database URL generation."""
        config = DatabaseConfig(
            host="db.example.com",
            port=5433,
            name="mydb",
            user="myuser"
        )
        assert config.url == "postgresql://myuser@db.example.com:5433/mydb"

    def test_connection_string_with_password(self):
        """Test connection string includes password."""
        config = DatabaseConfig(
            host="db.example.com",
            port=5432,
            name="mydb",
            user="myuser",
            password="secret123"
        )
        assert "secret123" in config.connection_string

    def test_to_dict_excludes_password(self):
        """Test to_dict excludes password."""
        config = DatabaseConfig(password="secret123")
        d = config.to_dict()
        assert "password" not in d


class TestCacheConfig:
    """Test CacheConfig."""

    def test_default_values(self):
        """Test default cache config values."""
        config = CacheConfig()
        assert config.enabled is True
        assert config.backend == "memory"
        assert config.ttl_seconds == 3600

    def test_redis_url(self):
        """Test Redis URL generation."""
        config = CacheConfig(
            host="redis.example.com",
            port=6380,
            db=1
        )
        assert config.redis_url == "redis://redis.example.com:6380/1"

    def test_redis_url_with_password(self):
        """Test Redis URL with password."""
        config = CacheConfig(
            host="redis.example.com",
            password="secret"
        )
        assert ":secret@" in config.redis_url


class TestLoggingConfig:
    """Test LoggingConfig."""

    def test_default_values(self):
        """Test default logging config values."""
        config = LoggingConfig()
        assert config.level == "INFO"
        assert config.format == "json"
        assert config.redact_sensitive is True


class TestSecurityConfig:
    """Test SecurityConfig."""

    def test_default_values(self):
        """Test default security config values."""
        config = SecurityConfig()
        assert config.token_expiry_hours == 24
        assert config.enable_cors is True
        assert config.rate_limit_per_minute == 100

    def test_to_dict_excludes_secret_key(self):
        """Test to_dict excludes secret_key."""
        config = SecurityConfig(secret_key="super-secret")
        d = config.to_dict()
        assert "secret_key" not in d


class TestAIConfig:
    """Test AIConfig."""

    def test_default_values(self):
        """Test default AI config values."""
        config = AIConfig()
        assert config.anomaly_detection_window == 20
        assert config.anomaly_detection_threshold == 2.5
        assert config.default_confidence_threshold == 0.7


class TestSettings:
    """Test Settings."""

    def setup_method(self):
        """Reset settings before each test."""
        reset_settings()
        # Clear relevant env vars
        for key in list(os.environ.keys()):
            if key.startswith("GOODAI_"):
                del os.environ[key]

    def teardown_method(self):
        """Cleanup after each test."""
        reset_settings()
        for key in list(os.environ.keys()):
            if key.startswith("GOODAI_"):
                del os.environ[key]

    def test_default_environment(self):
        """Test default environment is development."""
        settings = Settings()
        assert settings.environment == Environment.DEVELOPMENT

    def test_development_defaults(self):
        """Test development environment applies defaults."""
        settings = Settings(environment=Environment.DEVELOPMENT)
        assert settings.debug is True
        assert settings.logging.level == "DEBUG"
        assert settings.logging.format == "human"

    def test_production_requires_secret_key(self):
        """Test production requires secret key."""
        with pytest.raises(ValueError, match="GOODAI_SECRET_KEY"):
            Settings(
                environment=Environment.PRODUCTION,
                security=SecurityConfig(secret_key="")
            )

    def test_production_with_secret_key(self):
        """Test production works with secret key."""
        settings = Settings(
            environment=Environment.PRODUCTION,
            security=SecurityConfig(secret_key="my-secret-key")
        )
        assert settings.is_production is True
        assert settings.debug is False

    def test_is_properties(self):
        """Test is_* helper properties."""
        dev = Settings(environment=Environment.DEVELOPMENT)
        assert dev.is_development is True
        assert dev.is_production is False
        assert dev.is_testing is False

        test = Settings(environment=Environment.TESTING)
        assert test.is_testing is True

    def test_feature_flags(self):
        """Test feature flag management."""
        settings = Settings(
            features={"new_ui": True, "beta_api": False}
        )
        assert settings.get_feature("new_ui") is True
        assert settings.get_feature("beta_api") is False
        assert settings.get_feature("nonexistent") is False
        assert settings.get_feature("nonexistent", default=True) is True

    def test_to_dict(self):
        """Test to_dict conversion."""
        settings = Settings(service_name="test-service")
        d = settings.to_dict()

        assert d["service_name"] == "test-service"
        assert d["environment"] == "development"
        assert "database" in d
        assert "cache" in d
        assert "logging" in d
        assert "security" in d
        assert "ai" in d

    def test_to_dict_with_secrets(self):
        """Test to_dict with secrets flag."""
        settings = Settings(
            security=SecurityConfig(secret_key="secret"),
            database=DatabaseConfig(password="dbpass")
        )
        d = settings.to_dict(include_secrets=True)

        assert d["security"]["secret_key"] == "[SET]"
        assert d["database"]["password"] == "[SET]"


class TestGetSettings:
    """Test get_settings function."""

    def setup_method(self):
        """Reset settings before each test."""
        reset_settings()
        for key in list(os.environ.keys()):
            if key.startswith("GOODAI_"):
                del os.environ[key]

    def teardown_method(self):
        """Cleanup after each test."""
        reset_settings()
        for key in list(os.environ.keys()):
            if key.startswith("GOODAI_"):
                del os.environ[key]

    def test_get_settings_caches(self):
        """Test get_settings returns cached instance."""
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_get_settings_reload(self):
        """Test get_settings with reload."""
        s1 = get_settings()
        s2 = get_settings(reload=True)
        assert s1 is not s2

    def test_loads_from_env(self):
        """Test settings load from environment."""
        os.environ["GOODAI_SERVICE_NAME"] = "my-service"
        os.environ["GOODAI_DEBUG"] = "true"

        settings = get_settings(reload=True)

        assert settings.service_name == "my-service"
        assert settings.debug is True

    def test_loads_database_config_from_env(self):
        """Test database config loads from environment."""
        os.environ["GOODAI_DB_HOST"] = "db.example.com"
        os.environ["GOODAI_DB_PORT"] = "5433"
        os.environ["GOODAI_DB_NAME"] = "mydb"

        settings = get_settings(reload=True)

        assert settings.database.host == "db.example.com"
        assert settings.database.port == 5433
        assert settings.database.name == "mydb"

    def test_loads_feature_flags_from_env(self):
        """Test feature flags load from environment."""
        os.environ["GOODAI_FEATURE_NEW_UI"] = "true"
        os.environ["GOODAI_FEATURE_BETA_API"] = "false"

        settings = get_settings(reload=True)

        assert settings.get_feature("new_ui") is True
        assert settings.get_feature("beta_api") is False


class TestCreateSettings:
    """Test create_settings function."""

    def setup_method(self):
        """Reset settings before each test."""
        reset_settings()

    def test_create_with_overrides(self):
        """Test creating settings with overrides."""
        settings = create_settings(
            environment=Environment.TESTING,
            debug=True,
            service_name="test-service"
        )

        assert settings.environment == Environment.TESTING
        assert settings.debug is True
        assert settings.service_name == "test-service"


class TestGetEnvHelper:
    """Test _get_env helper function."""

    def test_string_value(self):
        """Test string value retrieval."""
        os.environ["TEST_STRING"] = "hello"
        assert _get_env("TEST_STRING") == "hello"
        del os.environ["TEST_STRING"]

    def test_bool_values(self):
        """Test boolean value parsing."""
        os.environ["TEST_BOOL"] = "true"
        assert _get_env("TEST_BOOL", type_=bool) is True

        os.environ["TEST_BOOL"] = "1"
        assert _get_env("TEST_BOOL", type_=bool) is True

        os.environ["TEST_BOOL"] = "false"
        assert _get_env("TEST_BOOL", type_=bool) is False

        del os.environ["TEST_BOOL"]

    def test_int_values(self):
        """Test integer value parsing."""
        os.environ["TEST_INT"] = "42"
        assert _get_env("TEST_INT", type_=int) == 42
        del os.environ["TEST_INT"]

    def test_float_values(self):
        """Test float value parsing."""
        os.environ["TEST_FLOAT"] = "3.14"
        assert _get_env("TEST_FLOAT", type_=float) == 3.14
        del os.environ["TEST_FLOAT"]

    def test_list_values(self):
        """Test list value parsing."""
        os.environ["TEST_LIST"] = "a,b,c"
        result = _get_env("TEST_LIST", type_=list)
        assert result == ["a", "b", "c"]
        del os.environ["TEST_LIST"]

    def test_default_value(self):
        """Test default value is returned for missing key."""
        assert _get_env("NONEXISTENT", "default") == "default"

    def test_invalid_int_returns_default(self):
        """Test invalid int returns default."""
        os.environ["TEST_INVALID"] = "not-a-number"
        assert _get_env("TEST_INVALID", 10, int) == 10
        del os.environ["TEST_INVALID"]
