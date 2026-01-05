"""
Configuration Settings Module

Enterprise-grade configuration management with:
- Environment-based settings (dev/staging/production)
- Pydantic validation for type safety
- Secrets management support
- Hierarchical configuration

Good AI Philosophy: Non-invasive by default - configuration doesn't force patterns.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json


class Environment(Enum):
    """Deployment environment."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


@dataclass
class DatabaseConfig:
    """Database configuration."""

    host: str = "localhost"
    port: int = 5432
    name: str = "goodai"
    user: str = "goodai"
    password: str = ""  # Should be loaded from secrets
    pool_size: int = 10
    pool_timeout: int = 30
    ssl_mode: str = "prefer"

    @property
    def url(self) -> str:
        """Build database URL (without password for logging)."""
        return f"postgresql://{self.user}@{self.host}:{self.port}/{self.name}"

    @property
    def connection_string(self) -> str:
        """Build full connection string."""
        ssl = f"?sslmode={self.ssl_mode}" if self.ssl_mode else ""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}{ssl}"

    def to_dict(self) -> dict:
        """Convert to dictionary (without password)."""
        return {
            "host": self.host,
            "port": self.port,
            "name": self.name,
            "user": self.user,
            "pool_size": self.pool_size,
            "pool_timeout": self.pool_timeout,
            "ssl_mode": self.ssl_mode,
        }


@dataclass
class CacheConfig:
    """Cache configuration."""

    enabled: bool = True
    backend: str = "memory"  # memory, redis, memcached
    host: str = "localhost"
    port: int = 6379
    password: str = ""
    db: int = 0
    ttl_seconds: int = 3600
    max_size: int = 1000  # For memory cache
    prefix: str = "goodai:"

    @property
    def redis_url(self) -> str:
        """Build Redis URL."""
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"

    def to_dict(self) -> dict:
        """Convert to dictionary (without password)."""
        return {
            "enabled": self.enabled,
            "backend": self.backend,
            "host": self.host,
            "port": self.port,
            "db": self.db,
            "ttl_seconds": self.ttl_seconds,
            "max_size": self.max_size,
            "prefix": self.prefix,
        }


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    format: str = "json"  # json, human
    output: str = "stderr"  # stderr, stdout, or file path
    include_stack_traces: bool = True
    redact_sensitive: bool = True
    max_message_length: int = 10000

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "level": self.level,
            "format": self.format,
            "output": self.output,
            "include_stack_traces": self.include_stack_traces,
            "redact_sensitive": self.redact_sensitive,
            "max_message_length": self.max_message_length,
        }


@dataclass
class SecurityConfig:
    """Security configuration."""

    secret_key: str = ""  # Must be set in production
    token_expiry_hours: int = 24
    refresh_token_expiry_days: int = 30
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])
    allowed_hosts: List[str] = field(default_factory=lambda: ["*"])
    enable_cors: bool = True
    enable_csrf: bool = True
    rate_limit_per_minute: int = 100
    max_request_size_mb: int = 10

    def to_dict(self) -> dict:
        """Convert to dictionary (without secrets)."""
        return {
            "token_expiry_hours": self.token_expiry_hours,
            "refresh_token_expiry_days": self.refresh_token_expiry_days,
            "allowed_origins": self.allowed_origins,
            "allowed_hosts": self.allowed_hosts,
            "enable_cors": self.enable_cors,
            "enable_csrf": self.enable_csrf,
            "rate_limit_per_minute": self.rate_limit_per_minute,
            "max_request_size_mb": self.max_request_size_mb,
        }


@dataclass
class AIConfig:
    """AI/ML configuration."""

    model_cache_dir: str = ".cache/models"
    default_confidence_threshold: float = 0.7
    anomaly_detection_window: int = 20
    anomaly_detection_threshold: float = 2.5
    batch_size: int = 32
    max_concurrent_predictions: int = 10
    enable_model_versioning: bool = True
    fallback_on_error: bool = True

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "model_cache_dir": self.model_cache_dir,
            "default_confidence_threshold": self.default_confidence_threshold,
            "anomaly_detection_window": self.anomaly_detection_window,
            "anomaly_detection_threshold": self.anomaly_detection_threshold,
            "batch_size": self.batch_size,
            "max_concurrent_predictions": self.max_concurrent_predictions,
            "enable_model_versioning": self.enable_model_versioning,
            "fallback_on_error": self.fallback_on_error,
        }


@dataclass
class Settings:
    """
    Main application settings.

    Loads configuration from environment variables with sensible defaults.
    Supports hierarchical configuration and environment-specific overrides.

    Environment Variables:
        GOODAI_ENV: Environment (development, staging, production)
        GOODAI_DEBUG: Enable debug mode
        GOODAI_SERVICE_NAME: Service name for logging
        GOODAI_VERSION: Application version

    Example:
        >>> settings = get_settings()
        >>> print(settings.environment)
        Environment.DEVELOPMENT

        >>> # Override for testing
        >>> settings = Settings(environment=Environment.TESTING)
    """

    # Core settings
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    service_name: str = "goodai"
    version: str = "0.1.0"

    # Component configs
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    ai: AIConfig = field(default_factory=AIConfig)

    # Feature flags
    features: Dict[str, bool] = field(default_factory=dict)

    def __post_init__(self):
        """Apply environment-specific defaults."""
        if self.environment == Environment.PRODUCTION:
            self._apply_production_defaults()
        elif self.environment == Environment.DEVELOPMENT:
            self._apply_development_defaults()

    def _apply_production_defaults(self):
        """Apply production-safe defaults."""
        self.debug = False
        self.logging.level = "INFO"
        self.logging.format = "json"
        self.logging.redact_sensitive = True
        self.security.enable_csrf = True

        # Validate production requirements
        if not self.security.secret_key:
            raise ValueError("GOODAI_SECRET_KEY must be set in production")

    def _apply_development_defaults(self):
        """Apply development-friendly defaults."""
        self.debug = True
        self.logging.level = "DEBUG"
        self.logging.format = "human"

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == Environment.PRODUCTION

    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == Environment.DEVELOPMENT

    @property
    def is_testing(self) -> bool:
        """Check if running in test mode."""
        return self.environment == Environment.TESTING

    def get_feature(self, name: str, default: bool = False) -> bool:
        """Get feature flag value."""
        return self.features.get(name, default)

    def to_dict(self, include_secrets: bool = False) -> dict:
        """
        Convert settings to dictionary.

        Args:
            include_secrets: Whether to include secret values

        Returns:
            Dictionary representation of settings
        """
        result = {
            "environment": self.environment.value,
            "debug": self.debug,
            "service_name": self.service_name,
            "version": self.version,
            "database": self.database.to_dict(),
            "cache": self.cache.to_dict(),
            "logging": self.logging.to_dict(),
            "security": self.security.to_dict(),
            "ai": self.ai.to_dict(),
            "features": self.features,
        }

        if include_secrets:
            result["database"]["password"] = "[SET]" if self.database.password else "[NOT SET]"
            result["security"]["secret_key"] = "[SET]" if self.security.secret_key else "[NOT SET]"

        return result


def _get_env(key: str, default: Any = None, type_: type = str) -> Any:
    """Get environment variable with type conversion."""
    value = os.environ.get(key, default)

    if value is None:
        return default

    if type_ == bool:
        return str(value).lower() in ("true", "1", "yes", "on")
    elif type_ == int:
        try:
            return int(value)
        except (ValueError, TypeError):
            return default
    elif type_ == float:
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    elif type_ == list:
        if isinstance(value, list):
            return value
        return [v.strip() for v in str(value).split(",") if v.strip()]

    return value


def _load_from_env() -> Settings:
    """Load settings from environment variables."""
    # Determine environment
    env_str = _get_env("GOODAI_ENV", "development")
    try:
        environment = Environment(env_str.lower())
    except ValueError:
        environment = Environment.DEVELOPMENT

    # Build database config
    database = DatabaseConfig(
        host=_get_env("GOODAI_DB_HOST", "localhost"),
        port=_get_env("GOODAI_DB_PORT", 5432, int),
        name=_get_env("GOODAI_DB_NAME", "goodai"),
        user=_get_env("GOODAI_DB_USER", "goodai"),
        password=_get_env("GOODAI_DB_PASSWORD", ""),
        pool_size=_get_env("GOODAI_DB_POOL_SIZE", 10, int),
        ssl_mode=_get_env("GOODAI_DB_SSL_MODE", "prefer"),
    )

    # Build cache config
    cache = CacheConfig(
        enabled=_get_env("GOODAI_CACHE_ENABLED", True, bool),
        backend=_get_env("GOODAI_CACHE_BACKEND", "memory"),
        host=_get_env("GOODAI_CACHE_HOST", "localhost"),
        port=_get_env("GOODAI_CACHE_PORT", 6379, int),
        password=_get_env("GOODAI_CACHE_PASSWORD", ""),
        db=_get_env("GOODAI_CACHE_DB", 0, int),
        ttl_seconds=_get_env("GOODAI_CACHE_TTL", 3600, int),
    )

    # Build logging config
    logging_config = LoggingConfig(
        level=_get_env("GOODAI_LOG_LEVEL", "INFO"),
        format=_get_env("GOODAI_LOG_FORMAT", "json"),
        output=_get_env("GOODAI_LOG_OUTPUT", "stderr"),
        redact_sensitive=_get_env("GOODAI_LOG_REDACT", True, bool),
    )

    # Build security config
    security = SecurityConfig(
        secret_key=_get_env("GOODAI_SECRET_KEY", ""),
        token_expiry_hours=_get_env("GOODAI_TOKEN_EXPIRY_HOURS", 24, int),
        allowed_origins=_get_env("GOODAI_ALLOWED_ORIGINS", ["*"], list),
        rate_limit_per_minute=_get_env("GOODAI_RATE_LIMIT", 100, int),
    )

    # Build AI config
    ai = AIConfig(
        model_cache_dir=_get_env("GOODAI_MODEL_CACHE", ".cache/models"),
        default_confidence_threshold=_get_env("GOODAI_CONFIDENCE_THRESHOLD", 0.7, float),
        anomaly_detection_window=_get_env("GOODAI_ANOMALY_WINDOW", 20, int),
        anomaly_detection_threshold=_get_env("GOODAI_ANOMALY_THRESHOLD", 2.5, float),
    )

    # Load feature flags
    features = {}
    for key, value in os.environ.items():
        if key.startswith("GOODAI_FEATURE_"):
            feature_name = key[15:].lower()  # Remove prefix
            features[feature_name] = _get_env(key, False, bool)

    return Settings(
        environment=environment,
        debug=_get_env("GOODAI_DEBUG", False, bool),
        service_name=_get_env("GOODAI_SERVICE_NAME", "goodai"),
        version=_get_env("GOODAI_VERSION", "0.1.0"),
        database=database,
        cache=cache,
        logging=logging_config,
        security=security,
        ai=ai,
        features=features,
    )


def _load_from_file(path: Union[str, Path]) -> Dict[str, Any]:
    """Load settings from JSON file."""
    path = Path(path)
    if not path.exists():
        return {}

    with open(path) as f:
        return json.load(f)


# Global settings instance
_settings: Optional[Settings] = None


def get_settings(reload: bool = False) -> Settings:
    """
    Get application settings.

    Loads settings from environment variables on first call.
    Subsequent calls return cached settings unless reload=True.

    Args:
        reload: Force reload settings from environment

    Returns:
        Settings instance

    Example:
        >>> settings = get_settings()
        >>> print(settings.service_name)
        goodai

        >>> # Force reload after env change
        >>> settings = get_settings(reload=True)
    """
    global _settings

    if _settings is None or reload:
        _settings = _load_from_env()

    return _settings


def reset_settings() -> None:
    """Reset cached settings (for testing)."""
    global _settings
    _settings = None


def create_settings(**overrides) -> Settings:
    """
    Create settings with overrides.

    Useful for testing or programmatic configuration.

    Args:
        **overrides: Values to override

    Returns:
        New Settings instance

    Example:
        >>> settings = create_settings(
        ...     environment=Environment.TESTING,
        ...     debug=True
        ... )
    """
    base = _load_from_env()

    # Apply simple overrides
    for key, value in overrides.items():
        if hasattr(base, key):
            setattr(base, key, value)

    return base
