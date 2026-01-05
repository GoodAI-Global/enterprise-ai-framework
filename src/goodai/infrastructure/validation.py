"""
Data Validation with Schema Versioning

Provides schema-based validation with version management,
migration support, and backward compatibility.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
import re
import copy


class ValidationLevel(Enum):
    """Validation strictness levels."""

    STRICT = "strict"  # All errors are fatal
    NORMAL = "normal"  # Warnings allowed
    LENIENT = "lenient"  # Best effort validation


@dataclass
class ValidationError:
    """A single validation error."""

    path: str
    message: str
    value: Any = None
    expected: Any = None
    error_code: str = "validation_error"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "message": self.message,
            "value": self.value,
            "expected": self.expected,
            "error_code": self.error_code,
        }


@dataclass
class ValidationResult:
    """Result of validation."""

    valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    validated_data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "valid": self.valid,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "validated_data": self.validated_data,
        }


@dataclass
class SchemaVersion:
    """Schema version information."""

    major: int
    minor: int
    patch: int = 0

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def __lt__(self, other: "SchemaVersion") -> bool:
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    def __le__(self, other: "SchemaVersion") -> bool:
        return (self.major, self.minor, self.patch) <= (other.major, other.minor, other.patch)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SchemaVersion):
            return False
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)

    def __hash__(self) -> int:
        return hash((self.major, self.minor, self.patch))

    def is_compatible_with(self, other: "SchemaVersion") -> bool:
        """Check if this version is compatible with another (same major)."""
        return self.major == other.major

    @classmethod
    def from_string(cls, version_str: str) -> "SchemaVersion":
        """Parse version from string."""
        parts = version_str.split(".")
        return cls(
            major=int(parts[0]),
            minor=int(parts[1]) if len(parts) > 1 else 0,
            patch=int(parts[2]) if len(parts) > 2 else 0,
        )


@dataclass
class FieldSchema:
    """Schema for a single field."""

    name: str
    field_type: str  # string, integer, float, boolean, array, object, any
    required: bool = True
    nullable: bool = False
    default: Any = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    enum_values: Optional[List[Any]] = None
    items_schema: Optional["FieldSchema"] = None  # For arrays
    properties: Optional[Dict[str, "FieldSchema"]] = None  # For objects
    description: str = ""
    deprecated: bool = False
    added_in_version: Optional[str] = None
    removed_in_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "name": self.name,
            "type": self.field_type,
            "required": self.required,
            "nullable": self.nullable,
        }
        if self.default is not None:
            result["default"] = self.default
        if self.min_value is not None:
            result["min_value"] = self.min_value
        if self.max_value is not None:
            result["max_value"] = self.max_value
        if self.pattern:
            result["pattern"] = self.pattern
        if self.enum_values:
            result["enum"] = self.enum_values
        return result


@dataclass
class Schema:
    """A versioned data schema."""

    name: str
    version: SchemaVersion
    fields: Dict[str, FieldSchema] = field(default_factory=dict)
    description: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    strict_mode: bool = False  # Disallow extra fields

    def add_field(self, field_schema: FieldSchema) -> None:
        """Add a field to the schema."""
        self.fields[field_schema.name] = field_schema

    def remove_field(self, name: str) -> None:
        """Remove a field from the schema."""
        if name in self.fields:
            del self.fields[name]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": str(self.version),
            "fields": {k: v.to_dict() for k, v in self.fields.items()},
            "description": self.description,
            "strict_mode": self.strict_mode,
        }


class SchemaValidator:
    """Validates data against a schema."""

    TYPE_VALIDATORS = {
        "string": lambda v: isinstance(v, str),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "float": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "boolean": lambda v: isinstance(v, bool),
        "array": lambda v: isinstance(v, list),
        "object": lambda v: isinstance(v, dict),
        "any": lambda v: True,
    }

    def __init__(self, level: ValidationLevel = ValidationLevel.NORMAL):
        """
        Initialize validator.

        Args:
            level: Validation strictness level
        """
        self.level = level

    def validate(
        self,
        data: Dict[str, Any],
        schema: Schema,
    ) -> ValidationResult:
        """
        Validate data against schema.

        Args:
            data: Data to validate
            schema: Schema to validate against

        Returns:
            ValidationResult
        """
        errors: List[ValidationError] = []
        warnings: List[ValidationError] = []
        validated_data = copy.deepcopy(data)

        # Check for extra fields in strict mode
        if schema.strict_mode:
            extra_fields = set(data.keys()) - set(schema.fields.keys())
            for field_name in extra_fields:
                errors.append(ValidationError(
                    path=field_name,
                    message=f"Unknown field '{field_name}' not allowed in strict mode",
                    error_code="unknown_field",
                ))

        # Validate each field
        for field_name, field_schema in schema.fields.items():
            field_errors, field_warnings = self._validate_field(
                data.get(field_name),
                field_schema,
                field_name,
                validated_data,
            )
            errors.extend(field_errors)
            warnings.extend(field_warnings)

        valid = len(errors) == 0
        return ValidationResult(
            valid=valid,
            errors=errors,
            warnings=warnings,
            validated_data=validated_data if valid else None,
        )

    def _validate_field(
        self,
        value: Any,
        field_schema: FieldSchema,
        path: str,
        validated_data: Dict[str, Any],
    ) -> Tuple[List[ValidationError], List[ValidationError]]:
        """Validate a single field."""
        errors: List[ValidationError] = []
        warnings: List[ValidationError] = []

        # Check deprecated
        if field_schema.deprecated and value is not None:
            warnings.append(ValidationError(
                path=path,
                message=f"Field '{path}' is deprecated",
                error_code="deprecated_field",
            ))

        # Handle missing value
        if value is None:
            if field_schema.required and not field_schema.nullable:
                if field_schema.default is not None:
                    # Apply default
                    validated_data[path] = field_schema.default
                else:
                    errors.append(ValidationError(
                        path=path,
                        message=f"Required field '{path}' is missing",
                        error_code="required_field",
                    ))
            elif field_schema.default is not None:
                validated_data[path] = field_schema.default
            return errors, warnings

        # Check nullable
        if value is None and not field_schema.nullable:
            errors.append(ValidationError(
                path=path,
                message=f"Field '{path}' cannot be null",
                value=value,
                error_code="null_not_allowed",
            ))
            return errors, warnings

        # Type validation
        if value is not None:
            type_validator = self.TYPE_VALIDATORS.get(field_schema.field_type)
            if type_validator and not type_validator(value):
                errors.append(ValidationError(
                    path=path,
                    message=f"Field '{path}' must be of type {field_schema.field_type}",
                    value=value,
                    expected=field_schema.field_type,
                    error_code="invalid_type",
                ))
                return errors, warnings

        # Numeric range validation
        if field_schema.field_type in ("integer", "float") and isinstance(value, (int, float)):
            if field_schema.min_value is not None and value < field_schema.min_value:
                errors.append(ValidationError(
                    path=path,
                    message=f"Field '{path}' must be >= {field_schema.min_value}",
                    value=value,
                    expected=f">= {field_schema.min_value}",
                    error_code="value_too_small",
                ))
            if field_schema.max_value is not None and value > field_schema.max_value:
                errors.append(ValidationError(
                    path=path,
                    message=f"Field '{path}' must be <= {field_schema.max_value}",
                    value=value,
                    expected=f"<= {field_schema.max_value}",
                    error_code="value_too_large",
                ))

        # String validation
        if field_schema.field_type == "string" and isinstance(value, str):
            if field_schema.min_length is not None and len(value) < field_schema.min_length:
                errors.append(ValidationError(
                    path=path,
                    message=f"Field '{path}' must have length >= {field_schema.min_length}",
                    value=value,
                    error_code="string_too_short",
                ))
            if field_schema.max_length is not None and len(value) > field_schema.max_length:
                errors.append(ValidationError(
                    path=path,
                    message=f"Field '{path}' must have length <= {field_schema.max_length}",
                    value=value,
                    error_code="string_too_long",
                ))
            if field_schema.pattern:
                if not re.match(field_schema.pattern, value):
                    errors.append(ValidationError(
                        path=path,
                        message=f"Field '{path}' must match pattern {field_schema.pattern}",
                        value=value,
                        expected=field_schema.pattern,
                        error_code="pattern_mismatch",
                    ))

        # Enum validation
        if field_schema.enum_values and value not in field_schema.enum_values:
            errors.append(ValidationError(
                path=path,
                message=f"Field '{path}' must be one of {field_schema.enum_values}",
                value=value,
                expected=field_schema.enum_values,
                error_code="invalid_enum_value",
            ))

        # Array validation
        if field_schema.field_type == "array" and isinstance(value, list):
            if field_schema.items_schema:
                for i, item in enumerate(value):
                    item_errors, item_warnings = self._validate_field(
                        item,
                        field_schema.items_schema,
                        f"{path}[{i}]",
                        {},
                    )
                    errors.extend(item_errors)
                    warnings.extend(item_warnings)

        # Object validation
        if field_schema.field_type == "object" and isinstance(value, dict):
            if field_schema.properties:
                for prop_name, prop_schema in field_schema.properties.items():
                    prop_errors, prop_warnings = self._validate_field(
                        value.get(prop_name),
                        prop_schema,
                        f"{path}.{prop_name}",
                        value,
                    )
                    errors.extend(prop_errors)
                    warnings.extend(prop_warnings)

        return errors, warnings


class SchemaRegistry:
    """
    Enterprise Schema Registry.

    Manages schema versions and migrations.

    Example:
        registry = SchemaRegistry()

        # Register schema
        schema = Schema(
            name="user",
            version=SchemaVersion(1, 0, 0),
            fields={
                "id": FieldSchema("id", "string", required=True),
                "email": FieldSchema("email", "string", pattern=r".*@.*"),
            }
        )
        registry.register(schema)

        # Validate data
        result = registry.validate("user", data)
    """

    def __init__(self):
        """Initialize the registry."""
        self._schemas: Dict[str, Dict[SchemaVersion, Schema]] = {}
        self._migrations: Dict[str, List[Tuple[SchemaVersion, SchemaVersion, Callable]]] = {}
        self._validator = SchemaValidator()

    def register(self, schema: Schema) -> None:
        """
        Register a schema.

        Args:
            schema: Schema to register
        """
        if schema.name not in self._schemas:
            self._schemas[schema.name] = {}

        self._schemas[schema.name][schema.version] = schema

    def get_schema(
        self,
        name: str,
        version: Optional[SchemaVersion] = None,
    ) -> Optional[Schema]:
        """
        Get a schema by name and optional version.

        Args:
            name: Schema name
            version: Specific version (or latest if None)

        Returns:
            Schema or None
        """
        if name not in self._schemas:
            return None

        versions = self._schemas[name]
        if not versions:
            return None

        if version:
            return versions.get(version)

        # Return latest version
        return versions[max(versions.keys())]

    def list_versions(self, name: str) -> List[SchemaVersion]:
        """List all versions of a schema."""
        if name not in self._schemas:
            return []
        return sorted(self._schemas[name].keys())

    def register_migration(
        self,
        schema_name: str,
        from_version: SchemaVersion,
        to_version: SchemaVersion,
        migration_func: Callable[[Dict[str, Any]], Dict[str, Any]],
    ) -> None:
        """
        Register a migration function.

        Args:
            schema_name: Schema name
            from_version: Source version
            to_version: Target version
            migration_func: Function to migrate data
        """
        if schema_name not in self._migrations:
            self._migrations[schema_name] = []

        self._migrations[schema_name].append((from_version, to_version, migration_func))

    def validate(
        self,
        schema_name: str,
        data: Dict[str, Any],
        version: Optional[SchemaVersion] = None,
        level: ValidationLevel = ValidationLevel.NORMAL,
    ) -> ValidationResult:
        """
        Validate data against a schema.

        Args:
            schema_name: Schema name
            data: Data to validate
            version: Schema version (or latest)
            level: Validation level

        Returns:
            ValidationResult
        """
        schema = self.get_schema(schema_name, version)
        if not schema:
            return ValidationResult(
                valid=False,
                errors=[ValidationError(
                    path="",
                    message=f"Schema '{schema_name}' not found",
                    error_code="schema_not_found",
                )],
            )

        validator = SchemaValidator(level)
        return validator.validate(data, schema)

    def migrate(
        self,
        schema_name: str,
        data: Dict[str, Any],
        from_version: SchemaVersion,
        to_version: SchemaVersion,
    ) -> Dict[str, Any]:
        """
        Migrate data from one schema version to another.

        Args:
            schema_name: Schema name
            data: Data to migrate
            from_version: Source version
            to_version: Target version

        Returns:
            Migrated data
        """
        if schema_name not in self._migrations:
            return data

        migrations = self._migrations[schema_name]
        current_data = copy.deepcopy(data)
        current_version = from_version

        while current_version < to_version:
            # Find next migration
            next_migration = None
            for fv, tv, func in migrations:
                if fv == current_version:
                    if next_migration is None or tv < next_migration[1]:
                        next_migration = (fv, tv, func)

            if next_migration is None:
                break  # No migration path found

            # Apply migration
            _, to_ver, migration_func = next_migration
            current_data = migration_func(current_data)
            current_version = to_ver

        return current_data

    def get_migration_path(
        self,
        schema_name: str,
        from_version: SchemaVersion,
        to_version: SchemaVersion,
    ) -> List[Tuple[SchemaVersion, SchemaVersion]]:
        """
        Get the migration path between versions.

        Args:
            schema_name: Schema name
            from_version: Source version
            to_version: Target version

        Returns:
            List of (from, to) version tuples
        """
        if schema_name not in self._migrations:
            return []

        migrations = self._migrations[schema_name]
        path = []
        current = from_version

        while current < to_version:
            next_step = None
            for fv, tv, _ in migrations:
                if fv == current and tv <= to_version:
                    if next_step is None or tv < next_step[1]:
                        next_step = (fv, tv)

            if next_step is None:
                break

            path.append(next_step)
            current = next_step[1]

        return path

    def clear(self) -> None:
        """Clear all schemas (for testing)."""
        self._schemas.clear()
        self._migrations.clear()


def validate(
    schema_name: str,
    data: Dict[str, Any],
    version: Optional[SchemaVersion] = None,
) -> ValidationResult:
    """
    Convenience function using global registry.

    Example:
        result = validate("user", {"id": "123", "email": "test@example.com"})
    """
    return get_schema_registry().validate(schema_name, data, version)


# Global singleton
_schema_registry: Optional[SchemaRegistry] = None


def get_schema_registry() -> SchemaRegistry:
    """Get the global schema registry instance."""
    global _schema_registry
    if _schema_registry is None:
        _schema_registry = SchemaRegistry()
    return _schema_registry
