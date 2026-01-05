"""
Model Registry for Enterprise AI

Provides versioned model management with metadata tracking,
lifecycle management, and deployment status tracking.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import hashlib
import json


class ModelStatus(Enum):
    """Model lifecycle status."""

    DRAFT = "draft"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


@dataclass
class ModelMetrics:
    """Model performance metrics."""

    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_roc: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            k: v for k, v in {
                "accuracy": self.accuracy,
                "precision": self.precision,
                "recall": self.recall,
                "f1_score": self.f1_score,
                "auc_roc": self.auc_roc,
                "mse": self.mse,
                "mae": self.mae,
            }.items() if v is not None
        }
        result.update(self.custom_metrics)
        return result


@dataclass
class ModelVersion:
    """A specific version of a model."""

    version: str
    model_name: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: ModelStatus = ModelStatus.DRAFT
    description: str = ""
    framework: str = ""  # e.g., "pytorch", "tensorflow", "sklearn"
    framework_version: str = ""
    artifact_path: Optional[str] = None
    artifact_hash: Optional[str] = None
    metrics: Optional[ModelMetrics] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    created_by: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "model_name": self.model_name,
            "created_at": self.created_at.isoformat(),
            "status": self.status.value,
            "description": self.description,
            "framework": self.framework,
            "framework_version": self.framework_version,
            "artifact_path": self.artifact_path,
            "artifact_hash": self.artifact_hash,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "parameters": self.parameters,
            "tags": self.tags,
            "created_by": self.created_by,
        }


@dataclass
class Model:
    """A registered model with version history."""

    name: str
    description: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    owner: Optional[str] = None
    team: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    versions: List[ModelVersion] = field(default_factory=list)

    @property
    def latest_version(self) -> Optional[ModelVersion]:
        """Get the latest version."""
        if not self.versions:
            return None
        return max(self.versions, key=lambda v: v.created_at)

    @property
    def production_version(self) -> Optional[ModelVersion]:
        """Get the production version."""
        for version in self.versions:
            if version.status == ModelStatus.PRODUCTION:
                return version
        return None

    def get_version(self, version: str) -> Optional[ModelVersion]:
        """Get a specific version."""
        for v in self.versions:
            if v.version == version:
                return v
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "owner": self.owner,
            "team": self.team,
            "tags": self.tags,
            "versions": [v.to_dict() for v in self.versions],
            "latest_version": self.latest_version.version if self.latest_version else None,
            "production_version": self.production_version.version if self.production_version else None,
        }


class ModelRegistry:
    """
    Enterprise Model Registry.

    Manages model registration, versioning, and lifecycle.

    Example:
        registry = ModelRegistry()

        # Register a new model
        model = registry.register_model(
            name="fraud_detector",
            description="Credit card fraud detection model"
        )

        # Create a version
        version = registry.create_version(
            model_name="fraud_detector",
            version="1.0.0",
            framework="pytorch",
            metrics=ModelMetrics(accuracy=0.95, f1_score=0.92)
        )

        # Promote to production
        registry.transition_status(
            model_name="fraud_detector",
            version="1.0.0",
            status=ModelStatus.PRODUCTION
        )
    """

    def __init__(self):
        """Initialize the registry."""
        self._models: Dict[str, Model] = {}
        self._transition_hooks: List[Callable[[str, str, ModelStatus, ModelStatus], None]] = []

    def register_model(
        self,
        name: str,
        description: str = "",
        owner: Optional[str] = None,
        team: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Model:
        """
        Register a new model.

        Args:
            name: Unique model name
            description: Model description
            owner: Model owner
            team: Owning team
            tags: Metadata tags

        Returns:
            Registered model

        Raises:
            ValueError: If model already exists
        """
        if name in self._models:
            raise ValueError(f"Model '{name}' already exists")

        model = Model(
            name=name,
            description=description,
            owner=owner,
            team=team,
            tags=tags or {},
        )
        self._models[name] = model
        return model

    def get_model(self, name: str) -> Optional[Model]:
        """Get a model by name."""
        return self._models.get(name)

    def list_models(
        self,
        team: Optional[str] = None,
        tag_filter: Optional[Dict[str, str]] = None,
    ) -> List[Model]:
        """
        List all registered models.

        Args:
            team: Filter by team
            tag_filter: Filter by tags

        Returns:
            List of models
        """
        models = list(self._models.values())

        if team:
            models = [m for m in models if m.team == team]

        if tag_filter:
            filtered = []
            for model in models:
                if all(model.tags.get(k) == v for k, v in tag_filter.items()):
                    filtered.append(model)
            models = filtered

        return models

    def delete_model(self, name: str) -> bool:
        """
        Delete a model and all its versions.

        Args:
            name: Model name

        Returns:
            True if deleted, False if not found
        """
        if name in self._models:
            del self._models[name]
            return True
        return False

    def create_version(
        self,
        model_name: str,
        version: str,
        description: str = "",
        framework: str = "",
        framework_version: str = "",
        artifact_path: Optional[str] = None,
        metrics: Optional[ModelMetrics] = None,
        parameters: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        created_by: Optional[str] = None,
    ) -> ModelVersion:
        """
        Create a new model version.

        Args:
            model_name: Name of the model
            version: Version string (e.g., "1.0.0")
            description: Version description
            framework: ML framework used
            framework_version: Framework version
            artifact_path: Path to model artifact
            metrics: Performance metrics
            parameters: Training parameters
            tags: Metadata tags
            created_by: Creator identity

        Returns:
            Created model version

        Raises:
            ValueError: If model not found or version exists
        """
        model = self._models.get(model_name)
        if not model:
            raise ValueError(f"Model '{model_name}' not found")

        if model.get_version(version):
            raise ValueError(f"Version '{version}' already exists for model '{model_name}'")

        # Calculate artifact hash if path provided
        artifact_hash = None
        if artifact_path:
            artifact_hash = hashlib.sha256(artifact_path.encode()).hexdigest()[:16]

        model_version = ModelVersion(
            version=version,
            model_name=model_name,
            description=description,
            framework=framework,
            framework_version=framework_version,
            artifact_path=artifact_path,
            artifact_hash=artifact_hash,
            metrics=metrics,
            parameters=parameters or {},
            tags=tags or {},
            created_by=created_by,
        )

        model.versions.append(model_version)
        model.updated_at = datetime.utcnow()

        return model_version

    def get_version(self, model_name: str, version: str) -> Optional[ModelVersion]:
        """Get a specific model version."""
        model = self._models.get(model_name)
        if not model:
            return None
        return model.get_version(version)

    def list_versions(
        self,
        model_name: str,
        status: Optional[ModelStatus] = None,
    ) -> List[ModelVersion]:
        """
        List versions of a model.

        Args:
            model_name: Model name
            status: Filter by status

        Returns:
            List of versions
        """
        model = self._models.get(model_name)
        if not model:
            return []

        versions = model.versions
        if status:
            versions = [v for v in versions if v.status == status]

        return sorted(versions, key=lambda v: v.created_at, reverse=True)

    def transition_status(
        self,
        model_name: str,
        version: str,
        status: ModelStatus,
    ) -> ModelVersion:
        """
        Transition a model version to a new status.

        Args:
            model_name: Model name
            version: Version string
            status: New status

        Returns:
            Updated model version

        Raises:
            ValueError: If model or version not found
        """
        model_version = self.get_version(model_name, version)
        if not model_version:
            raise ValueError(f"Version '{version}' not found for model '{model_name}'")

        old_status = model_version.status

        # If transitioning to production, demote current production
        if status == ModelStatus.PRODUCTION:
            model = self._models[model_name]
            for v in model.versions:
                if v.status == ModelStatus.PRODUCTION and v.version != version:
                    v.status = ModelStatus.ARCHIVED

        model_version.status = status

        # Invoke transition hooks
        for hook in self._transition_hooks:
            try:
                hook(model_name, version, old_status, status)
            except Exception:
                pass  # Hooks should not break transitions

        return model_version

    def add_transition_hook(
        self,
        hook: Callable[[str, str, ModelStatus, ModelStatus], None],
    ) -> None:
        """
        Add a status transition hook.

        Args:
            hook: Callback function(model_name, version, old_status, new_status)
        """
        self._transition_hooks.append(hook)

    def compare_versions(
        self,
        model_name: str,
        version_a: str,
        version_b: str,
    ) -> Dict[str, Any]:
        """
        Compare two model versions.

        Args:
            model_name: Model name
            version_a: First version
            version_b: Second version

        Returns:
            Comparison results
        """
        v_a = self.get_version(model_name, version_a)
        v_b = self.get_version(model_name, version_b)

        if not v_a or not v_b:
            raise ValueError("One or both versions not found")

        result = {
            "model_name": model_name,
            "version_a": version_a,
            "version_b": version_b,
            "metrics_comparison": {},
            "parameter_diff": {},
        }

        # Compare metrics
        if v_a.metrics and v_b.metrics:
            metrics_a = v_a.metrics.to_dict()
            metrics_b = v_b.metrics.to_dict()

            all_metrics = set(metrics_a.keys()) | set(metrics_b.keys())
            for metric in all_metrics:
                val_a = metrics_a.get(metric)
                val_b = metrics_b.get(metric)
                if val_a is not None and val_b is not None:
                    result["metrics_comparison"][metric] = {
                        "a": val_a,
                        "b": val_b,
                        "diff": val_b - val_a,
                        "pct_change": ((val_b - val_a) / val_a * 100) if val_a != 0 else None,
                    }

        # Compare parameters
        all_params = set(v_a.parameters.keys()) | set(v_b.parameters.keys())
        for param in all_params:
            val_a = v_a.parameters.get(param)
            val_b = v_b.parameters.get(param)
            if val_a != val_b:
                result["parameter_diff"][param] = {
                    "a": val_a,
                    "b": val_b,
                }

        return result

    def export_lineage(self, model_name: str) -> Dict[str, Any]:
        """
        Export model lineage information.

        Args:
            model_name: Model name

        Returns:
            Lineage information
        """
        model = self._models.get(model_name)
        if not model:
            raise ValueError(f"Model '{model_name}' not found")

        return {
            "model": model.to_dict(),
            "version_timeline": [
                {
                    "version": v.version,
                    "created_at": v.created_at.isoformat(),
                    "status": v.status.value,
                    "framework": v.framework,
                }
                for v in sorted(model.versions, key=lambda x: x.created_at)
            ],
        }

    def clear(self) -> None:
        """Clear all models (for testing)."""
        self._models.clear()


# Global singleton
_registry: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """Get the global model registry instance."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
