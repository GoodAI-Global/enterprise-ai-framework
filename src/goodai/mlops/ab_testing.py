"""
A/B Testing Framework for ML Models

Provides experiment management, traffic splitting, and statistical analysis
for comparing model variants in production.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import hashlib
import random
import math


class ExperimentStatus(Enum):
    """Experiment lifecycle status."""

    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class VariantMetrics:
    """Metrics for a variant."""

    impressions: int = 0
    conversions: int = 0
    total_value: float = 0.0
    custom_metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def conversion_rate(self) -> float:
        """Calculate conversion rate."""
        if self.impressions == 0:
            return 0.0
        return self.conversions / self.impressions

    @property
    def average_value(self) -> float:
        """Calculate average value per conversion."""
        if self.conversions == 0:
            return 0.0
        return self.total_value / self.conversions

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "impressions": self.impressions,
            "conversions": self.conversions,
            "total_value": self.total_value,
            "conversion_rate": self.conversion_rate,
            "average_value": self.average_value,
            "custom_metrics": self.custom_metrics,
        }


@dataclass
class Variant:
    """A variant in an experiment."""

    name: str
    model_name: str
    model_version: str
    weight: float = 1.0  # Traffic weight
    description: str = ""
    is_control: bool = False
    metrics: VariantMetrics = field(default_factory=VariantMetrics)
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "weight": self.weight,
            "description": self.description,
            "is_control": self.is_control,
            "metrics": self.metrics.to_dict(),
            "config": self.config,
        }


@dataclass
class Experiment:
    """An A/B test experiment."""

    id: str
    name: str
    description: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    status: ExperimentStatus = ExperimentStatus.DRAFT
    variants: List[Variant] = field(default_factory=list)
    target_sample_size: int = 1000
    min_runtime_hours: int = 24
    owner: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)

    @property
    def control(self) -> Optional[Variant]:
        """Get the control variant."""
        for variant in self.variants:
            if variant.is_control:
                return variant
        return None

    @property
    def total_impressions(self) -> int:
        """Get total impressions across all variants."""
        return sum(v.metrics.impressions for v in self.variants)

    def get_variant(self, name: str) -> Optional[Variant]:
        """Get a variant by name."""
        for variant in self.variants:
            if variant.name == name:
                return variant
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "status": self.status.value,
            "variants": [v.to_dict() for v in self.variants],
            "target_sample_size": self.target_sample_size,
            "min_runtime_hours": self.min_runtime_hours,
            "total_impressions": self.total_impressions,
            "owner": self.owner,
            "tags": self.tags,
        }


class ABTestingFramework:
    """
    Enterprise A/B Testing Framework.

    Manages experiments, assigns users to variants, and provides
    statistical analysis of results.

    Example:
        ab = ABTestingFramework()

        # Create experiment
        exp = ab.create_experiment(
            name="model_comparison",
            description="Compare v1 vs v2 of fraud detector"
        )

        # Add variants
        ab.add_variant(exp.id, "control", "fraud_detector", "1.0.0", is_control=True)
        ab.add_variant(exp.id, "treatment", "fraud_detector", "2.0.0")

        # Start experiment
        ab.start_experiment(exp.id)

        # Assign user to variant
        variant = ab.assign_variant(exp.id, user_id="user123")

        # Record conversion
        ab.record_conversion(exp.id, variant.name, value=100.0)
    """

    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize the A/B testing framework.

        Args:
            random_seed: Seed for reproducibility
        """
        self._experiments: Dict[str, Experiment] = {}
        self._assignments: Dict[str, Dict[str, str]] = {}  # exp_id -> {user_id -> variant_name}
        self._random = random.Random(random_seed)

    def create_experiment(
        self,
        name: str,
        description: str = "",
        target_sample_size: int = 1000,
        min_runtime_hours: int = 24,
        owner: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Experiment:
        """
        Create a new experiment.

        Args:
            name: Experiment name
            description: Description
            target_sample_size: Target sample size per variant
            min_runtime_hours: Minimum runtime before stopping
            owner: Experiment owner
            tags: Metadata tags

        Returns:
            Created experiment
        """
        exp_id = hashlib.sha256(
            f"{name}:{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]

        experiment = Experiment(
            id=exp_id,
            name=name,
            description=description,
            target_sample_size=target_sample_size,
            min_runtime_hours=min_runtime_hours,
            owner=owner,
            tags=tags or {},
        )

        self._experiments[exp_id] = experiment
        self._assignments[exp_id] = {}

        return experiment

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get an experiment by ID."""
        return self._experiments.get(experiment_id)

    def list_experiments(
        self,
        status: Optional[ExperimentStatus] = None,
    ) -> List[Experiment]:
        """
        List experiments.

        Args:
            status: Filter by status

        Returns:
            List of experiments
        """
        experiments = list(self._experiments.values())
        if status:
            experiments = [e for e in experiments if e.status == status]
        return experiments

    def add_variant(
        self,
        experiment_id: str,
        name: str,
        model_name: str,
        model_version: str,
        weight: float = 1.0,
        is_control: bool = False,
        description: str = "",
        config: Optional[Dict[str, Any]] = None,
    ) -> Variant:
        """
        Add a variant to an experiment.

        Args:
            experiment_id: Experiment ID
            name: Variant name
            model_name: Model name
            model_version: Model version
            weight: Traffic weight
            is_control: Whether this is the control variant
            description: Variant description
            config: Additional configuration

        Returns:
            Created variant

        Raises:
            ValueError: If experiment not found or already running
        """
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment '{experiment_id}' not found")

        if experiment.status != ExperimentStatus.DRAFT:
            raise ValueError("Cannot add variants to non-draft experiment")

        if experiment.get_variant(name):
            raise ValueError(f"Variant '{name}' already exists")

        variant = Variant(
            name=name,
            model_name=model_name,
            model_version=model_version,
            weight=weight,
            is_control=is_control,
            description=description,
            config=config or {},
        )

        experiment.variants.append(variant)
        return variant

    def start_experiment(self, experiment_id: str) -> Experiment:
        """
        Start an experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            Started experiment

        Raises:
            ValueError: If experiment not found or has no variants
        """
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment '{experiment_id}' not found")

        if len(experiment.variants) < 2:
            raise ValueError("Experiment must have at least 2 variants")

        if not experiment.control:
            raise ValueError("Experiment must have a control variant")

        experiment.status = ExperimentStatus.RUNNING
        experiment.started_at = datetime.utcnow()

        return experiment

    def pause_experiment(self, experiment_id: str) -> Experiment:
        """Pause a running experiment."""
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment '{experiment_id}' not found")

        experiment.status = ExperimentStatus.PAUSED
        return experiment

    def resume_experiment(self, experiment_id: str) -> Experiment:
        """Resume a paused experiment."""
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment '{experiment_id}' not found")

        if experiment.status != ExperimentStatus.PAUSED:
            raise ValueError("Can only resume paused experiments")

        experiment.status = ExperimentStatus.RUNNING
        return experiment

    def stop_experiment(self, experiment_id: str) -> Experiment:
        """Stop and complete an experiment."""
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment '{experiment_id}' not found")

        experiment.status = ExperimentStatus.COMPLETED
        experiment.ended_at = datetime.utcnow()

        return experiment

    def assign_variant(
        self,
        experiment_id: str,
        user_id: str,
        sticky: bool = True,
    ) -> Optional[Variant]:
        """
        Assign a user to a variant.

        Args:
            experiment_id: Experiment ID
            user_id: User identifier
            sticky: Whether to return same variant for same user

        Returns:
            Assigned variant or None if experiment not running
        """
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            return None

        if experiment.status != ExperimentStatus.RUNNING:
            return None

        # Check for sticky assignment
        if sticky and user_id in self._assignments.get(experiment_id, {}):
            variant_name = self._assignments[experiment_id][user_id]
            return experiment.get_variant(variant_name)

        # Weighted random assignment
        total_weight = sum(v.weight for v in experiment.variants)
        rand_val = self._random.random() * total_weight

        cumulative = 0.0
        for variant in experiment.variants:
            cumulative += variant.weight
            if rand_val <= cumulative:
                if sticky:
                    self._assignments[experiment_id][user_id] = variant.name
                variant.metrics.impressions += 1
                return variant

        # Fallback to last variant
        variant = experiment.variants[-1]
        if sticky:
            self._assignments[experiment_id][user_id] = variant.name
        variant.metrics.impressions += 1
        return variant

    def record_conversion(
        self,
        experiment_id: str,
        variant_name: str,
        value: float = 1.0,
    ) -> bool:
        """
        Record a conversion for a variant.

        Args:
            experiment_id: Experiment ID
            variant_name: Variant name
            value: Conversion value

        Returns:
            True if recorded, False otherwise
        """
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            return False

        variant = experiment.get_variant(variant_name)
        if not variant:
            return False

        variant.metrics.conversions += 1
        variant.metrics.total_value += value

        return True

    def get_results(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get experiment results with statistical analysis.

        Args:
            experiment_id: Experiment ID

        Returns:
            Results with statistical significance
        """
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment '{experiment_id}' not found")

        control = experiment.control
        if not control:
            raise ValueError("No control variant found")

        results = {
            "experiment_id": experiment_id,
            "experiment_name": experiment.name,
            "status": experiment.status.value,
            "total_impressions": experiment.total_impressions,
            "variants": {},
        }

        for variant in experiment.variants:
            variant_results = variant.metrics.to_dict()

            # Calculate statistical significance vs control
            if not variant.is_control and control.metrics.impressions > 0:
                significance = self._calculate_significance(
                    control.metrics, variant.metrics
                )
                variant_results["vs_control"] = significance

            results["variants"][variant.name] = variant_results

        # Determine winner
        results["winner"] = self._determine_winner(experiment)

        return results

    def _calculate_significance(
        self,
        control_metrics: VariantMetrics,
        treatment_metrics: VariantMetrics,
    ) -> Dict[str, Any]:
        """
        Calculate statistical significance between control and treatment.

        Uses a two-proportion z-test.
        """
        n_c = control_metrics.impressions
        n_t = treatment_metrics.impressions

        if n_c < 10 or n_t < 10:
            return {
                "is_significant": False,
                "reason": "Insufficient sample size",
            }

        p_c = control_metrics.conversion_rate
        p_t = treatment_metrics.conversion_rate

        # Pooled proportion
        p_pool = (
            (control_metrics.conversions + treatment_metrics.conversions) /
            (n_c + n_t)
        )

        if p_pool == 0 or p_pool == 1:
            return {
                "is_significant": False,
                "reason": "No variance in conversions",
            }

        # Standard error
        se = math.sqrt(p_pool * (1 - p_pool) * (1/n_c + 1/n_t))

        if se == 0:
            return {
                "is_significant": False,
                "reason": "Zero standard error",
            }

        # Z-score
        z_score = (p_t - p_c) / se

        # Two-tailed p-value (using normal approximation)
        p_value = 2 * (1 - self._normal_cdf(abs(z_score)))

        # Effect size (relative lift)
        relative_lift = ((p_t - p_c) / p_c * 100) if p_c > 0 else 0

        return {
            "is_significant": p_value < 0.05,
            "p_value": round(p_value, 4),
            "z_score": round(z_score, 4),
            "relative_lift_percent": round(relative_lift, 2),
            "confidence_level": 0.95 if p_value < 0.05 else None,
        }

    def _normal_cdf(self, x: float) -> float:
        """
        Cumulative distribution function for standard normal.

        Uses the error function approximation.
        """
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def _determine_winner(self, experiment: Experiment) -> Optional[str]:
        """Determine the winning variant."""
        if experiment.status != ExperimentStatus.COMPLETED:
            return None

        control = experiment.control
        if not control:
            return None

        best_variant = None
        best_lift = 0.0

        for variant in experiment.variants:
            if variant.is_control:
                continue

            if variant.metrics.impressions < 100:
                continue

            # Calculate significance
            sig = self._calculate_significance(control.metrics, variant.metrics)

            if sig.get("is_significant") and sig.get("relative_lift_percent", 0) > best_lift:
                best_lift = sig["relative_lift_percent"]
                best_variant = variant.name

        return best_variant

    def clear(self) -> None:
        """Clear all experiments (for testing)."""
        self._experiments.clear()
        self._assignments.clear()


# Global singleton
_ab_framework: Optional[ABTestingFramework] = None


def get_ab_framework() -> ABTestingFramework:
    """Get the global A/B testing framework instance."""
    global _ab_framework
    if _ab_framework is None:
        _ab_framework = ABTestingFramework()
    return _ab_framework
