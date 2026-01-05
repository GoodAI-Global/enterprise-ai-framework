"""
Explainability Module for ML Models

Provides model-agnostic explanation methods for predictions,
supporting transparency and regulatory compliance.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import hashlib


class ExplanationType(Enum):
    """Types of explanations."""

    FEATURE_IMPORTANCE = "feature_importance"
    SHAP = "shap"
    LIME = "lime"
    COUNTERFACTUAL = "counterfactual"
    RULE_BASED = "rule_based"
    ANCHOR = "anchor"
    DECISION_PATH = "decision_path"
    ATTENTION = "attention"
    CUSTOM = "custom"


@dataclass
class FeatureImportance:
    """Feature importance for a prediction."""

    feature: str
    importance: float
    direction: str = "neutral"  # positive, negative, neutral
    value: Any = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature": self.feature,
            "importance": self.importance,
            "direction": self.direction,
            "value": self.value,
        }


@dataclass
class Counterfactual:
    """A counterfactual explanation."""

    original_prediction: Any
    counterfactual_prediction: Any
    changes: Dict[str, Tuple[Any, Any]]  # feature -> (original, changed)
    distance: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_prediction": self.original_prediction,
            "counterfactual_prediction": self.counterfactual_prediction,
            "changes": {
                k: {"from": v[0], "to": v[1]}
                for k, v in self.changes.items()
            },
            "distance": self.distance,
        }


@dataclass
class Rule:
    """A rule-based explanation."""

    condition: str
    coverage: float
    precision: float
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "condition": self.condition,
            "coverage": self.coverage,
            "precision": self.precision,
            "description": self.description,
        }


@dataclass
class Explanation:
    """A complete explanation for a prediction."""

    id: str
    explanation_type: ExplanationType
    model_name: str
    model_version: str
    prediction_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    input_data: Dict[str, Any] = field(default_factory=dict)
    prediction: Any = None
    confidence: Optional[float] = None
    feature_importance: List[FeatureImportance] = field(default_factory=list)
    counterfactuals: List[Counterfactual] = field(default_factory=list)
    rules: List[Rule] = field(default_factory=list)
    decision_path: List[str] = field(default_factory=list)
    summary: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    tenant_id: Optional[str] = None

    @property
    def top_features(self) -> List[FeatureImportance]:
        """Get top 5 most important features."""
        sorted_features = sorted(
            self.feature_importance,
            key=lambda f: abs(f.importance),
            reverse=True
        )
        return sorted_features[:5]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "explanation_type": self.explanation_type.value,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "prediction_id": self.prediction_id,
            "created_at": self.created_at.isoformat(),
            "input_data": self.input_data,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "feature_importance": [f.to_dict() for f in self.feature_importance],
            "counterfactuals": [c.to_dict() for c in self.counterfactuals],
            "rules": [r.to_dict() for r in self.rules],
            "decision_path": self.decision_path,
            "summary": self.summary,
            "metadata": self.metadata,
            "tenant_id": self.tenant_id,
        }

    def to_natural_language(self) -> str:
        """Generate natural language explanation."""
        parts = []

        if self.prediction is not None:
            parts.append(f"The model predicted: {self.prediction}")

        if self.confidence is not None:
            parts.append(f"with {self.confidence:.1%} confidence.")

        if self.top_features:
            feature_parts = []
            for f in self.top_features[:3]:
                direction = "increased" if f.direction == "positive" else "decreased"
                feature_parts.append(f"'{f.feature}' {direction} the likelihood")
            parts.append("Key factors: " + "; ".join(feature_parts) + ".")

        if self.rules:
            rule = self.rules[0]
            parts.append(f"Rule: {rule.condition}")

        if self.counterfactuals:
            cf = self.counterfactuals[0]
            changes = [f"{k}: {v[0]} -> {v[1]}" for k, v in list(cf.changes.items())[:2]]
            parts.append(f"If {', '.join(changes)}, the prediction would change to {cf.counterfactual_prediction}.")

        return " ".join(parts) if parts else self.summary


class ExplainabilityEngine:
    """
    Enterprise Explainability Engine.

    Provides model-agnostic explanations for predictions.

    Example:
        engine = ExplainabilityEngine()

        # Register explainer for a model
        engine.register_explainer(
            model_name="fraud_detector",
            explainer=my_shap_explainer
        )

        # Get explanation
        explanation = engine.explain(
            model_name="fraud_detector",
            model_version="1.0.0",
            input_data={"amount": 1000, "merchant": "online"},
            prediction="fraud",
            confidence=0.95
        )

        # Get natural language
        print(explanation.to_natural_language())
    """

    def __init__(self, store_explanations: bool = True):
        """
        Initialize the engine.

        Args:
            store_explanations: Whether to store explanations
        """
        self._explainers: Dict[str, Callable] = {}
        self._explanations: List[Explanation] = []
        self._store = store_explanations
        self._default_explainer: Optional[Callable] = None

    def register_explainer(
        self,
        model_name: str,
        explainer: Callable[[Dict[str, Any], Any], Dict[str, Any]],
    ) -> None:
        """
        Register an explainer for a model.

        Args:
            model_name: Model name
            explainer: Explainer function(input_data, prediction) -> explanation_dict

        The explainer should return a dict with optional keys:
            - feature_importance: List of {feature, importance, direction}
            - counterfactuals: List of counterfactual dicts
            - rules: List of rule dicts
            - decision_path: List of decision steps
            - summary: Natural language summary
        """
        self._explainers[model_name] = explainer

    def set_default_explainer(
        self,
        explainer: Callable[[Dict[str, Any], Any], Dict[str, Any]],
    ) -> None:
        """Set a default explainer for models without registered explainers."""
        self._default_explainer = explainer

    def explain(
        self,
        model_name: str,
        model_version: str,
        input_data: Dict[str, Any],
        prediction: Any,
        confidence: Optional[float] = None,
        prediction_id: Optional[str] = None,
        explanation_type: ExplanationType = ExplanationType.FEATURE_IMPORTANCE,
        tenant_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Explanation:
        """
        Generate an explanation for a prediction.

        Args:
            model_name: Model name
            model_version: Model version
            input_data: Input data used for prediction
            prediction: The model's prediction
            confidence: Prediction confidence
            prediction_id: ID of the prediction
            explanation_type: Type of explanation
            tenant_id: Tenant ID
            metadata: Additional metadata

        Returns:
            Generated explanation
        """
        exp_id = hashlib.sha256(
            f"{datetime.utcnow().isoformat()}:{model_name}:{prediction_id}".encode()
        ).hexdigest()[:16]

        explanation = Explanation(
            id=exp_id,
            explanation_type=explanation_type,
            model_name=model_name,
            model_version=model_version,
            prediction_id=prediction_id,
            input_data=input_data,
            prediction=prediction,
            confidence=confidence,
            tenant_id=tenant_id,
            metadata=metadata or {},
        )

        # Get explainer
        explainer = self._explainers.get(model_name) or self._default_explainer

        if explainer:
            try:
                result = explainer(input_data, prediction)
                self._apply_explainer_result(explanation, result)
            except Exception as e:
                explanation.metadata["explainer_error"] = str(e)
        else:
            # Fallback: simple feature importance based on input values
            explanation.feature_importance = self._simple_feature_importance(
                input_data, prediction
            )
            explanation.summary = self._generate_simple_summary(explanation)

        if self._store:
            self._explanations.append(explanation)

        return explanation

    def _apply_explainer_result(
        self,
        explanation: Explanation,
        result: Dict[str, Any],
    ) -> None:
        """Apply explainer result to explanation."""
        if "feature_importance" in result:
            for fi in result["feature_importance"]:
                explanation.feature_importance.append(
                    FeatureImportance(
                        feature=fi.get("feature", ""),
                        importance=fi.get("importance", 0.0),
                        direction=fi.get("direction", "neutral"),
                        value=fi.get("value"),
                    )
                )

        if "counterfactuals" in result:
            for cf in result["counterfactuals"]:
                explanation.counterfactuals.append(
                    Counterfactual(
                        original_prediction=cf.get("original_prediction"),
                        counterfactual_prediction=cf.get("counterfactual_prediction"),
                        changes=cf.get("changes", {}),
                        distance=cf.get("distance", 0.0),
                    )
                )

        if "rules" in result:
            for rule in result["rules"]:
                explanation.rules.append(
                    Rule(
                        condition=rule.get("condition", ""),
                        coverage=rule.get("coverage", 0.0),
                        precision=rule.get("precision", 0.0),
                        description=rule.get("description", ""),
                    )
                )

        if "decision_path" in result:
            explanation.decision_path = result["decision_path"]

        if "summary" in result:
            explanation.summary = result["summary"]

    def _simple_feature_importance(
        self,
        input_data: Dict[str, Any],
        prediction: Any,
    ) -> List[FeatureImportance]:
        """Generate simple feature importance based on input values."""
        features = []

        for feature, value in input_data.items():
            # Simple heuristic: numeric values get importance based on magnitude
            if isinstance(value, (int, float)):
                importance = min(1.0, abs(value) / 1000)  # Normalize
                direction = "positive" if value > 0 else "negative"
            else:
                importance = 0.5
                direction = "neutral"

            features.append(
                FeatureImportance(
                    feature=feature,
                    importance=importance,
                    direction=direction,
                    value=value,
                )
            )

        return sorted(features, key=lambda f: f.importance, reverse=True)

    def _generate_simple_summary(self, explanation: Explanation) -> str:
        """Generate a simple summary."""
        if not explanation.feature_importance:
            return f"Predicted: {explanation.prediction}"

        top = explanation.top_features[:3]
        factors = [f.feature for f in top]

        return (
            f"Predicted {explanation.prediction} based primarily on: "
            f"{', '.join(factors)}."
        )

    def get_explanation(self, explanation_id: str) -> Optional[Explanation]:
        """Get an explanation by ID."""
        for exp in self._explanations:
            if exp.id == explanation_id:
                return exp
        return None

    def get_explanations(
        self,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        prediction_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Explanation]:
        """
        Query explanations.

        Args:
            model_name: Filter by model name
            model_version: Filter by version
            prediction_id: Filter by prediction ID
            limit: Maximum results

        Returns:
            List of explanations
        """
        results = self._explanations

        if model_name:
            results = [e for e in results if e.model_name == model_name]

        if model_version:
            results = [e for e in results if e.model_version == model_version]

        if prediction_id:
            results = [e for e in results if e.prediction_id == prediction_id]

        return sorted(results, key=lambda e: e.created_at, reverse=True)[:limit]

    def generate_audit_report(
        self,
        model_name: str,
        model_version: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Generate an audit report for model explanations.

        Useful for regulatory compliance (GDPR, CCPA, etc.).

        Args:
            model_name: Model name
            model_version: Model version
            start_date: Start of period
            end_date: End of period

        Returns:
            Audit report
        """
        explanations = self.get_explanations(
            model_name=model_name,
            model_version=model_version,
            limit=10000,
        )

        if start_date:
            explanations = [e for e in explanations if e.created_at >= start_date]
        if end_date:
            explanations = [e for e in explanations if e.created_at <= end_date]

        # Aggregate statistics
        prediction_counts: Dict[str, int] = {}
        feature_frequency: Dict[str, int] = {}
        confidence_values = []

        for exp in explanations:
            # Count predictions
            pred_key = str(exp.prediction)
            prediction_counts[pred_key] = prediction_counts.get(pred_key, 0) + 1

            # Track feature frequency in explanations
            for fi in exp.feature_importance[:5]:  # Top 5
                feature_frequency[fi.feature] = feature_frequency.get(fi.feature, 0) + 1

            if exp.confidence:
                confidence_values.append(exp.confidence)

        return {
            "model_name": model_name,
            "model_version": model_version,
            "period": {
                "start": start_date.isoformat() if start_date else None,
                "end": end_date.isoformat() if end_date else None,
            },
            "total_explanations": len(explanations),
            "prediction_distribution": prediction_counts,
            "most_important_features": sorted(
                feature_frequency.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10],
            "average_confidence": (
                sum(confidence_values) / len(confidence_values)
                if confidence_values else None
            ),
            "explanation_coverage": len(explanations),
        }

    def clear(self) -> None:
        """Clear all stored explanations (for testing)."""
        self._explanations.clear()


# Global singleton
_engine: Optional[ExplainabilityEngine] = None


def get_explainability_engine() -> ExplainabilityEngine:
    """Get the global explainability engine instance."""
    global _engine
    if _engine is None:
        _engine = ExplainabilityEngine()
    return _engine
