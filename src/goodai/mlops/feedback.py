"""
Feedback Loop System for ML Models

Collects, processes, and analyzes feedback to improve model performance
and enable continuous learning pipelines.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import statistics
import hashlib


class FeedbackType(Enum):
    """Types of feedback."""

    # Explicit feedback
    RATING = "rating"  # Numeric rating (1-5, 1-10, etc.)
    THUMBS = "thumbs"  # Binary thumbs up/down
    CORRECTION = "correction"  # User corrected the output
    COMMENT = "comment"  # Free-text comment

    # Implicit feedback
    CLICK = "click"  # User clicked on result
    DWELL_TIME = "dwell_time"  # Time spent on result
    CONVERSION = "conversion"  # User converted
    BOUNCE = "bounce"  # User left without action

    # Quality feedback
    ACCURACY = "accuracy"  # Was prediction accurate?
    LATENCY = "latency"  # Response time
    ERROR = "error"  # Error occurred


@dataclass
class Feedback:
    """A single feedback entry."""

    id: str
    feedback_type: FeedbackType
    model_name: str
    model_version: str
    prediction_id: Optional[str] = None
    user_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    value: Any = None  # The feedback value (rating, correction, etc.)
    context: Dict[str, Any] = field(default_factory=dict)  # Input/output context
    metadata: Dict[str, Any] = field(default_factory=dict)
    tenant_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "feedback_type": self.feedback_type.value,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "prediction_id": self.prediction_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "context": self.context,
            "metadata": self.metadata,
            "tenant_id": self.tenant_id,
        }


@dataclass
class FeedbackSummary:
    """Summary of feedback for a model version."""

    model_name: str
    model_version: str
    period_start: datetime
    period_end: datetime
    total_feedback: int = 0
    positive_feedback: int = 0
    negative_feedback: int = 0
    average_rating: Optional[float] = None
    correction_rate: float = 0.0
    error_rate: float = 0.0
    feedback_by_type: Dict[str, int] = field(default_factory=dict)

    @property
    def satisfaction_rate(self) -> float:
        """Calculate satisfaction rate."""
        total = self.positive_feedback + self.negative_feedback
        if total == 0:
            return 0.0
        return self.positive_feedback / total

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_feedback": self.total_feedback,
            "positive_feedback": self.positive_feedback,
            "negative_feedback": self.negative_feedback,
            "average_rating": self.average_rating,
            "satisfaction_rate": self.satisfaction_rate,
            "correction_rate": self.correction_rate,
            "error_rate": self.error_rate,
            "feedback_by_type": self.feedback_by_type,
        }


@dataclass
class AlertConfig:
    """Configuration for feedback alerts."""

    min_satisfaction_rate: float = 0.7
    max_error_rate: float = 0.1
    max_correction_rate: float = 0.2
    min_sample_size: int = 100
    check_interval_hours: int = 1


class FeedbackLoop:
    """
    Enterprise Feedback Loop System.

    Collects and analyzes feedback to enable continuous model improvement.

    Example:
        feedback_loop = FeedbackLoop()

        # Record explicit feedback
        feedback_loop.record(
            FeedbackType.RATING,
            model_name="fraud_detector",
            model_version="1.0.0",
            value=4,  # Rating out of 5
            prediction_id="pred123",
            user_id="user456"
        )

        # Record correction
        feedback_loop.record(
            FeedbackType.CORRECTION,
            model_name="fraud_detector",
            model_version="1.0.0",
            value={"predicted": "fraud", "corrected": "legitimate"},
            context={"transaction": {...}}
        )

        # Get summary
        summary = feedback_loop.get_summary(
            model_name="fraud_detector",
            model_version="1.0.0",
            hours=24
        )
    """

    def __init__(
        self,
        alert_config: Optional[AlertConfig] = None,
        max_history_days: int = 90,
    ):
        """
        Initialize the feedback loop.

        Args:
            alert_config: Alert configuration
            max_history_days: Maximum days to retain feedback
        """
        self._feedback: List[Feedback] = []
        self._alert_config = alert_config or AlertConfig()
        self._max_history = timedelta(days=max_history_days)
        self._alert_handlers: List[Callable[[str, Dict[str, Any]], None]] = []
        self._processing_callbacks: List[Callable[[Feedback], None]] = []

    def record(
        self,
        feedback_type: FeedbackType,
        model_name: str,
        model_version: str,
        value: Any = None,
        prediction_id: Optional[str] = None,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
    ) -> Feedback:
        """
        Record feedback.

        Args:
            feedback_type: Type of feedback
            model_name: Model name
            model_version: Model version
            value: Feedback value
            prediction_id: Associated prediction ID
            user_id: User who provided feedback
            context: Context (input/output)
            metadata: Additional metadata
            tenant_id: Tenant ID

        Returns:
            Created feedback entry
        """
        feedback_id = hashlib.sha256(
            f"{datetime.utcnow().isoformat()}:{model_name}:{prediction_id}".encode()
        ).hexdigest()[:16]

        feedback = Feedback(
            id=feedback_id,
            feedback_type=feedback_type,
            model_name=model_name,
            model_version=model_version,
            value=value,
            prediction_id=prediction_id,
            user_id=user_id,
            context=context or {},
            metadata=metadata or {},
            tenant_id=tenant_id,
        )

        self._feedback.append(feedback)
        self._cleanup_old_feedback()

        # Invoke processing callbacks
        for callback in self._processing_callbacks:
            try:
                callback(feedback)
            except Exception:
                pass

        # Check for alerts
        self._check_alerts(model_name, model_version)

        return feedback

    def get_feedback(
        self,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        feedback_type: Optional[FeedbackType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        user_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Feedback]:
        """
        Query feedback entries.

        Args:
            model_name: Filter by model name
            model_version: Filter by model version
            feedback_type: Filter by feedback type
            start_time: Start of time range
            end_time: End of time range
            user_id: Filter by user
            limit: Maximum entries to return

        Returns:
            List of feedback entries
        """
        results = self._feedback

        if model_name:
            results = [f for f in results if f.model_name == model_name]

        if model_version:
            results = [f for f in results if f.model_version == model_version]

        if feedback_type:
            results = [f for f in results if f.feedback_type == feedback_type]

        if start_time:
            results = [f for f in results if f.timestamp >= start_time]

        if end_time:
            results = [f for f in results if f.timestamp <= end_time]

        if user_id:
            results = [f for f in results if f.user_id == user_id]

        # Sort by timestamp descending
        results = sorted(results, key=lambda f: f.timestamp, reverse=True)

        return results[:limit]

    def get_summary(
        self,
        model_name: str,
        model_version: str,
        hours: int = 24,
    ) -> FeedbackSummary:
        """
        Get feedback summary for a model version.

        Args:
            model_name: Model name
            model_version: Model version
            hours: Number of hours to summarize

        Returns:
            Feedback summary
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)

        feedback = self.get_feedback(
            model_name=model_name,
            model_version=model_version,
            start_time=start_time,
            end_time=end_time,
            limit=10000,
        )

        summary = FeedbackSummary(
            model_name=model_name,
            model_version=model_version,
            period_start=start_time,
            period_end=end_time,
            total_feedback=len(feedback),
        )

        ratings = []
        corrections = 0
        errors = 0

        for f in feedback:
            # Count by type
            type_name = f.feedback_type.value
            summary.feedback_by_type[type_name] = summary.feedback_by_type.get(type_name, 0) + 1

            # Process different feedback types
            if f.feedback_type == FeedbackType.RATING:
                if isinstance(f.value, (int, float)):
                    ratings.append(f.value)
                    # Assume 1-5 scale, 4+ is positive
                    if f.value >= 4:
                        summary.positive_feedback += 1
                    else:
                        summary.negative_feedback += 1

            elif f.feedback_type == FeedbackType.THUMBS:
                if f.value in (True, 1, "up", "positive"):
                    summary.positive_feedback += 1
                else:
                    summary.negative_feedback += 1

            elif f.feedback_type == FeedbackType.CORRECTION:
                corrections += 1
                summary.negative_feedback += 1

            elif f.feedback_type == FeedbackType.ERROR:
                errors += 1
                summary.negative_feedback += 1

            elif f.feedback_type == FeedbackType.CONVERSION:
                summary.positive_feedback += 1

            elif f.feedback_type == FeedbackType.BOUNCE:
                summary.negative_feedback += 1

        # Calculate aggregates
        if ratings:
            summary.average_rating = statistics.mean(ratings)

        if summary.total_feedback > 0:
            summary.correction_rate = corrections / summary.total_feedback
            summary.error_rate = errors / summary.total_feedback

        return summary

    def get_trends(
        self,
        model_name: str,
        model_version: str,
        days: int = 7,
        bucket_hours: int = 24,
    ) -> List[Dict[str, Any]]:
        """
        Get feedback trends over time.

        Args:
            model_name: Model name
            model_version: Model version
            days: Number of days to analyze
            bucket_hours: Hours per bucket

        Returns:
            List of time buckets with metrics
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)

        feedback = self.get_feedback(
            model_name=model_name,
            model_version=model_version,
            start_time=start_time,
            end_time=end_time,
            limit=100000,
        )

        # Create buckets
        bucket_duration = timedelta(hours=bucket_hours)
        buckets = []

        current_start = start_time
        while current_start < end_time:
            bucket_end = min(current_start + bucket_duration, end_time)

            bucket_feedback = [
                f for f in feedback
                if current_start <= f.timestamp < bucket_end
            ]

            bucket_data = {
                "start": current_start.isoformat(),
                "end": bucket_end.isoformat(),
                "count": len(bucket_feedback),
                "positive": 0,
                "negative": 0,
            }

            for f in bucket_feedback:
                if f.feedback_type in (FeedbackType.RATING, FeedbackType.THUMBS):
                    if f.feedback_type == FeedbackType.RATING and f.value >= 4:
                        bucket_data["positive"] += 1
                    elif f.feedback_type == FeedbackType.THUMBS and f.value in (True, 1, "up"):
                        bucket_data["positive"] += 1
                    else:
                        bucket_data["negative"] += 1

            if bucket_data["positive"] + bucket_data["negative"] > 0:
                bucket_data["satisfaction_rate"] = (
                    bucket_data["positive"] /
                    (bucket_data["positive"] + bucket_data["negative"])
                )
            else:
                bucket_data["satisfaction_rate"] = None

            buckets.append(bucket_data)
            current_start = bucket_end

        return buckets

    def add_processing_callback(
        self,
        callback: Callable[[Feedback], None],
    ) -> None:
        """
        Add a callback for processing new feedback.

        Args:
            callback: Callback function
        """
        self._processing_callbacks.append(callback)

    def add_alert_handler(
        self,
        handler: Callable[[str, Dict[str, Any]], None],
    ) -> None:
        """
        Add an alert handler.

        Args:
            handler: Handler function(alert_type, details)
        """
        self._alert_handlers.append(handler)

    def _check_alerts(self, model_name: str, model_version: str) -> None:
        """Check if alerts should be triggered."""
        summary = self.get_summary(
            model_name=model_name,
            model_version=model_version,
            hours=self._alert_config.check_interval_hours,
        )

        if summary.total_feedback < self._alert_config.min_sample_size:
            return

        alerts = []

        if summary.satisfaction_rate < self._alert_config.min_satisfaction_rate:
            alerts.append({
                "type": "low_satisfaction",
                "current": summary.satisfaction_rate,
                "threshold": self._alert_config.min_satisfaction_rate,
            })

        if summary.error_rate > self._alert_config.max_error_rate:
            alerts.append({
                "type": "high_error_rate",
                "current": summary.error_rate,
                "threshold": self._alert_config.max_error_rate,
            })

        if summary.correction_rate > self._alert_config.max_correction_rate:
            alerts.append({
                "type": "high_correction_rate",
                "current": summary.correction_rate,
                "threshold": self._alert_config.max_correction_rate,
            })

        for alert in alerts:
            alert["model_name"] = model_name
            alert["model_version"] = model_version
            alert["timestamp"] = datetime.utcnow().isoformat()

            for handler in self._alert_handlers:
                try:
                    handler(alert["type"], alert)
                except Exception:
                    pass

    def _cleanup_old_feedback(self) -> None:
        """Remove old feedback entries."""
        cutoff = datetime.utcnow() - self._max_history
        self._feedback = [f for f in self._feedback if f.timestamp > cutoff]

    def get_corrections_for_retraining(
        self,
        model_name: str,
        model_version: str,
        min_count: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get corrections suitable for retraining.

        Args:
            model_name: Model name
            model_version: Model version
            min_count: Minimum corrections required

        Returns:
            List of training examples from corrections
        """
        corrections = self.get_feedback(
            model_name=model_name,
            model_version=model_version,
            feedback_type=FeedbackType.CORRECTION,
            limit=10000,
        )

        if len(corrections) < min_count:
            return []

        training_examples = []
        for correction in corrections:
            if correction.context and correction.value:
                example = {
                    "input": correction.context.get("input"),
                    "original_output": correction.context.get("output"),
                    "corrected_output": correction.value,
                    "timestamp": correction.timestamp.isoformat(),
                    "user_id": correction.user_id,
                }
                if example["input"] and example["corrected_output"]:
                    training_examples.append(example)

        return training_examples

    def clear(self) -> None:
        """Clear all feedback (for testing)."""
        self._feedback.clear()


# Global singleton
_feedback_loop: Optional[FeedbackLoop] = None


def get_feedback_loop() -> FeedbackLoop:
    """Get the global feedback loop instance."""
    global _feedback_loop
    if _feedback_loop is None:
        _feedback_loop = FeedbackLoop()
    return _feedback_loop
