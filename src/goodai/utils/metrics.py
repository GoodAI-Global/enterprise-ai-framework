"""
Metrics Module

Standard manufacturing metrics calculations.
Includes OEE, TEEP, and custom KPI tracking.

Good AI Philosophy: Evidence over opinions.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime

import pandas as pd
import numpy as np


@dataclass
class OEEResult:
    """OEE calculation result."""

    availability: float
    performance: float
    quality: float
    oee: float
    losses: Dict[str, float]
    period: str
    data_quality_score: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "availability": self.availability,
            "performance": self.performance,
            "quality": self.quality,
            "oee": self.oee,
            "losses": self.losses,
            "period": self.period,
            "data_quality_score": self.data_quality_score,
        }


@dataclass
class MetricValue:
    """A single metric value with metadata."""

    name: str
    value: float
    unit: str
    timestamp: str
    target: Optional[float] = None
    previous: Optional[float] = None

    @property
    def on_target(self) -> Optional[bool]:
        """Check if value meets target."""
        if self.target is None:
            return None
        return self.value >= self.target

    @property
    def change(self) -> Optional[float]:
        """Calculate change from previous."""
        if self.previous is None:
            return None
        if self.previous == 0:
            return None
        return (self.value - self.previous) / self.previous


class OEECalculator:
    """
    Calculate OEE (Overall Equipment Effectiveness).

    OEE = Availability x Performance x Quality

    Provides detailed loss analysis and trend tracking.

    Example:
        >>> calc = OEECalculator()
        >>> data = pd.DataFrame({
        ...     "uptime_minutes": [55, 58, 30, 57],
        ...     "planned_minutes": [60, 60, 60, 60],
        ...     "actual_output": [95, 98, 45, 92],
        ...     "theoretical_output": [100, 100, 100, 100],
        ...     "good_units": [94, 97, 44, 90],
        ...     "total_units": [95, 98, 45, 92]
        ... })
        >>> result = calc.calculate(data)
        >>> print(f"OEE: {result.oee:.1%}")
    """

    def __init__(
        self,
        availability_target: float = 0.90,
        performance_target: float = 0.95,
        quality_target: float = 0.99
    ):
        self.targets = {
            "availability": availability_target,
            "performance": performance_target,
            "quality": quality_target,
            "oee": availability_target * performance_target * quality_target
        }

    def calculate(
        self,
        data: pd.DataFrame,
        uptime_col: str = "uptime_minutes",
        planned_col: str = "planned_minutes",
        actual_output_col: str = "actual_output",
        theoretical_output_col: str = "theoretical_output",
        good_units_col: str = "good_units",
        total_units_col: str = "total_units",
        period: Optional[str] = None
    ) -> OEEResult:
        """
        Calculate OEE from production data.

        Args:
            data: DataFrame with production metrics
            uptime_col: Column for actual uptime
            planned_col: Column for planned time
            actual_output_col: Column for actual production
            theoretical_output_col: Column for theoretical max
            good_units_col: Column for good quality units
            total_units_col: Column for total units produced
            period: Optional period description

        Returns:
            OEEResult with detailed metrics
        """
        # Validate data
        required_cols = [
            uptime_col, planned_col, actual_output_col,
            theoretical_output_col, good_units_col, total_units_col
        ]

        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            # Try alternative column names
            data = self._map_columns(data)

        # Calculate totals
        total_uptime = data[uptime_col].sum()
        total_planned = data[planned_col].sum()
        total_actual = data[actual_output_col].sum()
        total_theoretical = data[theoretical_output_col].sum()
        total_good = data[good_units_col].sum()
        total_produced = data[total_units_col].sum()

        # Calculate OEE components
        availability = total_uptime / total_planned if total_planned > 0 else 0
        performance = total_actual / total_theoretical if total_theoretical > 0 else 0
        quality = total_good / total_produced if total_produced > 0 else 0
        oee = availability * performance * quality

        # Calculate losses
        losses = self._calculate_losses(
            availability, performance, quality,
            total_planned, total_theoretical, total_produced
        )

        # Assess data quality
        data_quality = self._assess_data_quality(data, required_cols)

        return OEEResult(
            availability=round(availability, 4),
            performance=round(performance, 4),
            quality=round(quality, 4),
            oee=round(oee, 4),
            losses=losses,
            period=period or datetime.now().strftime("%Y-%m-%d"),
            data_quality_score=data_quality
        )

    def _map_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Map alternative column names."""
        mappings = {
            "uptime_minutes": ["uptime", "runtime", "actual_time"],
            "planned_minutes": ["planned", "scheduled", "available_time"],
            "actual_output": ["output", "production", "actual_units"],
            "theoretical_output": ["target", "planned_output", "max_output"],
            "good_units": ["good", "quality_units", "conforming"],
            "total_units": ["total", "produced", "all_units"]
        }

        df = data.copy()
        for standard, alternatives in mappings.items():
            if standard not in df.columns:
                for alt in alternatives:
                    if alt in df.columns:
                        df[standard] = df[alt]
                        break

        return df

    def _calculate_losses(
        self,
        availability: float,
        performance: float,
        quality: float,
        planned_time: float,
        theoretical_output: float,
        total_produced: float
    ) -> Dict[str, float]:
        """Calculate loss breakdown."""
        # Availability loss (downtime)
        availability_loss = (1 - availability)

        # Performance loss (speed loss during uptime)
        performance_loss = availability * (1 - performance)

        # Quality loss (defects during effective operation)
        quality_loss = availability * performance * (1 - quality)

        # Convert to percentages of total possible output
        return {
            "availability_loss_pct": round(availability_loss * 100, 2),
            "performance_loss_pct": round(performance_loss * 100, 2),
            "quality_loss_pct": round(quality_loss * 100, 2),
            "total_loss_pct": round((availability_loss + performance_loss + quality_loss) * 100, 2),
            "effective_utilization_pct": round(availability * performance * quality * 100, 2),
        }

    def _assess_data_quality(
        self,
        data: pd.DataFrame,
        required_cols: List[str]
    ) -> float:
        """Assess quality of input data."""
        score = 1.0

        for col in required_cols:
            if col in data.columns:
                # Penalize missing values
                null_ratio = data[col].isnull().sum() / len(data)
                score -= null_ratio * 0.1

                # Penalize impossible values
                if (data[col] < 0).any():
                    score -= 0.1
            else:
                score -= 0.15

        return max(0, round(score, 2))

    def compare_periods(
        self,
        current: OEEResult,
        previous: OEEResult
    ) -> Dict[str, Any]:
        """Compare two OEE periods."""
        return {
            "oee_change": round(current.oee - previous.oee, 4),
            "oee_change_pct": round((current.oee - previous.oee) / previous.oee * 100, 2) if previous.oee > 0 else 0,
            "availability_change": round(current.availability - previous.availability, 4),
            "performance_change": round(current.performance - previous.performance, 4),
            "quality_change": round(current.quality - previous.quality, 4),
            "improved": current.oee > previous.oee,
            "biggest_gain": self._identify_biggest_change(current, previous, positive=True),
            "biggest_loss": self._identify_biggest_change(current, previous, positive=False),
        }

    def _identify_biggest_change(
        self,
        current: OEEResult,
        previous: OEEResult,
        positive: bool
    ) -> str:
        """Identify the component with biggest change."""
        changes = {
            "availability": current.availability - previous.availability,
            "performance": current.performance - previous.performance,
            "quality": current.quality - previous.quality,
        }

        if positive:
            biggest = max(changes.items(), key=lambda x: x[1])
            return biggest[0] if biggest[1] > 0 else "none"
        else:
            biggest = min(changes.items(), key=lambda x: x[1])
            return biggest[0] if biggest[1] < 0 else "none"


class MetricsCollector:
    """
    Collect and track custom metrics over time.

    Example:
        >>> collector = MetricsCollector()
        >>> collector.record("throughput", 150, unit="units/hr", target=160)
        >>> collector.record("defect_rate", 0.02, unit="%", target=0.01)
        >>> summary = collector.get_summary()
    """

    def __init__(self):
        self._metrics: Dict[str, List[MetricValue]] = {}
        self._targets: Dict[str, float] = {}

    def record(
        self,
        name: str,
        value: float,
        unit: str = "",
        target: Optional[float] = None,
        timestamp: Optional[str] = None
    ):
        """Record a metric value."""
        if name not in self._metrics:
            self._metrics[name] = []

        # Get previous value
        previous = None
        if self._metrics[name]:
            previous = self._metrics[name][-1].value

        # Use provided target or stored target
        if target is not None:
            self._targets[name] = target
        effective_target = self._targets.get(name)

        metric = MetricValue(
            name=name,
            value=value,
            unit=unit,
            timestamp=timestamp or datetime.now().isoformat(),
            target=effective_target,
            previous=previous
        )

        self._metrics[name].append(metric)

    def get_latest(self, name: str) -> Optional[MetricValue]:
        """Get latest value for a metric."""
        if name in self._metrics and self._metrics[name]:
            return self._metrics[name][-1]
        return None

    def get_history(
        self,
        name: str,
        limit: Optional[int] = None
    ) -> List[MetricValue]:
        """Get historical values for a metric."""
        history = self._metrics.get(name, [])
        if limit:
            return history[-limit:]
        return history

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        summary = {}

        for name, values in self._metrics.items():
            if not values:
                continue

            latest = values[-1]
            all_values = [v.value for v in values]

            summary[name] = {
                "current": latest.value,
                "unit": latest.unit,
                "target": latest.target,
                "on_target": latest.on_target,
                "change_from_previous": latest.change,
                "min": min(all_values),
                "max": max(all_values),
                "mean": sum(all_values) / len(all_values),
                "count": len(values),
            }

        return summary

    def get_dashboard_data(self) -> List[Dict[str, Any]]:
        """Get data formatted for dashboard display."""
        dashboard = []

        for name, values in self._metrics.items():
            if not values:
                continue

            latest = values[-1]

            # Determine status
            if latest.on_target is None:
                status = "neutral"
            elif latest.on_target:
                status = "good"
            else:
                status = "warning"

            # Calculate trend
            if len(values) >= 2:
                recent = [v.value for v in values[-5:]]
                trend = "up" if recent[-1] > recent[0] else "down" if recent[-1] < recent[0] else "flat"
            else:
                trend = "flat"

            dashboard.append({
                "name": name,
                "value": latest.value,
                "unit": latest.unit,
                "target": latest.target,
                "status": status,
                "trend": trend,
                "change": latest.change,
            })

        return dashboard

    def export_to_dataframe(self, name: Optional[str] = None) -> pd.DataFrame:
        """Export metrics to DataFrame."""
        records = []

        metrics_to_export = {name: self._metrics[name]} if name else self._metrics

        for metric_name, values in metrics_to_export.items():
            for v in values:
                records.append({
                    "metric": metric_name,
                    "value": v.value,
                    "unit": v.unit,
                    "target": v.target,
                    "timestamp": v.timestamp,
                })

        return pd.DataFrame(records)

    def clear(self, name: Optional[str] = None):
        """Clear metric history."""
        if name:
            self._metrics.pop(name, None)
        else:
            self._metrics.clear()
