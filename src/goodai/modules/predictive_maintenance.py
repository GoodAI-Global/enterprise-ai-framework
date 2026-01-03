"""
Predictive Maintenance Module

Simple, explainable predictive maintenance using statistical methods.
No complex ML for v1 - focuses on reliability and transparency.

Good AI Philosophy: Augment first.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum

import pandas as pd
import numpy as np


class AlertSeverity(Enum):
    """Maintenance alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class HealthStatus(Enum):
    """Equipment health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    AT_RISK = "at_risk"
    CRITICAL = "critical"


@dataclass
class MaintenanceAlert:
    """A maintenance alert or recommendation."""

    equipment_id: str
    alert_type: str
    severity: AlertSeverity
    message: str
    recommended_action: str
    estimated_days_to_failure: Optional[float]
    confidence: float
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "equipment_id": self.equipment_id,
            "alert_type": self.alert_type,
            "severity": self.severity.value,
            "message": self.message,
            "recommended_action": self.recommended_action,
            "estimated_days_to_failure": self.estimated_days_to_failure,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }


@dataclass
class EquipmentHealth:
    """Equipment health assessment."""

    equipment_id: str
    status: HealthStatus
    health_score: float  # 0-100
    degradation_rate: float  # Change per day
    estimated_remaining_life_days: Optional[float]
    key_indicators: Dict[str, float]
    alerts: List[MaintenanceAlert]
    last_updated: str


class PredictiveMaintenance:
    """
    Predictive maintenance using statistical degradation analysis.

    Uses simple, explainable methods:
    - Trend analysis for degradation
    - Threshold monitoring
    - Pattern matching against known failure signatures

    Args:
        config: Configuration dictionary with thresholds

    Example:
        >>> pm = PredictiveMaintenance()
        >>> sensor_data = pd.DataFrame({
        ...     "timestamp": [...],
        ...     "temperature": [...],
        ...     "vibration": [...],
        ...     "pressure": [...]
        ... })
        >>> health = pm.assess_health("pump-001", sensor_data)
        >>> print(f"Health Score: {health.health_score}")
    """

    # Default thresholds for manufacturing equipment
    DEFAULT_THRESHOLDS = {
        "temperature": {
            "warning": 80,
            "critical": 95,
            "unit": "°C",
        },
        "vibration": {
            "warning": 4.5,  # mm/s RMS
            "critical": 7.1,
            "unit": "mm/s",
        },
        "pressure": {
            "warning_low": 0.8,  # ratio to nominal
            "warning_high": 1.2,
            "critical_low": 0.6,
            "critical_high": 1.4,
            "unit": "ratio",
        },
        "current": {
            "warning": 1.15,  # ratio to nominal
            "critical": 1.30,
            "unit": "ratio",
        },
    }

    # Known degradation patterns
    DEGRADATION_PATTERNS = {
        "bearing_wear": {
            "indicators": ["vibration", "temperature"],
            "pattern": "increasing",
            "typical_days_to_failure": 30,
        },
        "seal_leak": {
            "indicators": ["pressure"],
            "pattern": "decreasing",
            "typical_days_to_failure": 14,
        },
        "motor_overload": {
            "indicators": ["current", "temperature"],
            "pattern": "increasing",
            "typical_days_to_failure": 7,
        },
    }

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.thresholds = {
            **self.DEFAULT_THRESHOLDS,
            **self.config.get("thresholds", {})
        }
        self._anomaly_detector = None

    def assess_health(
        self,
        equipment_id: str,
        sensor_data: pd.DataFrame,
        timestamp_column: str = "timestamp",
        sensor_columns: Optional[List[str]] = None
    ) -> EquipmentHealth:
        """
        Assess equipment health from sensor data.

        Args:
            equipment_id: Unique equipment identifier
            sensor_data: DataFrame with sensor readings
            timestamp_column: Column name for timestamps
            sensor_columns: Columns to analyze (auto-detect if not specified)

        Returns:
            EquipmentHealth assessment
        """
        if len(sensor_data) == 0:
            return self._empty_health(equipment_id)

        # Determine sensor columns
        if sensor_columns is None:
            sensor_columns = [
                col for col in sensor_data.columns
                if col != timestamp_column and sensor_data[col].dtype in ['float64', 'int64', 'float32', 'int32']
            ]

        # Analyze each sensor
        indicators = {}
        alerts = []
        health_scores = []

        for column in sensor_columns:
            if column not in sensor_data.columns:
                continue

            analysis = self._analyze_sensor(
                equipment_id,
                sensor_data,
                column,
                timestamp_column
            )

            indicators[column] = analysis["health_score"]
            health_scores.append(analysis["health_score"])
            alerts.extend(analysis["alerts"])

        # Calculate overall health
        if health_scores:
            overall_score = min(health_scores)  # Weakest link
        else:
            overall_score = 100.0

        # Determine status
        status = self._score_to_status(overall_score)

        # Estimate degradation rate and remaining life
        degradation = self._estimate_degradation(sensor_data, sensor_columns)

        return EquipmentHealth(
            equipment_id=equipment_id,
            status=status,
            health_score=round(overall_score, 1),
            degradation_rate=degradation["rate"],
            estimated_remaining_life_days=degradation["remaining_days"],
            key_indicators=indicators,
            alerts=alerts,
            last_updated=datetime.now().isoformat()
        )

    def _analyze_sensor(
        self,
        equipment_id: str,
        data: pd.DataFrame,
        column: str,
        timestamp_column: str
    ) -> Dict[str, Any]:
        """Analyze a single sensor."""
        values = data[column].dropna()

        if len(values) == 0:
            return {"health_score": 100.0, "alerts": []}

        # Get thresholds for this sensor type
        sensor_type = self._identify_sensor_type(column)
        thresholds = self.thresholds.get(sensor_type, {})

        # Calculate statistics
        current = float(values.iloc[-1])
        mean = float(values.mean())
        std = float(values.std())
        trend = self._calculate_trend(values)

        # Assess against thresholds
        alerts = []
        health_score = 100.0

        if "critical" in thresholds:
            if current >= thresholds["critical"]:
                health_score = min(health_score, 20.0)
                alerts.append(MaintenanceAlert(
                    equipment_id=equipment_id,
                    alert_type=f"{column}_critical",
                    severity=AlertSeverity.CRITICAL,
                    message=f"{column} at critical level: {current:.2f}",
                    recommended_action="Immediate inspection required",
                    estimated_days_to_failure=3,
                    confidence=0.9,
                    evidence={"current": current, "threshold": thresholds["critical"]}
                ))
            elif current >= thresholds.get("warning", thresholds["critical"] * 0.85):
                health_score = min(health_score, 60.0)
                alerts.append(MaintenanceAlert(
                    equipment_id=equipment_id,
                    alert_type=f"{column}_warning",
                    severity=AlertSeverity.WARNING,
                    message=f"{column} above warning level: {current:.2f}",
                    recommended_action="Schedule inspection",
                    estimated_days_to_failure=14,
                    confidence=0.7,
                    evidence={"current": current, "threshold": thresholds.get("warning")}
                ))

        # Check for pressure-style thresholds
        if "critical_high" in thresholds:
            if current >= thresholds["critical_high"] or current <= thresholds["critical_low"]:
                health_score = min(health_score, 20.0)
                alerts.append(MaintenanceAlert(
                    equipment_id=equipment_id,
                    alert_type=f"{column}_critical",
                    severity=AlertSeverity.CRITICAL,
                    message=f"{column} outside critical limits: {current:.2f}",
                    recommended_action="Immediate inspection required",
                    estimated_days_to_failure=3,
                    confidence=0.85,
                    evidence={"current": current}
                ))

        # Trend-based health reduction
        if trend > 0.05:  # Positive trend (degradation)
            health_score = min(health_score, health_score - trend * 100)
            if trend > 0.10:
                alerts.append(MaintenanceAlert(
                    equipment_id=equipment_id,
                    alert_type=f"{column}_trending",
                    severity=AlertSeverity.WARNING,
                    message=f"{column} showing degradation trend",
                    recommended_action="Monitor closely, plan maintenance",
                    estimated_days_to_failure=21,
                    confidence=0.6,
                    evidence={"trend_per_day": trend, "current": current}
                ))

        return {
            "health_score": max(0, health_score),
            "alerts": alerts,
            "current": current,
            "mean": mean,
            "std": std,
            "trend": trend,
        }

    def _identify_sensor_type(self, column_name: str) -> str:
        """Identify sensor type from column name."""
        column_lower = column_name.lower()

        if "temp" in column_lower:
            return "temperature"
        elif "vib" in column_lower:
            return "vibration"
        elif "press" in column_lower:
            return "pressure"
        elif "curr" in column_lower or "amp" in column_lower:
            return "current"

        return "generic"

    def _calculate_trend(self, values: pd.Series) -> float:
        """Calculate trend as slope normalized by mean."""
        if len(values) < 2:
            return 0.0

        x = np.arange(len(values))
        y = values.values

        # Simple linear regression
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x * x)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x + 1e-10)

        # Normalize by mean
        mean = np.mean(y)
        if abs(mean) > 1e-10:
            return slope / mean
        return 0.0

    def _estimate_degradation(
        self,
        data: pd.DataFrame,
        columns: List[str]
    ) -> Dict[str, Optional[float]]:
        """Estimate overall degradation rate and remaining life."""
        trends = []

        for column in columns:
            if column in data.columns:
                values = data[column].dropna()
                if len(values) > 1:
                    trends.append(self._calculate_trend(values))

        if not trends:
            return {"rate": 0.0, "remaining_days": None}

        # Use max trend (worst degradation)
        max_trend = max(abs(t) for t in trends)

        # Estimate remaining days
        # Assume 100% health at trend=0, failure at some threshold
        if max_trend > 0.001:
            # Rough estimate: days until 50% degradation
            remaining_days = 0.5 / max_trend
            remaining_days = min(remaining_days, 365)  # Cap at 1 year
        else:
            remaining_days = None

        return {
            "rate": round(max_trend, 4),
            "remaining_days": round(remaining_days, 0) if remaining_days else None
        }

    def _score_to_status(self, score: float) -> HealthStatus:
        """Convert health score to status."""
        if score >= 80:
            return HealthStatus.HEALTHY
        elif score >= 60:
            return HealthStatus.DEGRADED
        elif score >= 40:
            return HealthStatus.AT_RISK
        return HealthStatus.CRITICAL

    def _empty_health(self, equipment_id: str) -> EquipmentHealth:
        """Return empty health assessment."""
        return EquipmentHealth(
            equipment_id=equipment_id,
            status=HealthStatus.HEALTHY,
            health_score=100.0,
            degradation_rate=0.0,
            estimated_remaining_life_days=None,
            key_indicators={},
            alerts=[MaintenanceAlert(
                equipment_id=equipment_id,
                alert_type="no_data",
                severity=AlertSeverity.INFO,
                message="No sensor data available",
                recommended_action="Verify data connection",
                estimated_days_to_failure=None,
                confidence=0.0,
            )],
            last_updated=datetime.now().isoformat()
        )

    def detect_anomalies(
        self,
        data: pd.DataFrame,
        column: str,
        window_size: int = 20,
        threshold: float = 2.5
    ) -> pd.DataFrame:
        """
        Detect anomalies in sensor data.

        Uses rolling z-score method from AnomalyDetector.

        Args:
            data: Sensor data DataFrame
            column: Column to analyze
            window_size: Rolling window size
            threshold: Z-score threshold

        Returns:
            DataFrame with anomaly flags
        """
        from goodai.modules.anomaly_detection import AnomalyDetector

        if self._anomaly_detector is None:
            self._anomaly_detector = AnomalyDetector(
                window_size=window_size,
                threshold=threshold
            )

        return self._anomaly_detector.detect(data, column)

    def generate_maintenance_schedule(
        self,
        equipment_health: List[EquipmentHealth],
        planning_horizon_days: int = 30
    ) -> List[dict]:
        """
        Generate maintenance schedule based on health assessments.

        Args:
            equipment_health: List of EquipmentHealth assessments
            planning_horizon_days: Days to plan ahead

        Returns:
            List of scheduled maintenance items
        """
        schedule = []
        today = datetime.now()

        for health in equipment_health:
            # Priority based on status
            if health.status == HealthStatus.CRITICAL:
                priority = 1
                suggested_date = today
            elif health.status == HealthStatus.AT_RISK:
                priority = 2
                suggested_date = today + timedelta(days=3)
            elif health.status == HealthStatus.DEGRADED:
                priority = 3
                if health.estimated_remaining_life_days:
                    days_ahead = min(
                        health.estimated_remaining_life_days * 0.7,
                        planning_horizon_days
                    )
                else:
                    days_ahead = 14
                suggested_date = today + timedelta(days=days_ahead)
            else:
                continue  # Healthy equipment doesn't need scheduling

            # Determine work type
            if health.status == HealthStatus.CRITICAL:
                work_type = "Emergency Repair"
            elif health.status == HealthStatus.AT_RISK:
                work_type = "Corrective Maintenance"
            else:
                work_type = "Preventive Maintenance"

            schedule.append({
                "equipment_id": health.equipment_id,
                "priority": priority,
                "work_type": work_type,
                "suggested_date": suggested_date.strftime("%Y-%m-%d"),
                "health_score": health.health_score,
                "status": health.status.value,
                "estimated_duration_hours": 4 if priority <= 2 else 2,
                "alerts_count": len(health.alerts),
            })

        # Sort by priority
        schedule.sort(key=lambda x: (x["priority"], x["suggested_date"]))

        return schedule

    def summarize_fleet_health(
        self,
        equipment_health: List[EquipmentHealth]
    ) -> dict:
        """
        Summarize health across multiple equipment.

        Args:
            equipment_health: List of health assessments

        Returns:
            Fleet health summary
        """
        if not equipment_health:
            return {
                "total_equipment": 0,
                "average_health": 0,
                "status_breakdown": {},
                "total_alerts": 0,
                "critical_equipment": [],
            }

        scores = [h.health_score for h in equipment_health]
        status_counts = {}
        total_alerts = 0
        critical_equipment = []

        for health in equipment_health:
            status = health.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
            total_alerts += len(health.alerts)

            if health.status in [HealthStatus.CRITICAL, HealthStatus.AT_RISK]:
                critical_equipment.append({
                    "equipment_id": health.equipment_id,
                    "health_score": health.health_score,
                    "status": health.status.value,
                })

        return {
            "total_equipment": len(equipment_health),
            "average_health": round(sum(scores) / len(scores), 1),
            "min_health": round(min(scores), 1),
            "max_health": round(max(scores), 1),
            "status_breakdown": status_counts,
            "total_alerts": total_alerts,
            "critical_equipment": critical_equipment,
            "equipment_requiring_attention": len(critical_equipment),
        }
