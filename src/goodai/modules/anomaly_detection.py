"""
Anomaly Detection Module

Simple, deterministic statistical anomaly detection using rolling z-score.
No complex ML - just robust statistics that work in production.

Good AI Philosophy: Leverage, not lore.
"""

from dataclasses import dataclass
from typing import List, Optional, Literal
import pandas as pd
import numpy as np


@dataclass
class AnomalyResult:
    """Result of anomaly detection on a single data point."""

    timestamp: Optional[str]
    value: float
    rolling_mean: float
    rolling_std: float
    z_score: float
    is_anomaly: bool
    anomaly_type: Optional[Literal["high", "low"]]
    explanation: str


class AnomalyDetector:
    """
    Simple statistical anomaly detection using rolling z-score.

    Uses rolling window statistics to detect values that deviate significantly
    from recent trends. This approach is:
    - Deterministic: Same data always produces same results
    - Explainable: Every anomaly has a human-readable explanation
    - Fast: O(n) complexity, works on large datasets
    - Robust: No training required, adapts to concept drift naturally

    Args:
        window_size: Number of observations for rolling statistics (default: 20)
        threshold: Z-score threshold for anomaly detection (default: 2.5)
        min_periods: Minimum observations before detection starts (default: window_size // 2)

    Example:
        >>> detector = AnomalyDetector(window_size=20, threshold=2.5)
        >>> df = pd.DataFrame({"value": [100, 101, 99, 102, 150, 98, 101]})
        >>> result = detector.detect(df, column="value")
        >>> print(result[result["is_anomaly"]])  # Shows anomalies
    """

    def __init__(
        self,
        window_size: int = 20,
        threshold: float = 2.5,
        min_periods: Optional[int] = None
    ):
        if window_size < 3:
            raise ValueError("window_size must be at least 3 for meaningful statistics")
        if threshold <= 0:
            raise ValueError("threshold must be positive")

        self.window_size = window_size
        self.threshold = threshold
        self.min_periods = min_periods if min_periods is not None else max(1, window_size // 2)

    def detect(
        self,
        data: pd.DataFrame,
        column: str,
        timestamp_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Detect anomalies using rolling z-score.

        Args:
            data: DataFrame containing the data to analyze
            column: Name of the column to analyze for anomalies
            timestamp_column: Optional column name for timestamps

        Returns:
            DataFrame with columns:
            - value: original value
            - rolling_mean: window mean
            - rolling_std: window standard deviation
            - z_score: standardized score
            - is_anomaly: boolean flag
            - anomaly_type: 'high', 'low', or None

        Raises:
            ValueError: If column not found or data is empty
        """
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame. Available: {list(data.columns)}")

        if len(data) == 0:
            raise ValueError("DataFrame is empty")

        df = data.copy()

        # Preserve timestamp if available
        if timestamp_column and timestamp_column in df.columns:
            df["_timestamp"] = df[timestamp_column]
        elif "timestamp" in df.columns:
            df["_timestamp"] = df["timestamp"]
        else:
            df["_timestamp"] = range(len(df))

        # Calculate rolling statistics
        df["value"] = df[column].astype(float)
        df["rolling_mean"] = df["value"].rolling(
            window=self.window_size,
            min_periods=self.min_periods
        ).mean()
        df["rolling_std"] = df["value"].rolling(
            window=self.window_size,
            min_periods=self.min_periods
        ).std()

        # Handle zero standard deviation (constant values)
        df["rolling_std"] = df["rolling_std"].replace(0, np.nan)

        # Calculate z-score
        df["z_score"] = (df["value"] - df["rolling_mean"]) / df["rolling_std"]

        # Fill NaN z-scores with 0 (no anomaly when we can't compute)
        df["z_score"] = df["z_score"].fillna(0)

        # Detect anomalies
        df["is_anomaly"] = df["z_score"].abs() > self.threshold

        # Classify anomaly type
        df["anomaly_type"] = df.apply(
            lambda row: self._classify_anomaly(row["z_score"]),
            axis=1
        )

        # Rename timestamp column back
        if "_timestamp" in df.columns:
            df["timestamp"] = df["_timestamp"]
            df = df.drop(columns=["_timestamp"])

        return df

    def _classify_anomaly(self, z_score: float) -> Optional[str]:
        """Classify anomaly as high, low, or None."""
        if z_score > self.threshold:
            return "high"
        elif z_score < -self.threshold:
            return "low"
        return None

    def explain_anomaly(self, row: pd.Series) -> str:
        """
        Generate human-readable explanation for a data point.

        Args:
            row: Series containing at least 'value', 'z_score',
                 'rolling_mean', and 'anomaly_type'

        Returns:
            Human-readable explanation string
        """
        value = row.get("value", row.get("production_rate", 0))
        z_score = row.get("z_score", 0)
        rolling_mean = row.get("rolling_mean", 0)
        anomaly_type = row.get("anomaly_type")

        if anomaly_type == "high":
            return (
                f"Value {value:.2f} is {z_score:.1f} standard deviations "
                f"above normal ({rolling_mean:.2f})"
            )
        elif anomaly_type == "low":
            return (
                f"Value {value:.2f} is {abs(z_score):.1f} standard deviations "
                f"below normal ({rolling_mean:.2f})"
            )
        return "No anomaly detected"

    def get_anomaly_summary(self, result_df: pd.DataFrame) -> dict:
        """
        Generate summary statistics for detected anomalies.

        Args:
            result_df: DataFrame returned by detect()

        Returns:
            Dictionary with anomaly statistics
        """
        anomalies = result_df[result_df["is_anomaly"]]

        return {
            "total_observations": len(result_df),
            "anomalies_detected": len(anomalies),
            "anomaly_rate": len(anomalies) / len(result_df) if len(result_df) > 0 else 0,
            "high_anomalies": len(anomalies[anomalies["anomaly_type"] == "high"]),
            "low_anomalies": len(anomalies[anomalies["anomaly_type"] == "low"]),
            "max_z_score": float(result_df["z_score"].abs().max()) if len(result_df) > 0 else 0,
            "threshold_used": self.threshold,
            "window_size_used": self.window_size,
        }

    def detect_with_results(
        self,
        data: pd.DataFrame,
        column: str,
        timestamp_column: Optional[str] = None
    ) -> List[AnomalyResult]:
        """
        Detect anomalies and return structured results.

        Args:
            data: DataFrame containing the data to analyze
            column: Name of the column to analyze
            timestamp_column: Optional column name for timestamps

        Returns:
            List of AnomalyResult objects for each anomaly detected
        """
        df = self.detect(data, column, timestamp_column)
        results = []

        for _, row in df[df["is_anomaly"]].iterrows():
            results.append(AnomalyResult(
                timestamp=str(row.get("timestamp", "")),
                value=float(row["value"]),
                rolling_mean=float(row["rolling_mean"]),
                rolling_std=float(row["rolling_std"]),
                z_score=float(row["z_score"]),
                is_anomaly=True,
                anomaly_type=row["anomaly_type"],
                explanation=self.explain_anomaly(row)
            ))

        return results
