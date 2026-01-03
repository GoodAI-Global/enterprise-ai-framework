"""
Tests for Anomaly Detection Module

Tests the rolling z-score anomaly detection implementation.
"""

import pytest
import pandas as pd
import numpy as np

from goodai.modules.anomaly_detection import AnomalyDetector, AnomalyResult


class TestAnomalyDetector:
    """Test suite for AnomalyDetector class."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        detector = AnomalyDetector()
        assert detector.window_size == 20
        assert detector.threshold == 2.5
        assert detector.min_periods == 10

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        detector = AnomalyDetector(window_size=30, threshold=3.0, min_periods=5)
        assert detector.window_size == 30
        assert detector.threshold == 3.0
        assert detector.min_periods == 5

    def test_init_invalid_window_size(self):
        """Test that small window size raises error."""
        with pytest.raises(ValueError, match="window_size must be at least 3"):
            AnomalyDetector(window_size=2)

    def test_init_invalid_threshold(self):
        """Test that non-positive threshold raises error."""
        with pytest.raises(ValueError, match="threshold must be positive"):
            AnomalyDetector(threshold=0)
        with pytest.raises(ValueError, match="threshold must be positive"):
            AnomalyDetector(threshold=-1)

    def test_detect_finds_obvious_anomaly(self):
        """Test that obvious anomalies are detected."""
        # Create data with small variance and extreme spike
        # Use very stable data with a massive spike
        np.random.seed(42)
        base_values = np.random.normal(100, 2, 30).tolist()  # Low variance
        base_values[20] = 500.0  # Extreme spike - clearly an anomaly

        df = pd.DataFrame({"value": base_values})

        detector = AnomalyDetector(window_size=10, threshold=2.5, min_periods=5)
        result = detector.detect(df, column="value")

        # Should detect the spike as anomaly
        assert result["is_anomaly"].any(), f"No anomalies detected, max z-score: {result['z_score'].abs().max()}"
        # Check that there's at least one high anomaly
        high_anomalies = result[result["anomaly_type"] == "high"]
        assert len(high_anomalies) >= 1

    def test_detect_finds_low_anomaly(self):
        """Test that low anomalies are detected."""
        # Create data with low variance and extreme drop
        np.random.seed(42)
        values = np.random.normal(100, 2, 30).tolist()  # Low variance
        values[20] = -50.0  # Extreme drop - clearly an anomaly
        df = pd.DataFrame({"value": values})

        detector = AnomalyDetector(window_size=10, threshold=2.5, min_periods=5)
        result = detector.detect(df, column="value")

        # Should detect the drop as anomaly
        assert result["is_anomaly"].any(), f"No anomalies detected, max z-score: {result['z_score'].abs().max()}"
        # Check that there's at least one low anomaly
        low_anomalies = result[result["anomaly_type"] == "low"]
        assert len(low_anomalies) >= 1

    def test_detect_no_anomalies_in_stable_data(self):
        """Test that stable data has no anomalies."""
        # Create stable data with small variance
        np.random.seed(42)
        values = np.random.normal(100, 1, 100)  # Very low variance
        df = pd.DataFrame({"value": values})

        detector = AnomalyDetector(window_size=20, threshold=3.0)
        result = detector.detect(df, column="value")

        # Should have very few or no anomalies
        anomaly_rate = result["is_anomaly"].sum() / len(result)
        assert anomaly_rate < 0.05  # Less than 5%

    def test_detect_missing_column_raises_error(self):
        """Test that missing column raises appropriate error."""
        df = pd.DataFrame({"value": [1, 2, 3]})
        detector = AnomalyDetector()

        with pytest.raises(ValueError, match="Column 'missing' not found"):
            detector.detect(df, column="missing")

    def test_detect_empty_dataframe_raises_error(self):
        """Test that empty DataFrame raises appropriate error."""
        df = pd.DataFrame({"value": []})
        detector = AnomalyDetector()

        with pytest.raises(ValueError, match="DataFrame is empty"):
            detector.detect(df, column="value")

    def test_detect_preserves_original_data(self):
        """Test that original DataFrame is not modified."""
        original_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        df = pd.DataFrame({"value": original_values.copy()})

        detector = AnomalyDetector(window_size=3)
        detector.detect(df, column="value")

        # Original should be unchanged
        assert list(df["value"]) == original_values

    def test_detect_output_columns(self):
        """Test that output has expected columns."""
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

        detector = AnomalyDetector(window_size=3)
        result = detector.detect(df, column="value")

        expected_columns = [
            "value", "rolling_mean", "rolling_std",
            "z_score", "is_anomaly", "anomaly_type"
        ]
        for col in expected_columns:
            assert col in result.columns

    def test_explain_anomaly_high(self):
        """Test explanation for high anomaly."""
        detector = AnomalyDetector()

        row = pd.Series({
            "value": 150.0,
            "z_score": 3.5,
            "rolling_mean": 100.0,
            "anomaly_type": "high"
        })

        explanation = detector.explain_anomaly(row)

        assert "150.00" in explanation
        assert "3.5" in explanation
        assert "above normal" in explanation

    def test_explain_anomaly_low(self):
        """Test explanation for low anomaly."""
        detector = AnomalyDetector()

        row = pd.Series({
            "value": 50.0,
            "z_score": -3.0,
            "rolling_mean": 100.0,
            "anomaly_type": "low"
        })

        explanation = detector.explain_anomaly(row)

        assert "50.00" in explanation
        assert "below normal" in explanation

    def test_explain_anomaly_none(self):
        """Test explanation when no anomaly."""
        detector = AnomalyDetector()

        row = pd.Series({
            "value": 100.0,
            "z_score": 0.5,
            "rolling_mean": 99.0,
            "anomaly_type": None
        })

        explanation = detector.explain_anomaly(row)

        assert "No anomaly detected" in explanation

    def test_get_anomaly_summary(self):
        """Test anomaly summary generation."""
        # Create data with low variance and extreme anomalies
        np.random.seed(42)
        values = np.random.normal(100, 2, 40).tolist()  # Low variance, longer series
        values[20] = 500.0  # Extreme high anomaly
        values[30] = -50.0  # Extreme low anomaly
        df = pd.DataFrame({"value": values})

        detector = AnomalyDetector(window_size=10, threshold=2.5, min_periods=5)
        result = detector.detect(df, column="value")
        summary = detector.get_anomaly_summary(result)

        assert "total_observations" in summary
        assert "anomalies_detected" in summary
        assert "anomaly_rate" in summary
        assert "high_anomalies" in summary
        assert "low_anomalies" in summary
        assert summary["total_observations"] == 40
        assert summary["high_anomalies"] >= 1, f"Expected high anomalies, got {summary}"
        assert summary["low_anomalies"] >= 1, f"Expected low anomalies, got {summary}"

    def test_detect_with_results(self):
        """Test detect_with_results returns AnomalyResult objects."""
        # Create data with low variance and extreme anomaly
        np.random.seed(42)
        values = np.random.normal(100, 2, 30).tolist()  # Low variance
        values[20] = 500.0  # Extreme spike
        df = pd.DataFrame({"value": values, "timestamp": range(30)})

        detector = AnomalyDetector(window_size=10, threshold=2.5, min_periods=5)
        results = detector.detect_with_results(df, column="value")

        assert len(results) > 0, "Expected at least one anomaly"
        assert isinstance(results[0], AnomalyResult)
        assert results[0].is_anomaly == True

    def test_deterministic_results(self):
        """Test that same input produces same output."""
        values = [100.0, 102.0, 98.0, 150.0, 99.0, 101.0, 97.0, 103.0, 100.0, 98.0]
        df = pd.DataFrame({"value": values})

        detector = AnomalyDetector(window_size=5, threshold=2.0)

        result1 = detector.detect(df.copy(), column="value")
        result2 = detector.detect(df.copy(), column="value")

        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)


class TestAnomalyDetectorEdgeCases:
    """Test edge cases for AnomalyDetector."""

    def test_single_value(self):
        """Test handling of single value DataFrame."""
        df = pd.DataFrame({"value": [100.0]})
        detector = AnomalyDetector(window_size=3, min_periods=1)

        result = detector.detect(df, column="value")

        assert len(result) == 1
        # Single value can't be an anomaly (no reference)
        assert result["is_anomaly"].iloc[0] == False

    def test_constant_values(self):
        """Test handling of constant values (zero std)."""
        df = pd.DataFrame({"value": [100.0] * 20})
        detector = AnomalyDetector(window_size=5)

        result = detector.detect(df, column="value")

        # Constant values should not produce anomalies
        assert result["is_anomaly"].sum() == 0

    def test_with_nan_values(self):
        """Test handling of NaN values."""
        values = [100.0, 102.0, np.nan, 98.0, 103.0, 99.0, 101.0, 100.0, 102.0, 98.0]
        df = pd.DataFrame({"value": values})

        detector = AnomalyDetector(window_size=5)
        result = detector.detect(df, column="value")

        # Should handle NaN gracefully
        assert len(result) == 10

    def test_large_dataset_performance(self):
        """Test performance with larger dataset."""
        np.random.seed(42)
        values = np.random.normal(100, 10, 10000)
        df = pd.DataFrame({"value": values})

        detector = AnomalyDetector(window_size=50, threshold=3.0)
        result = detector.detect(df, column="value")

        assert len(result) == 10000
        # With normal distribution, ~0.3% should be beyond 3 std
        anomaly_rate = result["is_anomaly"].sum() / len(result)
        assert 0.001 < anomaly_rate < 0.01
