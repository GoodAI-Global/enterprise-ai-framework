"""
Tests for Manufacturing OEE Demo

Integration tests for the manufacturing demo.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import pandas as pd

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from goodai.modules.anomaly_detection import AnomalyDetector


class TestManufacturingDemo:
    """Test suite for Manufacturing OEE Demo."""

    @pytest.fixture
    def sample_data_path(self):
        """Get path to sample data."""
        return Path(__file__).parent.parent / "examples" / "manufacturing_oee" / "sample_data.csv"

    @pytest.fixture
    def sample_data(self, sample_data_path):
        """Load sample data."""
        return pd.read_csv(sample_data_path)

    def test_sample_data_exists(self, sample_data_path):
        """Test that sample data file exists."""
        assert sample_data_path.exists()

    def test_sample_data_has_required_columns(self, sample_data):
        """Test that sample data has required columns."""
        required_columns = [
            "timestamp",
            "production_rate",
            "planned_rate",
            "good_units",
            "total_units",
            "uptime_minutes",
            "planned_minutes"
        ]
        for col in required_columns:
            assert col in sample_data.columns

    def test_sample_data_not_empty(self, sample_data):
        """Test that sample data is not empty."""
        assert len(sample_data) > 0
        assert len(sample_data) >= 20  # Need enough for anomaly detection

    def test_anomaly_detection_on_sample_data(self, sample_data):
        """Test anomaly detection on sample data."""
        detector = AnomalyDetector(window_size=10, threshold=2.0)
        result = detector.detect(sample_data, column="production_rate")

        # Should detect some anomalies (we have intentional spikes/drops)
        assert result["is_anomaly"].any()

    def test_oee_calculation(self, sample_data):
        """Test OEE calculation from sample data."""
        # Calculate OEE components
        total_uptime = sample_data["uptime_minutes"].sum()
        total_planned = sample_data["planned_minutes"].sum()
        availability = total_uptime / total_planned

        total_actual = sample_data["production_rate"].sum()
        total_theoretical = sample_data["planned_rate"].sum()
        performance = total_actual / total_theoretical

        total_good = sample_data["good_units"].sum()
        total_produced = sample_data["total_units"].sum()
        quality = total_good / total_produced

        oee = availability * performance * quality

        # Validate ranges
        assert 0 <= availability <= 1
        assert 0 <= performance <= 1.5  # Can exceed if over-producing
        assert 0 <= quality <= 1
        assert 0 <= oee <= 1.5

    def test_demo_script_runs(self):
        """Test that demo script runs without error."""
        demo_path = Path(__file__).parent.parent / "examples" / "manufacturing_oee" / "run_demo.py"

        result = subprocess.run(
            [sys.executable, str(demo_path)],
            capture_output=True,
            text=True,
            timeout=30
        )

        # Should exit successfully
        assert result.returncode == 0, f"Demo failed: {result.stderr}"

    def test_demo_output_is_valid_json(self):
        """Test that demo outputs valid JSON."""
        demo_path = Path(__file__).parent.parent / "examples" / "manufacturing_oee" / "run_demo.py"

        result = subprocess.run(
            [sys.executable, str(demo_path)],
            capture_output=True,
            text=True,
            timeout=30
        )

        # Should output valid JSON
        output = json.loads(result.stdout)
        assert isinstance(output, dict)

    def test_demo_output_has_required_fields(self):
        """Test that demo output has required fields."""
        demo_path = Path(__file__).parent.parent / "examples" / "manufacturing_oee" / "run_demo.py"

        result = subprocess.run(
            [sys.executable, str(demo_path)],
            capture_output=True,
            text=True,
            timeout=30
        )

        output = json.loads(result.stdout)

        # Check required fields
        assert "oee_metrics" in output
        assert "anomaly_detection" in output
        assert "recommendations" in output

        # Check OEE metrics
        oee_metrics = output["oee_metrics"]
        assert "availability" in oee_metrics
        assert "performance" in oee_metrics
        assert "quality" in oee_metrics
        assert "overall_oee" in oee_metrics

        # Check anomaly detection
        anomaly_data = output["anomaly_detection"]
        assert "total_anomalies" in anomaly_data
        assert anomaly_data["total_anomalies"] >= 0

    def test_demo_detects_expected_anomalies(self):
        """Test that demo detects expected anomalies in sample data."""
        demo_path = Path(__file__).parent.parent / "examples" / "manufacturing_oee" / "run_demo.py"

        result = subprocess.run(
            [sys.executable, str(demo_path)],
            capture_output=True,
            text=True,
            timeout=30
        )

        output = json.loads(result.stdout)

        # Sample data has intentional anomalies (e.g., production_rate of 45, 25, 150, 35)
        # Should detect at least some of these
        assert output["anomaly_detection"]["total_anomalies"] >= 1

    def test_demo_generates_recommendations(self):
        """Test that demo generates recommendations."""
        demo_path = Path(__file__).parent.parent / "examples" / "manufacturing_oee" / "run_demo.py"

        result = subprocess.run(
            [sys.executable, str(demo_path)],
            capture_output=True,
            text=True,
            timeout=30
        )

        output = json.loads(result.stdout)

        # Should have recommendations
        assert "recommendations" in output
        assert len(output["recommendations"]) > 0


class TestVisionConnectorHeadless:
    """Test that VisionConnector works in headless mode."""

    def test_vision_connector_imports(self):
        """Test that VisionConnector can be imported."""
        from goodai.connectors.vision_connector import VisionConnector
        assert VisionConnector is not None

    def test_vision_connector_init_headless(self):
        """Test that VisionConnector initializes in headless mode."""
        from goodai.connectors.vision_connector import VisionConnector

        connector = VisionConnector()

        # Should report headless mode
        status = connector.get_status()
        assert status["headless_mode"] == True

    def test_vision_connector_calibrate_without_display(self):
        """Test calibrate raises appropriate error without display and config."""
        import os
        from goodai.connectors.vision_connector import VisionConnector

        # Ensure no display
        old_display = os.environ.get("DISPLAY")
        if "DISPLAY" in os.environ:
            del os.environ["DISPLAY"]

        try:
            connector = VisionConnector()

            with pytest.raises(ValueError) as exc_info:
                connector.calibrate("nonexistent.png")

            # Should provide helpful error message
            assert "No display available" in str(exc_info.value)
            assert "roi_config" in str(exc_info.value)
        finally:
            # Restore display
            if old_display:
                os.environ["DISPLAY"] = old_display

    def test_vision_connector_calibrate_with_config(self):
        """Test calibrate works when roi_config is provided."""
        from goodai.connectors.vision_connector import VisionConnector

        connector = VisionConnector()

        roi_config = {
            "temperature": {"x": 100, "y": 200, "w": 80, "h": 30, "type": "number"},
            "pressure": {"x": 100, "y": 250, "w": 80, "h": 30, "type": "number"},
        }

        # Should return the config when provided
        result = connector.calibrate("any_image.png", roi_config=roi_config)
        assert result == roi_config


class TestDataIntegrity:
    """Test data integrity and determinism."""

    def test_anomaly_detection_deterministic(self):
        """Test that anomaly detection produces deterministic results."""
        data = pd.DataFrame({
            "value": [100, 101, 99, 102, 150, 98, 101, 100, 99, 102,
                     98, 101, 100, 45, 99, 102, 98, 101, 100, 99]
        })

        detector = AnomalyDetector(window_size=5, threshold=2.0)

        result1 = detector.detect(data.copy(), column="value")
        result2 = detector.detect(data.copy(), column="value")

        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)

    def test_oee_calculation_deterministic(self):
        """Test that OEE calculations are deterministic."""
        from goodai.utils.metrics import OEECalculator

        data = pd.DataFrame({
            "uptime_minutes": [55, 58, 30, 57],
            "planned_minutes": [60, 60, 60, 60],
            "actual_output": [95, 98, 45, 92],
            "theoretical_output": [100, 100, 100, 100],
            "good_units": [94, 97, 44, 90],
            "total_units": [95, 98, 45, 92],
        })

        calc = OEECalculator()

        result1 = calc.calculate(data.copy())
        result2 = calc.calculate(data.copy())

        assert result1.oee == result2.oee
        assert result1.availability == result2.availability
        assert result1.performance == result2.performance
        assert result1.quality == result2.quality
