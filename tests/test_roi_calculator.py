"""
Tests for ROI Calculator Module.

Ensures deterministic calculations for ROI projections.
"""

import pytest
from goodai.core.roi_calculator import ROICalculator, ConfidenceLevel


class TestROICalculator:
    """Test ROI Calculator functionality."""

    def test_calculate_basic(self):
        """Test basic ROI calculation."""
        calculator = ROICalculator()

        result = calculator.calculate(
            project_name="Test Project",
            use_case="predictive_maintenance",
            investment={
                "software": 50000,
                "implementation": 30000,
                "training": 10000,
            },
            baseline_metrics={
                "downtime_hours_annual": 500,
                "downtime_cost_per_hour": 5000,
            },
        )

        assert result.project_name == "Test Project"
        assert result.investment_total > 0  # Includes contingency
        assert result.annual_benefit > 0
        assert result.roi_percent > 0
        assert result.payback_months > 0

    def test_deterministic_results(self):
        """Test that ROI calculations are deterministic - same inputs produce same outputs."""
        calculator = ROICalculator(
            discount_rate=0.10,
            project_years=3,
            conservative_factor=0.7,
        )

        inputs = {
            "project_name": "Predictive Maintenance Pilot",
            "use_case": "predictive_maintenance",
            "investment": {
                "software": 50000,
                "implementation": 30000,
                "training": 10000,
            },
            "baseline_metrics": {
                "downtime_hours_annual": 500,
                "downtime_cost_per_hour": 5000,
            },
            "readiness_score": 6.0,
        }

        # Run calculation multiple times
        result1 = calculator.calculate(**inputs)
        result2 = calculator.calculate(**inputs)
        result3 = calculator.calculate(**inputs)

        # All results should be identical
        assert result1.investment_total == result2.investment_total == result3.investment_total
        assert result1.annual_benefit == result2.annual_benefit == result3.annual_benefit
        assert result1.roi_percent == result2.roi_percent == result3.roi_percent
        assert result1.payback_months == result2.payback_months == result3.payback_months
        assert result1.npv == result2.npv == result3.npv
        assert result1.confidence == result2.confidence == result3.confidence

        # Verify consistency with specific values
        assert result1.investment_total > 0
        assert result1.annual_benefit > 0

    def test_quick_estimate(self):
        """Test quick ROI estimation."""
        calculator = ROICalculator()

        estimate = calculator.quick_estimate(
            use_case="predictive_maintenance",
            annual_baseline_cost=1000000,
            investment=100000,
        )

        assert "estimated_roi_percent" in estimate
        assert "estimated_payback_months" in estimate
        assert "confidence" in estimate

    def test_quick_estimate_deterministic(self):
        """Test that quick estimates are deterministic."""
        calculator = ROICalculator()

        inputs = {
            "use_case": "quality_inspection",
            "annual_baseline_cost": 500000,
            "investment": 75000,
        }

        result1 = calculator.quick_estimate(**inputs)
        result2 = calculator.quick_estimate(**inputs)

        assert result1["estimated_roi_percent"] == result2["estimated_roi_percent"]
        assert result1["estimated_payback_months"] == result2["estimated_payback_months"]
        assert result1["confidence"] == result2["confidence"]

    def test_different_use_cases(self):
        """Test ROI calculation for different use cases."""
        calculator = ROICalculator()

        use_cases = [
            "predictive_maintenance",
            "quality_inspection",
            "demand_forecasting",
        ]

        for use_case in use_cases:
            estimate = calculator.quick_estimate(
                use_case=use_case,
                annual_baseline_cost=1000000,
                investment=100000,
            )
            assert estimate["estimated_roi_percent"] > 0
            assert estimate["confidence"] in ["low", "medium", "high"]

    def test_conservative_factor_impact(self):
        """Test that conservative factor affects results appropriately."""
        aggressive = ROICalculator(conservative_factor=1.0)
        conservative = ROICalculator(conservative_factor=0.5)

        inputs = {
            "project_name": "Test",
            "use_case": "predictive_maintenance",
            "investment": {"total": 100000},
            "baseline_metrics": {
                "downtime_hours_annual": 500,
                "downtime_cost_per_hour": 5000,
            },
        }

        aggressive_result = aggressive.calculate(**inputs)
        conservative_result = conservative.calculate(**inputs)

        # Conservative factor should reduce benefits
        assert aggressive_result.annual_benefit >= conservative_result.annual_benefit

    def test_result_to_dict(self):
        """Test ROI result serialization."""
        calculator = ROICalculator()

        result = calculator.calculate(
            project_name="Serialization Test",
            use_case="predictive_maintenance",
            investment={"software": 50000},
            baseline_metrics={
                "downtime_hours_annual": 100,
                "downtime_cost_per_hour": 1000,
            },
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["project_name"] == "Serialization Test"
        assert "investment_total" in result_dict
        assert "roi_percent" in result_dict
        assert "confidence" in result_dict
        assert isinstance(result_dict["confidence"], str)
