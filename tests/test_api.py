"""
Tests for API Module

Tests for FastAPI application and middleware.
"""

import pytest

# Skip all tests if FastAPI not installed
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from goodai.api.app import create_app, get_app, reset_app
from goodai.api.dependencies import (
    get_anomaly_detector,
    get_assessment_engine,
    clear_dependency_cache,
)
from goodai.config import Settings, Environment, reset_settings


@pytest.fixture
def app():
    """Create test application."""
    reset_app()
    reset_settings()
    clear_dependency_cache()

    # Create with testing settings
    settings = Settings(environment=Environment.TESTING)
    return create_app(settings=settings)


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_health_endpoint(self, client):
        """Test /health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "uptime_seconds" in data

    def test_liveness_endpoint(self, client):
        """Test /health/live endpoint."""
        response = client.get("/health/live")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "alive"

    def test_readiness_endpoint(self, client):
        """Test /health/ready endpoint."""
        response = client.get("/health/ready")
        assert response.status_code == 200

        data = response.json()
        assert "ready" in data

    def test_startup_endpoint(self, client):
        """Test /health/startup endpoint."""
        response = client.get("/health/startup")
        assert response.status_code == 200

        data = response.json()
        assert "started" in data


class TestCorrelationMiddleware:
    """Test correlation ID middleware."""

    def test_generates_correlation_id(self, client):
        """Test correlation ID is generated."""
        response = client.get("/health")
        assert "X-Correlation-ID" in response.headers
        assert len(response.headers["X-Correlation-ID"]) == 36  # UUID

    def test_uses_provided_correlation_id(self, client):
        """Test provided correlation ID is used."""
        response = client.get(
            "/health",
            headers={"X-Correlation-ID": "custom-id-123"}
        )
        assert response.headers["X-Correlation-ID"] == "custom-id-123"


class TestAnomalyEndpoints:
    """Test anomaly detection endpoints."""

    def test_detect_anomalies(self, client):
        """Test anomaly detection endpoint."""
        # Normal values with one anomaly
        values = [100.0] * 25 + [500.0]  # Obvious anomaly at end

        response = client.post(
            "/api/v1/anomaly/detect",
            json={
                "values": values,
                "window_size": 10,
                "threshold": 2.5
            }
        )
        assert response.status_code == 200

        data = response.json()
        assert "summary" in data
        assert "anomalies" in data
        assert data["summary"]["total_observations"] == 26

    def test_detect_anomalies_with_timestamps(self, client):
        """Test anomaly detection with timestamps."""
        values = [100.0, 101.0, 99.0, 102.0, 200.0]
        timestamps = ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05"]

        response = client.post(
            "/api/v1/anomaly/detect",
            json={
                "values": values,
                "timestamps": timestamps,
                "window_size": 3,
                "threshold": 2.0
            }
        )
        assert response.status_code == 200


class TestAssessmentEndpoints:
    """Test assessment endpoints."""

    def test_assess_readiness(self, client):
        """Test readiness assessment endpoint."""
        response = client.post(
            "/api/v1/assessment/readiness",
            json={
                "company_data": {
                    "data_sources": ["erp", "scada"],
                    "data_quality_score": 0.7,
                    "has_data_warehouse": True
                },
                "industry": "manufacturing"
            }
        )
        assert response.status_code == 200

        data = response.json()
        assert "overall_score" in data
        assert "dimensions" in data
        assert "recommendations" in data

    def test_identify_bottlenecks(self, client):
        """Test bottleneck identification endpoint."""
        response = client.post(
            "/api/v1/assessment/bottlenecks",
            json={
                "availability": 0.85,
                "performance": 0.80,
                "quality": 0.95
            }
        )
        assert response.status_code == 200

        data = response.json()
        assert "bottlenecks" in data
        assert "count" in data


class TestROIEndpoints:
    """Test ROI calculation endpoints."""

    def test_calculate_roi(self, client):
        """Test ROI calculation endpoint."""
        response = client.post(
            "/api/v1/roi/calculate",
            json={
                "project_name": "Predictive Maintenance",
                "use_case": "predictive_maintenance",
                "investment": {
                    "software": 50000,
                    "implementation": 30000
                },
                "baseline_metrics": {
                    "downtime_hours_annual": 500,
                    "downtime_cost_per_hour": 5000
                },
                "readiness_score": 6.0
            }
        )
        assert response.status_code == 200

        data = response.json()
        assert "roi_percent" in data
        assert "payback_months" in data
        assert "npv" in data

    def test_quick_estimate(self, client):
        """Test quick estimate endpoint."""
        response = client.post(
            "/api/v1/roi/quick-estimate",
            json={
                "use_case": "predictive_maintenance",
                "investment": 100000,
                "annual_baseline_cost": 500000
            }
        )
        assert response.status_code == 200

        data = response.json()
        assert "estimated_roi_percent" in data
        assert "estimated_payback_months" in data


class TestMetricsEndpoints:
    """Test metrics endpoints."""

    def test_calculate_oee(self, client):
        """Test OEE calculation endpoint."""
        response = client.post(
            "/api/v1/metrics/oee",
            json={
                "availability": 0.90,
                "performance": 0.85,
                "quality": 0.95
            }
        )
        assert response.status_code == 200

        data = response.json()
        assert "oee" in data
        assert "availability" in data
        assert "performance" in data
        assert "quality" in data


class TestDependencies:
    """Test dependency injection."""

    def test_get_anomaly_detector(self):
        """Test anomaly detector dependency."""
        clear_dependency_cache()
        detector = get_anomaly_detector()
        assert detector is not None
        assert detector.window_size == 20  # Default from config

    def test_get_anomaly_detector_cached(self):
        """Test anomaly detector is cached."""
        clear_dependency_cache()
        d1 = get_anomaly_detector()
        d2 = get_anomaly_detector()
        assert d1 is d2

    def test_get_assessment_engine(self):
        """Test assessment engine dependency."""
        clear_dependency_cache()
        engine = get_assessment_engine()
        assert engine is not None
        assert engine.industry == "manufacturing"


class TestErrorHandling:
    """Test error handling."""

    def test_invalid_json(self, client):
        """Test invalid JSON handling."""
        response = client.post(
            "/api/v1/anomaly/detect",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422  # Validation error

    def test_missing_required_field(self, client):
        """Test missing required field."""
        response = client.post(
            "/api/v1/roi/calculate",
            json={
                "project_name": "Test",
                # Missing required fields
            }
        )
        assert response.status_code == 422


class TestAppFactory:
    """Test application factory."""

    def test_create_app(self):
        """Test create_app returns FastAPI instance."""
        from fastapi import FastAPI

        reset_app()
        app = create_app(title="Test API", version="2.0.0")

        assert isinstance(app, FastAPI)
        assert app.title == "Test API"
        assert app.version == "2.0.0"

    def test_get_app_singleton(self):
        """Test get_app returns singleton."""
        reset_app()
        app1 = get_app()
        app2 = get_app()
        assert app1 is app2
