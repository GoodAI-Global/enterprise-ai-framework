"""
API Dependencies

Dependency injection for FastAPI routes.
Provides configured instances of core services.

Good AI Philosophy: Leverage, not lore - dependency injection enables testing.
"""

from functools import lru_cache
from typing import Optional

from goodai.config import get_settings
from goodai.modules.anomaly_detection import AnomalyDetector
from goodai.core.assessment_engine import AssessmentEngine
from goodai.core.roi_calculator import ROICalculator
from goodai.utils.metrics import OEECalculator


@lru_cache()
def get_anomaly_detector(
    window_size: Optional[int] = None,
    threshold: Optional[float] = None
) -> AnomalyDetector:
    """
    Get configured anomaly detector.

    Uses settings from configuration if not overridden.

    Args:
        window_size: Override window size
        threshold: Override threshold

    Returns:
        Configured AnomalyDetector instance
    """
    settings = get_settings()
    ai_config = settings.ai

    return AnomalyDetector(
        window_size=window_size or ai_config.anomaly_detection_window,
        threshold=threshold or ai_config.anomaly_detection_threshold
    )


@lru_cache()
def get_assessment_engine(industry: str = "manufacturing") -> AssessmentEngine:
    """
    Get configured assessment engine.

    Args:
        industry: Industry for benchmarks

    Returns:
        Configured AssessmentEngine instance
    """
    return AssessmentEngine(industry=industry)


@lru_cache()
def get_roi_calculator(
    discount_rate: float = 0.10,
    project_years: int = 3,
    conservative_factor: float = 0.7
) -> ROICalculator:
    """
    Get configured ROI calculator.

    Args:
        discount_rate: Annual discount rate
        project_years: Project horizon
        conservative_factor: Conservative adjustment factor

    Returns:
        Configured ROICalculator instance
    """
    return ROICalculator(
        discount_rate=discount_rate,
        project_years=project_years,
        conservative_factor=conservative_factor
    )


@lru_cache()
def get_oee_calculator() -> OEECalculator:
    """
    Get configured OEE calculator.

    Returns:
        Configured OEECalculator instance
    """
    return OEECalculator()


def clear_dependency_cache() -> None:
    """Clear all cached dependencies (for testing)."""
    get_anomaly_detector.cache_clear()
    get_assessment_engine.cache_clear()
    get_roi_calculator.cache_clear()
    get_oee_calculator.cache_clear()
