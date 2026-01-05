"""API layer for HTTP/REST access to Good AI functionality."""

from goodai.api.app import create_app, get_app
from goodai.api.middleware import CorrelationMiddleware, RequestLoggingMiddleware
from goodai.api.dependencies import get_anomaly_detector, get_assessment_engine

__all__ = [
    "create_app",
    "get_app",
    "CorrelationMiddleware",
    "RequestLoggingMiddleware",
    "get_anomaly_detector",
    "get_assessment_engine",
]
