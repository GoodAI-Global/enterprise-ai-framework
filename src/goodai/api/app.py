"""
FastAPI Application Factory

Enterprise-grade API application with:
- Health check endpoints
- Correlation ID propagation
- Request/response logging
- Error handling

Good AI Philosophy: Non-invasive by default - API layer is optional.
"""

from typing import Optional

from goodai.config import get_settings, Settings
from goodai.monitoring import get_logger, HealthCheck, health_check

# Check if FastAPI is available
try:
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None

logger = get_logger(__name__)

# Global app instance
_app: Optional["FastAPI"] = None


def create_app(
    title: str = "Good AI API",
    version: str = "1.0.0",
    settings: Optional[Settings] = None
) -> "FastAPI":
    """
    Create FastAPI application.

    Args:
        title: API title
        version: API version
        settings: Application settings (uses get_settings() if not provided)

    Returns:
        Configured FastAPI application

    Example:
        >>> app = create_app(title="Manufacturing API")
        >>> # Run with: uvicorn module:app --reload
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI is required for the API layer. "
            "Install with: pip install fastapi uvicorn"
        )

    settings = settings or get_settings()

    app = FastAPI(
        title=title,
        version=version,
        description="Enterprise AI Implementation Framework API",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
    )

    # Configure CORS
    if settings.security.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.security.allowed_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Add custom middleware (order matters - first added is last executed)
    from goodai.api.middleware import (
        CorrelationMiddleware,
        RequestLoggingMiddleware,
        AuditMiddleware,
        TenantMiddleware,
    )
    # Execution order: Correlation -> Tenant -> Audit -> Logging -> Handler
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(AuditMiddleware)
    app.add_middleware(TenantMiddleware)
    app.add_middleware(CorrelationMiddleware)

    # Register routes
    _register_health_routes(app, settings)
    _register_core_routes(app)
    _register_mlops_routes(app)

    # Error handlers
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(
            "Unhandled exception",
            exc_info=exc,
            path=request.url.path,
            method=request.method
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc) if settings.debug else "An error occurred"
            }
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "status_code": exc.status_code
            }
        )

    logger.info(
        "API application created",
        title=title,
        version=version,
        debug=settings.debug
    )

    return app


def _register_health_routes(app: "FastAPI", settings: Settings):
    """Register health check routes."""
    from fastapi import APIRouter

    router = APIRouter(prefix="/health", tags=["Health"])
    hc = health_check(version=settings.version)

    # Register comprehensive health checks for all modules
    _register_module_health_checks(hc)

    @router.get("")
    async def health():
        """Full health check with component details."""
        report = hc.check()
        return report.to_dict()

    @router.get("/live")
    async def liveness():
        """Liveness probe for Kubernetes."""
        return hc.liveness()

    @router.get("/ready")
    async def readiness():
        """Readiness probe for Kubernetes."""
        result = hc.readiness()
        if not result["ready"]:
            raise HTTPException(status_code=503, detail="Not ready")
        return result

    @router.get("/startup")
    async def startup():
        """Startup probe for Kubernetes."""
        result = hc.startup()
        if not result["started"]:
            raise HTTPException(status_code=503, detail="Not started")
        return result

    app.include_router(router)


def _register_core_routes(app: "FastAPI"):
    """Register core functionality routes."""
    from fastapi import APIRouter, Body
    from typing import Dict, List, Any
    from pydantic import BaseModel

    # Models for request/response
    class AnomalyRequest(BaseModel):
        values: List[float]
        timestamps: Optional[List[str]] = None
        window_size: int = 20
        threshold: float = 2.5

    class AssessmentRequest(BaseModel):
        company_data: Dict[str, Any]
        industry: str = "manufacturing"

    class ROIRequest(BaseModel):
        project_name: str
        use_case: str
        investment: Dict[str, float]
        baseline_metrics: Dict[str, float]
        readiness_score: float = 5.0

    # Anomaly detection routes
    anomaly_router = APIRouter(prefix="/api/v1/anomaly", tags=["Anomaly Detection"])

    @anomaly_router.post("/detect")
    async def detect_anomalies(request: AnomalyRequest):
        """Detect anomalies in time series data."""
        import pandas as pd
        from goodai.modules.anomaly_detection import AnomalyDetector

        detector = AnomalyDetector(
            window_size=request.window_size,
            threshold=request.threshold
        )

        df = pd.DataFrame({"value": request.values})
        if request.timestamps:
            df["timestamp"] = request.timestamps

        result = detector.detect(df, column="value")
        summary = detector.get_anomaly_summary(result)

        # Get anomaly details
        anomalies = result[result["is_anomaly"]].to_dict("records")

        return {
            "summary": summary,
            "anomalies": anomalies
        }

    # Assessment routes
    assessment_router = APIRouter(prefix="/api/v1/assessment", tags=["Assessment"])

    @assessment_router.post("/readiness")
    async def assess_readiness(request: AssessmentRequest):
        """Assess AI readiness."""
        from goodai.core.assessment_engine import AssessmentEngine

        engine = AssessmentEngine(industry=request.industry)
        result = engine.assess_readiness(request.company_data)

        # Convert to dict, handling nested dataclasses
        result_dict = {
            "overall_score": result.overall_score,
            "overall_level": result.overall_level.value,
            "dimensions": {
                name: {
                    "name": dim.name,
                    "score": dim.score,
                    "level": dim.level.value,
                    "findings": dim.findings,
                    "recommendations": dim.recommendations,
                }
                for name, dim in result.dimensions.items()
            },
            "recommendations": result.recommendations,
            "red_flags": result.red_flags,
            "quick_wins": result.quick_wins,
            "metadata": result.metadata,
        }
        return result_dict

    @assessment_router.post("/bottlenecks")
    async def identify_bottlenecks(process_data: Dict[str, Any] = Body(...)):
        """Identify process bottlenecks."""
        from goodai.core.assessment_engine import AssessmentEngine

        engine = AssessmentEngine()
        bottlenecks = engine.identify_bottlenecks(process_data)

        return {
            "bottlenecks": [b.to_dict() for b in bottlenecks],
            "count": len(bottlenecks)
        }

    # ROI routes
    roi_router = APIRouter(prefix="/api/v1/roi", tags=["ROI"])

    @roi_router.post("/calculate")
    async def calculate_roi(request: ROIRequest):
        """Calculate ROI for AI project."""
        from goodai.core.roi_calculator import ROICalculator

        calculator = ROICalculator()
        result = calculator.calculate(
            project_name=request.project_name,
            use_case=request.use_case,
            investment=request.investment,
            baseline_metrics=request.baseline_metrics,
            readiness_score=request.readiness_score
        )

        return result.to_dict()

    @roi_router.post("/quick-estimate")
    async def quick_estimate(
        use_case: str = Body(...),
        investment: float = Body(...),
        annual_baseline_cost: float = Body(...)
    ):
        """Quick ROI estimate."""
        from goodai.core.roi_calculator import ROICalculator

        calculator = ROICalculator()
        return calculator.quick_estimate(use_case, investment, annual_baseline_cost)

    # Metrics routes
    metrics_router = APIRouter(prefix="/api/v1/metrics", tags=["Metrics"])

    @metrics_router.post("/oee")
    async def calculate_oee(
        availability: float = Body(...),
        performance: float = Body(...),
        quality: float = Body(...)
    ):
        """Calculate OEE from component values."""
        # Direct OEE calculation from components
        oee = availability * performance * quality

        return {
            "availability": round(availability, 4),
            "performance": round(performance, 4),
            "quality": round(quality, 4),
            "oee": round(oee, 4),
            "oee_percent": f"{oee * 100:.1f}%",
            "world_class_gap": round(0.85 - oee, 4) if oee < 0.85 else 0,
            "status": "world_class" if oee >= 0.85 else "good" if oee >= 0.65 else "needs_improvement"
        }

    # Include routers
    app.include_router(anomaly_router)
    app.include_router(assessment_router)
    app.include_router(roi_router)
    app.include_router(metrics_router)


def get_app() -> "FastAPI":
    """
    Get or create the global FastAPI application.

    Returns:
        FastAPI application instance
    """
    global _app
    if _app is None:
        _app = create_app()
    return _app


def reset_app() -> None:
    """Reset the global app instance (for testing)."""
    global _app
    _app = None


def _register_mlops_routes(app: "FastAPI"):
    """Register MLOps routes for model registry, experiments, feedback, and explainability."""
    from goodai.api.mlops_routes import (
        model_router,
        experiment_router,
        feedback_router,
        explain_router,
    )

    # Add prefix for versioned API
    app.include_router(model_router, prefix="/api/v1")
    app.include_router(experiment_router, prefix="/api/v1")
    app.include_router(feedback_router, prefix="/api/v1")
    app.include_router(explain_router, prefix="/api/v1")


def _register_module_health_checks(hc: HealthCheck):
    """Register health checks for all enterprise modules."""
    from goodai.monitoring import HealthStatus

    # Model Registry health check
    def check_model_registry():
        try:
            from goodai.mlops.registry import get_model_registry
            registry = get_model_registry()
            model_count = len(registry.list_models())
            return {
                "status": HealthStatus.HEALTHY,
                "model_count": model_count,
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "error": str(e),
            }

    hc.add_check("model_registry", check_model_registry, critical=False)

    # A/B Testing Framework health check
    def check_ab_testing():
        try:
            from goodai.mlops.ab_testing import get_ab_framework
            framework = get_ab_framework()
            experiment_count = len(framework.list_experiments())
            return {
                "status": HealthStatus.HEALTHY,
                "experiment_count": experiment_count,
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "error": str(e),
            }

    hc.add_check("ab_testing", check_ab_testing, critical=False)

    # Feedback Loop health check
    def check_feedback_loop():
        try:
            from goodai.mlops.feedback import get_feedback_loop
            get_feedback_loop()  # Verify it can be instantiated
            return {
                "status": HealthStatus.HEALTHY,
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "error": str(e),
            }

    hc.add_check("feedback_loop", check_feedback_loop, critical=False)

    # Cache health check
    def check_cache():
        try:
            from goodai.infrastructure.cache import get_cache_manager
            manager = get_cache_manager()
            stats = manager.get_all_stats()
            return {
                "status": HealthStatus.HEALTHY,
                "namespaces": list(stats.keys()),
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "error": str(e),
            }

    hc.add_check("cache", check_cache, critical=False)

    # Async Executor health check
    def check_async_executor():
        try:
            from goodai.infrastructure.async_utils import get_async_executor
            executor = get_async_executor()
            stats = executor.get_stats()
            return {
                "status": HealthStatus.HEALTHY,
                "total_executions": stats.get("total_executions", 0),
                "success_rate": stats.get("success_rate", 0),
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "error": str(e),
            }

    hc.add_check("async_executor", check_async_executor, critical=False)

    # Schema Registry health check
    def check_schema_registry():
        try:
            from goodai.infrastructure.validation import get_schema_registry
            get_schema_registry()  # Verify it can be instantiated
            return {
                "status": HealthStatus.HEALTHY,
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "error": str(e),
            }

    hc.add_check("schema_registry", check_schema_registry, critical=False)

    # Audit Logger health check
    def check_audit_logger():
        try:
            from goodai.security.audit import get_audit_logger
            audit = get_audit_logger()
            audit.query(limit=1)
            return {
                "status": HealthStatus.HEALTHY,
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "error": str(e),
            }

    hc.add_check("audit_logger", check_audit_logger, critical=False)

    # RBAC Manager health check
    def check_rbac():
        try:
            from goodai.security.rbac import get_rbac_manager
            manager = get_rbac_manager()
            roles = manager.list_roles()
            return {
                "status": HealthStatus.HEALTHY,
                "role_count": len(roles),
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "error": str(e),
            }

    hc.add_check("rbac", check_rbac, critical=False)

    # Tenant Manager health check
    def check_tenant_manager():
        try:
            from goodai.security.tenancy import get_tenant_manager
            manager = get_tenant_manager()
            tenants = manager.list_tenants()
            return {
                "status": HealthStatus.HEALTHY,
                "tenant_count": len(tenants),
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "error": str(e),
            }

    hc.add_check("tenant_manager", check_tenant_manager, critical=False)
