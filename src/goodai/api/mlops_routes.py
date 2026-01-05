"""
MLOps REST API Routes

Exposes model registry, A/B testing, feedback, and explainability
functionality via REST endpoints.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from goodai.mlops.registry import (
    ModelMetrics,
    ModelStatus,
    get_model_registry,
)
from goodai.mlops.ab_testing import (
    ExperimentStatus,
    get_ab_framework,
)
from goodai.mlops.feedback import (
    FeedbackType,
    get_feedback_loop,
)
from goodai.mlops.explainability import (
    get_explainability_engine,
)
from goodai.monitoring import get_logger

logger = get_logger(__name__)

# ============== Pydantic Models ==============

class ModelCreate(BaseModel):
    """Request to register a new model."""
    name: str
    description: str = ""
    owner: Optional[str] = None
    team: Optional[str] = None
    tags: Dict[str, str] = Field(default_factory=dict)


class VersionCreate(BaseModel):
    """Request to create a new model version."""
    version: str
    description: str = ""
    framework: str = ""
    framework_version: str = ""
    artifact_path: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    tags: Dict[str, str] = Field(default_factory=dict)
    metrics: Optional[Dict[str, float]] = None


class StatusTransition(BaseModel):
    """Request to transition model status."""
    status: str  # draft, staging, production, archived, deprecated


class ExperimentCreate(BaseModel):
    """Request to create an experiment."""
    name: str
    description: str = ""
    target_sample_size: int = 1000
    min_runtime_hours: int = 24
    owner: Optional[str] = None


class VariantCreate(BaseModel):
    """Request to add a variant to an experiment."""
    name: str
    model_name: str
    model_version: str
    weight: float = 1.0
    is_control: bool = False
    description: str = ""
    config: Dict[str, Any] = Field(default_factory=dict)


class FeedbackCreate(BaseModel):
    """Request to record feedback."""
    feedback_type: str  # rating, thumbs, correction, etc.
    model_name: str
    model_version: str
    value: Any = None
    prediction_id: Optional[str] = None
    user_id: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExplainRequest(BaseModel):
    """Request for model explanation."""
    model_name: str
    model_version: str
    input_data: Dict[str, Any]
    prediction: Any
    confidence: Optional[float] = None
    prediction_id: Optional[str] = None


# ============== Routers ==============

model_router = APIRouter(prefix="/models", tags=["Model Registry"])
experiment_router = APIRouter(prefix="/experiments", tags=["A/B Testing"])
feedback_router = APIRouter(prefix="/feedback", tags=["Feedback"])
explain_router = APIRouter(prefix="/explain", tags=["Explainability"])


# ============== Model Registry Routes ==============

@model_router.post("/", response_model=Dict[str, Any])
async def register_model(request: ModelCreate):
    """Register a new model."""
    registry = get_model_registry()
    try:
        model = registry.register_model(
            name=request.name,
            description=request.description,
            owner=request.owner,
            team=request.team,
            tags=request.tags,
        )
        logger.info(f"Model registered: {request.name}")
        return model.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@model_router.get("/", response_model=List[Dict[str, Any]])
async def list_models(
    team: Optional[str] = None,
    tag_key: Optional[str] = None,
    tag_value: Optional[str] = None,
):
    """List all registered models."""
    registry = get_model_registry()
    tag_filter = {tag_key: tag_value} if tag_key and tag_value else None
    models = registry.list_models(team=team, tag_filter=tag_filter)
    return [m.to_dict() for m in models]


@model_router.get("/{model_name}", response_model=Dict[str, Any])
async def get_model(model_name: str):
    """Get a model by name."""
    registry = get_model_registry()
    model = registry.get_model(model_name)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    return model.to_dict()


@model_router.delete("/{model_name}")
async def delete_model(model_name: str):
    """Delete a model."""
    registry = get_model_registry()
    if registry.delete_model(model_name):
        logger.info(f"Model deleted: {model_name}")
        return {"status": "deleted", "model_name": model_name}
    raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")


@model_router.post("/{model_name}/versions", response_model=Dict[str, Any])
async def create_version(model_name: str, request: VersionCreate):
    """Create a new model version."""
    registry = get_model_registry()

    metrics = None
    if request.metrics:
        metrics = ModelMetrics(
            accuracy=request.metrics.get("accuracy"),
            precision=request.metrics.get("precision"),
            recall=request.metrics.get("recall"),
            f1_score=request.metrics.get("f1_score"),
            auc_roc=request.metrics.get("auc_roc"),
            mse=request.metrics.get("mse"),
            mae=request.metrics.get("mae"),
        )

    try:
        version = registry.create_version(
            model_name=model_name,
            version=request.version,
            description=request.description,
            framework=request.framework,
            framework_version=request.framework_version,
            artifact_path=request.artifact_path,
            parameters=request.parameters,
            tags=request.tags,
            metrics=metrics,
        )
        logger.info(f"Version created: {model_name}:{request.version}")
        return version.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@model_router.get("/{model_name}/versions", response_model=List[Dict[str, Any]])
async def list_versions(
    model_name: str,
    status: Optional[str] = None,
):
    """List versions of a model."""
    registry = get_model_registry()
    status_enum = None
    if status:
        try:
            status_enum = ModelStatus(status)
        except ValueError:
            valid_values = [s.value for s in ModelStatus]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status '{status}'. Valid values: {valid_values}"
            )
    versions = registry.list_versions(model_name, status=status_enum)
    return [v.to_dict() for v in versions]


@model_router.get("/{model_name}/versions/{version}", response_model=Dict[str, Any])
async def get_version(model_name: str, version: str):
    """Get a specific model version."""
    registry = get_model_registry()
    model_version = registry.get_version(model_name, version)
    if not model_version:
        raise HTTPException(
            status_code=404,
            detail=f"Version '{version}' not found for model '{model_name}'"
        )
    return model_version.to_dict()


@model_router.post("/{model_name}/versions/{version}/transition", response_model=Dict[str, Any])
async def transition_status(model_name: str, version: str, request: StatusTransition):
    """Transition model version to a new status."""
    registry = get_model_registry()
    try:
        status_enum = ModelStatus(request.status)
        model_version = registry.transition_status(model_name, version, status_enum)
        logger.info(f"Status transitioned: {model_name}:{version} -> {request.status}")
        return model_version.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@model_router.get("/{model_name}/compare", response_model=Dict[str, Any])
async def compare_versions(
    model_name: str,
    version_a: str = Query(..., description="First version to compare"),
    version_b: str = Query(..., description="Second version to compare"),
):
    """Compare two model versions."""
    registry = get_model_registry()
    try:
        return registry.compare_versions(model_name, version_a, version_b)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@model_router.get("/{model_name}/lineage", response_model=Dict[str, Any])
async def get_lineage(model_name: str):
    """Get model lineage information."""
    registry = get_model_registry()
    try:
        return registry.export_lineage(model_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ============== A/B Testing Routes ==============

@experiment_router.post("/", response_model=Dict[str, Any])
async def create_experiment(request: ExperimentCreate):
    """Create a new experiment."""
    ab = get_ab_framework()
    exp = ab.create_experiment(
        name=request.name,
        description=request.description,
        target_sample_size=request.target_sample_size,
        min_runtime_hours=request.min_runtime_hours,
        owner=request.owner,
    )
    logger.info(f"Experiment created: {exp.id}")
    return exp.to_dict()


@experiment_router.get("/", response_model=List[Dict[str, Any]])
async def list_experiments(status: Optional[str] = None):
    """List all experiments."""
    ab = get_ab_framework()
    status_enum = None
    if status:
        try:
            status_enum = ExperimentStatus(status)
        except ValueError:
            valid_values = [s.value for s in ExperimentStatus]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status '{status}'. Valid values: {valid_values}"
            )
    experiments = ab.list_experiments(status=status_enum)
    return [e.to_dict() for e in experiments]


@experiment_router.get("/{experiment_id}", response_model=Dict[str, Any])
async def get_experiment(experiment_id: str):
    """Get an experiment by ID."""
    ab = get_ab_framework()
    exp = ab.get_experiment(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found")
    return exp.to_dict()


@experiment_router.post("/{experiment_id}/variants", response_model=Dict[str, Any])
async def add_variant(experiment_id: str, request: VariantCreate):
    """Add a variant to an experiment."""
    ab = get_ab_framework()
    try:
        variant = ab.add_variant(
            experiment_id=experiment_id,
            name=request.name,
            model_name=request.model_name,
            model_version=request.model_version,
            weight=request.weight,
            is_control=request.is_control,
            description=request.description,
            config=request.config,
        )
        logger.info(f"Variant added: {experiment_id}/{request.name}")
        return variant.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@experiment_router.post("/{experiment_id}/start", response_model=Dict[str, Any])
async def start_experiment(experiment_id: str):
    """Start an experiment."""
    ab = get_ab_framework()
    try:
        exp = ab.start_experiment(experiment_id)
        logger.info(f"Experiment started: {experiment_id}")
        return exp.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@experiment_router.post("/{experiment_id}/pause", response_model=Dict[str, Any])
async def pause_experiment(experiment_id: str):
    """Pause an experiment."""
    ab = get_ab_framework()
    try:
        exp = ab.pause_experiment(experiment_id)
        logger.info(f"Experiment paused: {experiment_id}")
        return exp.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@experiment_router.post("/{experiment_id}/resume", response_model=Dict[str, Any])
async def resume_experiment(experiment_id: str):
    """Resume a paused experiment."""
    ab = get_ab_framework()
    try:
        exp = ab.resume_experiment(experiment_id)
        logger.info(f"Experiment resumed: {experiment_id}")
        return exp.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@experiment_router.post("/{experiment_id}/stop", response_model=Dict[str, Any])
async def stop_experiment(experiment_id: str):
    """Stop and complete an experiment."""
    ab = get_ab_framework()
    try:
        exp = ab.stop_experiment(experiment_id)
        logger.info(f"Experiment stopped: {experiment_id}")
        return exp.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@experiment_router.get("/{experiment_id}/assign", response_model=Dict[str, Any])
async def assign_variant(experiment_id: str, user_id: str = Query(...)):
    """Assign a user to a variant."""
    ab = get_ab_framework()
    variant = ab.assign_variant(experiment_id, user_id)
    if not variant:
        raise HTTPException(
            status_code=400,
            detail="Experiment not running or not found"
        )
    return {"variant": variant.to_dict()}


@experiment_router.post("/{experiment_id}/convert", response_model=Dict[str, Any])
async def record_conversion(
    experiment_id: str,
    variant_name: str = Query(...),
    value: float = Query(default=1.0),
):
    """Record a conversion for a variant."""
    ab = get_ab_framework()
    if ab.record_conversion(experiment_id, variant_name, value):
        return {"status": "recorded", "variant": variant_name, "value": value}
    raise HTTPException(status_code=400, detail="Failed to record conversion")


@experiment_router.get("/{experiment_id}/results", response_model=Dict[str, Any])
async def get_results(experiment_id: str):
    """Get experiment results with statistical analysis."""
    ab = get_ab_framework()
    try:
        return ab.get_results(experiment_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============== Feedback Routes ==============

@feedback_router.post("/", response_model=Dict[str, Any])
async def record_feedback(request: FeedbackCreate):
    """Record feedback for a model prediction."""
    loop = get_feedback_loop()
    try:
        feedback_type = FeedbackType(request.feedback_type)
        feedback = loop.record(
            feedback_type=feedback_type,
            model_name=request.model_name,
            model_version=request.model_version,
            value=request.value,
            prediction_id=request.prediction_id,
            user_id=request.user_id,
            context=request.context,
            metadata=request.metadata,
        )
        logger.info(f"Feedback recorded: {feedback.id}")
        return feedback.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@feedback_router.get("/", response_model=List[Dict[str, Any]])
async def get_feedback(
    model_name: Optional[str] = None,
    model_version: Optional[str] = None,
    feedback_type: Optional[str] = None,
    user_id: Optional[str] = None,
    limit: int = Query(default=100, le=1000),
):
    """Query feedback entries."""
    loop = get_feedback_loop()
    type_enum = None
    if feedback_type:
        try:
            type_enum = FeedbackType(feedback_type)
        except ValueError:
            valid_values = [t.value for t in FeedbackType]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid feedback_type '{feedback_type}'. Valid values: {valid_values}"
            )
    entries = loop.get_feedback(
        model_name=model_name,
        model_version=model_version,
        feedback_type=type_enum,
        user_id=user_id,
        limit=limit,
    )
    return [f.to_dict() for f in entries]


@feedback_router.get("/summary", response_model=Dict[str, Any])
async def get_feedback_summary(
    model_name: str = Query(...),
    model_version: str = Query(...),
    hours: int = Query(default=24, ge=1, le=720),
):
    """Get feedback summary for a model version."""
    loop = get_feedback_loop()
    summary = loop.get_summary(model_name, model_version, hours)
    return summary.to_dict()


@feedback_router.get("/trends", response_model=List[Dict[str, Any]])
async def get_feedback_trends(
    model_name: str = Query(...),
    model_version: str = Query(...),
    days: int = Query(default=7, ge=1, le=30),
    bucket_hours: int = Query(default=24, ge=1, le=168),
):
    """Get feedback trends over time."""
    loop = get_feedback_loop()
    return loop.get_trends(model_name, model_version, days, bucket_hours)


@feedback_router.get("/corrections", response_model=List[Dict[str, Any]])
async def get_corrections_for_retraining(
    model_name: str = Query(...),
    model_version: str = Query(...),
    min_count: int = Query(default=10, ge=1),
):
    """Get corrections suitable for retraining."""
    loop = get_feedback_loop()
    return loop.get_corrections_for_retraining(model_name, model_version, min_count)


# ============== Explainability Routes ==============

@explain_router.post("/", response_model=Dict[str, Any])
async def explain_prediction(request: ExplainRequest):
    """Generate an explanation for a prediction."""
    engine = get_explainability_engine()
    explanation = engine.explain(
        model_name=request.model_name,
        model_version=request.model_version,
        input_data=request.input_data,
        prediction=request.prediction,
        confidence=request.confidence,
        prediction_id=request.prediction_id,
    )
    logger.info(f"Explanation generated: {explanation.id}")
    return explanation.to_dict()


@explain_router.get("/{explanation_id}", response_model=Dict[str, Any])
async def get_explanation(explanation_id: str):
    """Get an explanation by ID."""
    engine = get_explainability_engine()
    explanation = engine.get_explanation(explanation_id)
    if not explanation:
        raise HTTPException(status_code=404, detail="Explanation not found")
    return explanation.to_dict()


@explain_router.get("/", response_model=List[Dict[str, Any]])
async def list_explanations(
    model_name: Optional[str] = None,
    model_version: Optional[str] = None,
    prediction_id: Optional[str] = None,
    limit: int = Query(default=100, le=1000),
):
    """Query explanations."""
    engine = get_explainability_engine()
    explanations = engine.get_explanations(
        model_name=model_name,
        model_version=model_version,
        prediction_id=prediction_id,
        limit=limit,
    )
    return [e.to_dict() for e in explanations]


@explain_router.get("/audit-report", response_model=Dict[str, Any])
async def get_audit_report(
    model_name: str = Query(...),
    model_version: str = Query(...),
):
    """Generate an audit report for model explanations."""
    engine = get_explainability_engine()
    return engine.generate_audit_report(model_name, model_version)


@explain_router.post("/natural-language", response_model=Dict[str, str])
async def get_natural_language_explanation(request: ExplainRequest):
    """Generate a natural language explanation."""
    engine = get_explainability_engine()
    explanation = engine.explain(
        model_name=request.model_name,
        model_version=request.model_version,
        input_data=request.input_data,
        prediction=request.prediction,
        confidence=request.confidence,
        prediction_id=request.prediction_id,
    )
    return {
        "explanation_id": explanation.id,
        "natural_language": explanation.to_natural_language(),
    }
