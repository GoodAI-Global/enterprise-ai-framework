"""
Tests for ML Operations Module

Tests for model registry, A/B testing, feedback loops, and explainability.
"""

import pytest
from datetime import datetime, timedelta

from goodai.mlops.registry import (
    Model,
    ModelVersion,
    ModelStatus,
    ModelMetrics,
    ModelRegistry,
    get_model_registry,
)
from goodai.mlops.ab_testing import (
    ExperimentStatus,
    VariantMetrics,
    ABTestingFramework,
    get_ab_framework,
)
from goodai.mlops.feedback import (
    FeedbackType,
    Feedback,
    FeedbackSummary,
    FeedbackLoop,
    AlertConfig,
    get_feedback_loop,
)
from goodai.mlops.explainability import (
    ExplanationType,
    Explanation,
    FeatureImportance,
    Counterfactual,
    Rule,
    ExplainabilityEngine,
    get_explainability_engine,
)


# ============== Model Registry Tests ==============

class TestModelMetrics:
    """Test ModelMetrics dataclass."""

    def test_to_dict(self):
        """Test to_dict method."""
        metrics = ModelMetrics(
            accuracy=0.95,
            precision=0.92,
            recall=0.88,
            custom_metrics={"specificity": 0.97}
        )
        result = metrics.to_dict()
        assert result["accuracy"] == 0.95
        assert result["specificity"] == 0.97
        assert "mse" not in result  # None values excluded


class TestModelVersion:
    """Test ModelVersion dataclass."""

    def test_create_version(self):
        """Test version creation."""
        version = ModelVersion(
            version="1.0.0",
            model_name="test_model",
            framework="pytorch",
        )
        assert version.version == "1.0.0"
        assert version.status == ModelStatus.DRAFT

    def test_to_dict(self):
        """Test to_dict method."""
        version = ModelVersion(
            version="1.0.0",
            model_name="test_model",
            metrics=ModelMetrics(accuracy=0.95),
        )
        result = version.to_dict()
        assert result["version"] == "1.0.0"
        assert result["metrics"]["accuracy"] == 0.95


class TestModel:
    """Test Model dataclass."""

    def test_latest_version(self):
        """Test latest version property."""
        model = Model(name="test")
        v1 = ModelVersion(version="1.0.0", model_name="test")
        v2 = ModelVersion(version="2.0.0", model_name="test")
        v2.created_at = datetime.utcnow() + timedelta(seconds=1)
        model.versions = [v1, v2]

        assert model.latest_version.version == "2.0.0"

    def test_production_version(self):
        """Test production version property."""
        model = Model(name="test")
        v1 = ModelVersion(version="1.0.0", model_name="test", status=ModelStatus.PRODUCTION)
        v2 = ModelVersion(version="2.0.0", model_name="test", status=ModelStatus.STAGING)
        model.versions = [v1, v2]

        assert model.production_version.version == "1.0.0"


class TestModelRegistry:
    """Test ModelRegistry class."""

    def test_register_model(self):
        """Test model registration."""
        registry = ModelRegistry()
        model = registry.register_model(
            name="fraud_detector",
            description="Fraud detection model",
            owner="ml-team"
        )
        assert model.name == "fraud_detector"
        assert registry.get_model("fraud_detector") is not None

    def test_register_duplicate_raises(self):
        """Test duplicate registration raises error."""
        registry = ModelRegistry()
        registry.register_model(name="test")
        with pytest.raises(ValueError, match="already exists"):
            registry.register_model(name="test")

    def test_create_version(self):
        """Test version creation."""
        registry = ModelRegistry()
        registry.register_model(name="test")
        version = registry.create_version(
            model_name="test",
            version="1.0.0",
            framework="sklearn",
            metrics=ModelMetrics(accuracy=0.95)
        )
        assert version.version == "1.0.0"
        assert version.metrics.accuracy == 0.95

    def test_create_version_duplicate_raises(self):
        """Test duplicate version raises error."""
        registry = ModelRegistry()
        registry.register_model(name="test")
        registry.create_version(model_name="test", version="1.0.0")
        with pytest.raises(ValueError, match="already exists"):
            registry.create_version(model_name="test", version="1.0.0")

    def test_transition_status(self):
        """Test status transition."""
        registry = ModelRegistry()
        registry.register_model(name="test")
        registry.create_version(model_name="test", version="1.0.0")

        version = registry.transition_status(
            model_name="test",
            version="1.0.0",
            status=ModelStatus.PRODUCTION
        )
        assert version.status == ModelStatus.PRODUCTION

    def test_transition_to_production_demotes_existing(self):
        """Test that promoting to production demotes existing."""
        registry = ModelRegistry()
        registry.register_model(name="test")
        registry.create_version(model_name="test", version="1.0.0")
        registry.create_version(model_name="test", version="2.0.0")

        registry.transition_status("test", "1.0.0", ModelStatus.PRODUCTION)
        registry.transition_status("test", "2.0.0", ModelStatus.PRODUCTION)

        v1 = registry.get_version("test", "1.0.0")
        v2 = registry.get_version("test", "2.0.0")

        assert v1.status == ModelStatus.ARCHIVED
        assert v2.status == ModelStatus.PRODUCTION

    def test_compare_versions(self):
        """Test version comparison."""
        registry = ModelRegistry()
        registry.register_model(name="test")
        registry.create_version(
            model_name="test",
            version="1.0.0",
            metrics=ModelMetrics(accuracy=0.90),
            parameters={"learning_rate": 0.01}
        )
        registry.create_version(
            model_name="test",
            version="2.0.0",
            metrics=ModelMetrics(accuracy=0.95),
            parameters={"learning_rate": 0.001}
        )

        comparison = registry.compare_versions("test", "1.0.0", "2.0.0")
        assert comparison["metrics_comparison"]["accuracy"]["diff"] == pytest.approx(0.05)
        assert "learning_rate" in comparison["parameter_diff"]

    def test_list_models_with_filter(self):
        """Test listing models with filters."""
        registry = ModelRegistry()
        registry.register_model(name="model1", team="team-a", tags={"env": "prod"})
        registry.register_model(name="model2", team="team-b", tags={"env": "dev"})
        registry.register_model(name="model3", team="team-a", tags={"env": "prod"})

        team_a = registry.list_models(team="team-a")
        assert len(team_a) == 2

        prod = registry.list_models(tag_filter={"env": "prod"})
        assert len(prod) == 2

    def test_transition_hook(self):
        """Test transition hooks are called."""
        registry = ModelRegistry()
        transitions = []

        def hook(model, version, old_status, new_status):
            transitions.append((model, version, old_status, new_status))

        registry.add_transition_hook(hook)
        registry.register_model(name="test")
        registry.create_version(model_name="test", version="1.0.0")
        registry.transition_status("test", "1.0.0", ModelStatus.STAGING)

        assert len(transitions) == 1
        assert transitions[0] == ("test", "1.0.0", ModelStatus.DRAFT, ModelStatus.STAGING)


# ============== A/B Testing Tests ==============

class TestVariantMetrics:
    """Test VariantMetrics dataclass."""

    def test_conversion_rate(self):
        """Test conversion rate calculation."""
        metrics = VariantMetrics(impressions=100, conversions=15)
        assert metrics.conversion_rate == 0.15

    def test_conversion_rate_zero_impressions(self):
        """Test conversion rate with zero impressions."""
        metrics = VariantMetrics()
        assert metrics.conversion_rate == 0.0


class TestABTestingFramework:
    """Test ABTestingFramework class."""

    def test_create_experiment(self):
        """Test experiment creation."""
        ab = ABTestingFramework()
        exp = ab.create_experiment(
            name="test_exp",
            description="Test experiment"
        )
        assert exp.name == "test_exp"
        assert exp.status == ExperimentStatus.DRAFT

    def test_add_variant(self):
        """Test adding variants."""
        ab = ABTestingFramework()
        exp = ab.create_experiment(name="test")
        variant = ab.add_variant(
            exp.id,
            name="control",
            model_name="model",
            model_version="1.0",
            is_control=True
        )
        assert variant.name == "control"
        assert variant.is_control

    def test_add_variant_to_running_raises(self):
        """Test cannot add variant to running experiment."""
        ab = ABTestingFramework()
        exp = ab.create_experiment(name="test")
        ab.add_variant(exp.id, "control", "m", "1", is_control=True)
        ab.add_variant(exp.id, "treatment", "m", "2")
        ab.start_experiment(exp.id)

        with pytest.raises(ValueError, match="non-draft"):
            ab.add_variant(exp.id, "new", "m", "3")

    def test_start_requires_control(self):
        """Test start requires control variant."""
        ab = ABTestingFramework()
        exp = ab.create_experiment(name="test")
        ab.add_variant(exp.id, "v1", "m", "1")
        ab.add_variant(exp.id, "v2", "m", "2")

        with pytest.raises(ValueError, match="control"):
            ab.start_experiment(exp.id)

    def test_start_requires_two_variants(self):
        """Test start requires at least 2 variants."""
        ab = ABTestingFramework()
        exp = ab.create_experiment(name="test")
        ab.add_variant(exp.id, "control", "m", "1", is_control=True)

        with pytest.raises(ValueError, match="at least 2"):
            ab.start_experiment(exp.id)

    def test_assign_variant_sticky(self):
        """Test sticky variant assignment."""
        ab = ABTestingFramework(random_seed=42)
        exp = ab.create_experiment(name="test")
        ab.add_variant(exp.id, "control", "m", "1", is_control=True)
        ab.add_variant(exp.id, "treatment", "m", "2")
        ab.start_experiment(exp.id)

        v1 = ab.assign_variant(exp.id, "user1")
        v2 = ab.assign_variant(exp.id, "user1")
        assert v1.name == v2.name

    def test_assign_variant_tracks_impressions(self):
        """Test impressions are tracked."""
        ab = ABTestingFramework(random_seed=42)
        exp = ab.create_experiment(name="test")
        ab.add_variant(exp.id, "control", "m", "1", is_control=True)
        ab.add_variant(exp.id, "treatment", "m", "2")
        ab.start_experiment(exp.id)

        for i in range(10):
            ab.assign_variant(exp.id, f"user{i}")

        total = sum(v.metrics.impressions for v in exp.variants)
        assert total == 10

    def test_record_conversion(self):
        """Test recording conversions."""
        ab = ABTestingFramework(random_seed=42)
        exp = ab.create_experiment(name="test")
        ab.add_variant(exp.id, "control", "m", "1", is_control=True)
        ab.add_variant(exp.id, "treatment", "m", "2")
        ab.start_experiment(exp.id)

        ab.record_conversion(exp.id, "control", value=100.0)
        control = exp.get_variant("control")
        assert control.metrics.conversions == 1
        assert control.metrics.total_value == 100.0

    def test_get_results(self):
        """Test getting experiment results."""
        ab = ABTestingFramework(random_seed=42)
        exp = ab.create_experiment(name="test")
        ab.add_variant(exp.id, "control", "m", "1", is_control=True)
        ab.add_variant(exp.id, "treatment", "m", "2")
        ab.start_experiment(exp.id)

        # Generate some data
        for i in range(100):
            variant = ab.assign_variant(exp.id, f"user{i}", sticky=False)
            if i % 10 == 0:  # 10% conversion
                ab.record_conversion(exp.id, variant.name)

        results = ab.get_results(exp.id)
        assert "variants" in results
        assert "control" in results["variants"]


class TestExperimentLifecycle:
    """Test experiment lifecycle."""

    def test_pause_resume(self):
        """Test pause and resume."""
        ab = ABTestingFramework()
        exp = ab.create_experiment(name="test")
        ab.add_variant(exp.id, "control", "m", "1", is_control=True)
        ab.add_variant(exp.id, "treatment", "m", "2")
        ab.start_experiment(exp.id)

        ab.pause_experiment(exp.id)
        assert exp.status == ExperimentStatus.PAUSED

        ab.resume_experiment(exp.id)
        assert exp.status == ExperimentStatus.RUNNING

    def test_stop_experiment(self):
        """Test stopping experiment."""
        ab = ABTestingFramework()
        exp = ab.create_experiment(name="test")
        ab.add_variant(exp.id, "control", "m", "1", is_control=True)
        ab.add_variant(exp.id, "treatment", "m", "2")
        ab.start_experiment(exp.id)

        ab.stop_experiment(exp.id)
        assert exp.status == ExperimentStatus.COMPLETED
        assert exp.ended_at is not None


# ============== Feedback Loop Tests ==============

class TestFeedback:
    """Test Feedback dataclass."""

    def test_to_dict(self):
        """Test to_dict method."""
        feedback = Feedback(
            id="f1",
            feedback_type=FeedbackType.RATING,
            model_name="model",
            model_version="1.0",
            value=5
        )
        result = feedback.to_dict()
        assert result["feedback_type"] == "rating"
        assert result["value"] == 5


class TestFeedbackLoop:
    """Test FeedbackLoop class."""

    def test_record_feedback(self):
        """Test recording feedback."""
        loop = FeedbackLoop()
        feedback = loop.record(
            FeedbackType.RATING,
            model_name="model",
            model_version="1.0",
            value=4,
            user_id="user1"
        )
        assert feedback.value == 4
        assert feedback.user_id == "user1"

    def test_query_feedback(self):
        """Test querying feedback."""
        loop = FeedbackLoop()
        loop.record(FeedbackType.RATING, "m1", "1.0", value=4)
        loop.record(FeedbackType.RATING, "m2", "1.0", value=5)
        loop.record(FeedbackType.THUMBS, "m1", "1.0", value=True)

        m1_feedback = loop.get_feedback(model_name="m1")
        assert len(m1_feedback) == 2

        ratings = loop.get_feedback(feedback_type=FeedbackType.RATING)
        assert len(ratings) == 2

    def test_get_summary(self):
        """Test getting feedback summary."""
        loop = FeedbackLoop()
        for i in range(10):
            loop.record(FeedbackType.RATING, "model", "1.0", value=4 if i < 7 else 2)

        summary = loop.get_summary("model", "1.0", hours=1)
        assert summary.total_feedback == 10
        assert summary.positive_feedback == 7
        assert summary.negative_feedback == 3

    def test_get_summary_with_thumbs(self):
        """Test summary with thumbs feedback."""
        loop = FeedbackLoop()
        loop.record(FeedbackType.THUMBS, "model", "1.0", value=True)
        loop.record(FeedbackType.THUMBS, "model", "1.0", value=True)
        loop.record(FeedbackType.THUMBS, "model", "1.0", value=False)

        summary = loop.get_summary("model", "1.0", hours=1)
        assert summary.positive_feedback == 2
        assert summary.negative_feedback == 1
        assert summary.satisfaction_rate == pytest.approx(0.667, rel=0.01)

    def test_processing_callback(self):
        """Test processing callbacks."""
        loop = FeedbackLoop()
        received = []

        loop.add_processing_callback(lambda f: received.append(f))
        loop.record(FeedbackType.RATING, "model", "1.0", value=5)

        assert len(received) == 1
        assert received[0].value == 5

    def test_alert_handler(self):
        """Test alert handlers."""
        config = AlertConfig(
            min_satisfaction_rate=0.8,
            min_sample_size=5,
            check_interval_hours=1
        )
        loop = FeedbackLoop(alert_config=config)
        alerts = []

        loop.add_alert_handler(lambda t, d: alerts.append((t, d)))

        # Record enough low ratings to trigger alert
        for _ in range(5):
            loop.record(FeedbackType.RATING, "model", "1.0", value=2)

        assert len(alerts) > 0
        assert alerts[0][0] == "low_satisfaction"

    def test_corrections_for_retraining(self):
        """Test getting corrections for retraining."""
        loop = FeedbackLoop()

        for i in range(15):
            loop.record(
                FeedbackType.CORRECTION,
                "model", "1.0",
                value={"corrected": f"output{i}"},
                context={"input": f"input{i}", "output": f"wrong{i}"}
            )

        examples = loop.get_corrections_for_retraining("model", "1.0", min_count=10)
        assert len(examples) == 15
        assert "corrected_output" in examples[0]


class TestFeedbackSummary:
    """Test FeedbackSummary dataclass."""

    def test_satisfaction_rate(self):
        """Test satisfaction rate calculation."""
        summary = FeedbackSummary(
            model_name="m",
            model_version="1",
            period_start=datetime.utcnow(),
            period_end=datetime.utcnow(),
            positive_feedback=80,
            negative_feedback=20
        )
        assert summary.satisfaction_rate == 0.8


# ============== Explainability Tests ==============

class TestFeatureImportance:
    """Test FeatureImportance dataclass."""

    def test_to_dict(self):
        """Test to_dict method."""
        fi = FeatureImportance(
            feature="amount",
            importance=0.8,
            direction="positive",
            value=1000
        )
        result = fi.to_dict()
        assert result["feature"] == "amount"
        assert result["importance"] == 0.8


class TestExplanation:
    """Test Explanation dataclass."""

    def test_top_features(self):
        """Test top features property."""
        exp = Explanation(
            id="e1",
            explanation_type=ExplanationType.FEATURE_IMPORTANCE,
            model_name="model",
            model_version="1.0"
        )
        exp.feature_importance = [
            FeatureImportance("a", 0.5, "positive"),
            FeatureImportance("b", 0.9, "positive"),
            FeatureImportance("c", 0.3, "negative"),
        ]

        top = exp.top_features
        assert len(top) == 3
        assert top[0].feature == "b"

    def test_to_natural_language(self):
        """Test natural language generation."""
        exp = Explanation(
            id="e1",
            explanation_type=ExplanationType.FEATURE_IMPORTANCE,
            model_name="model",
            model_version="1.0",
            prediction="fraud",
            confidence=0.95
        )
        exp.feature_importance = [
            FeatureImportance("amount", 0.8, "positive"),
        ]

        nl = exp.to_natural_language()
        assert "fraud" in nl
        assert "95" in nl
        assert "amount" in nl


class TestExplainabilityEngine:
    """Test ExplainabilityEngine class."""

    def test_explain_with_default(self):
        """Test explanation with default explainer."""
        engine = ExplainabilityEngine()
        explanation = engine.explain(
            model_name="model",
            model_version="1.0",
            input_data={"amount": 1000, "merchant": "online"},
            prediction="fraud",
            confidence=0.95
        )

        assert explanation.prediction == "fraud"
        assert len(explanation.feature_importance) > 0

    def test_explain_with_custom_explainer(self):
        """Test explanation with custom explainer."""
        engine = ExplainabilityEngine()

        def custom_explainer(input_data, prediction):
            return {
                "feature_importance": [
                    {"feature": "amount", "importance": 0.9, "direction": "positive"},
                ],
                "summary": "High amount indicates fraud",
            }

        engine.register_explainer("model", custom_explainer)

        explanation = engine.explain(
            model_name="model",
            model_version="1.0",
            input_data={"amount": 1000},
            prediction="fraud"
        )

        assert explanation.feature_importance[0].feature == "amount"
        assert explanation.summary == "High amount indicates fraud"

    def test_explain_with_counterfactuals(self):
        """Test explanation with counterfactuals."""
        engine = ExplainabilityEngine()

        def cf_explainer(input_data, prediction):
            return {
                "counterfactuals": [
                    {
                        "original_prediction": "fraud",
                        "counterfactual_prediction": "legitimate",
                        "changes": {"amount": (1000, 100)},
                        "distance": 0.5,
                    }
                ],
            }

        engine.register_explainer("model", cf_explainer)

        explanation = engine.explain(
            model_name="model",
            model_version="1.0",
            input_data={"amount": 1000},
            prediction="fraud"
        )

        assert len(explanation.counterfactuals) == 1
        assert explanation.counterfactuals[0].counterfactual_prediction == "legitimate"

    def test_get_explanations(self):
        """Test querying explanations."""
        engine = ExplainabilityEngine()
        engine.explain("m1", "1.0", {"x": 1}, "a")
        engine.explain("m1", "1.0", {"x": 2}, "b")
        engine.explain("m2", "1.0", {"x": 3}, "c")

        m1_explanations = engine.get_explanations(model_name="m1")
        assert len(m1_explanations) == 2

    def test_generate_audit_report(self):
        """Test audit report generation."""
        engine = ExplainabilityEngine()

        for i in range(20):
            engine.explain(
                model_name="model",
                model_version="1.0",
                input_data={"amount": i * 100},
                prediction="fraud" if i % 3 == 0 else "legitimate",
                confidence=0.8 + (i % 3) * 0.05
            )

        report = engine.generate_audit_report("model", "1.0")

        assert report["total_explanations"] == 20
        assert "fraud" in report["prediction_distribution"]
        assert "legitimate" in report["prediction_distribution"]


class TestCounterfactual:
    """Test Counterfactual dataclass."""

    def test_to_dict(self):
        """Test to_dict method."""
        cf = Counterfactual(
            original_prediction="fraud",
            counterfactual_prediction="legitimate",
            changes={"amount": (1000, 100)},
            distance=0.5
        )
        result = cf.to_dict()
        assert result["changes"]["amount"]["from"] == 1000
        assert result["changes"]["amount"]["to"] == 100


class TestRule:
    """Test Rule dataclass."""

    def test_to_dict(self):
        """Test to_dict method."""
        rule = Rule(
            condition="amount > 500 AND merchant = 'online'",
            coverage=0.3,
            precision=0.95
        )
        result = rule.to_dict()
        assert "amount > 500" in result["condition"]


# ============== Global Singleton Tests ==============

class TestGlobalSingletons:
    """Test global singleton instances."""

    def test_get_model_registry(self):
        """Test model registry singleton."""
        r1 = get_model_registry()
        r2 = get_model_registry()
        assert r1 is r2

    def test_get_ab_framework(self):
        """Test A/B framework singleton."""
        ab1 = get_ab_framework()
        ab2 = get_ab_framework()
        assert ab1 is ab2

    def test_get_feedback_loop(self):
        """Test feedback loop singleton."""
        fl1 = get_feedback_loop()
        fl2 = get_feedback_loop()
        assert fl1 is fl2

    def test_get_explainability_engine(self):
        """Test explainability engine singleton."""
        e1 = get_explainability_engine()
        e2 = get_explainability_engine()
        assert e1 is e2
