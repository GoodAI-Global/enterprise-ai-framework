"""
Tests for Assessment Engine Module

Tests the AI readiness assessment logic.
"""


from goodai.core.assessment_engine import (
    AssessmentEngine,
    AssessmentResult,
    DimensionScore,
    ReadinessLevel
)


class TestAssessmentEngine:
    """Test suite for AssessmentEngine class."""

    def test_init_default_industry(self):
        """Test initialization with default industry."""
        engine = AssessmentEngine()
        assert engine.industry == "manufacturing"
        assert engine.benchmarks is not None

    def test_init_custom_industry(self):
        """Test initialization with custom industry."""
        engine = AssessmentEngine(industry="manufacturing")
        assert engine.industry == "manufacturing"

    def test_init_with_custom_benchmarks(self):
        """Test initialization with custom benchmarks."""
        custom = {"data_sources_minimum": 5}
        engine = AssessmentEngine(benchmarks=custom)
        assert engine.benchmarks["data_sources_minimum"] == 5

    def test_assess_readiness_returns_result(self):
        """Test that assess_readiness returns AssessmentResult."""
        engine = AssessmentEngine()
        company_data = {
            "data_sources": ["erp", "scada"],
            "data_quality_score": 0.7,
            "has_data_warehouse": True,
            "has_data_catalog": False,
            "it_staff_count": 5,
            "total_employees": 200,
            "cloud_adoption_level": 0.5,
            "has_ml_experience": False,
            "process_documentation_level": 0.6,
            "automation_level": 0.4,
            "has_continuous_improvement": True,
            "leadership_ai_support": 0.7,
            "change_success_rate": 0.6,
            "training_budget_exists": True,
        }

        result = engine.assess_readiness(company_data)

        assert isinstance(result, AssessmentResult)
        assert 1 <= result.overall_score <= 10
        assert isinstance(result.overall_level, ReadinessLevel)
        assert len(result.dimensions) == 4

    def test_assess_readiness_all_dimensions(self):
        """Test that all four dimensions are assessed."""
        engine = AssessmentEngine()
        company_data = {"data_sources": ["erp"]}

        result = engine.assess_readiness(company_data)

        expected_dimensions = [
            "data_readiness",
            "technical_capability",
            "process_maturity",
            "organizational_readiness"
        ]
        for dim in expected_dimensions:
            assert dim in result.dimensions
            assert isinstance(result.dimensions[dim], DimensionScore)

    def test_assess_high_readiness_company(self):
        """Test assessment of highly ready company."""
        engine = AssessmentEngine()
        company_data = {
            "data_sources": ["erp", "scada", "mes", "crm", "iot"],
            "data_quality_score": 0.90,
            "has_data_warehouse": True,
            "has_data_catalog": True,
            "has_data_governance": True,
            "has_privacy_compliance": True,
            "it_staff_count": 10,
            "total_employees": 200,
            "cloud_adoption_level": 0.8,
            "has_ml_experience": True,
            "ml_projects_count": 5,
            "has_cicd": True,
            "has_version_control": True,
            "process_documentation_level": 0.9,
            "automation_level": 0.75,
            "has_continuous_improvement": True,
            "has_process_metrics": True,
            "sop_coverage": 0.85,
            "leadership_ai_support": 0.9,
            "change_success_rate": 0.85,
            "training_budget_exists": True,
            "has_ai_training_program": True,
            "cross_functional_collaboration": 0.85,
        }

        result = engine.assess_readiness(company_data)

        # High-readiness company should score well
        assert result.overall_score >= 7.0
        assert result.overall_level in [ReadinessLevel.ADVANCED, ReadinessLevel.LEADING]

    def test_assess_low_readiness_company(self):
        """Test assessment of company with low readiness."""
        engine = AssessmentEngine()
        company_data = {
            "data_sources": ["excel"],
            "data_quality_score": 0.3,
            "has_data_warehouse": False,
            "has_data_catalog": False,
            "it_staff_count": 1,
            "total_employees": 100,
            "cloud_adoption_level": 0.1,
            "has_ml_experience": False,
            "process_documentation_level": 0.2,
            "automation_level": 0.1,
            "has_continuous_improvement": False,
            "leadership_ai_support": 0.2,
            "change_success_rate": 0.3,
            "training_budget_exists": False,
        }

        result = engine.assess_readiness(company_data)

        # Low-readiness company should score poorly
        assert result.overall_score < 5.0
        assert result.overall_level in [ReadinessLevel.NOT_READY, ReadinessLevel.FOUNDATIONAL]

    def test_assess_generates_recommendations(self):
        """Test that recommendations are generated."""
        engine = AssessmentEngine()
        company_data = {
            "data_sources": ["erp"],
            "data_quality_score": 0.5,
        }

        result = engine.assess_readiness(company_data)

        assert len(result.recommendations) > 0
        assert all(isinstance(r, str) for r in result.recommendations)

    def test_assess_identifies_red_flags(self):
        """Test that red flags are identified for critical issues."""
        engine = AssessmentEngine()
        company_data = {
            "data_sources": ["erp"],
            "data_quality_score": 0.3,  # Very low
            "leadership_ai_support": 0.2,  # Very low
            "change_success_rate": 0.2,  # Very low
        }

        result = engine.assess_readiness(company_data)

        # Should have red flags for critical issues
        assert len(result.red_flags) > 0

    def test_assess_identifies_quick_wins(self):
        """Test that quick wins are identified."""
        engine = AssessmentEngine()
        company_data = {
            "data_sources": ["erp", "scada", "mes"],
            "data_quality_score": 0.8,
            "has_data_warehouse": True,
            "has_ml_experience": False,
            "process_documentation_level": 0.8,
            "automation_level": 0.3,
            "change_success_rate": 0.8,
        }

        result = engine.assess_readiness(company_data)

        # Should identify quick wins
        assert len(result.quick_wins) > 0

    def test_dimension_score_level_property(self):
        """Test DimensionScore level property."""
        low = DimensionScore("test", 2.5, [], [])
        foundational = DimensionScore("test", 4.0, [], [])
        developing = DimensionScore("test", 6.0, [], [])
        advanced = DimensionScore("test", 8.0, [], [])
        leading = DimensionScore("test", 9.5, [], [])

        assert low.level == ReadinessLevel.NOT_READY
        assert foundational.level == ReadinessLevel.FOUNDATIONAL
        assert developing.level == ReadinessLevel.DEVELOPING
        assert advanced.level == ReadinessLevel.ADVANCED
        assert leading.level == ReadinessLevel.LEADING

    def test_assess_with_empty_data(self):
        """Test assessment with minimal data."""
        engine = AssessmentEngine()
        result = engine.assess_readiness({})

        # Should still return valid result
        assert isinstance(result, AssessmentResult)
        assert result.overall_score >= 1.0

    def test_assess_deterministic(self):
        """Test that same input produces same output."""
        engine = AssessmentEngine()
        company_data = {
            "data_sources": ["erp", "scada"],
            "data_quality_score": 0.7,
            "has_data_warehouse": True,
        }

        result1 = engine.assess_readiness(company_data.copy())
        result2 = engine.assess_readiness(company_data.copy())

        assert result1.overall_score == result2.overall_score
        assert result1.overall_level == result2.overall_level


class TestAssessmentEngineBottlenecks:
    """Test bottleneck identification functionality."""

    def test_identify_bottlenecks_returns_list(self):
        """Test that identify_bottlenecks returns a list."""
        engine = AssessmentEngine()
        process_data = {
            "oee": 0.65,
            "availability": 0.85,
            "performance": 0.80,
            "quality": 0.95,
            "downtime_hours_monthly": 40,
        }

        bottlenecks = engine.identify_bottlenecks(process_data)

        assert isinstance(bottlenecks, list)

    def test_identify_bottlenecks_with_issues(self):
        """Test bottleneck detection when issues exist."""
        engine = AssessmentEngine()
        process_data = {
            "availability": 0.70,  # Below target
            "performance": 0.75,  # Below target
            "quality": 0.90,  # Below target
            "downtime_hours_monthly": 50,
            "defect_rate": 0.10,
        }

        bottlenecks = engine.identify_bottlenecks(process_data)

        # Should identify bottlenecks
        assert len(bottlenecks) > 0
