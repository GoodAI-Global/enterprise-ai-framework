"""
Assessment Engine Module

Evaluates enterprise AI readiness across multiple dimensions.
Provides actionable insights based on evidence, not opinions.

Good AI Philosophy: Evidence over opinions.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class ReadinessLevel(Enum):
    """AI readiness level classification."""

    NOT_READY = "not_ready"
    FOUNDATIONAL = "foundational"
    DEVELOPING = "developing"
    ADVANCED = "advanced"
    LEADING = "leading"


@dataclass
class DimensionScore:
    """Score for a single assessment dimension."""

    name: str
    score: float  # 1-10 scale
    findings: List[str]
    recommendations: List[str]
    evidence: Dict[str, Any] = field(default_factory=dict)

    @property
    def level(self) -> ReadinessLevel:
        """Convert score to readiness level."""
        if self.score < 3:
            return ReadinessLevel.NOT_READY
        elif self.score < 5:
            return ReadinessLevel.FOUNDATIONAL
        elif self.score < 7:
            return ReadinessLevel.DEVELOPING
        elif self.score < 9:
            return ReadinessLevel.ADVANCED
        return ReadinessLevel.LEADING


@dataclass
class AssessmentResult:
    """Complete assessment result."""

    overall_score: float
    overall_level: ReadinessLevel
    dimensions: Dict[str, DimensionScore]
    recommendations: List[str]
    red_flags: List[str]
    quick_wins: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class AssessmentEngine:
    """
    Evaluate AI readiness across dimensions.

    Assesses organizations across four key dimensions:
    - Data Readiness: Quality, availability, and accessibility of data
    - Technical Capability: Infrastructure and technical skills
    - Process Maturity: Standardization and documentation of processes
    - Organizational Readiness: Culture, leadership, and change capacity

    Args:
        industry: Target industry for benchmarking (default: "manufacturing")
        benchmarks: Optional custom benchmarks dictionary

    Example:
        >>> engine = AssessmentEngine(industry="manufacturing")
        >>> company_data = {
        ...     "data_sources": ["erp", "scada", "mes"],
        ...     "data_quality_score": 0.75,
        ...     "it_staff_count": 5,
        ...     "has_data_warehouse": True,
        ...     # ... more attributes
        ... }
        >>> result = engine.assess_readiness(company_data)
        >>> print(f"Overall Score: {result.overall_score:.1f}/10")
    """

    # Industry benchmarks for manufacturing
    MANUFACTURING_BENCHMARKS = {
        "data_sources_minimum": 3,
        "data_sources_good": 5,
        "data_quality_minimum": 0.6,
        "data_quality_good": 0.85,
        "it_staff_ratio": 0.02,  # IT staff per employee
        "automation_level_minimum": 0.3,
        "automation_level_good": 0.7,
        "process_documentation_minimum": 0.5,
        "process_documentation_good": 0.9,
        "change_success_rate_minimum": 0.4,
        "change_success_rate_good": 0.8,
    }

    def __init__(
        self,
        industry: str = "manufacturing",
        benchmarks: Optional[Dict[str, float]] = None
    ):
        self.industry = industry
        self.benchmarks = benchmarks or self._load_benchmarks()

    def _load_benchmarks(self) -> Dict[str, float]:
        """Load industry-specific benchmarks."""
        if self.industry == "manufacturing":
            return self.MANUFACTURING_BENCHMARKS.copy()
        # Default to manufacturing for now
        return self.MANUFACTURING_BENCHMARKS.copy()

    def assess_readiness(self, company_data: dict) -> AssessmentResult:
        """
        Evaluate AI readiness across dimensions.

        Args:
            company_data: Dictionary containing company attributes:
                - data_sources: List of data source types
                - data_quality_score: 0-1 score for data quality
                - has_data_warehouse: Boolean
                - has_data_catalog: Boolean
                - it_staff_count: Number of IT staff
                - total_employees: Total employee count
                - cloud_adoption_level: 0-1 score
                - has_ml_experience: Boolean
                - process_documentation_level: 0-1 score
                - automation_level: 0-1 score
                - has_continuous_improvement: Boolean
                - leadership_ai_support: 0-1 score
                - change_success_rate: 0-1 score
                - training_budget_exists: Boolean

        Returns:
            AssessmentResult with scores and recommendations
        """
        # Assess each dimension
        dimensions = {
            "data_readiness": self._assess_data(company_data),
            "technical_capability": self._assess_technical(company_data),
            "process_maturity": self._assess_process(company_data),
            "organizational_readiness": self._assess_org(company_data),
        }

        # Calculate overall score (weighted average)
        weights = {
            "data_readiness": 0.30,
            "technical_capability": 0.25,
            "process_maturity": 0.25,
            "organizational_readiness": 0.20,
        }

        overall_score = sum(
            dimensions[dim].score * weights[dim]
            for dim in dimensions
        )

        # Determine overall level
        overall_level = self._score_to_level(overall_score)

        # Generate recommendations
        recommendations = self._generate_recommendations(dimensions)
        red_flags = self._identify_red_flags(dimensions, company_data)
        quick_wins = self._identify_quick_wins(dimensions, company_data)

        return AssessmentResult(
            overall_score=round(overall_score, 2),
            overall_level=overall_level,
            dimensions=dimensions,
            recommendations=recommendations,
            red_flags=red_flags,
            quick_wins=quick_wins,
            metadata={
                "industry": self.industry,
                "benchmarks_used": self.benchmarks,
            }
        )

    def _assess_data(self, data: dict) -> DimensionScore:
        """Assess data readiness dimension."""
        score = 0.0
        findings = []
        recommendations = []
        evidence = {}

        # Data sources (0-2.5 points)
        sources = data.get("data_sources", [])
        num_sources = len(sources)
        evidence["data_sources"] = sources

        if num_sources >= self.benchmarks["data_sources_good"]:
            score += 2.5
            findings.append(f"Strong data ecosystem with {num_sources} sources")
        elif num_sources >= self.benchmarks["data_sources_minimum"]:
            score += 1.5
            findings.append(f"Adequate data sources ({num_sources})")
            recommendations.append("Consider integrating additional data sources")
        else:
            score += 0.5
            findings.append(f"Limited data sources ({num_sources})")
            recommendations.append("Priority: Expand data collection capabilities")

        # Data quality (0-2.5 points)
        quality = data.get("data_quality_score", 0)
        evidence["data_quality"] = quality

        if quality >= self.benchmarks["data_quality_good"]:
            score += 2.5
            findings.append(f"High data quality ({quality*100:.0f}%)")
        elif quality >= self.benchmarks["data_quality_minimum"]:
            score += 1.5
            findings.append(f"Moderate data quality ({quality*100:.0f}%)")
            recommendations.append("Implement data quality monitoring")
        else:
            score += 0.5
            findings.append(f"Data quality needs improvement ({quality*100:.0f}%)")
            recommendations.append("Priority: Establish data quality standards and processes")

        # Data infrastructure (0-2.5 points)
        has_warehouse = data.get("has_data_warehouse", False)
        has_catalog = data.get("has_data_catalog", False)
        evidence["has_warehouse"] = has_warehouse
        evidence["has_catalog"] = has_catalog

        if has_warehouse and has_catalog:
            score += 2.5
            findings.append("Mature data infrastructure")
        elif has_warehouse:
            score += 1.5
            findings.append("Data warehouse in place")
            recommendations.append("Consider implementing a data catalog")
        else:
            score += 0.5
            findings.append("Basic data infrastructure")
            recommendations.append("Consider data warehouse for analytics")

        # Data governance (0-2.5 points)
        has_governance = data.get("has_data_governance", False)
        has_privacy = data.get("has_privacy_compliance", False)
        evidence["governance"] = has_governance

        if has_governance and has_privacy:
            score += 2.5
            findings.append("Strong data governance")
        elif has_governance or has_privacy:
            score += 1.5
            findings.append("Partial data governance")
            recommendations.append("Strengthen data governance framework")
        else:
            score += 0.5
            findings.append("Data governance needs attention")
            recommendations.append("Establish data governance policies")

        return DimensionScore(
            name="Data Readiness",
            score=min(10.0, score),
            findings=findings,
            recommendations=recommendations,
            evidence=evidence
        )

    def _assess_technical(self, data: dict) -> DimensionScore:
        """Assess technical capability dimension."""
        score = 0.0
        findings = []
        recommendations = []
        evidence = {}

        # IT staffing (0-2.5 points)
        it_staff = data.get("it_staff_count", 0)
        total_employees = data.get("total_employees", 100)
        it_ratio = it_staff / total_employees if total_employees > 0 else 0
        evidence["it_ratio"] = it_ratio

        if it_ratio >= self.benchmarks["it_staff_ratio"]:
            score += 2.5
            findings.append(f"Strong IT staffing ({it_ratio*100:.1f}%)")
        elif it_ratio >= self.benchmarks["it_staff_ratio"] * 0.5:
            score += 1.5
            findings.append(f"Adequate IT staffing ({it_ratio*100:.1f}%)")
            recommendations.append("Consider expanding technical team")
        else:
            score += 0.5
            findings.append("Limited IT resources")
            recommendations.append("Priority: Build or acquire technical talent")

        # Cloud adoption (0-2.5 points)
        cloud_level = data.get("cloud_adoption_level", 0)
        evidence["cloud_level"] = cloud_level

        if cloud_level >= 0.7:
            score += 2.5
            findings.append("Advanced cloud adoption")
        elif cloud_level >= 0.3:
            score += 1.5
            findings.append("Moderate cloud adoption")
            recommendations.append("Accelerate cloud migration for AI workloads")
        else:
            score += 0.5
            findings.append("Limited cloud infrastructure")
            recommendations.append("Develop cloud strategy for AI scalability")

        # ML/AI experience (0-2.5 points)
        has_ml = data.get("has_ml_experience", False)
        ml_projects = data.get("ml_projects_count", 0)
        evidence["has_ml_experience"] = has_ml
        evidence["ml_projects"] = ml_projects

        if has_ml and ml_projects >= 3:
            score += 2.5
            findings.append(f"Strong ML experience ({ml_projects} projects)")
        elif has_ml:
            score += 1.5
            findings.append("Some ML experience")
            recommendations.append("Expand ML pilot projects")
        else:
            score += 0.5
            findings.append("Limited ML experience")
            recommendations.append("Start with simple ML use cases")

        # Development practices (0-2.5 points)
        has_cicd = data.get("has_cicd", False)
        has_version_control = data.get("has_version_control", True)
        evidence["has_cicd"] = has_cicd

        if has_cicd and has_version_control:
            score += 2.5
            findings.append("Modern development practices")
        elif has_version_control:
            score += 1.5
            findings.append("Basic development practices")
            recommendations.append("Implement CI/CD for ML pipelines")
        else:
            score += 0.5
            findings.append("Development practices need improvement")
            recommendations.append("Establish version control and CI/CD")

        return DimensionScore(
            name="Technical Capability",
            score=min(10.0, score),
            findings=findings,
            recommendations=recommendations,
            evidence=evidence
        )

    def _assess_process(self, data: dict) -> DimensionScore:
        """Assess process maturity dimension."""
        score = 0.0
        findings = []
        recommendations = []
        evidence = {}

        # Process documentation (0-2.5 points)
        doc_level = data.get("process_documentation_level", 0)
        evidence["documentation_level"] = doc_level

        if doc_level >= self.benchmarks["process_documentation_good"]:
            score += 2.5
            findings.append("Well-documented processes")
        elif doc_level >= self.benchmarks["process_documentation_minimum"]:
            score += 1.5
            findings.append("Moderate process documentation")
            recommendations.append("Improve process documentation for AI training")
        else:
            score += 0.5
            findings.append("Limited process documentation")
            recommendations.append("Priority: Document key processes")

        # Automation level (0-2.5 points)
        automation = data.get("automation_level", 0)
        evidence["automation_level"] = automation

        if automation >= self.benchmarks["automation_level_good"]:
            score += 2.5
            findings.append(f"High automation ({automation*100:.0f}%)")
        elif automation >= self.benchmarks["automation_level_minimum"]:
            score += 1.5
            findings.append(f"Moderate automation ({automation*100:.0f}%)")
            recommendations.append("Identify additional automation opportunities")
        else:
            score += 0.5
            findings.append("Low automation level")
            recommendations.append("Start automation before AI")

        # Continuous improvement (0-2.5 points)
        has_ci = data.get("has_continuous_improvement", False)
        has_metrics = data.get("has_process_metrics", False)
        evidence["continuous_improvement"] = has_ci

        if has_ci and has_metrics:
            score += 2.5
            findings.append("Strong continuous improvement culture")
        elif has_ci:
            score += 1.5
            findings.append("Continuous improvement practices exist")
            recommendations.append("Add metrics to improvement initiatives")
        else:
            score += 0.5
            findings.append("Limited continuous improvement")
            recommendations.append("Establish improvement framework")

        # Standard operating procedures (0-2.5 points)
        sop_coverage = data.get("sop_coverage", 0)
        evidence["sop_coverage"] = sop_coverage

        if sop_coverage >= 0.8:
            score += 2.5
            findings.append("Comprehensive SOPs")
        elif sop_coverage >= 0.5:
            score += 1.5
            findings.append("Partial SOP coverage")
            recommendations.append("Expand SOP coverage")
        else:
            score += 0.5
            findings.append("Limited SOPs")
            recommendations.append("Develop SOPs for critical processes")

        return DimensionScore(
            name="Process Maturity",
            score=min(10.0, score),
            findings=findings,
            recommendations=recommendations,
            evidence=evidence
        )

    def _assess_org(self, data: dict) -> DimensionScore:
        """Assess organizational readiness dimension."""
        score = 0.0
        findings = []
        recommendations = []
        evidence = {}

        # Leadership support (0-2.5 points)
        leadership = data.get("leadership_ai_support", 0)
        evidence["leadership_support"] = leadership

        if leadership >= 0.8:
            score += 2.5
            findings.append("Strong executive support for AI")
        elif leadership >= 0.5:
            score += 1.5
            findings.append("Moderate leadership support")
            recommendations.append("Build executive AI literacy")
        else:
            score += 0.5
            findings.append("Limited leadership buy-in")
            recommendations.append("Priority: Secure executive sponsorship")

        # Change management (0-2.5 points)
        change_rate = data.get("change_success_rate", 0)
        evidence["change_success_rate"] = change_rate

        if change_rate >= self.benchmarks["change_success_rate_good"]:
            score += 2.5
            findings.append("Strong change management track record")
        elif change_rate >= self.benchmarks["change_success_rate_minimum"]:
            score += 1.5
            findings.append("Moderate change capability")
            recommendations.append("Strengthen change management for AI projects")
        else:
            score += 0.5
            findings.append("Change management needs improvement")
            recommendations.append("Build change management capability")

        # Training and development (0-2.5 points)
        has_training_budget = data.get("training_budget_exists", False)
        has_ai_training = data.get("has_ai_training_program", False)
        evidence["training_budget"] = has_training_budget

        if has_training_budget and has_ai_training:
            score += 2.5
            findings.append("AI training programs in place")
        elif has_training_budget:
            score += 1.5
            findings.append("Training budget available")
            recommendations.append("Develop AI-specific training")
        else:
            score += 0.5
            findings.append("Limited training resources")
            recommendations.append("Allocate budget for AI training")

        # Cross-functional collaboration (0-2.5 points)
        collaboration = data.get("cross_functional_collaboration", 0)
        evidence["collaboration"] = collaboration

        if collaboration >= 0.8:
            score += 2.5
            findings.append("Strong cross-functional collaboration")
        elif collaboration >= 0.5:
            score += 1.5
            findings.append("Moderate collaboration")
            recommendations.append("Foster data science/business partnerships")
        else:
            score += 0.5
            findings.append("Siloed organization")
            recommendations.append("Break down silos for AI success")

        return DimensionScore(
            name="Organizational Readiness",
            score=min(10.0, score),
            findings=findings,
            recommendations=recommendations,
            evidence=evidence
        )

    def _score_to_level(self, score: float) -> ReadinessLevel:
        """Convert numeric score to readiness level."""
        if score < 3:
            return ReadinessLevel.NOT_READY
        elif score < 5:
            return ReadinessLevel.FOUNDATIONAL
        elif score < 7:
            return ReadinessLevel.DEVELOPING
        elif score < 9:
            return ReadinessLevel.ADVANCED
        return ReadinessLevel.LEADING

    def _generate_recommendations(
        self,
        dimensions: Dict[str, DimensionScore]
    ) -> List[str]:
        """Generate prioritized recommendations."""
        all_recommendations = []

        # Sort dimensions by score (lowest first)
        sorted_dims = sorted(
            dimensions.items(),
            key=lambda x: x[1].score
        )

        # Get recommendations from lowest-scoring dimensions first
        for dim_name, dim_score in sorted_dims:
            for rec in dim_score.recommendations:
                if rec not in all_recommendations:
                    all_recommendations.append(rec)

        return all_recommendations[:10]  # Top 10 recommendations

    def _identify_red_flags(
        self,
        dimensions: Dict[str, DimensionScore],
        company_data: dict
    ) -> List[str]:
        """Identify critical issues that could derail AI initiatives."""
        red_flags = []

        # Check for critical gaps
        for dim_name, dim_score in dimensions.items():
            if dim_score.score < 3:
                red_flags.append(
                    f"Critical gap in {dim_score.name} "
                    f"(score: {dim_score.score:.1f}/10)"
                )

        # Specific red flags
        if company_data.get("data_quality_score", 0) < 0.4:
            red_flags.append(
                "Data quality below 40% - AI models will likely fail"
            )

        if not company_data.get("has_data_governance", False):
            red_flags.append(
                "No data governance - compliance and quality risks"
            )

        if company_data.get("leadership_ai_support", 0) < 0.3:
            red_flags.append(
                "Weak leadership support - projects likely to stall"
            )

        if company_data.get("change_success_rate", 0) < 0.3:
            red_flags.append(
                "Poor change track record - adoption will be challenging"
            )

        return red_flags

    def _identify_quick_wins(
        self,
        dimensions: Dict[str, DimensionScore],
        company_data: dict
    ) -> List[str]:
        """Identify opportunities for quick wins."""
        quick_wins = []

        # High data quality + low ML experience = easy pilots
        if (company_data.get("data_quality_score", 0) >= 0.7 and
                not company_data.get("has_ml_experience", False)):
            quick_wins.append(
                "Good data quality exists - start with simple analytics/ML pilots"
            )

        # Strong processes + low automation = RPA opportunities
        if (company_data.get("process_documentation_level", 0) >= 0.7 and
                company_data.get("automation_level", 0) < 0.4):
            quick_wins.append(
                "Well-documented processes ready for automation"
            )

        # Existing data warehouse = ready for analytics
        if company_data.get("has_data_warehouse", False):
            quick_wins.append(
                "Data warehouse ready - can start predictive analytics"
            )

        # Good change management = can handle AI adoption
        if company_data.get("change_success_rate", 0) >= 0.7:
            quick_wins.append(
                "Strong change capability - can manage AI transition"
            )

        return quick_wins

    def identify_bottlenecks(self, process_data: dict) -> List[dict]:
        """
        Rank operational bottlenecks by cost and AI solution fit.

        Args:
            process_data: Dictionary containing process metrics

        Returns:
            List of bottleneck dictionaries with rankings

        Note: For detailed bottleneck analysis, use BottleneckAnalyzer.
        """
        from goodai.core.bottleneck_analyzer import BottleneckAnalyzer

        analyzer = BottleneckAnalyzer(industry=self.industry)
        return analyzer.analyze(process_data)
