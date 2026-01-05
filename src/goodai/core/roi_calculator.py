"""
ROI Calculator Module

Calculate and project ROI for AI implementations.
Focuses on realistic, evidence-based projections.

Good AI Philosophy: Evidence over opinions.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class ConfidenceLevel(Enum):
    """Confidence level for ROI projections."""

    LOW = "low"  # <50% confidence
    MEDIUM = "medium"  # 50-75% confidence
    HIGH = "high"  # >75% confidence


@dataclass
class ROIResult:
    """ROI calculation result."""

    project_name: str
    investment_total: float
    annual_benefit: float
    roi_percent: float
    payback_months: float
    npv: float  # Net Present Value
    confidence: ConfidenceLevel
    benefit_breakdown: Dict[str, float]
    cost_breakdown: Dict[str, float]
    assumptions: List[str]
    risks: List[str]
    sensitivity: Dict[str, float]  # How ROI changes with assumptions

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "project_name": self.project_name,
            "investment_total": self.investment_total,
            "annual_benefit": self.annual_benefit,
            "roi_percent": self.roi_percent,
            "payback_months": self.payback_months,
            "npv": self.npv,
            "confidence": self.confidence.value,
            "benefit_breakdown": self.benefit_breakdown,
            "cost_breakdown": self.cost_breakdown,
            "assumptions": self.assumptions,
            "risks": self.risks,
            "sensitivity": self.sensitivity,
        }


class ROICalculator:
    """
    Calculate ROI for AI implementation projects.

    Provides realistic, conservative projections based on:
    - Industry benchmarks
    - Implementation complexity
    - Organizational readiness

    Args:
        discount_rate: Annual discount rate for NPV (default: 0.10)
        project_years: Project horizon in years (default: 3)
        conservative_factor: Factor to reduce optimistic estimates (default: 0.7)

    Example:
        >>> calculator = ROICalculator()
        >>> result = calculator.calculate(
        ...     project_name="Predictive Maintenance",
        ...     use_case="predictive_maintenance",
        ...     investment={
        ...         "software": 50000,
        ...         "implementation": 30000,
        ...         "training": 10000,
        ...     },
        ...     baseline_metrics={
        ...         "downtime_hours_annual": 500,
        ...         "downtime_cost_per_hour": 5000,
        ...     }
        ... )
    """

    # Industry benchmarks for improvement potential
    IMPROVEMENT_BENCHMARKS = {
        "predictive_maintenance": {
            "downtime_reduction": 0.35,  # 35% reduction typical
            "maintenance_cost_reduction": 0.25,
            "equipment_life_extension": 0.20,
            "confidence_base": ConfidenceLevel.HIGH,
        },
        "quality_inspection": {
            "defect_detection_improvement": 0.40,
            "inspection_time_reduction": 0.60,
            "rework_reduction": 0.30,
            "confidence_base": ConfidenceLevel.HIGH,
        },
        "process_optimization": {
            "efficiency_improvement": 0.15,
            "waste_reduction": 0.20,
            "throughput_increase": 0.10,
            "confidence_base": ConfidenceLevel.MEDIUM,
        },
        "demand_forecasting": {
            "forecast_accuracy_improvement": 0.30,
            "inventory_reduction": 0.20,
            "stockout_reduction": 0.40,
            "confidence_base": ConfidenceLevel.MEDIUM,
        },
    }

    def __init__(
        self,
        discount_rate: float = 0.10,
        project_years: int = 3,
        conservative_factor: float = 0.7
    ):
        self.discount_rate = discount_rate
        self.project_years = project_years
        self.conservative_factor = conservative_factor

    def calculate(
        self,
        project_name: str,
        use_case: str,
        investment: Dict[str, float],
        baseline_metrics: Dict[str, float],
        readiness_score: float = 5.0,
        custom_improvements: Optional[Dict[str, float]] = None
    ) -> ROIResult:
        """
        Calculate ROI for an AI implementation project.

        Args:
            project_name: Name of the project
            use_case: Type of use case (predictive_maintenance, quality_inspection, etc.)
            investment: Cost breakdown dictionary
            baseline_metrics: Current performance metrics
            readiness_score: Organization AI readiness (1-10)
            custom_improvements: Override default improvement estimates

        Returns:
            ROIResult with complete analysis
        """
        # Get benchmarks for use case
        benchmarks = self.IMPROVEMENT_BENCHMARKS.get(
            use_case,
            self.IMPROVEMENT_BENCHMARKS["process_optimization"]
        )

        # Apply custom improvements if provided
        improvements = custom_improvements or {}
        for key in benchmarks:
            if key not in improvements and key != "confidence_base":
                improvements[key] = benchmarks[key]

        # Calculate total investment
        cost_breakdown = self._calculate_costs(investment)
        investment_total = sum(cost_breakdown.values())

        # Calculate benefits based on use case
        benefit_breakdown = self._calculate_benefits(
            use_case,
            baseline_metrics,
            improvements,
            readiness_score
        )
        annual_benefit = sum(benefit_breakdown.values())

        # Apply conservative factor
        annual_benefit_adjusted = annual_benefit * self.conservative_factor

        # Calculate ROI metrics
        roi_percent = self._calculate_roi(investment_total, annual_benefit_adjusted)
        payback_months = self._calculate_payback(investment_total, annual_benefit_adjusted)
        npv = self._calculate_npv(investment_total, annual_benefit_adjusted)

        # Determine confidence level
        confidence = self._assess_confidence(
            benchmarks.get("confidence_base", ConfidenceLevel.MEDIUM),
            readiness_score,
            len(baseline_metrics)
        )

        # Build assumptions and risks
        assumptions = self._build_assumptions(use_case, improvements)
        risks = self._identify_risks(readiness_score, investment_total)

        # Sensitivity analysis
        sensitivity = self._calculate_sensitivity(
            investment_total,
            annual_benefit_adjusted
        )

        return ROIResult(
            project_name=project_name,
            investment_total=round(investment_total, 2),
            annual_benefit=round(annual_benefit_adjusted, 2),
            roi_percent=round(roi_percent, 1),
            payback_months=round(payback_months, 1),
            npv=round(npv, 2),
            confidence=confidence,
            benefit_breakdown={k: round(v * self.conservative_factor, 2)
                              for k, v in benefit_breakdown.items()},
            cost_breakdown={k: round(v, 2) for k, v in cost_breakdown.items()},
            assumptions=assumptions,
            risks=risks,
            sensitivity=sensitivity
        )

    def _calculate_costs(self, investment: Dict[str, float]) -> Dict[str, float]:
        """Calculate total cost breakdown including hidden costs."""
        costs = investment.copy()

        # Add contingency if not specified
        subtotal = sum(investment.values())
        if "contingency" not in costs:
            costs["contingency"] = subtotal * 0.15

        # Add ongoing costs for year 1
        if "software" in investment and "annual_license" not in costs:
            costs["annual_license"] = investment["software"] * 0.20

        return costs

    def _calculate_benefits(
        self,
        use_case: str,
        baseline: Dict[str, float],
        improvements: Dict[str, float],
        readiness: float
    ) -> Dict[str, float]:
        """Calculate benefit breakdown based on use case."""
        benefits = {}

        # Readiness adjustment (lower readiness = lower realized benefits)
        readiness_factor = 0.5 + (readiness / 10) * 0.5  # 0.5 to 1.0

        if use_case == "predictive_maintenance":
            benefits = self._calc_predictive_maintenance_benefits(
                baseline, improvements, readiness_factor
            )
        elif use_case == "quality_inspection":
            benefits = self._calc_quality_benefits(
                baseline, improvements, readiness_factor
            )
        elif use_case == "demand_forecasting":
            benefits = self._calc_forecasting_benefits(
                baseline, improvements, readiness_factor
            )
        else:
            benefits = self._calc_generic_benefits(
                baseline, improvements, readiness_factor
            )

        return benefits

    def _calc_predictive_maintenance_benefits(
        self,
        baseline: Dict[str, float],
        improvements: Dict[str, float],
        readiness_factor: float
    ) -> Dict[str, float]:
        """Calculate predictive maintenance benefits."""
        benefits = {}

        # Downtime reduction
        downtime_hours = baseline.get("downtime_hours_annual", 0)
        downtime_cost = baseline.get("downtime_cost_per_hour", 5000)
        reduction = improvements.get("downtime_reduction", 0.35)

        benefits["downtime_savings"] = (
            downtime_hours * downtime_cost * reduction * readiness_factor
        )

        # Maintenance cost reduction
        maintenance_cost = baseline.get("maintenance_cost_annual", 0)
        maint_reduction = improvements.get("maintenance_cost_reduction", 0.25)

        benefits["maintenance_savings"] = (
            maintenance_cost * maint_reduction * readiness_factor
        )

        # Equipment life extension (avoided capital)
        equipment_value = baseline.get("equipment_replacement_value", 0)
        life_extension = improvements.get("equipment_life_extension", 0.20)

        # Annualized benefit of life extension
        benefits["equipment_life_savings"] = (
            equipment_value * life_extension / 10 * readiness_factor
        )

        return benefits

    def _calc_quality_benefits(
        self,
        baseline: Dict[str, float],
        improvements: Dict[str, float],
        readiness_factor: float
    ) -> Dict[str, float]:
        """Calculate quality inspection benefits."""
        benefits = {}

        # Defect detection improvement
        defect_cost = baseline.get("annual_defect_cost", 0)
        detection_improvement = improvements.get("defect_detection_improvement", 0.40)

        benefits["defect_reduction"] = (
            defect_cost * detection_improvement * readiness_factor
        )

        # Inspection time savings
        inspection_hours = baseline.get("inspection_hours_annual", 0)
        labor_cost = baseline.get("labor_cost_per_hour", 50)
        time_reduction = improvements.get("inspection_time_reduction", 0.60)

        benefits["inspection_savings"] = (
            inspection_hours * labor_cost * time_reduction * readiness_factor
        )

        # Rework reduction
        rework_cost = baseline.get("annual_rework_cost", 0)
        rework_reduction = improvements.get("rework_reduction", 0.30)

        benefits["rework_savings"] = (
            rework_cost * rework_reduction * readiness_factor
        )

        return benefits

    def _calc_forecasting_benefits(
        self,
        baseline: Dict[str, float],
        improvements: Dict[str, float],
        readiness_factor: float
    ) -> Dict[str, float]:
        """Calculate demand forecasting benefits."""
        benefits = {}

        # Inventory reduction
        inventory_value = baseline.get("average_inventory_value", 0)
        carrying_cost = baseline.get("inventory_carrying_cost_percent", 0.25)
        inv_reduction = improvements.get("inventory_reduction", 0.20)

        benefits["inventory_savings"] = (
            inventory_value * carrying_cost * inv_reduction * readiness_factor
        )

        # Stockout reduction
        stockout_cost = baseline.get("annual_stockout_cost", 0)
        stockout_reduction = improvements.get("stockout_reduction", 0.40)

        benefits["stockout_savings"] = (
            stockout_cost * stockout_reduction * readiness_factor
        )

        return benefits

    def _calc_generic_benefits(
        self,
        baseline: Dict[str, float],
        improvements: Dict[str, float],
        readiness_factor: float
    ) -> Dict[str, float]:
        """Calculate generic process optimization benefits."""
        benefits = {}

        # Efficiency improvement
        labor_cost = baseline.get("annual_labor_cost", 0)
        efficiency = improvements.get("efficiency_improvement", 0.15)

        benefits["efficiency_savings"] = (
            labor_cost * efficiency * readiness_factor
        )

        # Waste reduction
        material_cost = baseline.get("annual_material_cost", 0)
        waste_rate = baseline.get("waste_rate", 0.05)
        waste_reduction = improvements.get("waste_reduction", 0.20)

        benefits["waste_savings"] = (
            material_cost * waste_rate * waste_reduction * readiness_factor
        )

        return benefits

    def _calculate_roi(self, investment: float, annual_benefit: float) -> float:
        """Calculate ROI percentage."""
        if investment <= 0:
            return 0
        # Simple ROI over project years
        total_benefit = annual_benefit * self.project_years
        return ((total_benefit - investment) / investment) * 100

    def _calculate_payback(self, investment: float, annual_benefit: float) -> float:
        """Calculate payback period in months."""
        if annual_benefit <= 0:
            return float("inf")
        return (investment / annual_benefit) * 12

    def _calculate_npv(self, investment: float, annual_benefit: float) -> float:
        """Calculate Net Present Value."""
        npv = -investment

        for year in range(1, self.project_years + 1):
            npv += annual_benefit / ((1 + self.discount_rate) ** year)

        return npv

    def _assess_confidence(
        self,
        base_confidence: ConfidenceLevel,
        readiness: float,
        data_points: int
    ) -> ConfidenceLevel:
        """Assess confidence level based on multiple factors."""
        # Start with base confidence score
        confidence_scores = {
            ConfidenceLevel.LOW: 1,
            ConfidenceLevel.MEDIUM: 2,
            ConfidenceLevel.HIGH: 3,
        }
        score = confidence_scores.get(base_confidence, 2)

        # Adjust for readiness
        if readiness < 4:
            score -= 1
        elif readiness >= 7:
            score += 0.5

        # Adjust for data quality
        if data_points < 3:
            score -= 0.5
        elif data_points >= 6:
            score += 0.5

        # Convert back to level
        if score >= 2.5:
            return ConfidenceLevel.HIGH
        elif score >= 1.5:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW

    def _build_assumptions(
        self,
        use_case: str,
        improvements: Dict[str, float]
    ) -> List[str]:
        """Build list of assumptions."""
        assumptions = [
            f"Project timeline: {self.project_years} years",
            f"Discount rate: {self.discount_rate*100:.0f}%",
            f"Conservative factor applied: {self.conservative_factor:.0%}",
        ]

        for key, value in improvements.items():
            if key != "confidence_base" and isinstance(value, (int, float)):
                readable_key = key.replace("_", " ").title()
                assumptions.append(f"{readable_key}: {value*100:.0f}%")

        return assumptions

    def _identify_risks(
        self,
        readiness: float,
        investment: float
    ) -> List[str]:
        """Identify project risks."""
        risks = []

        if readiness < 5:
            risks.append("Low organizational readiness may delay benefits realization")

        if readiness < 3:
            risks.append("CRITICAL: Organization not ready for AI implementation")

        if investment > 500000:
            risks.append("Large investment requires strong change management")

        risks.extend([
            "Benefits dependent on user adoption",
            "Technology integration complexity",
            "Data quality issues may reduce effectiveness",
        ])

        return risks

    def _calculate_sensitivity(
        self,
        investment: float,
        annual_benefit: float
    ) -> Dict[str, float]:
        """Calculate sensitivity of ROI to key variables."""
        return {
            "roi_if_benefits_minus_20pct": round(
                self._calculate_roi(investment, annual_benefit * 0.8), 1
            ),
            "roi_if_benefits_plus_20pct": round(
                self._calculate_roi(investment, annual_benefit * 1.2), 1
            ),
            "roi_if_costs_plus_20pct": round(
                self._calculate_roi(investment * 1.2, annual_benefit), 1
            ),
            "roi_if_delayed_6_months": round(
                self._calculate_roi(investment, annual_benefit * 0.85), 1
            ),
        }

    def quick_estimate(
        self,
        use_case: str,
        investment: float,
        annual_baseline_cost: float
    ) -> dict:
        """
        Quick ROI estimate without detailed inputs.

        Args:
            use_case: Type of use case
            investment: Total investment amount
            annual_baseline_cost: Current annual cost in problem area

        Returns:
            Quick estimate dictionary
        """
        benchmarks = self.IMPROVEMENT_BENCHMARKS.get(
            use_case,
            self.IMPROVEMENT_BENCHMARKS["process_optimization"]
        )

        # Get primary improvement metric
        improvement = 0.15  # Default
        for key, value in benchmarks.items():
            if key != "confidence_base" and isinstance(value, (int, float)):
                improvement = max(improvement, value)
                break

        # Apply conservative factor
        estimated_benefit = annual_baseline_cost * improvement * self.conservative_factor

        roi = self._calculate_roi(investment, estimated_benefit)
        payback = self._calculate_payback(investment, estimated_benefit)

        return {
            "estimated_annual_benefit": round(estimated_benefit, 2),
            "estimated_roi_percent": round(roi, 1),
            "estimated_payback_months": round(payback, 1),
            "improvement_assumption": f"{improvement*100:.0f}%",
            "confidence": benchmarks.get("confidence_base", ConfidenceLevel.MEDIUM).value,
            "note": "Quick estimate - detailed analysis recommended",
        }
