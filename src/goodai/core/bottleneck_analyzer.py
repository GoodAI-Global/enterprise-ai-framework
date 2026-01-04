"""
Bottleneck Analyzer Module

Identifies and ranks operational bottlenecks by cost impact and AI solution fit.
Focuses on manufacturing operations.

Good AI Philosophy: Leverage, not lore.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
import math


class BottleneckSeverity(Enum):
    """Bottleneck severity classification."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AISolutionFit(Enum):
    """How well AI can address the bottleneck."""

    POOR = "poor"  # AI unlikely to help
    MODERATE = "moderate"  # AI might help with effort
    GOOD = "good"  # AI well-suited
    EXCELLENT = "excellent"  # Classic AI use case


@dataclass
class Bottleneck:
    """Identified operational bottleneck."""

    name: str
    category: str
    severity: BottleneckSeverity
    estimated_cost_impact: float  # Annual cost in currency
    ai_solution_fit: AISolutionFit
    ai_solutions: List[str]
    root_causes: List[str]
    metrics_affected: List[str]
    priority_score: float  # 0-100, higher = more urgent
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "category": self.category,
            "severity": self.severity.value,
            "estimated_cost_impact": self.estimated_cost_impact,
            "ai_solution_fit": self.ai_solution_fit.value,
            "ai_solutions": self.ai_solutions,
            "root_causes": self.root_causes,
            "metrics_affected": self.metrics_affected,
            "priority_score": self.priority_score,
            "evidence": self.evidence,
        }


class BottleneckAnalyzer:
    """
    Analyze operations to identify and rank bottlenecks.

    Evaluates processes across multiple dimensions:
    - Throughput bottlenecks
    - Quality issues
    - Downtime/availability problems
    - Resource inefficiencies

    Args:
        industry: Target industry for analysis patterns

    Example:
        >>> analyzer = BottleneckAnalyzer(industry="manufacturing")
        >>> process_data = {
        ...     "oee": 0.65,
        ...     "availability": 0.85,
        ...     "performance": 0.80,
        ...     "quality": 0.95,
        ...     "downtime_hours_monthly": 40,
        ...     "defect_rate": 0.05,
        ...     "cycle_time_variance": 0.15,
        ... }
        >>> bottlenecks = analyzer.analyze(process_data)
    """

    # Industry benchmarks
    BENCHMARKS = {
        "manufacturing": {
            "oee_target": 0.85,
            "availability_target": 0.90,
            "performance_target": 0.95,
            "quality_target": 0.99,
            "downtime_threshold_hours": 20,  # per month
            "defect_rate_threshold": 0.02,
            "cycle_time_variance_threshold": 0.10,
            "changeover_time_threshold_minutes": 30,
            "inventory_turns_minimum": 6,
        }
    }

    # Cost estimation factors (per hour/unit)
    COST_FACTORS = {
        "downtime_cost_per_hour": 5000,
        "defect_cost_per_unit": 50,
        "overtime_cost_multiplier": 1.5,
        "inventory_carrying_cost_percent": 0.25,
    }

    def __init__(
        self,
        industry: str = "manufacturing",
        cost_factors: Optional[Dict[str, float]] = None
    ):
        self.industry = industry
        self.benchmarks = self.BENCHMARKS.get(
            industry,
            self.BENCHMARKS["manufacturing"]
        )
        self.cost_factors = cost_factors or self.COST_FACTORS.copy()

    def analyze(self, process_data: dict) -> List[Bottleneck]:
        """
        Analyze process data to identify bottlenecks.

        Args:
            process_data: Dictionary with process metrics:
                - oee: Overall Equipment Effectiveness (0-1)
                - availability: Equipment availability (0-1)
                - performance: Equipment performance (0-1)
                - quality: Quality rate (0-1)
                - downtime_hours_monthly: Unplanned downtime hours
                - defect_rate: Defect rate (0-1)
                - cycle_time_variance: Variance in cycle time (0-1)
                - changeover_time_minutes: Average changeover time
                - production_volume_monthly: Units produced per month
                - planned_volume_monthly: Target units per month

        Returns:
            List of Bottleneck objects, sorted by priority
        """
        bottlenecks = []

        # Check availability/downtime
        bottleneck = self._analyze_availability(process_data)
        if bottleneck:
            bottlenecks.append(bottleneck)

        # Check performance/speed
        bottleneck = self._analyze_performance(process_data)
        if bottleneck:
            bottlenecks.append(bottleneck)

        # Check quality
        bottleneck = self._analyze_quality(process_data)
        if bottleneck:
            bottlenecks.append(bottleneck)

        # Check cycle time variance
        bottleneck = self._analyze_variability(process_data)
        if bottleneck:
            bottlenecks.append(bottleneck)

        # Check changeover efficiency
        bottleneck = self._analyze_changeover(process_data)
        if bottleneck:
            bottlenecks.append(bottleneck)

        # Sort by priority score (descending)
        bottlenecks.sort(key=lambda b: b.priority_score, reverse=True)

        return bottlenecks

    def _analyze_availability(self, data: dict) -> Optional[Bottleneck]:
        """Analyze availability/downtime bottlenecks."""
        availability = data.get("availability", 1.0)
        downtime_hours = data.get("downtime_hours_monthly", 0)
        target = self.benchmarks["availability_target"]
        threshold = self.benchmarks["downtime_threshold_hours"]

        # Check if there's a problem
        if availability >= target and downtime_hours <= threshold:
            return None

        # Calculate severity
        gap = target - availability
        if gap >= 0.15 or downtime_hours >= threshold * 2:
            severity = BottleneckSeverity.CRITICAL
        elif gap >= 0.10 or downtime_hours >= threshold * 1.5:
            severity = BottleneckSeverity.HIGH
        elif gap >= 0.05:
            severity = BottleneckSeverity.MEDIUM
        else:
            severity = BottleneckSeverity.LOW

        # Estimate cost impact
        cost_per_hour = self.cost_factors["downtime_cost_per_hour"]
        annual_cost = downtime_hours * 12 * cost_per_hour

        # Determine AI solution fit
        ai_fit = AISolutionFit.EXCELLENT  # Predictive maintenance is classic AI

        # Calculate priority score
        priority = self._calculate_priority(
            cost_impact=annual_cost,
            severity=severity,
            ai_fit=ai_fit
        )

        return Bottleneck(
            name="Equipment Downtime",
            category="availability",
            severity=severity,
            estimated_cost_impact=annual_cost,
            ai_solution_fit=ai_fit,
            ai_solutions=[
                "Predictive maintenance using sensor data",
                "Anomaly detection for early warning",
                "Root cause analysis automation",
                "Maintenance scheduling optimization"
            ],
            root_causes=[
                "Unplanned equipment failures",
                "Lack of predictive maintenance",
                "Reactive maintenance culture",
                "Insufficient spare parts inventory"
            ],
            metrics_affected=["OEE", "Availability", "Throughput"],
            priority_score=priority,
            evidence={
                "availability": availability,
                "downtime_hours_monthly": downtime_hours,
                "target": target,
                "gap": gap,
            }
        )

    def _analyze_performance(self, data: dict) -> Optional[Bottleneck]:
        """Analyze performance/speed bottlenecks."""
        performance = data.get("performance", 1.0)
        target = self.benchmarks["performance_target"]

        if performance >= target:
            return None

        gap = target - performance

        if gap >= 0.20:
            severity = BottleneckSeverity.CRITICAL
        elif gap >= 0.15:
            severity = BottleneckSeverity.HIGH
        elif gap >= 0.10:
            severity = BottleneckSeverity.MEDIUM
        else:
            severity = BottleneckSeverity.LOW

        # Estimate cost (lost production)
        planned_volume = data.get("planned_volume_monthly", 10000)
        actual_volume = data.get("production_volume_monthly", planned_volume * performance)
        lost_units = (planned_volume - actual_volume) * 12
        unit_margin = data.get("unit_margin", 10)
        annual_cost = lost_units * unit_margin

        ai_fit = AISolutionFit.GOOD  # Process optimization is suitable

        priority = self._calculate_priority(
            cost_impact=annual_cost,
            severity=severity,
            ai_fit=ai_fit
        )

        return Bottleneck(
            name="Speed/Performance Loss",
            category="performance",
            severity=severity,
            estimated_cost_impact=annual_cost,
            ai_solution_fit=ai_fit,
            ai_solutions=[
                "Process parameter optimization",
                "Bottleneck detection and analysis",
                "Real-time performance monitoring",
                "Operator guidance systems"
            ],
            root_causes=[
                "Suboptimal machine settings",
                "Minor stoppages and idling",
                "Operator skill variance",
                "Material quality variations"
            ],
            metrics_affected=["OEE", "Performance", "Throughput", "Unit Cost"],
            priority_score=priority,
            evidence={
                "performance": performance,
                "target": target,
                "gap": gap,
                "estimated_lost_units_annual": lost_units,
            }
        )

    def _analyze_quality(self, data: dict) -> Optional[Bottleneck]:
        """Analyze quality bottlenecks."""
        quality = data.get("quality", 1.0)
        defect_rate = data.get("defect_rate", 1 - quality)
        target = self.benchmarks["quality_target"]
        threshold = self.benchmarks["defect_rate_threshold"]

        if quality >= target and defect_rate <= threshold:
            return None

        if defect_rate >= threshold * 3:
            severity = BottleneckSeverity.CRITICAL
        elif defect_rate >= threshold * 2:
            severity = BottleneckSeverity.HIGH
        elif defect_rate >= threshold:
            severity = BottleneckSeverity.MEDIUM
        else:
            severity = BottleneckSeverity.LOW

        # Estimate cost
        volume = data.get("production_volume_monthly", 10000)
        defect_cost = self.cost_factors["defect_cost_per_unit"]
        annual_cost = volume * defect_rate * 12 * defect_cost

        ai_fit = AISolutionFit.EXCELLENT  # Vision inspection is excellent fit

        priority = self._calculate_priority(
            cost_impact=annual_cost,
            severity=severity,
            ai_fit=ai_fit
        )

        return Bottleneck(
            name="Quality Defects",
            category="quality",
            severity=severity,
            estimated_cost_impact=annual_cost,
            ai_solution_fit=ai_fit,
            ai_solutions=[
                "Computer vision inspection",
                "Statistical process control with AI",
                "Defect pattern recognition",
                "Predictive quality models"
            ],
            root_causes=[
                "Inspection gaps",
                "Process variations",
                "Material inconsistencies",
                "Equipment wear"
            ],
            metrics_affected=["OEE", "Quality", "Customer Satisfaction", "Rework Cost"],
            priority_score=priority,
            evidence={
                "quality": quality,
                "defect_rate": defect_rate,
                "target": target,
            }
        )

    def _analyze_variability(self, data: dict) -> Optional[Bottleneck]:
        """Analyze process variability bottlenecks."""
        variance = data.get("cycle_time_variance", 0)
        threshold = self.benchmarks["cycle_time_variance_threshold"]

        if variance <= threshold:
            return None

        if variance >= threshold * 3:
            severity = BottleneckSeverity.HIGH
        elif variance >= threshold * 2:
            severity = BottleneckSeverity.MEDIUM
        else:
            severity = BottleneckSeverity.LOW

        # Variability has indirect costs
        annual_cost = data.get("production_volume_monthly", 10000) * 12 * variance * 5

        ai_fit = AISolutionFit.GOOD

        priority = self._calculate_priority(
            cost_impact=annual_cost,
            severity=severity,
            ai_fit=ai_fit
        )

        return Bottleneck(
            name="Process Variability",
            category="variability",
            severity=severity,
            estimated_cost_impact=annual_cost,
            ai_solution_fit=ai_fit,
            ai_solutions=[
                "Process optimization algorithms",
                "Real-time parameter adjustment",
                "Operator decision support",
                "Adaptive control systems"
            ],
            root_causes=[
                "Inconsistent operator procedures",
                "Equipment calibration drift",
                "Material variation",
                "Environmental factors"
            ],
            metrics_affected=["Cycle Time", "Predictability", "Scheduling Accuracy"],
            priority_score=priority,
            evidence={
                "cycle_time_variance": variance,
                "threshold": threshold,
            }
        )

    def _analyze_changeover(self, data: dict) -> Optional[Bottleneck]:
        """Analyze changeover/setup bottlenecks."""
        changeover_time = data.get("changeover_time_minutes", 0)
        threshold = self.benchmarks["changeover_time_threshold_minutes"]

        if changeover_time <= threshold:
            return None

        if changeover_time >= threshold * 3:
            severity = BottleneckSeverity.HIGH
        elif changeover_time >= threshold * 2:
            severity = BottleneckSeverity.MEDIUM
        else:
            severity = BottleneckSeverity.LOW

        # Estimate cost
        changeovers_monthly = data.get("changeovers_monthly", 20)
        hours_lost = (changeover_time - threshold) / 60 * changeovers_monthly
        cost_per_hour = self.cost_factors["downtime_cost_per_hour"]
        annual_cost = hours_lost * 12 * cost_per_hour

        ai_fit = AISolutionFit.MODERATE  # More suited for SMED/process improvement

        priority = self._calculate_priority(
            cost_impact=annual_cost,
            severity=severity,
            ai_fit=ai_fit
        )

        return Bottleneck(
            name="Changeover Inefficiency",
            category="changeover",
            severity=severity,
            estimated_cost_impact=annual_cost,
            ai_solution_fit=ai_fit,
            ai_solutions=[
                "Changeover sequence optimization",
                "Setup verification automation",
                "Digital work instructions",
                "Parameter preloading"
            ],
            root_causes=[
                "Complex changeover procedures",
                "Missing or incomplete documentation",
                "Skill-dependent setups",
                "Equipment design limitations"
            ],
            metrics_affected=["Changeover Time", "Availability", "Flexibility"],
            priority_score=priority,
            evidence={
                "changeover_time_minutes": changeover_time,
                "threshold": threshold,
                "changeovers_monthly": changeovers_monthly,
            }
        )

    def _calculate_priority(
        self,
        cost_impact: float,
        severity: BottleneckSeverity,
        ai_fit: AISolutionFit
    ) -> float:
        """Calculate priority score (0-100)."""
        # Cost score (0-40 points)
        # Logarithmic scale: $100K = 20, $1M = 30, $10M = 40
        if cost_impact <= 0:
            cost_score = 0
        else:
            cost_score = min(40, 10 * math.log10(cost_impact / 10000 + 1))

        # Severity score (0-30 points)
        severity_scores = {
            BottleneckSeverity.LOW: 5,
            BottleneckSeverity.MEDIUM: 15,
            BottleneckSeverity.HIGH: 25,
            BottleneckSeverity.CRITICAL: 30,
        }
        severity_score = severity_scores.get(severity, 0)

        # AI fit score (0-30 points)
        ai_scores = {
            AISolutionFit.POOR: 5,
            AISolutionFit.MODERATE: 15,
            AISolutionFit.GOOD: 25,
            AISolutionFit.EXCELLENT: 30,
        }
        ai_score = ai_scores.get(ai_fit, 0)

        return round(cost_score + severity_score + ai_score, 1)

    def get_summary(self, bottlenecks: List[Bottleneck]) -> dict:
        """Generate summary of bottleneck analysis."""
        if not bottlenecks:
            return {
                "total_bottlenecks": 0,
                "total_cost_impact": 0,
                "top_priority": None,
                "by_category": {},
                "by_severity": {},
            }

        total_cost = sum(b.estimated_cost_impact for b in bottlenecks)

        by_category = {}
        for b in bottlenecks:
            by_category[b.category] = by_category.get(b.category, 0) + 1

        by_severity = {}
        for b in bottlenecks:
            severity_name = b.severity.value
            by_severity[severity_name] = by_severity.get(severity_name, 0) + 1

        return {
            "total_bottlenecks": len(bottlenecks),
            "total_cost_impact": total_cost,
            "top_priority": bottlenecks[0].name if bottlenecks else None,
            "by_category": by_category,
            "by_severity": by_severity,
            "top_ai_opportunities": [
                b.name for b in bottlenecks
                if b.ai_solution_fit in [AISolutionFit.GOOD, AISolutionFit.EXCELLENT]
            ][:3],
        }
