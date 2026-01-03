# Quickstart Guide

Get started with the Good AI Enterprise Framework in under 5 minutes.

## Installation

```bash
pip install -e .
```

## Your First Analysis

### 1. Anomaly Detection

```python
from goodai.modules import AnomalyDetector
import pandas as pd

# Load your data
df = pd.read_csv("your_data.csv")

# Create detector with default settings
detector = AnomalyDetector()

# Detect anomalies in a specific column
results = detector.detect(df, column="production_rate")

# See what was found
print(f"Found {results['is_anomaly'].sum()} anomalies")

# Get explanations
for _, row in results[results["is_anomaly"]].iterrows():
    print(detector.explain_anomaly(row))
```

### 2. OEE Calculation

```python
from goodai.utils.metrics import OEECalculator
import pandas as pd

# Your production data
data = pd.DataFrame({
    "uptime_minutes": [55, 58, 57, 54],
    "planned_minutes": [60, 60, 60, 60],
    "actual_output": [95, 98, 92, 88],
    "theoretical_output": [100, 100, 100, 100],
    "good_units": [94, 97, 90, 86],
    "total_units": [95, 98, 92, 88],
})

# Calculate OEE
calc = OEECalculator()
result = calc.calculate(data)

print(f"Availability: {result.availability:.1%}")
print(f"Performance: {result.performance:.1%}")
print(f"Quality: {result.quality:.1%}")
print(f"OEE: {result.oee:.1%}")
```

### 3. AI Readiness Assessment

```python
from goodai.core import AssessmentEngine

engine = AssessmentEngine(industry="manufacturing")

# Describe your organization
company_data = {
    "data_sources": ["erp", "scada", "mes"],
    "data_quality_score": 0.75,
    "has_data_warehouse": True,
    "has_data_catalog": False,
    "it_staff_count": 5,
    "total_employees": 200,
    "cloud_adoption_level": 0.5,
    "has_ml_experience": False,
    "process_documentation_level": 0.6,
    "automation_level": 0.4,
    "leadership_ai_support": 0.7,
    "change_success_rate": 0.6,
    "training_budget_exists": True,
}

# Get assessment
result = engine.assess_readiness(company_data)

print(f"Overall Score: {result.overall_score}/10")
print(f"Readiness Level: {result.overall_level.value}")

print("\nTop Recommendations:")
for rec in result.recommendations[:3]:
    print(f"  - {rec}")

if result.red_flags:
    print("\nRed Flags:")
    for flag in result.red_flags:
        print(f"  ! {flag}")

if result.quick_wins:
    print("\nQuick Wins:")
    for win in result.quick_wins:
        print(f"  + {win}")
```

## Run the Demo

The manufacturing demo shows all components working together:

```bash
python examples/manufacturing_oee/run_demo.py
```

This demonstrates:
- Loading production data from CSV
- Detecting anomalies in production rate
- Calculating OEE metrics
- Generating recommendations

## Next Steps

1. **Customize thresholds**: Adjust anomaly detection sensitivity
2. **Add your data**: Replace sample data with real production data
3. **Generate reports**: Use the reporting module for stakeholder presentations
4. **Assess bottlenecks**: Use the bottleneck analyzer for improvement priorities

See [ARCHITECTURE.md](../ARCHITECTURE.md) for detailed component documentation.
