# Good AI Enterprise Framework

A production-ready framework for enterprise AI implementations.

**Leverage, not lore. Evidence over opinions. Augment first.**

## 90 Second Quickstart

```bash
# Install
pip install -e .

# Run the manufacturing demo
python examples/manufacturing_oee/run_demo.py

# Run tests
pytest
```

That's it. You'll see JSON output with OEE metrics, detected anomalies, and recommendations.

## What This Is

Good AI Enterprise Framework is a Python toolkit for implementing AI in manufacturing environments. It focuses on:

- **Simple, explainable methods** - Rolling z-score for anomaly detection, not black-box ML
- **Deterministic results** - Same data always produces same output
- **Headless operation** - Works in CI/CD pipelines and Docker containers
- **Production-ready** - Real logic, no stubs, comprehensive tests

## Core Components

### Anomaly Detection

Simple statistical anomaly detection using rolling z-score:

```python
from goodai.modules import AnomalyDetector
import pandas as pd

# Create detector
detector = AnomalyDetector(window_size=20, threshold=2.5)

# Load your data
df = pd.read_csv("production_data.csv")

# Detect anomalies
results = detector.detect(df, column="production_rate")

# Get human-readable explanations
for _, row in results[results["is_anomaly"]].iterrows():
    print(detector.explain_anomaly(row))
```

### Assessment Engine

Evaluate AI readiness across four dimensions:

```python
from goodai.core import AssessmentEngine

engine = AssessmentEngine(industry="manufacturing")

company_data = {
    "data_sources": ["erp", "scada", "mes"],
    "data_quality_score": 0.75,
    "has_data_warehouse": True,
    "it_staff_count": 5,
    "total_employees": 200,
    # ... more attributes
}

result = engine.assess_readiness(company_data)

print(f"Overall Score: {result.overall_score}/10")
print(f"Level: {result.overall_level.value}")
print(f"Red Flags: {result.red_flags}")
print(f"Quick Wins: {result.quick_wins}")
```

### Vision Connector

Non-invasive data capture from visual sources (headless-safe):

```python
from goodai.connectors import VisionConnector

connector = VisionConnector()

# Define regions to read
regions = {
    "temperature": {"x": 100, "y": 200, "w": 80, "h": 30, "type": "number"},
    "pressure": {"x": 100, "y": 250, "w": 80, "h": 30, "type": "number"},
}

# Extract values
values = connector.read_from_image("dashboard.png", regions)
print(values)  # {"temperature": 72.5, "pressure": 14.7}
```

### Predictive Maintenance

Statistical degradation analysis for equipment:

```python
from goodai.modules import PredictiveMaintenance
import pandas as pd

pm = PredictiveMaintenance()

# Load sensor data
sensor_data = pd.DataFrame({
    "timestamp": [...],
    "temperature": [...],
    "vibration": [...],
})

# Assess equipment health
health = pm.assess_health("pump-001", sensor_data)

print(f"Health Score: {health.health_score}")
print(f"Status: {health.status.value}")

for alert in health.alerts:
    print(f"Alert: {alert.message}")
```

## Manufacturing Example Walkthrough

The `examples/manufacturing_oee/` directory contains a complete working example:

1. **sample_data.csv** - Production data with intentional anomalies
2. **config.json** - Configuration for anomaly detection and OEE targets
3. **run_demo.py** - Complete demo that:
   - Loads production data
   - Detects anomalies using rolling z-score
   - Calculates OEE (Availability × Performance × Quality)
   - Generates actionable recommendations

```bash
python examples/manufacturing_oee/run_demo.py
```

Output:
```json
{
  "oee_metrics": {
    "availability": "93.5%",
    "performance": "89.2%",
    "quality": "98.7%",
    "overall_oee": "82.3%"
  },
  "anomaly_detection": {
    "total_anomalies": 4,
    "details": [...]
  },
  "recommendations": [
    "Performance at 89.2%. Look for speed losses...",
    "Detected 4 production drops..."
  ]
}
```

## Good AI Methodology

Our implementation approach:

```
Identify → Pilot → Instrument → Learn → Scale/Sunset
```

1. **Identify**: Find bottlenecks with highest cost and best AI fit
2. **Pilot**: Small-scale proof of concept with measurable goals
3. **Instrument**: Add data collection and monitoring
4. **Learn**: Analyze results, iterate on approach
5. **Scale/Sunset**: Expand successful pilots or retire failed ones

## Architecture

```
goodai/
├── core/
│   ├── assessment_engine.py  # AI readiness assessment
│   ├── bottleneck_analyzer.py # Operational bottleneck detection
│   └── roi_calculator.py      # ROI projections
├── connectors/
│   ├── vision_connector.py    # OCR from images (headless-safe)
│   └── data_connector.py      # CSV, JSON, database connections
├── modules/
│   ├── anomaly_detection.py   # Rolling z-score detection
│   └── predictive_maintenance.py # Equipment health monitoring
└── utils/
    ├── metrics.py             # OEE and custom metrics
    └── reporting.py           # JSON, Markdown, HTML reports
```

## Key Design Decisions

### Why Rolling Z-Score for Anomaly Detection?

- **Simple**: Easy to understand and explain to stakeholders
- **Deterministic**: Same data → same results
- **Fast**: O(n) complexity, works on large datasets
- **No training**: Works immediately, adapts to drift naturally

### Why Headless Vision Connector?

- Works in CI/CD pipelines
- Runs in Docker containers
- No display server required
- Uses `opencv-python-headless`

### Why Manufacturing First?

Manufacturing has:
- Clear, measurable KPIs (OEE)
- Rich sensor data
- High cost of downtime
- Strong ROI for AI implementations

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=goodai
```

## License

MIT License - see [LICENSE](LICENSE)

## About Good AI

Good AI is an enterprise AI consultancy focused on practical implementations that deliver measurable results.

**Our principles:**
- Leverage, not lore
- Evidence over opinions
- Augment first
- Non-invasive by default

---

Built with pragmatism by Good AI.
