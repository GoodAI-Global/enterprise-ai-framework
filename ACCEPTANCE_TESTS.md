# Acceptance Tests

Manual verification steps to confirm the framework is working correctly.

## Prerequisites

- Python 3.9+
- pip

## Test Commands

### 1. Installation

```bash
pip install -e .
```

**Expected Result:**
- Command completes without errors
- Package `goodai-enterprise` is installed

**Verification:**
```bash
python -c "import goodai; print(goodai.__version__)"
```
Should print: `0.1.0`

### 2. Run Manufacturing Demo

```bash
python examples/manufacturing_oee/run_demo.py
```

**Expected Result:**
- Command completes without errors
- JSON output is printed to stdout

**Verification:**
Output should contain:
```json
{
  "demo": "Manufacturing OEE Demo",
  "oee_metrics": {
    "availability": "...",
    "performance": "...",
    "quality": "...",
    "overall_oee": "..."
  },
  "anomaly_detection": {
    "total_anomalies": <number >= 1>,
    ...
  },
  "recommendations": [...]
}
```

### 3. Run Tests

```bash
pytest
```

**Expected Result:**
- All tests pass
- No errors or failures
- Minimum 4 test files executed

**Verification:**
```
====== X passed in Y.YYs ======
```

## Detailed Verification

### Anomaly Detection

```python
from goodai.modules import AnomalyDetector
import pandas as pd

detector = AnomalyDetector(window_size=5, threshold=2.0)
df = pd.DataFrame({"value": [100, 101, 99, 200, 98, 102, 99, 101]})
result = detector.detect(df, column="value")

print(result[result["is_anomaly"]])
```

**Expected:** Should detect the value 200 as a "high" anomaly.

### Assessment Engine

```python
from goodai.core import AssessmentEngine

engine = AssessmentEngine()
result = engine.assess_readiness({
    "data_sources": ["erp", "scada"],
    "data_quality_score": 0.7,
})

print(f"Score: {result.overall_score}")
print(f"Level: {result.overall_level.value}")
```

**Expected:** Returns valid score and level.

### Vision Connector (Headless)

```python
from goodai.connectors import VisionConnector

connector = VisionConnector()
status = connector.get_status()

print(f"Headless mode: {status['headless_mode']}")
```

**Expected:** `headless_mode: True`

### OEE Calculation

```python
from goodai.utils.metrics import OEECalculator
import pandas as pd

calc = OEECalculator()
data = pd.DataFrame({
    "uptime_minutes": [55, 58, 57],
    "planned_minutes": [60, 60, 60],
    "actual_output": [95, 98, 92],
    "theoretical_output": [100, 100, 100],
    "good_units": [94, 97, 90],
    "total_units": [95, 98, 92],
})

result = calc.calculate(data)
print(f"OEE: {result.oee:.1%}")
```

**Expected:** Returns valid OEE percentage.

## Checklist

- [ ] `pip install -e .` works
- [ ] `python examples/manufacturing_oee/run_demo.py` outputs valid JSON
- [ ] Demo detects anomalies (total_anomalies >= 1)
- [ ] Demo calculates OEE metrics
- [ ] Demo generates recommendations
- [ ] `pytest` passes all tests
- [ ] Anomaly detection is deterministic (same input = same output)
- [ ] Vision connector reports headless mode
- [ ] No insurance or aquaculture code exists

## Known Behaviors

1. **Anomaly detection requires warm-up**: First `window_size // 2` points cannot be anomalies (insufficient history).

2. **Vision connector requires OCR dependencies**: Some tests skip if tesseract is not installed.

3. **OEE can exceed 100%**: If actual output exceeds theoretical output (over-performance), OEE components can exceed 1.0.

## Troubleshooting

### Import errors

```bash
pip install -e ".[dev]"
```

### Missing tesseract

Vision connector tests may skip. Install tesseract-ocr for full testing:
```bash
# Ubuntu/Debian
apt-get install tesseract-ocr

# macOS
brew install tesseract
```

### pytest not found

```bash
pip install pytest
```
