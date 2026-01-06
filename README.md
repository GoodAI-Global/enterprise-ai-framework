# Good AI Enterprise Framework

[![CI](https://github.com/GoodAI-Global/enterprise-ai-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/GoodAI-Global/enterprise-ai-framework/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Enterprise AI implementation toolkit for manufacturing environments.

## What This Is

A Python framework providing:

- **Anomaly Detection** - Rolling z-score statistical detection (deterministic, explainable)
- **AI Readiness Assessment** - Evaluate organizational readiness across 4 dimensions
- **OEE Metrics** - Overall Equipment Effectiveness calculations
- **MLOps Primitives** - Model registry, A/B testing, feedback loops, explainability
- **Enterprise Security** - Multi-tenancy, RBAC, audit logging
- **Infrastructure** - Caching, async utilities, schema validation

## What This Isn't

- **Not a plug-and-play AI solution** - Requires integration work for your environment
- **Not ML model training** - Focuses on deployment, monitoring, and operations
- **Not battle-tested at scale** - v0.1.0, use with appropriate caution
- **Not a replacement for domain expertise** - Tools to augment, not replace, engineers

## Quickstart (< 5 minutes)

```bash
# Clone and install
git clone https://github.com/GoodAI-Global/enterprise-ai-framework.git
cd enterprise-ai-framework
pip install -e ".[dev]"

# Run tests (300 tests, ~10 seconds)
pytest

# Run the manufacturing demo
python examples/manufacturing_oee/run_demo.py
```

### Basic Usage

```python
from goodai.modules import AnomalyDetector
import pandas as pd

# Detect anomalies in production data
detector = AnomalyDetector(window_size=20, threshold=2.5)
df = pd.read_csv("production_data.csv")
results = detector.detect(df, column="production_rate")

# Results are deterministic - same input always produces same output
print(f"Found {results['is_anomaly'].sum()} anomalies")
```

## Project Status

| Component | Status | Tests |
|-----------|--------|-------|
| Core (Assessment, ROI, Bottleneck) | Stable | Yes |
| Anomaly Detection | Stable | Yes |
| MLOps (Registry, A/B, Feedback) | Beta | Yes |
| Enterprise Security | Beta | Yes |
| Infrastructure (Cache, Async, Validation) | Beta | Yes |
| REST API | Beta | Yes |

**Test Coverage**: 73% (300 tests passing)

## Architecture

```
src/goodai/
├── api/           # FastAPI REST endpoints
├── config/        # Configuration management
├── connectors/    # Data source connectors (CSV, vision OCR)
├── core/          # Assessment engine, ROI calculator, bottleneck analyzer
├── infrastructure/# Caching, async utilities, schema validation
├── mlops/         # Model registry, A/B testing, feedback, explainability
├── modules/       # Anomaly detection, predictive maintenance
├── monitoring/    # Structured logging, health checks
├── security/      # Multi-tenancy, RBAC, audit logging
└── utils/         # Metrics (OEE), reporting
```

## Development

```bash
# Setup
make setup

# Run linting
make lint

# Run tests
make test

# Run tests with coverage
make coverage
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - System design and module details
- [CHANGELOG.md](CHANGELOG.md) - Version history
- [RELEASING.md](RELEASING.md) - Release process
- [SECURITY.md](SECURITY.md) - Security policy

## License

MIT License - see [LICENSE](LICENSE)

## Philosophy

**Leverage, not lore. Evidence over opinions. Augment first.**

Built by Good AI - enterprise AI consultancy focused on practical implementations.
