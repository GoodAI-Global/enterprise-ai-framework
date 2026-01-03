# Architecture

## Overview

The Good AI Enterprise Framework is designed around simplicity, explainability, and production readiness. This document describes the architectural decisions and component interactions.

## Core Principles

1. **Simple over complex**: Prefer statistical methods over deep learning
2. **Deterministic**: Same input always produces same output
3. **Headless-first**: All components work without display
4. **No stubs**: Every method has real, working logic
5. **Vertical focus**: Manufacturing complete before expanding

## Component Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Application Layer                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Manufacturing   │  │ (Future)        │  │ (Future)        │  │
│  │ OEE Demo        │  │ Insurance       │  │ Aquaculture     │  │
│  └────────┬────────┘  └─────────────────┘  └─────────────────┘  │
└───────────┼─────────────────────────────────────────────────────┘
            │
┌───────────┼─────────────────────────────────────────────────────┐
│           │              Framework Layer                         │
│  ┌────────▼────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │     Core        │  │    Modules      │  │   Connectors    │  │
│  ├─────────────────┤  ├─────────────────┤  ├─────────────────┤  │
│  │ Assessment      │  │ Anomaly         │  │ Vision          │  │
│  │ Engine          │  │ Detection       │  │ Connector       │  │
│  ├─────────────────┤  ├─────────────────┤  ├─────────────────┤  │
│  │ Bottleneck      │  │ Predictive      │  │ Data            │  │
│  │ Analyzer        │  │ Maintenance     │  │ Connector       │  │
│  ├─────────────────┤  └─────────────────┘  └─────────────────┘  │
│  │ ROI             │                                             │
│  │ Calculator      │                                             │
│  └─────────────────┘                                             │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                        Utils                               │  │
│  │  ┌─────────────────────┐  ┌─────────────────────────────┐ │  │
│  │  │ Metrics (OEE, KPIs) │  │ Reporting (JSON, MD, HTML)  │ │  │
│  │  └─────────────────────┘  └─────────────────────────────┘ │  │
│  └───────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
            │
┌───────────┼──────────────────────────────────────────────────────┐
│           │              Data Layer                               │
│  ┌────────▼────────┐  ┌─────────────────┐  ┌─────────────────┐   │
│  │ CSV Files       │  │ JSON/Config     │  │ Images (OCR)    │   │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

## Component Details

### Core Layer

#### Assessment Engine
- **Purpose**: Evaluate organization's AI readiness
- **Method**: Weighted scoring across 4 dimensions
- **Output**: Scores, recommendations, red flags, quick wins

Dimensions:
1. Data Readiness (30% weight)
2. Technical Capability (25% weight)
3. Process Maturity (25% weight)
4. Organizational Readiness (20% weight)

#### Bottleneck Analyzer
- **Purpose**: Identify operational bottlenecks
- **Method**: Compare metrics against industry benchmarks
- **Output**: Ranked bottlenecks with AI solution fit

Categories analyzed:
- Availability/downtime
- Performance/speed
- Quality/defects
- Variability
- Changeover time

#### ROI Calculator
- **Purpose**: Project ROI for AI implementations
- **Method**: Conservative benefit estimation with NPV
- **Output**: ROI%, payback period, sensitivity analysis

Applies conservative factor (0.7x) to all estimates.

### Modules Layer

#### Anomaly Detection
- **Purpose**: Detect unusual values in time series
- **Method**: Rolling z-score (NOT machine learning)
- **Key properties**:
  - Deterministic
  - Explainable (human-readable explanations)
  - O(n) complexity

Algorithm:
```
z_score = (value - rolling_mean) / rolling_std
is_anomaly = |z_score| > threshold
```

Default parameters:
- Window size: 20
- Threshold: 2.5 standard deviations

#### Predictive Maintenance
- **Purpose**: Monitor equipment health
- **Method**: Trend analysis + threshold monitoring
- **Output**: Health scores, alerts, maintenance schedule

Health scoring:
- 80-100: Healthy
- 60-79: Degraded
- 40-59: At Risk
- 0-39: Critical

### Connectors Layer

#### Vision Connector
- **Purpose**: Extract data from images/screens
- **Method**: OpenCV + Tesseract OCR
- **Key property**: Headless-safe

Headless mode:
- Uses `opencv-python-headless`
- No `cv2.imshow()` calls
- Requires `roi_config` when no display available

#### Data Connector
- **Purpose**: Unified data source access
- **Supports**: CSV, JSON, Excel, Parquet
- **Features**: Schema validation, caching, merging

### Utils Layer

#### Metrics
- OEE calculation (Availability × Performance × Quality)
- Loss breakdown analysis
- Custom KPI tracking

#### Reporting
- JSON export
- Markdown generation
- HTML reports with styling

## Data Flow

### Manufacturing OEE Demo Flow

```
1. Load Data
   sample_data.csv → pd.DataFrame

2. Anomaly Detection
   DataFrame → AnomalyDetector.detect() → DataFrame with anomaly flags

3. OEE Calculation
   DataFrame → calculate_oee() → {availability, performance, quality, overall}

4. Recommendations
   OEE + Anomalies → generate_recommendations() → List[str]

5. Output
   All results → JSON → stdout
```

### Assessment Flow

```
1. Collect Company Data
   Survey/Interview → Dict of attributes

2. Dimension Assessment
   Data → _assess_data(), _assess_technical(), etc.

3. Scoring
   Dimension scores → Weighted average → Overall score

4. Analysis
   Scores → Recommendations, Red Flags, Quick Wins

5. Report
   AssessmentResult → ReportGenerator → Markdown/HTML
```

## Key Design Patterns

### Dataclass Results
All result objects are dataclasses with `to_dict()` methods for serialization.

### Optional Dependencies
Vision connector uses lazy imports - works even without opencv/tesseract installed.

### Configurable Thresholds
All thresholds are configurable via constructor or config files.

### No Side Effects
All functions are pure - same input always produces same output.

## Extension Points

### Adding New Industries
1. Add industry benchmarks to `AssessmentEngine.BENCHMARKS`
2. Add industry-specific bottleneck patterns to `BottleneckAnalyzer`
3. Create new example in `examples/`

### Adding New Modules
1. Create module in `src/goodai/modules/`
2. Add exports to `__init__.py`
3. Write tests in `tests/`
4. Update documentation

### Adding New Connectors
1. Create connector in `src/goodai/connectors/`
2. Implement headless-safe operation
3. Add tests including headless scenarios

## Testing Strategy

### Unit Tests
- Test each component in isolation
- Mock dependencies where needed
- Focus on edge cases

### Integration Tests
- Test component interactions
- Verify demo runs end-to-end
- Check output format

### Determinism Tests
- Same input → same output
- No random behavior
- Reproducible results

## Performance Considerations

- Anomaly detection: O(n) for n data points
- Assessment: O(1) for scoring, O(k) for k recommendations
- Vision connector: O(r) for r regions, limited by OCR speed

For large datasets (>1M rows), consider:
- Chunked processing
- Sampling for initial analysis
- Caching intermediate results

## Security Notes

- No network calls in core modules
- Config files are local only
- No credential storage
- Image processing is local only

## Future Considerations

Potential additions (not implemented):
- Real-time streaming support
- Database connectors
- Cloud integration
- Advanced ML models (optional)

These would be added as new modules without changing core architecture.
