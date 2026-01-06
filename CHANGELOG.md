# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Nothing yet

### Changed
- Nothing yet

### Fixed
- Nothing yet

## [0.1.0] - 2025-01-06

### Added

#### Core Modules
- Assessment Engine for AI readiness evaluation across 4 dimensions
- ROI Calculator with NPV, payback period, and sensitivity analysis
- Bottleneck Analyzer for operational process analysis

#### Anomaly Detection
- Rolling z-score based anomaly detection
- Deterministic results (same input = same output)
- Human-readable anomaly explanations

#### MLOps
- Model Registry with version control and status transitions
- A/B Testing Framework with statistical significance testing
- Feedback Loop for collecting and processing user feedback
- Explainability Engine with feature importance and counterfactuals

#### Enterprise Security
- Multi-tenancy support with tenant isolation
- Role-Based Access Control (RBAC) with hierarchical permissions
- Audit Logging for compliance and tracking

#### Infrastructure
- Memory-based caching with TTL and tag invalidation
- Async execution utilities with retry and timeout support
- Schema validation with versioning and migration support

#### API Layer
- FastAPI REST endpoints for all modules
- Health check endpoints (liveness, readiness, startup)
- Correlation ID middleware for request tracing
- Audit and tenant middleware

#### Monitoring
- Structured JSON logging with correlation context
- Health check system with configurable probes
- Sensitive data redaction in logs

### Technical Details
- Python 3.9+ support
- 300 tests with 73% coverage
- Deterministic test suite
- Headless operation support (no display required)

[Unreleased]: https://github.com/GoodAI-Global/enterprise-ai-framework/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/GoodAI-Global/enterprise-ai-framework/releases/tag/v0.1.0
