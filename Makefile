# Makefile for Good AI Enterprise Framework
.PHONY: setup lint test coverage clean help

# Default target
.DEFAULT_GOAL := help

# Python interpreter
PYTHON := python3
PYTEST := $(PYTHON) -m pytest

help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

setup: ## Install package with dev dependencies
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"
	$(PYTHON) -m pip install ruff bandit

lint: ## Run linting checks
	$(PYTHON) -m ruff check src/
	$(PYTHON) -m ruff check tests/

lint-fix: ## Run linting and auto-fix issues
	$(PYTHON) -m ruff check src/ --fix
	$(PYTHON) -m ruff check tests/ --fix

test: ## Run tests
	$(PYTEST) tests/ -v

test-fast: ## Run tests without verbose output
	$(PYTEST) tests/ -q

coverage: ## Run tests with coverage report
	$(PYTEST) tests/ --cov=goodai --cov-report=term-missing --cov-report=html

security: ## Run security checks
	$(PYTHON) -m bandit -r src/goodai -ll

deterministic: ## Run deterministic tests twice to verify consistency
	$(PYTEST) tests/ -k "deterministic" -v
	$(PYTEST) tests/ -k "deterministic" -v

clean: ## Remove build artifacts and cache files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

build: ## Build distribution packages
	$(PYTHON) -m pip install --upgrade build
	$(PYTHON) -m build

all: lint test ## Run lint and test
