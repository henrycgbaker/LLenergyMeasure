.PHONY: format lint typecheck check test test-integration test-all install dev clean
.PHONY: docker-build docker-build-all docker-build-vllm docker-build-tensorrt
.PHONY: docker-build-dev docker-check experiment datasets validate docker-shell docker-dev
.PHONY: setup lem-clean lem-clean-all generate-docs check-docs

# PUID/PGID for correct file ownership on bind mounts (LinuxServer.io pattern)
# These are also set in .env by setup.sh - Makefile exports for convenience
export PUID := $(shell id -u)
export PGID := $(shell id -g)

# =============================================================================
# Quick Start (new users should use: ./setup.sh)
# =============================================================================

setup:
	@./setup.sh

# =============================================================================
# Local Development
# =============================================================================

format:
	poetry run ruff format src/ tests/

lint:
	poetry run ruff check src/ tests/ --fix

typecheck:
	poetry run mypy src/

check: format lint typecheck

test:
	poetry run pytest tests/unit/ -v

test-integration:
	poetry run pytest tests/integration/ -v

test-all:
	poetry run pytest tests/ -v --ignore=tests/runtime/

test-runtime:
	poetry run pytest tests/runtime/ -v

test-runtime-quick:
	poetry run pytest tests/runtime/ -v --quick

install:
	poetry install

dev:
	poetry install --with dev
	poetry run pre-commit install

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage dist/ build/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Generate documentation from SSOT sources
generate-docs:
	python scripts/generate_invalid_combos_doc.py
	python scripts/generate_param_matrix.py
	python scripts/generate_config_docs.py
	@echo "Generated docs in docs/generated/"

# Check if generated docs are stale (CI validation)
check-docs:
	@python scripts/generate_invalid_combos_doc.py
	@git diff --quiet docs/generated/ || \
		(echo "ERROR: Generated docs are stale. Run 'make generate-docs' and commit." && exit 1)
	@echo "Generated docs are up to date"

ci: check test check-docs

# =============================================================================
# Docker Commands (Production)
# =============================================================================

# Build PyTorch backend (default, recommended for most users)
docker-build-pytorch:
	docker compose build base pytorch

# Build all backends (pytorch, vllm, tensorrt)
docker-build-all:
	docker compose build base pytorch vllm tensorrt

# Build specific backends
docker-build-vllm:
	docker compose build base vllm

docker-build-tensorrt:
	docker compose build base tensorrt

# Validate Docker setup
docker-check:
	@docker compose config -q || (echo "Error: Invalid docker-compose config"; exit 1)
	@echo "Docker config OK"

# Run any lem command in Docker
# Usage: make lem CMD="experiment configs/my_experiment.yaml"
#        make lem CMD="config validate configs/test.yaml"
#        make lem CMD="results list"
CMD ?= --help
lem: docker-check
	docker compose run --rm pytorch lem $(CMD)

# Run experiment (num_processes auto-inferred from config)
# Usage: make experiment CONFIG=test_tiny.yaml DATASET=alpaca SAMPLES=100
CONFIG ?= test_tiny.yaml
DATASET ?= alpaca
SAMPLES ?= 100
experiment: docker-check
	docker compose run --rm pytorch \
		lem experiment /app/configs/$(CONFIG) \
		--dataset $(DATASET) -n $(SAMPLES)

# List available datasets
datasets:
	docker compose run --rm pytorch lem datasets

# Validate a config file
# Usage: make validate CONFIG=test_tiny.yaml
validate: docker-check
	docker compose run --rm pytorch lem config validate /app/configs/$(CONFIG)

# Interactive shell in production container
docker-shell:
	docker compose run --rm pytorch /bin/bash

# =============================================================================
# Docker Commands (Development)
# =============================================================================

# Build the dev Docker image
docker-build-dev:
	docker compose --profile dev build base pytorch-dev

# Interactive dev shell with source mounted
docker-dev:
	docker compose --profile dev run --rm pytorch-dev

# =============================================================================
# Volume Management
# =============================================================================

# Clean experiment state volume (preserves caches)
lem-clean-state:
	docker volume rm lem-experiment-state 2>/dev/null || true
	@echo "Cleared experiment state"

# Clean HuggingFace cache volume (will need to re-download models)
lem-clean-cache:
	docker volume rm lem-hf-cache 2>/dev/null || true
	@echo "Cleared HuggingFace cache"

# Clean TensorRT engine cache
lem-clean-trt:
	docker volume rm lem-trt-engine-cache 2>/dev/null || true
	@echo "Cleared TensorRT engine cache"

# Clean all named volumes (state + caches)
lem-clean-all:
	docker volume rm lem-experiment-state lem-hf-cache lem-trt-engine-cache 2>/dev/null || true
	@echo "Cleared all LEM volumes"
