.PHONY: format lint typecheck check test test-integration test-all install dev clean
.PHONY: docker-build docker-build-all docker-build-vllm docker-build-tensorrt
.PHONY: docker-build-dev docker-check experiment datasets validate docker-shell docker-dev lem

# PUID/PGID for correct file ownership on bind mounts (LinuxServer.io pattern)
export PUID := $(shell id -u)
export PGID := $(shell id -g)

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

ci: check test

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
