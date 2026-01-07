.PHONY: format lint typecheck check test test-integration test-all install dev clean
.PHONY: docker-build docker-check experiment datasets validate docker-shell

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
	poetry run pytest tests/ -v

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
# Docker Commands
# =============================================================================

# Build the Docker image
docker-build:
	docker compose build

# Validate Docker setup
docker-check:
	@docker compose config -q || (echo "Error: Invalid docker-compose config"; exit 1)
	@echo "Docker config OK"

# Run experiment (num_processes auto-inferred from config)
# Usage: make experiment CONFIG=test_tiny.yaml DATASET=alpaca SAMPLES=100
CONFIG ?= test_tiny.yaml
DATASET ?= alpaca
SAMPLES ?= 100
experiment: docker-check
	docker compose run --rm bench \
		llm-energy-measure experiment /app/configs/$(CONFIG) \
		--dataset $(DATASET) -n $(SAMPLES)

# List available datasets
datasets:
	docker compose run --rm bench llm-energy-measure datasets

# Validate a config file
# Usage: make validate CONFIG=test_tiny.yaml
validate: docker-check
	docker compose run --rm bench llm-energy-measure config validate /app/configs/$(CONFIG)

# Interactive shell in container
docker-shell:
	docker compose run --rm bench /bin/bash
