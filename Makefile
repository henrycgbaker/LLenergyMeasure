.PHONY: format lint typecheck check test test-integration test-all install dev clean

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
