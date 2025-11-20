# Testing Guide

This document describes the testing strategy for the LLM Efficiency Measurement Tool.

## Testing Philosophy

We maintain comprehensive test coverage to catch bugs before they reach production:

1. **Unit Tests** - Test individual components in isolation
2. **Integration Tests** - Test component interactions
3. **CLI Tests** - Test command-line interface functionality
4. **CI/CD** - Automated testing on every push/PR

## Running Tests Locally

### Install Development Dependencies

```bash
pip install -e ".[dev]"
```

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test Suites

```bash
# Unit tests only
pytest tests/unit/ -v

# CLI tests specifically
pytest tests/unit/test_cli.py -v

# Integration tests
pytest tests/integration/ -v

# With coverage report
pytest tests/ --cov=src/llm_efficiency --cov-report=html
```

### Run Pre-commit Hooks

Pre-commit hooks automatically run linting, type checking, and tests before each commit:

```bash
# Install pre-commit
pip install pre-commit

# Install git hooks
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

## Test Categories

### Unit Tests (`tests/unit/`)

Test individual modules in isolation with mocked dependencies:

- **test_cli.py** - CLI command testing (new!)
  - Tests `init`, `run`, `list`, `show` commands
  - Tests config file loading and CLI argument overrides
  - Uses mocks to avoid downloading models

- **test_config.py** - Configuration validation
- **test_flops_calculator.py** - FLOPs calculation
- **test_energy.py** - Energy tracking
- **test_inference.py** - Inference engine
- **test_results.py** - Results storage
- **test_logging.py** - Structured logging
- **test_profiling.py** - Performance profiling
- **test_cache.py** - Caching utilities

### Integration Tests (`tests/integration/`)

Test complete workflows with real components:

- **test_full_workflow.py** - End-to-end experiment execution

### CLI Tests - What They Catch

The CLI tests (`test_cli.py`) specifically prevent issues like:

1. **Missing imports** - Would catch `Optional` import error
2. **Wrong method calls** - Would catch `from_dict()` vs `model_validate()`
3. **Config file loading** - Tests both file-based and CLI-based configs
4. **Argument overrides** - Ensures CLI args override config file values
5. **Error handling** - Tests missing files, invalid inputs

## Continuous Integration

### GitHub Actions Workflows

**`.github/workflows/tests.yml`**:
- Runs on: Python 3.11, 3.12
- Platforms: Ubuntu, macOS
- Steps:
  1. Lint with ruff
  2. Type check with mypy
  3. Run unit tests with coverage
  4. Run CLI tests
  5. Run integration tests
  6. Upload coverage to Codecov

### What Gets Tested in CI

✅ All unit tests
✅ CLI functionality
✅ Integration workflows
✅ Code formatting (ruff)
✅ Type checking (mypy)
✅ Coverage reporting

## Writing New Tests

### CLI Tests Example

```python
from typer.testing import CliRunner
from unittest.mock import patch

def test_my_cli_command():
    """Test a new CLI command."""
    runner = CliRunner()

    # Mock heavy dependencies
    with patch('llm_efficiency.cli.main.load_model_and_tokenizer'):
        result = runner.invoke(app, ["my-command", "--arg", "value"])

    assert result.exit_code == 0
    assert "expected output" in result.stdout
```

### Unit Test Best Practices

1. **Mock external dependencies** - Don't download models in tests
2. **Test edge cases** - Missing files, invalid inputs, etc.
3. **Use fixtures** - Share setup code across tests
4. **Keep tests fast** - Unit tests should run in milliseconds
5. **Test error paths** - Not just happy paths

## Test Coverage Goals

- **Unit tests**: >90% coverage
- **CLI tests**: 100% command coverage
- **Integration tests**: Key workflows covered

## Debugging Test Failures

### Run with verbose output

```bash
pytest tests/unit/test_cli.py -vv
```

### Run with debugging on failure

```bash
pytest tests/unit/test_cli.py --pdb
```

### Run specific test

```bash
pytest tests/unit/test_cli.py::TestRunCommand::test_run_with_config_file -v
```

## CI Troubleshooting

If tests pass locally but fail in CI:

1. Check Python version (CI uses 3.11, 3.12)
2. Check OS differences (CI tests Ubuntu + macOS)
3. Review CI logs in GitHub Actions
4. Check for environment-specific issues

## Adding New CLI Commands

When adding new CLI commands, **always add tests** to `tests/unit/test_cli.py`:

```python
class TestMyNewCommand:
    """Test my new command."""

    def test_basic_usage(self):
        """Test basic command usage."""
        result = runner.invoke(app, ["my-command"])
        assert result.exit_code == 0

    def test_with_options(self):
        """Test command with options."""
        result = runner.invoke(app, ["my-command", "--option", "value"])
        assert result.exit_code == 0

    def test_error_handling(self):
        """Test error conditions."""
        result = runner.invoke(app, ["my-command", "--invalid"])
        assert result.exit_code == 1
```

## Pre-commit Checklist

Before committing, ensure:

- [ ] All tests pass: `pytest tests/`
- [ ] Linting passes: `ruff check src tests`
- [ ] Formatting is correct: `ruff format src tests`
- [ ] Type checking passes: `mypy src`
- [ ] New features have tests
- [ ] CLI changes have CLI tests

## Questions?

If tests are failing and you're not sure why:

1. Read the test output carefully
2. Run with `-vv` for more detail
3. Check recent changes to related code
4. Review the test implementation
5. Ask for help with specific error messages
