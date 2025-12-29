# tests/ - Test Suite

Unit, integration, and end-to-end tests for the LLM Energy Measure framework.

## Structure

```
tests/
├── conftest.py          # Shared pytest fixtures
├── fixtures/            # Test data and fixtures
├── unit/                # Unit tests (fast, isolated)
├── integration/         # Integration tests (component interaction)
└── e2e/                 # End-to-end tests (full workflows)
```

## Running Tests

```bash
# Run all tests
make test-all
# or
poetry run pytest tests/ -v

# Unit tests only (fast)
make test
# or
poetry run pytest tests/unit/ -v

# Integration tests
make test-integration
# or
poetry run pytest tests/integration/ -v

# Specific test file
poetry run pytest tests/unit/test_config_models.py -v

# With coverage
poetry run pytest tests/ --cov=llm_energy_measure --cov-report=html
```

## Test Categories

### Unit Tests (`tests/unit/`)

Fast, isolated tests for individual components.

| File | Tests |
|------|-------|
| `test_config_models.py` | Pydantic config validation |
| `test_config_loader.py` | Config loading, inheritance |
| `test_core_inference.py` | Inference metrics calculation |
| `test_core_prompts.py` | Batch creation, tokenization |
| `test_core_distributed.py` | Distributed utilities |
| `test_core_model_loader.py` | Model loading logic |
| `test_core_energy_backends.py` | Energy backend interface |
| `test_core_compute_metrics.py` | Memory/utilization stats |
| `test_domain_metrics.py` | Metric model validation |
| `test_domain_experiment.py` | Result model validation |
| `test_domain_model_info.py` | Model info models |
| `test_orchestration_*.py` | Orchestration components |
| `test_results_aggregation.py` | Aggregation logic |
| `test_repository.py` | FileSystemRepository |
| `test_results_exporters.py` | Export functionality |
| `test_flops_estimator.py` | FLOPs estimation |
| `test_resilience.py` | Retry/circuit breaker |
| `test_security.py` | Path sanitization |
| `test_exceptions.py` | Exception hierarchy |
| `test_logging.py` | Logging setup |
| `test_constants.py` | Constant values |
| `test_protocols.py` | Protocol definitions |
| `test_state.py` | State management |
| `test_cli.py` | CLI command parsing |

### Integration Tests (`tests/integration/`)

Tests for component interaction.

| File | Tests |
|------|-------|
| `test_config_aggregation_pipeline.py` | Config -> Aggregation flow |
| `test_cli_workflows.py` | CLI multi-step workflows |
| `test_repository_operations.py` | Repository CRUD operations |
| `test_error_handling.py` | Error propagation |

### E2E Tests (`tests/e2e/`)

Full workflow tests (may require GPU).

| File | Tests |
|------|-------|
| `test_cli_e2e.py` | Full CLI workflows |

## Writing Tests

### Fixtures

Common fixtures in `conftest.py`:
```python
@pytest.fixture
def sample_config():
    return ExperimentConfig(
        config_name="test",
        model_name="test/model",
    )

@pytest.fixture
def temp_results_dir(tmp_path):
    return FileSystemRepository(tmp_path)
```

### Mocking GPU Operations

For tests that would require GPU:
```python
@pytest.fixture
def mock_cuda():
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.device_count", return_value=4):
            yield
```

### Test Naming

- `test_<function>_<scenario>` for functions
- `test_<class>_<method>_<scenario>` for methods
- Use descriptive names: `test_load_config_with_inheritance_resolves_extends`

## CI Integration

Tests run via GitHub Actions:
```yaml
# .github/workflows/ci.yml
- run: make test
```

Coverage reports uploaded to Codecov.

## Related

- See `src/llm_energy_measure/README.md` for package structure
- See `Makefile` for test commands
