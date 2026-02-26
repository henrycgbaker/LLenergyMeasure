# Testing Design

**Last updated**: 2026-02-19
**Source decisions**: [../decisions/testing-strategy.md](../decisions/testing-strategy.md)
**Status**: Confirmed

---

## Test Pyramid

```
tests/
  conftest.py                    ← shared fixtures (MockBackend, tmp_results_dir, etc.)
  unit/                          ← no GPU; always run in CI; must be fast
    test_config.py               ← Pydantic model validation, extra=forbid, field renames
    test_grid.py                 ← sweep: → list[ExperimentConfig] expansion
    test_validation.py           ← SSOT backend compatibility constraints (PRECISION_SUPPORT, etc.)
    test_vram.py                 ← VRAM estimation math (formula correctness, not GPU)
    test_result_schema.py        ← ExperimentResult / StudyResult field validation
    test_filenames.py            ← human-readable filename generation
    test_cli_parsing.py          ← Typer argument parsing (CliRunner, no actual experiment runs)
    test_preflight.py            ← pre-flight logic with mock backends
    test_config_hash.py          ← config_hash stability and collision resistance
    test_user_config.py          ← UserConfig parsing, runner format validation
    test_dataset.py              ← dataset loader (built-ins, JSONL, synthetic generation)
    test_co2.py                  ← CO2 calculation, lookup table, PUE adjustment
    test_schema_migration.py     ← backwards-compatible loading across schema versions
  integration/                   ← GPU required; gated with @pytest.mark.gpu
    test_pytorch_backend.py      ← real inference on GPU
    test_vllm_backend.py         ← vLLM inference (Linux only)
    test_experiment_run.py       ← end-to-end single experiment
    test_study_run.py            ← end-to-end study with subprocess isolation
    test_energy_backend.py       ← NVML / Zeus energy measurement
    test_reproducibility.py      ← same config_hash → same config_hash on re-run
```

---

## CI Path-Filter — Phase 5 Implementation Note (item 15)

When setting up GitHub Actions CI, use `dorny/paths-filter@v3` for path-based job skipping.
`contains(github.event.pull_request.changed_files, ...)` is not a valid GitHub Actions expression
and will silently fail.

```yaml
# .github/workflows/ci.yml — correct path filter pattern
jobs:
  changes:
    runs-on: ubuntu-latest
    outputs:
      backend: ${{ steps.filter.outputs.backend }}
      docs: ${{ steps.filter.outputs.docs }}
    steps:
      - uses: dorny/paths-filter@v3
        id: filter
        with:
          filters: |
            backend:
              - 'src/llenergymeasure/core/**'
              - 'src/llenergymeasure/orchestration/**'
            docs:
              - 'docs/**'
              - '*.md'

  test-backend:
    needs: changes
    if: needs.changes.outputs.backend == 'true'
    # ... run GPU tests only when backend code changes
```

---

## Mock Backend Pattern

```python
# tests/conftest.py
from llenergymeasure.core.backends.protocol import InferenceBackend
from llenergymeasure.core.energy.protocol import EnergyBackend

class MockInferenceBackend:
    """Deterministic fake backend — implements InferenceBackend Protocol."""
    def load_model(self, config): pass
    def run_inference(self, prompts, config):
        return [
            MockTokenResult(
                prompt=p,
                output_tokens=100,
                ttft_ms=50.0,
                itl_ms_per_token=5.0,
            )
            for p in prompts
        ]
    def unload_model(self): pass


class MockEnergyBackend:
    """Deterministic fake energy measurement — implements EnergyBackend Protocol."""
    def start_window(self): pass
    def end_window(self) -> float:
        return 42.0   # joules — deterministic for testing

    @property
    def power_w(self) -> float:
        return 200.0  # watts


@pytest.fixture
def mock_inference_backend():
    return MockInferenceBackend()

@pytest.fixture
def mock_energy_backend():
    return MockEnergyBackend()

@pytest.fixture
def tmp_results_dir(tmp_path):
    return tmp_path / "results"
```

Protocols are structural — `MockInferenceBackend` does not inherit from a base class.
It just implements the protocol methods. Same pattern as lm-eval's `LM` Protocol.

---

## GPU Gating

```python
# tests/conftest.py (also)
import pytest

def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires a real GPU to run")
```

```python
# tests/integration/test_pytorch_backend.py
@pytest.mark.gpu
def test_pytorch_inference_completes():
    ...
```

Run unit tests only (CI on every push):
```bash
pytest tests/unit/ -m "not gpu"
```

Run full suite (post-merge, weekly):
```bash
pytest tests/ -m "gpu"   # or just pytest tests/
```

---

## CI Trigger Policy

```yaml
# .github/workflows/tests.yml (sketch)

on:
  push:
    branches: ["*"]
  pull_request:
  schedule:
    - cron: "0 2 * * 0"   # Sunday 02:00 UTC
  workflow_dispatch:

jobs:
  unit:
    runs-on: ubuntu-latest
    steps:
      - run: pytest tests/unit/ -m "not gpu" --tb=short

  integration:
    runs-on: [self-hosted, gpu]
    if: |
      github.event_name == 'push' && github.ref == 'refs/heads/main' ||
      github.event_name == 'schedule' ||
      github.event_name == 'workflow_dispatch' ||
      contains(github.event.pull_request.changed_files, 'src/llenergymeasure/core/') ||
      contains(github.event.pull_request.changed_files, 'src/llenergymeasure/orchestration/')
    steps:
      - run: pytest tests/integration/ --tb=short
```

<!-- TODO: The path-filter trigger above uses a non-existent GitHub Actions feature
     (changed_files is not a native filter). Real implementation needs dorny/paths-filter
     or similar action. Clarify before CI setup in Phase 5. -->

<!-- TODO: Self-hosted GPU runner setup is not yet defined. What machine? What CUDA version?
     How is it maintained? This is infrastructure that needs to exist before integration
     tests can run in CI. -->

---

## Key Unit Test Cases

### `test_config.py` — `extra="forbid"` and typo detection

```python
def test_extra_forbid_rejects_typo():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ExperimentConfig(
            model="meta-llama/Llama-3.1-8B",
            backend="pytorch",
            bachsize=8,   # typo: should be pytorch.batch_size
        )

def test_extra_field_accepted():
    config = ExperimentConfig(
        model="meta-llama/Llama-3.1-8B",
        backend="pytorch",
        extra={"torch_compile_mode": "max-autotune"},
    )
    assert config.extra["torch_compile_mode"] == "max-autotune"
```

### `test_validation.py` — SSOT constraint enforcement

```python
def test_tensorrt_rejects_fp32():
    with pytest.raises(ValidationError, match="does not support precision='fp32'"):
        ExperimentConfig(
            model="meta-llama/Llama-3.1-8B",
            backend="tensorrt",
            precision="fp32",
        )

def test_tensorrt_rejects_beam_search():
    with pytest.raises(ValidationError, match="does not support decoding_strategy"):
        ExperimentConfig(
            model="meta-llama/Llama-3.1-8B",
            backend="tensorrt",
            decoder=DecoderConfig(decoding_strategy="beam"),
        )
```

### `test_grid.py` — sweep expansion

```python
def test_grid_cartesian_product():
    study = StudyConfig(
        model="meta-llama/Llama-3.1-8B",
        backend="pytorch",
        sweep={"precision": ["fp16", "bf16"], "n": [100, 500]},
    )
    experiments = expand_grid(study)
    assert len(experiments) == 4   # 2 × 2

def test_grid_skips_invalid_combinations():
    study = StudyConfig(
        model="meta-llama/Llama-3.1-8B",
        backend="tensorrt",
        sweep={"precision": ["fp16", "fp32"]},   # fp32 invalid for tensorrt
    )
    experiments = expand_grid(study)
    assert len(experiments) == 1   # only fp16
    assert study.skipped[0]["reason"] contains "does not support precision='fp32'"
```

---

## Not Included

- **Property-based testing (Hypothesis)**: Considered for sweep grammar edge cases. Not
  adopted — add if config validation bugs emerge in practice.
- **Measurement accuracy tests**: No automated approach is feasible. Calibration procedure
  may be documented separately. Out of scope.
- **Mutation testing**: Not warranted at this stage.

---

## Related

- [../decisions/testing-strategy.md](../decisions/testing-strategy.md): Decision rationale
- [architecture.md](architecture.md): Module structure (what to test)
- [experiment-isolation.md](experiment-isolation.md): StudyRunner subprocess (integration test target)
