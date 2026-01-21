# Parameter Validation Testing Framework

A systematic, programmatically-driven test framework that validates every parameter across all backends (PyTorch, vLLM, TensorRT) - both for passthrough and runtime behaviour.

## Architecture

```
tests/param_validation/
├── conftest.py                   # Hardware detection, skip markers
├── test_runner.py                # Test generation utilities
├── test_config_parsing.py        # CI-safe Pydantic parsing tests
├── test_vllm_passthrough.py      # GPU: vLLM passthrough verification
├── test_pytorch_runtime.py       # GPU: PyTorch runtime verification
├── registry/
│   ├── models.py                 # ParamSpec, VerificationResult dataclasses
│   ├── discovery.py              # Extract params from Pydantic models
│   ├── param_registry.py         # Central registry of all param specs
│   └── hardware_caps.py          # GPU/software capability detection
├── verifiers/
│   ├── passthrough.py            # Verify param reaches backend config
│   ├── behaviour.py              # Verify observable output change
│   ├── introspection.py          # Inspect model/engine state
│   └── mock_verifiers.py         # CI-safe mock-based verification
├── backends/
│   ├── pytorch/param_specs.py    # PyTorch param definitions
│   ├── vllm/param_specs.py       # vLLM param definitions
│   └── tensorrt/param_specs.py   # TensorRT param definitions
└── shared/param_specs.py         # Shared params (decoder, batching)
```

## Running Tests

### CI-Safe Tests (No GPU Required)

```bash
# Config parsing validation - verifies Pydantic models
pytest tests/param_validation/test_config_parsing.py -v

# All non-GPU tests
pytest tests/param_validation/ -v -m "not requires_gpu"
```

### GPU Tests

```bash
# All GPU tests
pytest tests/param_validation/ -v

# vLLM passthrough tests only
pytest tests/param_validation/test_vllm_passthrough.py -v

# PyTorch runtime tests only
pytest tests/param_validation/test_pytorch_runtime.py -v

# Filter by marker
pytest tests/param_validation/ -v -m requires_vllm
pytest tests/param_validation/ -v -m requires_hopper
```

### Discovery and Coverage

```bash
# Show hardware profile
python -c "from tests.param_validation.registry import get_hardware_summary; print(get_hardware_summary())"

# Show coverage report
python -c "
from tests.param_validation.backends.vllm import register_vllm_params
from tests.param_validation.backends.pytorch import register_pytorch_params
from tests.param_validation.backends.tensorrt import register_tensorrt_params
from tests.param_validation.shared import register_shared_params
from tests.param_validation.registry import get_coverage_report, registry

registry.reset()
register_vllm_params()
register_pytorch_params()
register_tensorrt_params()
register_shared_params()

report = get_coverage_report(registry.all_names)
print(f'Total: {report[\"total_discovered\"]} params discovered')
print(f'Covered: {report[\"total_covered\"]} params')
print(f'Coverage: {report[\"coverage_percent\"]:.1f}%')
for backend, pct in report['coverage_by_backend'].items():
    print(f'  {backend}: {pct:.1f}%')
"
```

## Core Concepts

### ParamSpec - Declarative Parameter Definition

Each parameter is defined as a `ParamSpec` with:

```python
ParamSpec(
    name="max_num_seqs",              # Parameter name
    backend="vllm",                    # Backend (vllm, pytorch, tensorrt, shared)
    config_path="vllm.max_num_seqs",   # Path in experiment config
    test_values=[64, 128, 256],        # Values to test
    verification_type=VerificationType.PASSTHROUGH,
    hardware_requirements={HardwareRequirement.GPU, HardwareRequirement.VLLM},
    passthrough_path="llm_engine.scheduler_config.max_num_seqs",
    category="memory",
    energy_impact=True,
)
```

### Verification Types

| Type | Purpose | GPU Required |
|------|---------|--------------|
| `PASSTHROUGH` | Verify param value reaches backend config | Yes (or mock) |
| `BEHAVIOUR` | Verify observable output/perf change | Yes |
| `INTROSPECTION` | Inspect model/engine internal state | Yes |
| `MOCK` | CI-safe via patching | No |

### Hardware Requirements

Specs declare their hardware requirements:

- `HardwareRequirement.GPU` - Any CUDA GPU
- `HardwareRequirement.VLLM` - vLLM installed
- `HardwareRequirement.TENSORRT` - TensorRT-LLM installed
- `HardwareRequirement.HOPPER` - Hopper (SM 9.0+) GPU for FP8
- `HardwareRequirement.AMPERE` - Ampere (SM 8.0+) GPU for BF16
- `HardwareRequirement.FLASH_ATTN` - Flash Attention installed
- `HardwareRequirement.MULTI_GPU` - Multiple GPUs

Tests are automatically skipped if requirements aren't met.

## Adding New Parameters

1. **Add ParamSpec** to the appropriate `param_specs.py`:

```python
ParamSpec(
    name="new_param",
    backend="vllm",
    config_path="vllm.new_param",
    test_values=[...],
    verification_type=VerificationType.PASSTHROUGH,
    passthrough_path="llm_engine.config.new_param",
    category="memory",
)
```

2. **Auto-discover** from Pydantic model (optional):

```python
from tests.param_validation.registry import discover_model_fields, infer_test_values
from llm_energy_measure.config.backend_configs import VLLMConfig

fields = discover_model_fields(VLLMConfig)
new_param = fields["new_param"]
test_values = infer_test_values(new_param)  # Auto-infer from constraints
```

3. **Register**:

```python
from tests.param_validation.registry import register
register(spec)
```

## Skip Markers

Use markers for conditional test execution:

```python
@pytest.mark.requires_gpu
@pytest.mark.requires_vllm
def test_vllm_feature():
    ...

@pytest.mark.requires_hopper  # FP8 requires Hopper
def test_fp8_kv_cache():
    ...
```

## Comparison with Manual Tests

| Aspect | Old Manual Tests | New Framework |
|--------|------------------|---------------|
| Organisation | Ad-hoc files | Systematic by backend |
| Discovery | Manual | Auto from Pydantic |
| Parametrisation | Manual per-test | Auto from ParamSpec |
| CI-safe tests | Limited | Full mock coverage |
| Coverage tracking | None | Built-in reports |
| Hardware skipping | Manual markers | Auto from requirements |

The new framework provides:
- **Systematic coverage** - All params derived from config models
- **Automatic test generation** - One ParamSpec generates multiple tests
- **CI-safe testing** - Mock-based tests for config parsing
- **Coverage reporting** - Track which params have tests
- **Declarative definitions** - Easy to add/modify params
