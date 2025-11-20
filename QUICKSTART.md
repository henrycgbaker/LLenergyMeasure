# Quick Start Guide - v2.0

Get started with the LLM Efficiency Measurement Tool v2.0 in minutes!

## Installation

```bash
# Clone the repository
git clone https://github.com/henrycgbaker/llm-efficiency-measurement-tool.git
cd llm-efficiency-measurement-tool

# Checkout v2.0 branch
git checkout claude/document-and-refactor-plan-01XBPnTQP7nnv93nBmCgMcuB

# Install in development mode
pip install -e ".[dev]"
```

## Verify Installation

```bash
# Run tests to verify everything works
pytest tests/ -v

# Check package is installed
python -c "from llm_efficiency import __version__; print(__version__)"
```

## Your First Experiment (v2.0 Style)

```python
from llm_efficiency.config import ExperimentConfig, BatchingConfig
from llm_efficiency.core.model_loader import load_model_and_tokenizer
from llm_efficiency.metrics import FLOPsCalculator
from llm_efficiency.utils.logging import setup_logging

# Setup logging
setup_logging(level="INFO")

# Create configuration
config = ExperimentConfig(
    model_name="hf-internal-testing/tiny-random-gpt2",  # Use tiny model for testing
    precision="float16",
    batching=BatchingConfig(batch_size=4),
)

# Load model
model, tokenizer = load_model_and_tokenizer(config)

# Calculate FLOPs (now accurate for quantized models!)
calculator = FLOPsCalculator()
device = next(model.parameters()).device

flops = calculator.get_flops(
    model=model,
    model_name=config.model_name,
    sequence_length=128,
    device=device,
    is_quantized=config.quantization.enabled,
)

print(f"FLOPs: {flops:,}")
```

## Run Example Scripts

```bash
# Simple experiment
python examples/simple_experiment.py

# Compare quantization methods (shows v2.0 FLOPs fix!)
python examples/quantization_comparison.py

# Migration guide
python examples/migrate_from_v1.py
```

## Migrating from v1.0

### Option 1: Automatic Migration

```bash
# Migrate single config file
python scripts/migrate_config.py configs/a_default_config.py configs_v2/base.json

# Test migration without saving
python scripts/migrate_config.py configs/a_default_config.py --dry-run
```

### Option 2: Manual Migration in Code

```python
from llm_efficiency.config import ExperimentConfig

# Your old v1.0 config
v1_config = {
    "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "fp_precision": "float16",
    "batching_options": {
        "batch_size___fixed_batching": 16
    }
}

# Automatically convert to v2.0
v2_config = ExperimentConfig.from_legacy_dict(v1_config)
```

### Option 3: Use Backward Compatibility Layer

```python
# Quick temporary solution - keeps working with deprecation warnings
from llm_efficiency.legacy import load_model_tokenizer

# Use old v1.0 dict config
model, tokenizer = load_model_tokenizer(old_config)
```

## Key Differences: v1.0 → v2.0

| Feature | v1.0 | v2.0 |
|---------|------|------|
| **Config format** | Nested dicts | Pydantic models |
| **Import path** | `from configs.a_default_config` | `from llm_efficiency.config` |
| **Precision key** | `fp_precision` | `precision` |
| **Batch size key** | `batch_size___fixed_batching` | `batching.batch_size` |
| **Quantized FLOPs** | ❌ Hardcoded (same for all) | ✅ Accurate (per model) |
| **Validation** | ❌ None | ✅ Automatic |
| **IDE support** | ❌ Limited | ✅ Full autocomplete |

## What's Fixed in v2.0? 🎉

### Critical: Quantized Model FLOPs

**v1.0 Problem:**
```python
# ALL quantized models used this same value!
"cached_flops_for_quantised_models": 52638582308864

# TinyLlama-1.1B quantized: 52,638,582,308,864 FLOPs
# Llama-3.1-8B quantized:    52,638,582,308,864 FLOPs  # WRONG!
```

**v2.0 Solution:**
```python
# Each model gets its accurate FLOPs!
calculator = FLOPsCalculator()  # Smart caching + multiple strategies

# TinyLlama-1.1B: ~1 trillion FLOPs
# Llama-3.1-8B:   ~50 trillion FLOPs  # Correct!
```

**Impact:** Energy efficiency metrics (FLOPs/Joule) are now accurate!

## Configuration Examples

### Basic Configuration

```python
from llm_efficiency.config import ExperimentConfig

config = ExperimentConfig(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    precision="float16",
)
```

### With Quantization

```python
from llm_efficiency.config import ExperimentConfig, QuantizationConfig

config = ExperimentConfig(
    model_name="meta-llama/Llama-3.2-1B",
    quantization=QuantizationConfig(
        load_in_4bit=True,
        compute_dtype="float16",
        quant_type="nf4",
    ),
)
```

### With Custom Batching & Decoder

```python
from llm_efficiency.config import (
    ExperimentConfig,
    BatchingConfig,
    DecoderConfig,
)

config = ExperimentConfig(
    model_name="meta-llama/Llama-3.2-3B",
    batching=BatchingConfig(
        batch_size=32,
        adaptive=False,
    ),
    decoder=DecoderConfig(
        mode="top_k",
        top_k=50,
        temperature=0.7,
    ),
)
```

### Save/Load Configuration

```python
import json

# Save to JSON
with open("my_config.json", "w") as f:
    json.dump(config.model_dump(exclude_none=True), f, indent=2)

# Load from JSON
with open("my_config.json") as f:
    config_dict = json.load(f)
    config = ExperimentConfig(**config_dict)
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/unit/test_config.py -v

# Run with coverage
pytest tests/ --cov=src/llm_efficiency --cov-report=html

# View coverage report
open htmlcov/index.html
```

## Common Issues

### Issue: Import Error

```
ModuleNotFoundError: No module named 'llm_efficiency'
```

**Solution:** Install in development mode:
```bash
pip install -e .
```

### Issue: CUDA Out of Memory

**Solution:** Use smaller batch size or quantization:
```python
config = ExperimentConfig(
    model_name="meta-llama/Llama-3.2-1B",
    batching=BatchingConfig(batch_size=4),  # Smaller
    quantization=QuantizationConfig(load_in_8bit=True),  # Less memory
)
```

### Issue: FLOPs Calculation Timeout

**Solution:** Adjust timeout or force architectural estimation:
```python
calculator = FLOPsCalculator()

# Custom timeout
flops = calculator._compute_flops_ptflops(
    model, sequence_length, device, timeout=30
)

# Or use architectural estimation
flops = calculator._estimate_flops_from_architecture(
    model, sequence_length
)
```

## Development

### Code Quality

```bash
# Format code
ruff format src tests

# Lint code
ruff check src tests --fix

# Type check
mypy src

# Run all quality checks
pre-commit run --all-files
```

### Adding Tests

```python
# tests/unit/test_my_feature.py
import pytest
from llm_efficiency.config import ExperimentConfig

def test_my_feature():
    """Test my new feature."""
    config = ExperimentConfig(model_name="test-model")
    assert config.model_name == "test-model"
```

## Next Steps

1. **Read the docs**: Check out `README.md` and module-specific docs in each directory
2. **Browse examples**: See `examples/` for more use cases
3. **Read CHANGELOG**: See `CHANGELOG.md` for full migration guide
4. **Check refactoring plan**: See `REFACTORING_STRATEGY.md` for future improvements

## Getting Help

- **Documentation**: See `README.md` and module `README.md` files
- **Examples**: Check `examples/` directory
- **Tests**: Look at `tests/` for usage examples
- **Issues**: Check GitHub issues or create a new one

## Version Compatibility

- **v1.0**: Original thesis code (tagged as `v1.0.0`)
- **v2.0-dev**: Current development version (this branch)
- **Migration**: Both versions can coexist using backward compatibility layer

## What's Coming Next?

See `REFACTORING_STRATEGY.md` for the full roadmap:

- ✅ **Phase 1**: Foundation + FLOPs fix (DONE!)
- ✅ **Phase 2**: Pydantic configs (DONE!)
- 🔄 **Phase 3**: Migrate remaining core modules
- 📋 **Phase 4**: Modern CLI with Typer + Rich
- 📋 **Phase 5**: Complete test coverage
- 📋 **Phase 6**: Storage & results management
- 📋 **Phase 7**: Full production release

---

**Author:** Henry Baker
**Version:** 2.0.0-dev
**Date:** 2025-11-20
