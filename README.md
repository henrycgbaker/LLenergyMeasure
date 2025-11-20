# LLM Efficiency Measurement Tool

A comprehensive, production-ready framework for measuring and analyzing Large Language Model (LLM) inference efficiency, performance, and energy consumption.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

Systematically evaluate LLM inference efficiency across multiple dimensions:

- **⚡ Energy Consumption**: CPU, GPU, and RAM power usage with CO2 emissions tracking
- **🚀 Performance Metrics**: Latency, throughput, tokens per second
- **💻 Computational Efficiency**: FLOPs calculation and compute utilization
- **🔧 Optimization Strategies**: Quantization (4-bit, 8-bit), precision tuning, batching
- **📊 Comprehensive Analysis**: Built-in profiling, caching, and reporting

Perfect for:
- Comparing model efficiency across architectures
- Evaluating quantization trade-offs
- Optimizing inference costs
- Environmental impact analysis
- Research and benchmarking

## Features

### Core Capabilities

- **🔬 Comprehensive Metrics**: Energy, performance, compute, and emissions in one tool
- **⚙️ Multiple Precision Types**: FP32, FP16, BF16, FP8 support
- **📦 Quantization**: 4-bit and 8-bit quantization via BitsAndBytes
- **🎯 Type-Safe Configuration**: Pydantic-based config with validation
- **💾 Flexible Storage**: JSON, CSV, and Pickle export formats
- **🔄 Automatic Retry Logic**: Network failure recovery with exponential backoff
- **🌍 Carbon Tracking**: Real-time CO2 emissions via CodeCarbon

### Performance & Optimization (v1.5.0+)

- **📈 Performance Profiling**: Track execution time, memory, CPU usage
- **💨 Advanced Caching**: LRU+TTL and disk-based caching
- **🎨 Beautiful CLI**: Rich terminal output with progress bars and tables
- **📊 Comprehensive Testing**: 186+ unit tests with 80%+ coverage

## Quick Start

### Installation

**⚠️ Important:** PyTorch requires platform-specific installation. See [INSTALLATION.md](INSTALLATION.md) for detailed instructions.

**Quick install (macOS/Linux):**

```bash
# 1. Install PyTorch first (required)
pip install torch

# 2. Install from source
git clone https://github.com/henrycgbaker/llm-efficiency-measurement-tool.git
cd llm-efficiency-measurement-tool
pip install -e .
```

**For specific platforms:**
- **macOS (Apple Silicon)**: [See macOS instructions](INSTALLATION.md#macos-apple-silicon--intel)
- **Linux with CUDA**: [See CUDA instructions](INSTALLATION.md#linux-with-cuda)
- **Windows**: [See Windows instructions](INSTALLATION.md#windows-with-cuda)
- **CPU-only**: [See CPU-only instructions](INSTALLATION.md#linux-cpu-only)

📖 **[Full Installation Guide →](INSTALLATION.md)**

### Basic Usage

#### Python API

```python
from pathlib import Path
from llm_efficiency.config import ExperimentConfig
from llm_efficiency.core.experiment import run_experiment

# Configure experiment
config = ExperimentConfig(
    model_name="gpt2",
    precision="float16",
    batch_size=4,
    num_batches=20,
    max_length=128,
    output_dir=Path("./results"),
)

# Run experiment
result = run_experiment(config)

# View results
print(f"Throughput: {result.metrics.tokens_per_second:.2f} tokens/sec")
print(f"Energy: {result.metrics.total_energy_kwh:.6f} kWh")
print(f"CO2: {result.metrics.co2_emissions:.6f} kg")
```

#### Command Line Interface

```bash
# Interactive configuration wizard
llm-efficiency init

# Run experiment
llm-efficiency run experiment_config.yaml

# List all experiments
llm-efficiency list

# Show detailed results
llm-efficiency show <experiment-id>

# Export results
llm-efficiency export results.csv

# Generate summary statistics
llm-efficiency summary
```

See [`examples/`](examples/) for 10+ comprehensive examples.

## Examples

We provide extensive examples for all use cases:

| Example | Description | Level |
|---------|-------------|-------|
| [01_basic_experiment.py](examples/01_basic_experiment.py) | Simple experiment workflow | Beginner |
| [02_cli_usage.sh](examples/02_cli_usage.sh) | Complete CLI reference | Beginner |
| [03_profiling_example.py](examples/03_profiling_example.py) | Performance profiling | Intermediate |
| [04_caching_example.py](examples/04_caching_example.py) | Caching strategies | Intermediate |
| [05_multi_model_comparison.py](examples/05_multi_model_comparison.py) | Compare multiple models | Intermediate |
| [06_quantization_comparison.py](examples/06_quantization_comparison.py) | Quantization benchmarking | Advanced |
| [07_custom_config.py](examples/07_custom_config.py) | Advanced configuration | Intermediate |
| [08_results_analysis.py](examples/08_results_analysis.py) | Results analysis | Intermediate |
| [09_error_handling.py](examples/09_error_handling.py) | Error handling patterns | Intermediate |
| [10_advanced_workflow.py](examples/10_advanced_workflow.py) | Production workflows | Advanced |

**See [`examples/README.md`](examples/README.md) for detailed documentation.**

## Architecture

Modern Python package with clean separation of concerns:

```
llm-efficiency-measurement-tool/
├── src/llm_efficiency/           # Main package
│   ├── __version__.py            # Version info
│   ├── config.py                 # Configuration system
│   ├── cli/                      # Command-line interface
│   │   └── main.py               # CLI commands
│   ├── core/                     # Core functionality
│   │   ├── distributed.py        # Multi-GPU support
│   │   ├── model_loader.py       # Model loading
│   │   ├── inference.py          # Inference engine
│   │   └── experiment.py         # Experiment runner
│   ├── metrics/                  # Metrics calculation
│   │   ├── compute.py            # FLOPs calculation
│   │   └── energy.py             # Energy tracking
│   ├── storage/                  # Results persistence
│   │   └── results.py            # Results manager
│   └── utils/                    # Utilities
│       ├── exceptions.py         # Custom exceptions
│       ├── retry.py              # Retry logic
│       ├── profiling.py          # Performance profiling
│       └── cache.py              # Caching utilities
├── tests/                        # 186+ unit tests
│   ├── unit/                     # Unit tests
│   └── integration/              # Integration tests
├── examples/                     # 10+ example scripts
│   ├── 01_basic_experiment.py
│   ├── ...
│   └── README.md
└── pyproject.toml                # Modern packaging
```

## Configuration

### Python API

```python
from llm_efficiency.config import ExperimentConfig, QuantizationConfig

config = ExperimentConfig(
    # Model settings
    model_name="gpt2",
    precision="float16",  # float32, float16, bfloat16, float8

    # Quantization (optional)
    quantization=QuantizationConfig(
        enabled=True,
        load_in_4bit=True,
        quant_type="nf4",  # nf4 or fp4
    ),

    # Inference settings
    batch_size=8,
    num_batches=50,
    max_length=256,

    # Dataset
    dataset_name="wikitext",
    dataset_config="wikitext-2-raw-v1",

    # Output
    output_dir=Path("./results"),
)
```

### YAML Configuration

```yaml
# experiment_config.yaml
model_name: "gpt2"
precision: "float16"
batch_size: 8
num_batches: 50
max_length: 256

quantization:
  enabled: true
  load_in_4bit: true
  quant_type: "nf4"

dataset_name: "wikitext"
dataset_config: "wikitext-2-raw-v1"
output_dir: "./results"
```

Use with CLI:
```bash
llm-efficiency run experiment_config.yaml
```

## Metrics Collected

### Performance Metrics
- **Throughput**: Tokens per second
- **Latency**: Per token and per sample
- **Total tokens generated**
- **Execution time breakdown**

### Energy Metrics
- **Total energy consumption** (kWh)
- **Energy per token** (kWh/token)
- **CO2 emissions** (kg)
- **Energy cost estimation**

### Compute Metrics
- **Total FLOPs**
- **FLOPs per token**
- **Memory usage**
- **GPU utilization**

## Supported Features

### Precision Types
- `float32` - Full precision
- `float16` - Half precision (recommended)
- `bfloat16` - Brain floating point
- `float8` - 8-bit floating point (PyTorch 2.1+)

### Quantization Methods
- **4-bit NF4** - Normalized Float 4-bit (best quality)
- **4-bit FP4** - Float 4-bit
- **8-bit INT8** - 8-bit integer quantization

### Models Supported
Any Hugging Face model that works with `AutoModelForCausalLM`:
- GPT-2 family (gpt2, gpt2-medium, gpt2-large, gpt2-xl)
- LLaMA family (meta-llama/Llama-2-7b-hf, etc.)
- OPT family (facebook/opt-125m, facebook/opt-6.7b, etc.)
- And many more...

## Advanced Features

### Performance Profiling

```python
from llm_efficiency.utils import PerformanceProfiler

profiler = PerformanceProfiler()

with profiler.profile("model_loading"):
    model, tokenizer = load_model_and_tokenizer(config)

with profiler.profile("inference"):
    result = run_experiment(config)

print(profiler.get_summary())
profiler.save("profiling_results.json")
```

### Caching

```python
from llm_efficiency.utils import DiskCache, cached_with_ttl

# Disk cache for large objects
cache = DiskCache(cache_dir="./cache", ttl=3600)
cache.set("model_outputs", results)
results = cache.get("model_outputs")

# Function caching decorator
@cached_with_ttl(ttl=3600, maxsize=100)
def expensive_computation(x):
    return complex_calculation(x)
```

### Error Handling

```python
from llm_efficiency.utils.exceptions import (
    ModelLoadingError,
    QuantizationError,
)

try:
    result = run_experiment(config)
except QuantizationError as e:
    # Fallback to full precision
    config.quantization.enabled = False
    result = run_experiment(config)
except ModelLoadingError as e:
    logger.error(f"Failed to load model: {e}")
```

## Results Export

### CSV Export

```bash
# CLI
llm-efficiency export results.csv

# Python
from llm_efficiency.storage.results import ResultsManager

manager = ResultsManager(results_dir="./results")
manager.export_to_csv("results.csv")
```

### Pickle Export (Fastest)

```bash
# CLI
llm-efficiency export results.pkl --format pickle

# Python
manager.export_to_pickle("results.pkl")
```

### JSON Export

```bash
# CLI
llm-efficiency export results.json --format json

# Python
manager.export("results.json", format="json")
```

## Testing

Comprehensive test suite with 186+ tests:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/llm_efficiency --cov-report=html

# Run specific test categories
pytest -m "not slow"          # Skip slow tests
pytest -m integration         # Integration tests only
pytest -m gpu                 # GPU tests only
```

## Development

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/henrycgbaker/llm-efficiency-measurement-tool.git
cd llm-efficiency-measurement-tool

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest
```

### Code Quality

We use:
- **Ruff** for linting and formatting
- **MyPy** for type checking
- **pytest** for testing
- **pre-commit** for automated checks

```bash
# Run linter
ruff check src/ tests/

# Type checking
mypy src/

# Format code
ruff format src/ tests/
```

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and release notes.

**Current Version**: 1.5.0

### Recent Releases

- **v1.5.0** (2025-01-XX) - Performance profiling + advanced caching
- **v1.4.0** (2025-01-XX) - Production polish + error handling
- **v1.3.0** (2025-01-XX) - Modern CLI + comprehensive tests
- **v1.2.0** (2025-01-XX) - Core modules + quantization FLOPs fix

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run the test suite
5. Submit a pull request

## Known Issues

- Quantization requires CUDA/GPU (CPU not supported by bitsandbytes)
- Some models may require authentication (use `huggingface-cli login`)
- Large models (7B+) require significant GPU memory

## Requirements

- Python 3.10+
- PyTorch 2.1+
- CUDA 11.8+ (for GPU/quantization)
- 8GB+ RAM (16GB+ recommended)
- Disk space for models (~500MB - 50GB depending on model)

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{llm_efficiency_tool,
  title = {LLM Efficiency Measurement Tool},
  author = {Baker, Henry},
  year = {2025},
  url = {https://github.com/henrycgbaker/llm-efficiency-measurement-tool}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Author

**Henry Baker**

- GitHub: [@henrycgbaker](https://github.com/henrycgbaker)
- Repository: [llm-efficiency-measurement-tool](https://github.com/henrycgbaker/llm-efficiency-measurement-tool)

## Acknowledgments

Built with:
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [CodeCarbon](https://github.com/mlco2/codecarbon)
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)
- [Typer](https://github.com/tiangolo/typer) & [Rich](https://github.com/Textualize/rich)

---

**Happy benchmarking!** 🚀

For questions or issues, please [open an issue](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/issues).
