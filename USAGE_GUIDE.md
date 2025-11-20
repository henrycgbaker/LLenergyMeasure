# LLM Efficiency Measurement Tool - Usage Guide

Complete guide to measuring and analyzing LLM inference efficiency.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Running Experiments](#running-experiments)
5. [Analyzing Results](#analyzing-results)
6. [CLI Reference](#cli-reference)
7. [Python API Reference](#python-api-reference)
8. [Advanced Topics](#advanced-topics)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/henrycgbaker/llm-efficiency-measurement-tool.git
cd llm-efficiency-measurement-tool

# Install in editable mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### From PyPI (when published)

```bash
pip install llm-efficiency
```

### Verify Installation

```bash
# Check CLI is available
llm-efficiency --version

# Check Python import
python -c "import llm_efficiency; print(llm_efficiency.__version__)"
```

## Quick Start

### Your First Experiment

```python
from pathlib import Path
from llm_efficiency.config import ExperimentConfig
from llm_efficiency.core.experiment import run_experiment

# Create configuration
config = ExperimentConfig(
    model_name="gpt2",              # Small model for testing
    precision="float16",            # Half precision
    batch_size=4,                   # Process 4 samples at once
    num_batches=10,                 # Run 10 batches
    max_length=128,                 # Max 128 tokens per sample
    output_dir=Path("./results"),   # Where to save results
)

# Run experiment
result = run_experiment(config)

# Print results
print(f"Throughput: {result.metrics.tokens_per_second:.2f} tokens/sec")
print(f"Energy: {result.metrics.total_energy_kwh:.6f} kWh")
print(f"CO2: {result.metrics.co2_emissions:.6f} kg")
```

### Using the CLI

```bash
# Create configuration file interactively
llm-efficiency init

# Run experiment
llm-efficiency run experiment_config.yaml

# List results
llm-efficiency list

# View specific result
llm-efficiency show exp_20250120_123456_abc123

# Export all results
llm-efficiency export results.csv
```

## Configuration

### Configuration Structure

```python
from llm_efficiency.config import ExperimentConfig, QuantizationConfig

config = ExperimentConfig(
    # === Model Settings ===
    model_name="gpt2",              # Hugging Face model name
    precision="float16",            # float32, float16, bfloat16, float8

    # === Quantization Settings ===
    quantization=QuantizationConfig(
        enabled=True,               # Enable quantization
        load_in_4bit=True,          # Use 4-bit quantization
        load_in_8bit=False,         # Or 8-bit (not both)
        quant_type="nf4",           # "nf4" or "fp4" for 4-bit
        compute_dtype="float16",    # Compute dtype
    ),

    # === Inference Settings ===
    batch_size=8,                   # Samples per batch
    num_batches=50,                 # Total batches to run
    max_length=256,                 # Max tokens per generation
    temperature=1.0,                # Sampling temperature
    top_p=0.95,                     # Top-p (nucleus) sampling
    top_k=50,                       # Top-k sampling
    repetition_penalty=1.0,         # Repetition penalty

    # === Dataset Settings ===
    dataset_name="wikitext",        # Dataset from Hugging Face
    dataset_config="wikitext-2-raw-v1",  # Dataset configuration
    dataset_split="test",           # Dataset split to use

    # === Performance Settings ===
    use_cache=True,                 # Enable KV cache
    num_warmup_batches=5,           # Warmup batches (excluded from metrics)
    seed=42,                        # Random seed for reproducibility

    # === Output Settings ===
    output_dir=Path("./results"),   # Results directory
    save_frequency=10,              # Save every N batches
)
```

### YAML Configuration

Create `experiment_config.yaml`:

```yaml
# Model configuration
model_name: "gpt2"
precision: "float16"

# Quantization
quantization:
  enabled: false
  load_in_4bit: false
  load_in_8bit: false
  quant_type: null
  compute_dtype: "float16"

# Inference settings
batch_size: 8
num_batches: 50
max_length: 256
temperature: 1.0
top_p: 0.95
top_k: 50
repetition_penalty: 1.0

# Dataset
dataset_name: "wikitext"
dataset_config: "wikitext-2-raw-v1"
dataset_split: "test"

# Performance
use_cache: true
num_warmup_batches: 5
seed: 42

# Output
output_dir: "./results"
save_frequency: 10
```

Use with:
```bash
llm-efficiency run experiment_config.yaml
```

### Configuration Presets

```python
# Quick test (fast, minimal)
quick_config = ExperimentConfig(
    model_name="gpt2",
    batch_size=2,
    num_batches=5,
    max_length=64,
    output_dir=Path("./results/quick"),
)

# Production benchmark (comprehensive)
production_config = ExperimentConfig(
    model_name="gpt2-large",
    precision="float16",
    quantization=QuantizationConfig(enabled=True, load_in_4bit=True),
    batch_size=16,
    num_batches=100,
    max_length=512,
    num_warmup_batches=10,
    output_dir=Path("./results/production"),
)

# Energy-optimized (minimal consumption)
energy_config = ExperimentConfig(
    model_name="distilgpt2",
    precision="float16",
    quantization=QuantizationConfig(enabled=True, load_in_8bit=True),
    batch_size=32,  # Larger batches = better GPU utilization
    num_batches=50,
    use_cache=True,
    output_dir=Path("./results/energy"),
)
```

## Running Experiments

### Single Experiment

```python
from llm_efficiency.config import ExperimentConfig
from llm_efficiency.core.experiment import run_experiment

config = ExperimentConfig(
    model_name="gpt2",
    batch_size=4,
    num_batches=20,
    output_dir=Path("./results/single"),
)

result = run_experiment(config)
```

### Multiple Models Comparison

```python
models = ["gpt2", "gpt2-medium", "gpt2-large"]
results = []

for model in models:
    config = ExperimentConfig(
        model_name=model,
        batch_size=4,
        num_batches=20,
        output_dir=Path(f"./results/{model}"),
    )
    result = run_experiment(config)
    results.append(result)

# Compare results
for result in results:
    print(f"{result.config.model_name}:")
    print(f"  Throughput: {result.metrics.tokens_per_second:.2f} tok/s")
    print(f"  Energy: {result.metrics.energy_per_token:.8f} kWh/tok")
```

### Quantization Comparison

```python
from llm_efficiency.config import QuantizationConfig

quantization_configs = [
    ("Full Precision", QuantizationConfig(enabled=False)),
    ("8-bit", QuantizationConfig(enabled=True, load_in_8bit=True)),
    ("4-bit NF4", QuantizationConfig(enabled=True, load_in_4bit=True, quant_type="nf4")),
]

for name, quant_config in quantization_configs:
    config = ExperimentConfig(
        model_name="gpt2",
        quantization=quant_config,
        batch_size=4,
        num_batches=20,
        output_dir=Path(f"./results/quant_{name.replace(' ', '_').lower()}"),
    )

    try:
        result = run_experiment(config)
        print(f"{name}: {result.metrics.tokens_per_second:.2f} tok/s")
    except Exception as e:
        print(f"{name}: Failed - {e}")
```

### With Performance Profiling

```python
from llm_efficiency.utils import PerformanceProfiler

profiler = PerformanceProfiler()

with profiler.profile("full_experiment"):
    result = run_experiment(config)

# Print profiling summary
print(profiler.get_summary())

# Save profiling data
profiler.save("profiling_results.json")
```

### With Caching

```python
from llm_efficiency.utils import DiskCache

cache = DiskCache(cache_dir="./cache", ttl=86400)  # 24h TTL

# Check cache first
cache_key = f"{config.model_name}:{config.precision}"
cached_result = cache.get(cache_key)

if cached_result:
    print("Using cached result")
    result = cached_result
else:
    print("Running new experiment")
    result = run_experiment(config)
    cache.set(cache_key, result)
```

## Analyzing Results

### Loading Results

```python
from llm_efficiency.storage.results import ResultsManager

# Initialize manager
manager = ResultsManager(results_dir=Path("./results"))

# List all experiments
experiment_ids = manager.list_experiments()
print(f"Found {len(experiment_ids)} experiments")

# Load specific experiment
result = manager.load_result(experiment_ids[0])

# Access metrics
metrics = result.metrics
print(f"Throughput: {metrics.tokens_per_second:.2f} tokens/sec")
print(f"Energy: {metrics.total_energy_kwh:.6f} kWh")
print(f"CO2: {metrics.co2_emissions:.6f} kg")
```

### Exporting Results

#### CSV Export

```python
# Export all results
manager.export_to_csv("all_results.csv")

# Export specific experiments
manager.export_to_csv(
    "filtered_results.csv",
    experiment_ids=["exp1", "exp2", "exp3"]
)
```

#### Pickle Export (Fast)

```python
# Fastest for large datasets
manager.export_to_pickle("results.pkl")
```

#### JSON Export

```python
manager.export("results.json", format="json")
```

### Statistical Analysis

```python
import statistics

# Load all results
results = [manager.load_result(exp_id) for exp_id in manager.list_experiments()]

# Calculate statistics
throughputs = [r.metrics.tokens_per_second for r in results]
energies = [r.metrics.energy_per_token for r in results]

print(f"Throughput - Mean: {statistics.mean(throughputs):.2f} tok/s")
print(f"Throughput - Median: {statistics.median(throughputs):.2f} tok/s")
print(f"Throughput - Std Dev: {statistics.stdev(throughputs):.2f} tok/s")

print(f"Energy - Mean: {statistics.mean(energies):.8f} kWh/tok")
print(f"Energy - Min: {min(energies):.8f} kWh/tok")
print(f"Energy - Max: {max(energies):.8f} kWh/tok")
```

### Grouping by Model

```python
from collections import defaultdict

# Group results by model
by_model = defaultdict(list)
for result in results:
    by_model[result.config.model_name].append(result)

# Analyze each model
for model, model_results in by_model.items():
    throughputs = [r.metrics.tokens_per_second for r in model_results]
    avg_throughput = statistics.mean(throughputs)
    print(f"{model}: {avg_throughput:.2f} tok/s (avg over {len(model_results)} runs)")
```

## CLI Reference

### Available Commands

```bash
llm-efficiency --help
```

### init - Create Configuration

```bash
# Interactive wizard
llm-efficiency init

# Specify output file
llm-efficiency init --output my_config.yaml
```

### run - Run Experiment

```bash
# From config file
llm-efficiency run experiment_config.yaml

# Override config parameters
llm-efficiency run config.yaml \
    --model gpt2-medium \
    --batch-size 8 \
    --num-batches 50

# Specify output directory
llm-efficiency run config.yaml --output-dir ./results/custom
```

### list - List Experiments

```bash
# List all experiments
llm-efficiency list

# Filter by model
llm-efficiency list --model gpt2

# Limit results
llm-efficiency list --limit 10

# Sort by throughput
llm-efficiency list --sort throughput
```

### show - Show Details

```bash
# Show experiment details
llm-efficiency show exp_20250120_123456_abc123

# Show with full configuration
llm-efficiency show exp_20250120_123456_abc123 --verbose
```

### export - Export Results

```bash
# Export to CSV
llm-efficiency export results.csv

# Export to JSON
llm-efficiency export results.json --format json

# Export to pickle
llm-efficiency export results.pkl --format pickle

# Export specific experiments
llm-efficiency export filtered.csv --ids exp1,exp2,exp3
```

### summary - Generate Summary

```bash
# Overall summary
llm-efficiency summary

# Group by model
llm-efficiency summary --group-by model

# Group by precision
llm-efficiency summary --group-by precision
```

## Python API Reference

### Core Functions

#### run_experiment()

```python
from llm_efficiency.core.experiment import run_experiment

result = run_experiment(config: ExperimentConfig) -> ExperimentResult
```

Runs a complete efficiency experiment.

**Parameters:**
- `config`: ExperimentConfig - Experiment configuration

**Returns:**
- `ExperimentResult` - Complete experiment results

**Raises:**
- `ModelLoadingError` - Model cannot be loaded
- `InferenceError` - Inference fails
- `ConfigurationError` - Invalid configuration

### Configuration Classes

#### ExperimentConfig

```python
from llm_efficiency.config import ExperimentConfig

config = ExperimentConfig(
    model_name: str,
    precision: str = "float16",
    quantization: QuantizationConfig = QuantizationConfig(),
    batch_size: int = 8,
    num_batches: int = 100,
    max_length: int = 512,
    # ... see Configuration section for full parameters
)
```

#### QuantizationConfig

```python
from llm_efficiency.config import QuantizationConfig

quant_config = QuantizationConfig(
    enabled: bool = False,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    quant_type: Optional[str] = None,  # "nf4" or "fp4"
    compute_dtype: str = "float16",
)
```

### Results Classes

#### ExperimentResult

```python
@dataclass
class ExperimentResult:
    experiment_id: str              # Unique experiment identifier
    timestamp: str                  # ISO format timestamp
    config: ExperimentConfig        # Experiment configuration
    metrics: EfficiencyMetrics      # All collected metrics
```

#### EfficiencyMetrics

```python
@dataclass
class EfficiencyMetrics:
    # Performance metrics
    total_samples: int
    total_tokens: int
    tokens_per_second: float
    latency_per_token: float
    latency_per_sample: float

    # Energy metrics
    total_energy_kwh: float
    energy_per_token: float
    energy_per_sample: float
    co2_emissions: float

    # Compute metrics
    total_flops: int
    flops_per_token: int
    flops_per_sample: int
```

### Utility Functions

#### Performance Profiling

```python
from llm_efficiency.utils import PerformanceProfiler, profile_function, timer

# Class-based profiling
profiler = PerformanceProfiler()
with profiler.profile("operation_name"):
    # Your code here
    pass

# Decorator-based profiling
@profile_function(name="my_function")
def my_function():
    pass

# Context manager timing
with timer("simple_timing"):
    # Your code here
    pass
```

#### Caching

```python
from llm_efficiency.utils import LRUCacheWithTTL, DiskCache, cached_with_ttl

# In-memory cache with TTL
cache = LRUCacheWithTTL(maxsize=100, ttl=3600)
cache.set("key", "value")
value = cache.get("key")

# Disk cache
disk_cache = DiskCache(cache_dir="./cache", ttl=86400)
disk_cache.set("key", large_object)
large_object = disk_cache.get("key")

# Decorator caching
@cached_with_ttl(ttl=3600, maxsize=100)
def expensive_function(x):
    return complex_calculation(x)
```

## Advanced Topics

### Multi-GPU Experiments

```python
# Experiments automatically use all available GPUs
# No code changes needed - handled by Accelerate

config = ExperimentConfig(
    model_name="gpt2-large",
    # ... other settings
)

result = run_experiment(config)  # Uses all GPUs automatically
```

### Custom Datasets

```python
# Use any Hugging Face dataset
config = ExperimentConfig(
    model_name="gpt2",
    dataset_name="openwebtext",  # Custom dataset
    dataset_config=None,         # No config needed
    dataset_split="train",       # Use train split
    # ... other settings
)
```

### Reproducibility

```python
# Set seed for reproducible results
config = ExperimentConfig(
    model_name="gpt2",
    seed=42,  # Fixed seed
    # ... other settings
)

# Multiple runs will produce identical results (same seed, config)
result1 = run_experiment(config)
result2 = run_experiment(config)
assert result1.metrics.total_tokens == result2.metrics.total_tokens
```

### Error Handling & Retry Logic

```python
from llm_efficiency.utils.exceptions import ModelLoadingError, QuantizationError
from llm_efficiency.utils.retry import retry_with_exponential_backoff

@retry_with_exponential_backoff(max_retries=3, initial_delay=2.0)
def run_with_retry(config):
    try:
        return run_experiment(config)
    except QuantizationError:
        # Fallback to full precision
        config.quantization.enabled = False
        return run_experiment(config)
    except ModelLoadingError as e:
        logger.error(f"Model loading failed: {e}")
        raise
```

## Best Practices

### 1. Start Small, Scale Up

```python
# Development: Quick iterations
dev_config = ExperimentConfig(
    model_name="gpt2",
    batch_size=2,
    num_batches=5,
    max_length=64,
)

# Production: Full scale
prod_config = ExperimentConfig(
    model_name="gpt2-large",
    batch_size=16,
    num_batches=100,
    max_length=512,
)
```

### 2. Use Warmup Batches

```python
# Exclude first few batches from metrics (model loading, CUDA init, etc.)
config = ExperimentConfig(
    model_name="gpt2",
    num_warmup_batches=5,  # First 5 batches excluded
    num_batches=55,        # 50 measured batches + 5 warmup
)
```

### 3. Enable Caching for Repeated Experiments

```python
from llm_efficiency.utils import DiskCache

cache = DiskCache(cache_dir="./experiment_cache", ttl=86400)

def run_cached_experiment(config):
    cache_key = f"{config.model_name}:{config.precision}:{config.batch_size}"

    result = cache.get(cache_key)
    if result:
        return result

    result = run_experiment(config)
    cache.set(cache_key, result)
    return result
```

### 4. Profile Before Optimizing

```python
from llm_efficiency.utils import PerformanceProfiler

profiler = PerformanceProfiler()

with profiler.profile("model_loading"):
    model, tokenizer = load_model_and_tokenizer(config)

with profiler.profile("data_prep"):
    dataset = prepare_dataset(config)

with profiler.profile("inference"):
    result = run_inference(model, dataset)

# Identify bottlenecks
print(profiler.get_summary())
```

### 5. Handle Errors Gracefully

```python
try:
    result = run_experiment(config)
except QuantizationError:
    # Fallback to full precision
    config.quantization.enabled = False
    result = run_experiment(config)
except torch.cuda.OutOfMemoryError:
    # Reduce batch size
    config.batch_size = max(1, config.batch_size // 2)
    result = run_experiment(config)
```

## Troubleshooting

### Out of Memory (OOM)

**Problem:** `torch.cuda.OutOfMemoryError`

**Solutions:**
1. Reduce batch size: `batch_size=1` or `batch_size=2`
2. Reduce max length: `max_length=128`
3. Enable quantization: `load_in_4bit=True`
4. Use smaller model: `gpt2` instead of `gpt2-large`

### Model Loading Fails

**Problem:** `ModelLoadingError: Could not load model`

**Solutions:**
1. Check internet connection
2. Verify model name on [Hugging Face](https://huggingface.co/models)
3. Check disk space (models can be large)
4. Try with `trust_remote_code=True` (for custom models)
5. Authenticate: `huggingface-cli login`

### Quantization Not Working

**Problem:** `QuantizationError: requires CUDA/GPU`

**Solutions:**
1. Verify CUDA availability:
   ```python
   import torch
   print(torch.cuda.is_available())
   ```
2. Use CPU-compatible config (disable quantization)
3. Install proper CUDA version
4. Check bitsandbytes installation: `pip install bitsandbytes>=0.39.0`

### Slow Performance

**Problem:** Experiments taking too long

**Solutions:**
1. Reduce `num_batches` for testing
2. Reduce `max_length`
3. Use smaller model
4. Enable GPU: set `CUDA_VISIBLE_DEVICES`
5. Check if running on CPU (much slower)

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'llm_efficiency'`

**Solutions:**
1. Verify installation: `pip list | grep llm-efficiency`
2. Reinstall: `pip install -e .`
3. Check Python version: `python --version` (requires 3.10+)
4. Activate correct environment

---

For more examples, see the [`examples/`](examples/) directory.

For issues and questions, please [open an issue](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/issues).
