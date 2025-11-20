# LLM Efficiency Examples

This directory contains comprehensive examples demonstrating all features of the LLM Efficiency Measurement Tool.

## Quick Start

Begin with these examples in order:

1. **[01_basic_experiment.py](01_basic_experiment.py)** - Start here! Simple experiment with GPT-2
2. **[02_cli_usage.sh](02_cli_usage.sh)** - Learn the command-line interface
3. **[03_profiling_example.py](03_profiling_example.py)** - Performance profiling basics

## Examples by Category

### Getting Started

- **01_basic_experiment.py** - Simplest possible experiment
  - Load model and tokenizer
  - Run inference with energy tracking
  - View and save results
  - ~5 minutes runtime

### Command Line Interface

- **02_cli_usage.sh** - Complete CLI reference
  - All available commands
  - Interactive configuration wizard
  - Listing and filtering results
  - Exporting data
  - Real-world workflows

### Performance Optimization

- **03_profiling_example.py** - Performance profiling
  - Track execution time and memory
  - CPU utilization monitoring
  - Bottleneck identification
  - Export profiling data

- **04_caching_example.py** - Caching strategies
  - LRU cache with TTL
  - Disk-based persistent caching
  - Function result caching
  - Hit rate optimization

### Model Comparison

- **05_multi_model_comparison.py** - Compare multiple models
  - Automated testing of multiple models
  - Side-by-side metrics comparison
  - Performance rankings
  - Efficiency insights

- **06_quantization_comparison.py** - Quantization benchmarking
  - Full precision vs quantized
  - 4-bit vs 8-bit comparison
  - Quality/efficiency trade-offs
  - Memory savings analysis

### Configuration

- **07_custom_config.py** - Advanced configuration
  - Programmatic configuration
  - YAML config files
  - Configuration templates
  - Environment-specific configs
  - Validation patterns

### Analysis & Reporting

- **08_results_analysis.py** - Results analysis
  - Load and inspect results
  - Statistical analysis
  - Export to CSV/JSON/pickle
  - Custom metrics calculation

- **09_error_handling.py** - Error handling
  - Common errors and solutions
  - Retry mechanisms
  - Graceful degradation
  - Production-ready patterns

### Advanced Usage

- **10_advanced_workflow.py** - Production workflows
  - Automated benchmarking
  - Profiling integration
  - Custom metrics
  - Comprehensive reporting

## Running the Examples

### Prerequisites

```bash
# Install the package
pip install llm-efficiency

# Or install from source
cd /path/to/llm-efficiency-measurement-tool
pip install -e .
```

### Run an Example

```bash
# Python examples
python examples/01_basic_experiment.py
python examples/03_profiling_example.py

# Shell script
bash examples/02_cli_usage.sh
```

### System Requirements

- **CPU Examples**: All examples can run on CPU (may be slow)
- **GPU Recommended**: For quantization examples (06) and production use
- **Disk Space**: ~500MB for GPT-2, more for larger models
- **Memory**: 8GB+ RAM recommended
- **Internet**: Required for downloading models

## Example Output

Most examples will create output in these directories:

```
./results/           # Experiment results (JSON files)
./exports/           # Exported data (CSV, pickle)
./cache/             # Cached data
./profiling_results.json  # Performance profiling data
```

## Learning Path

### Beginner Path (30 minutes)

1. `01_basic_experiment.py` - Understand the basics
2. `02_cli_usage.sh` - Learn CLI commands
3. `07_custom_config.py` - Create custom configurations
4. `08_results_analysis.py` - Analyze your results

### Intermediate Path (1 hour)

1. Complete Beginner Path
2. `05_multi_model_comparison.py` - Compare models
3. `03_profiling_example.py` - Profile performance
4. `04_caching_example.py` - Optimize with caching
5. `09_error_handling.py` - Handle errors gracefully

### Advanced Path (2 hours)

1. Complete Intermediate Path
2. `06_quantization_comparison.py` - Master quantization
3. `10_advanced_workflow.py` - Production workflows
4. Combine examples for custom workflows

## Common Use Cases

### Use Case 1: Quick Model Evaluation

```bash
# Run basic experiment
python examples/01_basic_experiment.py

# View results via CLI
llm-efficiency list
llm-efficiency show <experiment-id>
```

### Use Case 2: Compare Model Sizes

```python
# Edit 05_multi_model_comparison.py
models = [
    "gpt2",           # 124M
    "gpt2-medium",    # 355M
    "gpt2-large",     # 774M
]

# Run comparison
python examples/05_multi_model_comparison.py
```

### Use Case 3: Find Best Quantization

```python
# Run quantization comparison
python examples/06_quantization_comparison.py

# Review report
cat results/quantization_comparison/quantization_report.txt
```

### Use Case 4: Production Benchmarking

```python
# Use advanced workflow
python examples/10_advanced_workflow.py

# Or use CLI batch processing
for model in gpt2 gpt2-medium; do
    llm-efficiency run config.yaml --model $model
done

llm-efficiency export benchmark_results.csv
llm-efficiency summary --group-by model
```

## Tips and Tricks

### Speed Up Examples

For faster testing, reduce `num_batches` in any example:

```python
config = ExperimentConfig(
    # ... other settings ...
    num_batches=5,  # Instead of 20-50
)
```

### Use Smaller Models

Start with small models for quick iterations:

```python
models = [
    "gpt2",              # ✓ Fast
    "distilgpt2",        # ✓ Faster
    # "gpt2-xl",         # ✗ Slow without GPU
]
```

### Enable Caching

For repeated experiments, enable caching:

```python
from llm_efficiency.utils import DiskCache

cache = DiskCache(cache_dir="./cache", ttl=3600)
# Cache your results
```

### Profile Everything

Add profiling to understand performance:

```python
from llm_efficiency.utils import PerformanceProfiler

profiler = PerformanceProfiler()
with profiler.profile("my_operation"):
    # Your code here
    pass

print(profiler.get_summary())
```

## Troubleshooting

### Out of Memory Error

**Problem**: `torch.cuda.OutOfMemoryError`

**Solution**:
1. Reduce `batch_size` (try 1, 2, 4)
2. Reduce `max_length` (try 64, 128)
3. Enable quantization (4-bit or 8-bit)
4. Use smaller model

### Model Not Found

**Problem**: `ModelLoadingError: Could not load model`

**Solution**:
1. Check internet connection
2. Verify model name on [Hugging Face](https://huggingface.co/models)
3. Check disk space (~500MB per model)
4. Try with `gpt2` (always available)

### Quantization Fails

**Problem**: `QuantizationError: requires CUDA/GPU`

**Solution**:
1. Check CUDA availability: `torch.cuda.is_available()`
2. Use CPU-compatible config (disable quantization)
3. Use float16/float32 precision

### Slow Performance

**Problem**: Examples taking too long

**Solution**:
1. Reduce `num_batches` (try 5-10)
2. Reduce `max_length` (try 64-128)
3. Use smaller model (`gpt2` instead of `gpt2-large`)
4. Enable GPU if available

## Next Steps

After completing the examples:

1. **Read the Documentation**
   - Main README: `../README.md`
   - Usage Guide: `../USAGE_GUIDE.md` (if available)
   - API Documentation: Check docstrings

2. **Customize for Your Needs**
   - Modify examples for your models
   - Create custom configurations
   - Build automated workflows

3. **Share Your Results**
   - Export to CSV for analysis
   - Create custom visualizations
   - Share efficiency insights

4. **Contribute**
   - Report issues
   - Suggest improvements
   - Add new examples

## Additional Resources

- **Repository**: https://github.com/henrycgbaker/llm-efficiency-measurement-tool
- **Issues**: https://github.com/henrycgbaker/llm-efficiency-measurement-tool/issues
- **Hugging Face Models**: https://huggingface.co/models

## Example Index

| # | Name | Category | Difficulty | Runtime |
|---|------|----------|------------|---------|
| 01 | Basic Experiment | Getting Started | Easy | 5 min |
| 02 | CLI Usage | CLI | Easy | - |
| 03 | Profiling | Performance | Medium | 2 min |
| 04 | Caching | Performance | Medium | 2 min |
| 05 | Multi-Model | Comparison | Medium | 15 min |
| 06 | Quantization | Comparison | Advanced | 20 min |
| 07 | Custom Config | Configuration | Medium | - |
| 08 | Results Analysis | Analysis | Medium | 5 min |
| 09 | Error Handling | Best Practices | Medium | 2 min |
| 10 | Advanced Workflow | Production | Advanced | 30 min |

**Total learning time**: ~2-3 hours for all examples

---

**Happy benchmarking!** 🚀
