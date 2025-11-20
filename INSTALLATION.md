# Installation Guide

Complete installation instructions for the LLM Efficiency Measurement Tool.

## Prerequisites

- Python 3.10 or higher
- pip package manager
- Git (for development installation)

## Platform-Specific Installation

### macOS (Apple Silicon / Intel)

PyTorch requires special installation on macOS. Install in this order:

```bash
# 1. Install PyTorch first (with MPS support for Apple Silicon)
pip install torch torchvision torchaudio

# 2. Clone the repository
git clone https://github.com/henrycgbaker/llm-efficiency-measurement-tool.git
cd llm-efficiency-measurement-tool

# 3. Install the package
pip install -e .
```

**Note for Apple Silicon (M1/M2/M3):**
- MPS (Metal Performance Shaders) backend is available for GPU acceleration
- `bitsandbytes` may not work on Apple Silicon (quantization will be disabled)

### Linux (with CUDA)

```bash
# 1. Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 2. Clone the repository
git clone https://github.com/henrycgbaker/llm-efficiency-measurement-tool.git
cd llm-efficiency-measurement-tool

# 3. Install the package
pip install -e .
```

### Linux (CPU only)

```bash
# 1. Install PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 2. Clone the repository
git clone https://github.com/henrycgbaker/llm-efficiency-measurement-tool.git
cd llm-efficiency-measurement-tool

# 3. Install the package
pip install -e .
```

### Windows (with CUDA)

```bash
# 1. Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 2. Clone the repository
git clone https://github.com/henrycgbaker/llm-efficiency-measurement-tool.git
cd llm-efficiency-measurement-tool

# 3. Install the package
pip install -e .
```

## Troubleshooting

### bitsandbytes Installation Fails

**Issue:** `bitsandbytes` requires CUDA and won't install on CPU-only or Apple Silicon systems.

**Solution:** Install without quantization support:

```bash
# Remove bitsandbytes from requirements temporarily
pip install -e . --no-deps
pip install torch transformers accelerate codecarbon pydantic typer rich datasets ptflops psutil pynvml
```

Then disable quantization in your experiments:
```python
config = ExperimentConfig(
    model_name="gpt2",
    quantization=QuantizationConfig(enabled=False),  # Disable quantization
)
```

### CUDA Version Mismatch

**Issue:** PyTorch CUDA version doesn't match your system CUDA.

**Solution:** Check your CUDA version and install matching PyTorch:

```bash
# Check CUDA version
nvidia-smi

# Install PyTorch for CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Or for CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### codecarbon Fails to Track Energy

**Issue:** Energy tracking not working on your platform.

**Solution:** codecarbon has limited platform support. It works best on:
- Linux with Intel RAPL
- Windows with Intel Power Gadget
- Limited macOS support

The tool will still work without energy tracking - other metrics will be collected.

### psutil Not Found

**Issue:** `ModuleNotFoundError: No module named 'psutil'`

**Solution:** Install psutil separately:

```bash
pip install psutil>=5.9.0
```

## Development Installation

For contributing to the project:

```bash
# 1. Install PyTorch (platform-specific, see above)
pip install torch

# 2. Clone and install with dev dependencies
git clone https://github.com/henrycgbaker/llm-efficiency-measurement-tool.git
cd llm-efficiency-measurement-tool
pip install -e ".[dev]"

# 3. Install pre-commit hooks
pre-commit install
```

## Verification

Verify your installation:

```bash
# Check version
llm-efficiency --version

# Or in Python
python -c "from llm_efficiency import __version__; print(__version__)"
```

Expected output: `2.0.0`

## Minimal Installation (No GPU)

For testing or CPU-only usage:

```bash
# Install PyTorch CPU version
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Clone repository
git clone https://github.com/henrycgbaker/llm-efficiency-measurement-tool.git
cd llm-efficiency-measurement-tool

# Install without bitsandbytes
pip install transformers accelerate codecarbon pydantic typer rich datasets ptflops psutil pynvml
pip install -e . --no-deps
```

## Docker Installation (Coming Soon)

Docker support is planned for v2.1.0 to simplify installation across platforms.

## Getting Help

If you encounter installation issues:

1. Check the [Troubleshooting](#troubleshooting) section above
2. Search existing [GitHub Issues](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/issues)
3. Create a new issue with:
   - Your platform (OS, Python version)
   - Full error message
   - Installation method attempted

## Next Steps

After successful installation, see:
- [USAGE_GUIDE.md](USAGE_GUIDE.md) - Complete usage guide
- [examples/](examples/) - 10+ example scripts
- [README.md](README.md) - Quick start guide
