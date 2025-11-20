# LLM Efficiency Measurement Tool - Refactoring & Modernization Strategy

**Author:** Henry Baker
**Date:** 2025-11-20
**Version:** 1.0

---

## Executive Summary

This document outlines a comprehensive strategy to modernize and refactor the LLM Efficiency Measurement Tool. The codebase was originally developed as a thesis project and requires significant improvements to meet modern software engineering standards, improve maintainability, fix critical bugs, and incorporate state-of-the-art practices.

### Key Objectives

1. **Modularize Architecture**: Restructure code into clean, maintainable packages with clear separation of concerns
2. **Fix Critical Bugs**: Resolve quantized model FLOPs calculation issue
3. **Modernize Tooling**: Upgrade to modern libraries and best practices
4. **Add Comprehensive Testing**: Implement pytest-based test suite
5. **Improve Code Quality**: Apply type hints, linting, formatting standards
6. **Enhance Performance**: Optimize inference and metrics collection
7. **Better Documentation**: Maintain up-to-date documentation alongside code
8. **Production-Ready**: Transform from research code to production-quality tool

---

## Current State Analysis

### Strengths

1. **Comprehensive Metrics**: Excellent coverage of energy, performance, and compute metrics
2. **Flexible Configuration**: Well-designed configuration system with systematic variations
3. **Distributed Support**: Good integration with Hugging Face Accelerate
4. **Persistent Progress**: Smart progress tracking for long-running experiments
5. **Multiple Entry Points**: Various experiment modes (single, models, controlled, scenarios, grid)

### Critical Issues

#### 1. **Quantized Model FLOPs Calculation (HIGH PRIORITY)**

**Problem:**
```python
# configs/a_default_config.py:56
"cached_flops_for_quantised_models": 52638582308864  # Same value for ALL models!
```

The `ptflops` library cannot compute FLOPs for quantized models, so the code falls back to a hardcoded value. This is fundamentally broken - different model sizes have vastly different FLOPs.

**Impact:**
- All quantized model experiments report incorrect FLOPs
- Energy efficiency metrics (FLOPs/Joule) are meaningless
- Cannot compare quantized vs non-quantized models accurately

**Root Cause:**
`ptflops` analyzes the model graph, but quantized models have modified layers that `ptflops` doesn't understand.

#### 2. **Poor Code Organization**

- Alphabetic prefixes (a_, b_, c_) are non-standard and confusing
- Utilities mixed with core logic
- No clear package structure
- Duplicate code (e.g., `l_results_csv_cleaning.py` has functions from `j_results_saving.py`)
- Debug print statements instead of logging
- Minimal error handling

#### 3. **Lack of Testing**

- Zero test coverage
- No CI/CD
- Manual validation only
- High risk of regressions

#### 4. **Outdated Dependencies & Practices**

- 305 dependencies (excessive)
- Some deprecated libraries
- No dependency management (requirements.txt only)
- No virtual environment management
- No type hints
- No linting/formatting standards

#### 5. **Hard to Maintain**

- Nested dictionaries for configs (no IDE support)
- String-based keys prone to typos
- Long functions (>200 lines)
- Tight coupling between modules
- No abstractions/interfaces

---

## Proposed Architecture

### New Directory Structure

```
llm-efficiency-measurement-tool/
├── pyproject.toml                  # Modern Python project config
├── setup.py                        # Package setup
├── README.md                       # Main documentation
├── CHANGELOG.md                    # Version history
├── LICENSE                         # License file
│
├── docs/                           # Comprehensive documentation
│   ├── index.md
│   ├── quickstart.md
│   ├── configuration.md
│   ├── architecture.md
│   ├── api_reference.md
│   └── contributing.md
│
├── src/                            # Source code package
│   └── llm_efficiency/
│       ├── __init__.py
│       ├── __version__.py
│       │
│       ├── core/                   # Core functionality
│       │   ├── __init__.py
│       │   ├── distributed.py      # Distributed setup
│       │   ├── model_loader.py     # Model loading
│       │   ├── inference.py        # Inference engine
│       │   └── experiment_runner.py# Main runner
│       │
│       ├── config/                 # Configuration management
│       │   ├── __init__.py
│       │   ├── base.py             # Base config classes
│       │   ├── models.py           # Model configs
│       │   ├── scenarios.py        # Scenario configs
│       │   ├── validation.py       # Config validation
│       │   └── presets/            # Pre-defined configs
│       │       ├── controlled.py
│       │       ├── grid_search.py
│       │       └── scenarios.py
│       │
│       ├── metrics/                # Metrics collection
│       │   ├── __init__.py
│       │   ├── base.py             # Base metrics classes
│       │   ├── inference.py        # Inference metrics
│       │   ├── compute.py          # Compute metrics (FLOPs)
│       │   ├── energy.py           # Energy metrics
│       │   └── collectors.py       # Metric collectors
│       │
│       ├── data/                   # Data handling
│       │   ├── __init__.py
│       │   ├── datasets.py         # Dataset loading
│       │   ├── prompts.py          # Prompt processing
│       │   └── batching.py         # Batching strategies
│       │
│       ├── utils/                  # Utilities
│       │   ├── __init__.py
│       │   ├── logging.py          # Logging setup
│       │   ├── gpu.py              # GPU utilities
│       │   ├── timing.py           # Timing utilities
│       │   └── serialization.py    # JSON/CSV utilities
│       │
│       ├── storage/                # Results storage
│       │   ├── __init__.py
│       │   ├── results.py          # Results management
│       │   ├── formats/            # Output formats
│       │   │   ├── json.py
│       │   │   └── csv.py
│       │   └── progress.py         # Progress tracking
│       │
│       └── cli/                    # Command-line interface
│           ├── __init__.py
│           ├── main.py             # Main CLI entry
│           ├── commands/           # CLI commands
│           │   ├── single.py
│           │   ├── models.py
│           │   ├── controlled.py
│           │   ├── scenarios.py
│           │   └── grid.py
│           └── utils.py
│
├── tests/                          # Test suite
│   ├── __init__.py
│   ├── conftest.py                 # Pytest fixtures
│   ├── unit/                       # Unit tests
│   │   ├── test_config.py
│   │   ├── test_metrics.py
│   │   ├── test_inference.py
│   │   └── test_utils.py
│   ├── integration/                # Integration tests
│   │   ├── test_experiment_runner.py
│   │   └── test_full_workflow.py
│   └── fixtures/                   # Test fixtures
│       ├── configs.py
│       ├── models.py
│       └── data.py
│
├── scripts/                        # Utility scripts
│   ├── port_cleanup.py
│   ├── workspace_cleanup.sh
│   └── migrate_old_configs.py
│
├── examples/                       # Example usage
│   ├── simple_experiment.py
│   ├── custom_config.py
│   └── distributed_inference.py
│
└── .github/                        # CI/CD
    └── workflows/
        ├── tests.yml
        ├── lint.yml
        └── release.yml
```

### Key Architectural Changes

#### 1. **Package-Based Structure**
- Install via `pip install -e .`
- Import as `from llm_efficiency.core import ExperimentRunner`
- Clear namespace organization
- Better IDE support

#### 2. **Class-Based Metrics System**

**Before:**
```python
# Multiple disconnected functions
flops = get_flops(model, inputs)
memory = get_memory(device)
utilization = get_gpu_cpu_utilisation(device)
```

**After:**
```python
# Unified metrics collector
from llm_efficiency.metrics import MetricsCollector

collector = MetricsCollector(config)
collector.start()
# ... run inference ...
metrics = collector.stop()

# Access metrics
print(metrics.compute.flops)
print(metrics.energy.total_kwh)
print(metrics.inference.tokens_per_second)
```

#### 3. **Pydantic-Based Configuration**

**Before:**
```python
# Nested dictionaries, no validation
config = {
    "batching_options": {
        "batch_size___fixed_batching": 16
    }
}
```

**After:**
```python
# Type-safe, validated config
from llm_efficiency.config import ExperimentConfig, BatchingConfig

config = ExperimentConfig(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    batching=BatchingConfig(
        batch_size=16,
        adaptive=False
    ),
    precision="float16"
)
```

#### 4. **Modern CLI with Typer**

**Before:**
```bash
python MAIN_a_single_experiment.py  # No arguments
```

**After:**
```bash
llm-efficiency run single --model TinyLlama-1.1B --precision float16 --batch-size 16
llm-efficiency run models --suite controlled --cycles 5
llm-efficiency run grid --output results/grid_search.csv
llm-efficiency config validate my_config.yaml
llm-efficiency results analyze results/
```

---

## Critical Bug Fix: Quantized Model FLOPs

### Solution Strategy

Since `ptflops` cannot analyze quantized models, we'll compute FLOPs **before** quantization and cache it per model architecture.

### Implementation

#### Step 1: Create FLOPs Calculator

```python
# src/llm_efficiency/metrics/compute.py

from typing import Dict, Optional
import torch
import ptflops
from transformers import PreTrainedModel
import json
from pathlib import Path

class FLOPsCalculator:
    """
    Calculates FLOPs for models, with caching and support for quantized models.
    """

    def __init__(self, cache_dir: Path = Path("~/.cache/llm_efficiency/flops")):
        self.cache_dir = cache_dir.expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "flops_cache.json"
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict:
        """Load FLOPs cache from disk."""
        if self.cache_file.exists():
            with open(self.cache_file, "r") as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        """Save FLOPs cache to disk."""
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f, indent=2)

    def _get_cache_key(self, model_name: str, sequence_length: int) -> str:
        """Generate cache key for model and sequence length."""
        return f"{model_name}::{sequence_length}"

    def _compute_flops_unquantized(
        self,
        model: PreTrainedModel,
        sequence_length: int,
        device: torch.device
    ) -> int:
        """Compute FLOPs for unquantized model using ptflops."""
        def input_constructor(input_res):
            dummy_input = torch.zeros((1,) + input_res, dtype=torch.long).to(device)
            attention_mask = torch.ones_like(dummy_input)
            return {"input_ids": dummy_input, "attention_mask": attention_mask}

        flops, _ = ptflops.get_model_complexity_info(
            model,
            input_res=(sequence_length,),
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
            input_constructor=input_constructor
        )
        return flops

    def _estimate_flops_from_architecture(
        self,
        model: PreTrainedModel,
        sequence_length: int,
        num_output_tokens: int = 1
    ) -> int:
        """
        Estimate FLOPs based on model architecture parameters.

        For Transformer models:
        FLOPs per token ≈ 6 * N + 12 * L * H² * S
        Where:
        - N = total parameters
        - L = number of layers
        - H = hidden dimension
        - S = sequence length
        """
        config = model.config

        # Extract architecture parameters
        num_layers = getattr(config, "num_hidden_layers", 0)
        hidden_size = getattr(config, "hidden_size", 0)
        vocab_size = getattr(config, "vocab_size", 0)
        num_params = sum(p.numel() for p in model.parameters())

        if num_layers == 0 or hidden_size == 0:
            raise ValueError("Cannot estimate FLOPs: missing architecture info")

        # FLOPs for forward pass (per token)
        # Attention: 4 * L * H² * S (Q, K, V projections + output)
        attention_flops = 4 * num_layers * hidden_size ** 2 * sequence_length

        # FFN: 2 * L * H * (4H) = 8 * L * H²
        ffn_flops = 8 * num_layers * hidden_size ** 2

        # Embedding + output projection: H * V
        embedding_flops = hidden_size * vocab_size

        # Total per token
        flops_per_token = attention_flops + ffn_flops + embedding_flops

        # Total for generation (input + output tokens)
        total_tokens = sequence_length + num_output_tokens
        total_flops = flops_per_token * total_tokens

        return int(total_flops)

    def get_flops(
        self,
        model: PreTrainedModel,
        model_name: str,
        sequence_length: int,
        device: torch.device,
        is_quantized: bool = False,
        force_recompute: bool = False
    ) -> int:
        """
        Get FLOPs for a model, using cache when available.

        Args:
            model: The model to analyze
            model_name: Model identifier for caching
            sequence_length: Input sequence length
            device: Device model is on
            is_quantized: Whether model is quantized
            force_recompute: Force recomputation even if cached

        Returns:
            FLOPs as integer
        """
        cache_key = self._get_cache_key(model_name, sequence_length)

        # Check cache first
        if not force_recompute and cache_key in self.cache:
            return self.cache[cache_key]

        # Compute FLOPs
        if is_quantized:
            # For quantized models, we need the unquantized version
            # Strategy 1: Try architectural estimation
            try:
                flops = self._estimate_flops_from_architecture(
                    model, sequence_length
                )
            except Exception as e:
                # Strategy 2: Load unquantized model and compute
                from transformers import AutoModelForCausalLM

                print(f"Loading unquantized model to compute FLOPs...")
                temp_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="cpu"  # Load on CPU to save GPU memory
                )
                flops = self._compute_flops_unquantized(
                    temp_model, sequence_length, torch.device("cpu")
                )
                del temp_model
                torch.cuda.empty_cache()
        else:
            # Unquantized model - direct computation
            flops = self._compute_flops_unquantized(model, sequence_length, device)

        # Cache result
        self.cache[cache_key] = flops
        self._save_cache()

        return flops

    def get_flops_batch(
        self,
        model: PreTrainedModel,
        model_name: str,
        input_ids: torch.Tensor,
        device: torch.device,
        is_quantized: bool = False
    ) -> int:
        """
        Get FLOPs for a batch of inputs.

        Args:
            model: The model
            model_name: Model identifier
            input_ids: Batch of input token IDs [batch_size, seq_len]
            device: Device
            is_quantized: Whether model is quantized

        Returns:
            Total FLOPs for the batch
        """
        batch_size = input_ids.shape[0]
        sequence_lengths = [input_ids[i].shape[0] for i in range(batch_size)]

        # If all same length, compute once and multiply
        if len(set(sequence_lengths)) == 1:
            flops_single = self.get_flops(
                model, model_name, sequence_lengths[0], device, is_quantized
            )
            return flops_single * batch_size

        # Variable lengths - compute per sample
        total_flops = 0
        for seq_len in sequence_lengths:
            flops = self.get_flops(
                model, model_name, seq_len, device, is_quantized
            )
            total_flops += flops

        return total_flops
```

#### Step 2: Update Config to Remove Hardcoded FLOPs

```python
# src/llm_efficiency/config/base.py

from pydantic import BaseModel, Field
from typing import Optional, Literal

class QuantizationConfig(BaseModel):
    """Quantization configuration."""

    enabled: bool = False
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    compute_dtype: Literal["float16", "bfloat16"] = "float16"
    quant_type: Optional[Literal["nf4", "fp4", "int8"]] = None

    # NO MORE cached_flops_for_quantised_models!
```

#### Step 3: Update Metrics Collector

```python
# src/llm_efficiency/metrics/compute.py

from .compute import FLOPsCalculator

class ComputeMetrics:
    """Compute metrics collector."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.flops_calculator = FLOPsCalculator()

    def collect(
        self,
        model: PreTrainedModel,
        input_ids: torch.Tensor,
        device: torch.device
    ) -> Dict:
        """Collect compute metrics."""

        # Determine if model is quantized
        is_quantized = self.config.quantization.enabled

        # Compute FLOPs correctly, even for quantized models
        flops = self.flops_calculator.get_flops_batch(
            model=model,
            model_name=self.config.model_name,
            input_ids=input_ids,
            device=device,
            is_quantized=is_quantized
        )

        # Memory stats
        memory_stats = self._get_memory_stats(device)

        # Utilization
        utilization = self._get_utilization()

        return {
            "flops": flops,
            "memory": memory_stats,
            "utilization": utilization
        }
```

### Benefits of This Approach

1. **Accurate FLOPs**: Correct FLOPs for all models, quantized or not
2. **Cached**: Computations cached, no redundant calculations
3. **Fallback Strategy**: Multiple methods (ptflops → architectural estimation → unquantized loading)
4. **Per-Model Accuracy**: Each model gets its own correct FLOPs value
5. **Efficient**: Only loads unquantized model once for caching

---

## Modernization Plan

### 1. Dependency Management

#### Current: requirements.txt (305 packages)
```
torch
transformers
accelerate
...
```

#### Proposed: pyproject.toml + Poetry/uv

```toml
[tool.poetry]
name = "llm-efficiency"
version = "0.1.0"
description = "Comprehensive LLM efficiency measurement framework"
authors = ["Henry Baker"]

[tool.poetry.dependencies]
python = "^3.10"
torch = "^2.1.0"
transformers = "^4.36.0"
accelerate = "^0.25.0"
codecarbon = "^2.3.0"
pydantic = "^2.5.0"
typer = { extras = ["all"], version = "^0.9.0" }
rich = "^13.7.0"  # Beautiful terminal output
datasets = "^2.16.0"
bitsandbytes = "^0.41.0"

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.0"
pytest-cov = "^4.1.0"
pytest-mock = "^3.12.0"
ruff = "^0.1.9"  # Modern, fast linter/formatter
mypy = "^1.8.0"  # Type checking
pre-commit = "^3.6.0"

[tool.poetry.scripts]
llm-efficiency = "llm_efficiency.cli.main:app"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

**Benefits:**
- Dependency resolution (no conflicts)
- Lock files for reproducibility
- Separate dev/prod dependencies
- Virtual environment management
- Much fewer dependencies (core only)

### 2. Code Quality Tools

#### Ruff (Linter + Formatter)

Replace Black, isort, flake8, pylint with one tool:

```toml
[tool.ruff]
target-version = "py310"
line-length = 100
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "UP",  # pyupgrade
]

[tool.ruff.isort]
known-first-party = ["llm_efficiency"]
```

#### MyPy (Type Checking)

```toml
[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

#### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/python/mypy
    rev: v1.8.0
    hooks:
      - id: mypy
```

### 3. Type Hints

Add comprehensive type hints to all functions:

**Before:**
```python
def load_model_tokenizer(configs):
    model_name = configs.model_name
    # ...
    return model, tokenizer
```

**After:**
```python
from typing import Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer

def load_model_tokenizer(
    config: ExperimentConfig
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load model and tokenizer from Hugging Face.

    Args:
        config: Experiment configuration

    Returns:
        Tuple of (model, tokenizer)

    Raises:
        ValueError: If model cannot be loaded
    """
    model_name: str = config.model_name
    # ...
    return model, tokenizer
```

### 4. Logging Instead of Print

**Before:**
```python
print(f"[DEBUG] Computing FLOPs for sample {i}")
```

**After:**
```python
import logging

logger = logging.getLogger(__name__)

logger.debug("Computing FLOPs for sample %d", i)
```

**Benefits:**
- Configurable log levels
- Log to files
- Structured logging
- No debug prints in production

### 5. Modern CLI with Rich Output

```python
# src/llm_efficiency/cli/main.py

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.table import Table

app = typer.Typer()
console = Console()

@app.command()
def run_single(
    model: str = typer.Option(..., help="Model name from Hugging Face"),
    precision: str = typer.Option("float16", help="Precision: float32/float16/bfloat16"),
    batch_size: int = typer.Option(16, help="Batch size"),
    num_prompts: int = typer.Option(128, help="Number of prompts"),
    quantization: Optional[str] = typer.Option(None, help="Quantization: 4bit/8bit"),
):
    """Run a single experiment."""

    console.print(f"[bold green]Running experiment[/bold green]")
    console.print(f"Model: {model}")
    console.print(f"Precision: {precision}")

    # Create config
    config = ExperimentConfig(
        model_name=model,
        precision=precision,
        batching=BatchingConfig(batch_size=batch_size),
        num_prompts=num_prompts
    )

    # Run with progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console
    ) as progress:

        task1 = progress.add_task("Loading model...", total=None)
        # Load model
        progress.update(task1, completed=True)

        task2 = progress.add_task("Running inference...", total=num_prompts)
        # Run inference with updates
        progress.update(task2, advance=1)

    # Display results
    table = Table(title="Experiment Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Throughput", f"{results.tokens_per_second:.2f} tokens/s")
    table.add_row("Energy", f"{results.total_energy_kwh:.4f} kWh")
    table.add_row("FLOPs", f"{results.flops:,}")

    console.print(table)
```

---

## Testing Strategy

### Test Structure

```
tests/
├── unit/                           # Fast, isolated tests
│   ├── test_config.py              # Config validation, parsing
│   ├── test_metrics.py             # Metrics calculations
│   ├── test_batching.py            # Batching strategies
│   ├── test_flops_calculator.py    # FLOPs calculation
│   └── test_utils.py               # Utility functions
│
├── integration/                    # Multi-component tests
│   ├── test_experiment_runner.py   # Full experiment workflow
│   ├── test_distributed.py         # Distributed execution
│   └── test_storage.py             # Results saving/loading
│
├── e2e/                            # End-to-end tests
│   └── test_full_workflow.py       # Complete workflow with tiny model
│
└── fixtures/                       # Shared test fixtures
    ├── configs.py                  # Test configurations
    ├── models.py                   # Mock models
    └── data.py                     # Test datasets
```

### Key Test Cases

#### 1. Config Validation Tests

```python
# tests/unit/test_config.py

import pytest
from llm_efficiency.config import ExperimentConfig, BatchingConfig

def test_config_validation_valid():
    """Test valid configuration."""
    config = ExperimentConfig(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        precision="float16",
        batching=BatchingConfig(batch_size=16)
    )
    assert config.model_name == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    assert config.precision == "float16"

def test_config_validation_invalid_precision():
    """Test invalid precision raises error."""
    with pytest.raises(ValueError, match="Invalid precision"):
        ExperimentConfig(
            model_name="model",
            precision="float128"  # Invalid
        )

def test_config_from_dict():
    """Test creating config from dictionary."""
    config_dict = {
        "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "precision": "float16"
    }
    config = ExperimentConfig(**config_dict)
    assert config.model_name == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

#### 2. FLOPs Calculator Tests

```python
# tests/unit/test_flops_calculator.py

import pytest
import torch
from llm_efficiency.metrics.compute import FLOPsCalculator
from transformers import AutoModelForCausalLM

@pytest.fixture
def flops_calculator(tmp_path):
    """Create FLOPs calculator with temp cache."""
    return FLOPsCalculator(cache_dir=tmp_path / "flops_cache")

@pytest.fixture
def tiny_model():
    """Load tiny model for testing."""
    return AutoModelForCausalLM.from_pretrained(
        "hf-internal-testing/tiny-random-gpt2",
        torch_dtype=torch.float16
    )

def test_flops_calculation_unquantized(flops_calculator, tiny_model):
    """Test FLOPs calculation for unquantized model."""
    flops = flops_calculator.get_flops(
        model=tiny_model,
        model_name="tiny-gpt2",
        sequence_length=128,
        device=torch.device("cpu"),
        is_quantized=False
    )
    assert flops > 0
    assert isinstance(flops, int)

def test_flops_caching(flops_calculator, tiny_model):
    """Test that FLOPs are cached."""
    # First call - computed
    flops1 = flops_calculator.get_flops(
        tiny_model, "tiny-gpt2", 128, torch.device("cpu"), False
    )

    # Second call - from cache (should be fast)
    flops2 = flops_calculator.get_flops(
        tiny_model, "tiny-gpt2", 128, torch.device("cpu"), False
    )

    assert flops1 == flops2

def test_flops_batch_uniform_length(flops_calculator, tiny_model):
    """Test batch FLOPs with uniform length."""
    batch_size = 4
    seq_len = 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))

    total_flops = flops_calculator.get_flops_batch(
        model=tiny_model,
        model_name="tiny-gpt2",
        input_ids=input_ids,
        device=torch.device("cpu"),
        is_quantized=False
    )

    single_flops = flops_calculator.get_flops(
        tiny_model, "tiny-gpt2", seq_len, torch.device("cpu"), False
    )

    assert total_flops == single_flops * batch_size
```

#### 3. Experiment Runner Tests

```python
# tests/integration/test_experiment_runner.py

import pytest
from llm_efficiency.core.experiment_runner import ExperimentRunner
from llm_efficiency.config import ExperimentConfig

@pytest.fixture
def test_config():
    """Create test configuration with tiny model and few prompts."""
    return ExperimentConfig(
        model_name="hf-internal-testing/tiny-random-gpt2",
        precision="float16",
        num_prompts=10,  # Small for testing
        max_output_tokens=16,
        batching=BatchingConfig(batch_size=2)
    )

def test_experiment_runner_setup(test_config):
    """Test experiment runner setup."""
    runner = ExperimentRunner(test_config)
    runner.setup()

    assert runner.experiment_id is not None
    assert runner.accelerator is not None

def test_experiment_runner_full_workflow(test_config, tmp_path):
    """Test complete experiment workflow."""
    # Override results directory
    test_config.results_dir = tmp_path / "results"

    runner = ExperimentRunner(test_config)
    runner.setup()
    results = runner.run()

    # Verify results structure
    assert "inference_metrics" in results
    assert "compute_metrics" in results
    assert "energy_metrics" in results

    # Verify metrics
    assert results["inference_metrics"]["tokens_per_second"] > 0
    assert results["compute_metrics"]["flops"] > 0

    # Verify results saved
    assert (tmp_path / "results").exists()
```

#### 4. Integration Tests

```python
# tests/integration/test_full_workflow.py

import pytest
from llm_efficiency.core.experiment_runner import ExperimentRunner
from llm_efficiency.config.presets import get_test_config

def test_end_to_end_workflow(tmp_path):
    """Test complete end-to-end workflow."""

    # Get test config
    config = get_test_config()
    config.results_dir = tmp_path / "results"

    # Run experiment
    runner = ExperimentRunner(config)
    runner.setup()
    results = runner.run()

    # Verify results
    assert results is not None

    # Verify files created
    results_dir = tmp_path / "results" / "raw_results" / runner.experiment_id
    assert results_dir.exists()

    # Verify JSON files
    assert (results_dir / f"{runner.experiment_id}_1_experiment_setup.json").exists()
    assert (results_dir / f"{runner.experiment_id}_4_inference_metrics.json").exists()

    # Load and verify JSON
    import json
    with open(results_dir / f"{runner.experiment_id}_4_inference_metrics.json") as f:
        metrics = json.load(f)
        assert "tokens_per_second" in metrics
```

### Test Coverage Goals

- **Unit Tests**: 90%+ coverage
- **Integration Tests**: All major workflows
- **E2E Tests**: At least one complete workflow

### CI/CD Integration

```yaml
# .github/workflows/tests.yml

name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11"]

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        pip install poetry
        poetry install

    - name: Run tests
      run: |
        poetry run pytest --cov=src/llm_efficiency --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

**Priority: Critical**

1. **Setup New Project Structure**
   - Create `src/llm_efficiency` package
   - Setup `pyproject.toml`
   - Configure Poetry/uv
   - Setup pre-commit hooks

2. **Fix Quantized Model FLOPs Bug**
   - Implement `FLOPsCalculator` class
   - Add caching system
   - Add architectural estimation fallback
   - Write comprehensive tests

3. **Migrate Core Modules**
   - `distributed.py` → `src/llm_efficiency/core/distributed.py`
   - `model_loader.py` → `src/llm_efficiency/core/model_loader.py`
   - Add type hints
   - Add logging
   - Add tests

**Deliverables:**
- ✅ Working package structure
- ✅ Fixed FLOPs calculation
- ✅ Tests passing
- ✅ CI/CD running

### Phase 2: Configuration System (Week 3)

**Priority: High**

1. **Pydantic Config Models**
   - Create base config classes
   - Add validation
   - Migrate from dicts

2. **Config Presets**
   - Migrate existing config suites
   - Add new presets
   - Add YAML/JSON support

3. **Tests**
   - Config validation tests
   - Preset tests

**Deliverables:**
- ✅ Type-safe configs
- ✅ YAML/JSON config files
- ✅ Comprehensive validation

### Phase 3: Metrics Refactor (Week 4)

**Priority: High**

1. **Metrics Classes**
   - `MetricsCollector` base class
   - `InferenceMetrics`
   - `ComputeMetrics`
   - `EnergyMetrics`

2. **Unified Collection**
   - Single metrics collection API
   - Async collection support
   - Real-time streaming

3. **Tests**
   - Unit tests for each metric
   - Integration tests

**Deliverables:**
- ✅ Clean metrics API
- ✅ Better separation of concerns
- ✅ Tested metrics

### Phase 4: Storage & Results (Week 5)

**Priority: Medium**

1. **Results Management**
   - Refactor results saving
   - Add result querying
   - Add result comparison

2. **Multiple Formats**
   - JSON (structured)
   - CSV (tabular)
   - Parquet (efficient)
   - HDF5 (for large datasets)

3. **Tests**
   - Storage tests
   - Format conversion tests

**Deliverables:**
- ✅ Flexible storage system
- ✅ Multiple output formats
- ✅ Query API

### Phase 5: CLI & UX (Week 6)

**Priority: Medium**

1. **Modern CLI**
   - Typer-based CLI
   - Rich output
   - Progress bars
   - Interactive prompts

2. **Commands**
   - `run single`
   - `run models`
   - `run controlled`
   - `run grid`
   - `config validate`
   - `results analyze`

3. **Documentation**
   - CLI help
   - Examples
   - Tutorials

**Deliverables:**
- ✅ User-friendly CLI
- ✅ Beautiful output
- ✅ Good DX

### Phase 6: Testing & Polish (Week 7-8)

**Priority: High**

1. **Comprehensive Tests**
   - Increase coverage to 90%+
   - Add integration tests
   - Add E2E tests

2. **Documentation**
   - API documentation
   - User guide
   - Architecture docs
   - Contributing guide

3. **Performance Optimization**
   - Profile code
   - Optimize bottlenecks
   - Reduce memory usage

**Deliverables:**
- ✅ 90%+ test coverage
- ✅ Complete documentation
- ✅ Optimized performance

### Phase 7: Migration & Cleanup (Week 9)

**Priority: Medium**

1. **Migration Tools**
   - Script to migrate old configs
   - Script to migrate old results
   - Backward compatibility layer

2. **Deprecate Old Code**
   - Move old code to `legacy/`
   - Add deprecation warnings
   - Update main scripts

3. **Final Testing**
   - Test migration scripts
   - Verify backward compatibility
   - Test with real experiments

**Deliverables:**
- ✅ Migration complete
- ✅ Old code deprecated
- ✅ Smooth transition

---

## Migration Strategy

### Backward Compatibility

During transition, support both old and new interfaces:

```python
# src/llm_efficiency/legacy/__init__.py

import warnings
from llm_efficiency.core import ExperimentRunner as NewExperimentRunner

class ExperimentRunner(NewExperimentRunner):
    """Legacy wrapper for backward compatibility."""

    def __init__(self, config_dict):
        warnings.warn(
            "Using legacy ExperimentRunner. Please migrate to new config system.",
            DeprecationWarning,
            stacklevel=2
        )

        # Convert old dict config to new Pydantic config
        from llm_efficiency.config import ExperimentConfig
        new_config = ExperimentConfig.from_legacy_dict(config_dict)

        super().__init__(new_config)
```

### Migration Script

```python
# scripts/migrate_old_configs.py

"""Migrate old dictionary configs to new Pydantic configs."""

import json
from pathlib import Path
from llm_efficiency.config import ExperimentConfig

def migrate_config_dict(old_config: dict) -> ExperimentConfig:
    """Convert old config dict to new Pydantic model."""

    # Map old keys to new structure
    new_config = ExperimentConfig(
        model_name=old_config["model_name"],
        precision=old_config["fp_precision"],
        batching=BatchingConfig(
            batch_size=old_config["batching_options"]["batch_size___fixed_batching"],
            adaptive=old_config["batching_options"]["adaptive_batching"]
        ),
        # ... map other fields
    )

    return new_config

def migrate_file(input_path: Path, output_path: Path):
    """Migrate config file."""

    with open(input_path) as f:
        old_config = json.load(f)

    new_config = migrate_config_dict(old_config)

    with open(output_path, "w") as f:
        f.write(new_config.model_dump_json(indent=2))

if __name__ == "__main__":
    # Migrate all config files
    pass
```

---

## Risk Analysis

### High Risk Items

1. **Breaking Changes**
   - **Risk**: New architecture breaks existing workflows
   - **Mitigation**: Backward compatibility layer, migration scripts, thorough testing

2. **FLOPs Calculation Regression**
   - **Risk**: New FLOPs calculator introduces errors
   - **Mitigation**: Extensive testing, validation against known models, fallback strategies

3. **Performance Degradation**
   - **Risk**: New abstractions slow down experiments
   - **Mitigation**: Profiling, benchmarking, optimization before release

### Medium Risk Items

1. **Migration Complexity**
   - **Risk**: Users struggle to migrate
   - **Mitigation**: Clear docs, automated migration tools, gradual deprecation

2. **Dependency Conflicts**
   - **Risk**: New dependencies conflict with user environments
   - **Mitigation**: Poetry lock files, Docker containers, minimal dependencies

### Low Risk Items

1. **Learning Curve**
   - **Risk**: New CLI/API harder to use
   - **Mitigation**: Better docs, examples, tutorials, CLI help

---

## Success Metrics

### Code Quality
- [ ] 90%+ test coverage
- [ ] 100% type hint coverage
- [ ] Zero Ruff errors
- [ ] Zero MyPy errors
- [ ] All tests passing in CI

### Performance
- [ ] No slowdown in inference time
- [ ] <5% overhead from metrics collection
- [ ] Reduced memory usage

### Usability
- [ ] 50% reduction in lines of code for common tasks
- [ ] <5 minute setup time for new users
- [ ] Clear error messages
- [ ] Interactive CLI

### Correctness
- [ ] Accurate FLOPs for all models (quantized and unquantized)
- [ ] Energy metrics within 2% of CodeCarbon baseline
- [ ] Inference metrics match reference implementations

---

## Conclusion

This refactoring strategy transforms the LLM Efficiency Measurement Tool from research code into a production-ready package. The phased approach ensures minimal disruption while systematically addressing technical debt, fixing critical bugs, and modernizing the codebase.

**Key Improvements:**
1. ✅ Fixed quantized model FLOPs calculation
2. ✅ Modern package structure
3. ✅ Type-safe configuration
4. ✅ Comprehensive testing
5. ✅ Clean architecture
6. ✅ User-friendly CLI
7. ✅ Better documentation
8. ✅ Industry-standard tooling

**Timeline:** 9 weeks for complete refactor

**Next Steps:**
1. Review and approve strategy
2. Set up new project structure
3. Implement Phase 1 (Foundation + FLOPs fix)
4. Proceed with remaining phases

---

**Author:** Henry Baker
**Last Updated:** 2025-11-20
