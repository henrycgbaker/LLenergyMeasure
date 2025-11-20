# LLM Efficiency Measurement Tool

A comprehensive framework for measuring and analyzing Large Language Model (LLM) inference efficiency, performance, and energy consumption across various hardware configurations, precision levels, and optimization strategies.

## Overview

This tool enables systematic evaluation of LLM inference across multiple dimensions:
- **Energy Consumption**: CPU, GPU, and RAM power usage with carbon emissions tracking
- **Performance Metrics**: Latency, throughput, tokens per second, queries per second
- **Computational Efficiency**: FLOPs calculation and compute utilization
- **Model Variations**: Different model sizes, precision levels, and quantization methods
- **Optimization Strategies**: Batching, parallelization, and decoding configurations

## Features

- 🔬 **Comprehensive Metrics Collection**: Energy, performance, compute, and architectural data
- ⚡ **Distributed Execution**: Multi-GPU support via Hugging Face Accelerate
- 🔄 **Flexible Configuration System**: Base configs with systematic parameter variations
- 💾 **Persistent Progress Tracking**: Resume long-running experiments after interruptions
- 📊 **Multiple Experiment Modes**: Single runs, model comparisons, controlled variations, scenarios, and grid searches
- 🌍 **Energy & Emissions Tracking**: Powered by CodeCarbon for environmental impact analysis

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

**Run a single experiment:**
```bash
python MAIN_a_single_experiment.py
```

**Compare different models:**
```bash
python MAIN_b_diff_models.py
```

**Run controlled parameter experiments:**
```bash
python MAIN_c_controlled_experiments.py
```

**Execute pre-defined scenarios:**
```bash
python MAIN_d_scenarios.py
```

**Run grid search:**
```bash
python MAIN_e_grid_search.py
```

**Orchestrate large experimental suites:**
```bash
python MAIN_run_experimental_suite.py
```

## Architecture

The tool is organized into modular components:

```
llm-efficiency-measurement-tool/
├── configs/                              # Configuration management
│   ├── README.md                        # Config system documentation
│   ├── a_default_config.py              # Base configuration parameters
│   ├── config_class.py                  # Type-safe config dataclass
│   ├── config_utils.py                  # Config manipulation utilities
│   ├── b_models_config.py               # Model variations
│   ├── c_controlled_configs.py          # Controlled experiments
│   ├── d_scenario_configs.py            # Scenario definitions
│   └── e_grid_configs.py                # Grid search configurations
│
├── experiment_core_utils/                # Core measurement & inference
│   ├── README.md                        # Core utilities documentation
│   ├── a_distributed.py                 # Distributed setup
│   ├── b_model_loader.py                # Model & tokenizer loading
│   ├── c_prompt_processing.py           # Prompt preparation
│   ├── d_energy_tracking.py             # Energy monitoring
│   ├── e_inference.py                   # Inference execution
│   ├── f_experiment_info.py             # Experiment metadata
│   ├── g_metrics_inference.py           # Inference metrics
│   ├── h_metrics_compute.py             # Compute metrics (FLOPs)
│   ├── i_metrics_energy.py              # Energy metrics
│   ├── j_results_saving.py              # Results persistence
│   ├── k_results_aggregation.py         # Results aggregation
│   └── l_results_csv_cleaning.py        # CSV formatting
│
├── experiment_orchestration_utils/       # Experiment execution
│   ├── README.md                        # Orchestration documentation
│   ├── a_experiment_runner_class.py     # Main ExperimentRunner
│   ├── b_single_config_workflow.py      # Retry logic
│   └── c_launcher_utils.py              # Accelerate integration
│
├── persistent_progress_trackers/         # State persistence
│   ├── README.md                        # Progress tracking documentation
│   ├── experiment_id.txt                # Experiment counter
│   ├── cycle_id.txt                     # Cycle counter
│   └── configs_run_progress.json        # Completion tracking
│
├── MAIN_a_single_experiment.py          # Entry: single experiment
├── MAIN_b_diff_models.py                # Entry: model comparison
├── MAIN_c_controlled_experiments.py     # Entry: controlled variations
├── MAIN_d_scenarios.py                  # Entry: scenario execution
├── MAIN_e_grid_search.py                # Entry: grid search
├── MAIN_run_experimental_suite.py       # Entry: full suite orchestration
│
├── port_cleanup.py                      # Distributed port cleanup utility
├── thesis_workspace_cleanup.sh          # Disk cleanup script
└── requirements.txt                     # Python dependencies
```

## Configuration System

The tool uses a hierarchical configuration system:

1. **Base Config** (`configs/a_default_config.py`): Default parameters for all experiments
2. **Config Variations**: Systematic parameter variations for specific experiments
3. **Config Class** (`configs/config_class.py`): Type-safe configuration handling
4. **Config Utilities** (`configs/config_utils.py`): Helper functions for config manipulation

See [`configs/README.md`](configs/README.md) for detailed documentation.

## Supported Models

Currently configured models:
- TinyLlama 1.1B
- Llama-3.2 (1B, 3B)
- Llama-3.1 (8B)
- Custom models via Hugging Face Hub

## Supported Configurations

### Precision Levels
- `float32`: Full precision
- `float16`: Half precision
- `bfloat16`: Brain floating point
- `float8`: 8-bit floating point

### Quantization
- 8-bit quantization (INT8)
- 4-bit quantization (INT4/NF4)
- Powered by BitsAndBytes

### Batching Strategies
- Fixed batch sizes (1-64)
- Adaptive batching

### Parallelization
- Single-GPU execution
- Multi-GPU distributed inference (2-4 GPUs)

### Decoding Methods
- Greedy decoding
- Top-K sampling
- Top-P (nucleus) sampling
- Temperature-based sampling

### Latency Simulation
- Constant delays
- Bursty traffic patterns

## Results

Results are saved in multiple formats:

### Raw Results (`results/raw_results/{experiment_id}/`)
- Experiment setup and configuration
- Model architecture details
- Inference metrics
- Compute metrics (FLOPs, memory, utilization)
- Energy results (per-process and global)
- Generated text/tokens (optional)

### Aggregated Results
- `results/{task_type}_results.json`: Complete results in JSON format
- `results/{task_type}_results.csv`: Flattened tabular format for analysis

## Progress Tracking

The tool maintains persistent state for resumability:
- `experiment_id.txt`: Auto-incrementing experiment counter
- `cycle_id.txt`: Current experimental cycle
- `configs_run_progress.json`: Completion status for each configuration

## Documentation

Detailed documentation for each component:
- [`configs/README.md`](configs/README.md): Configuration system
- [`experiment_core_utils/README.md`](experiment_core_utils/README.md): Core utilities
- [`experiment_orchestration_utils/README.md`](experiment_orchestration_utils/README.md): Orchestration
- [`persistent_progress_trackers/README.md`](persistent_progress_trackers/README.md): Progress tracking

## Requirements

Key dependencies:
- PyTorch ecosystem (`torch`, `transformers`, `accelerate`)
- Energy monitoring (`codecarbon`)
- FLOPs calculation (`ptflops`)
- Quantization (`bitsandbytes`)
- Data loading (`datasets`)

See `requirements.txt` for complete dependency list.

## Utilities

### Port Cleanup
```bash
python port_cleanup.py
```
Cleans up distributed training ports that may be left open after interrupted experiments.

### Workspace Cleanup
```bash
bash thesis_workspace_cleanup.sh
```
Analyzes and cleans up disk usage in the workspace.

## Known Issues

1. **Quantized Model FLOPs**: Currently uses hardcoded FLOPs values for quantized models due to `ptflops` compatibility issues. This is being addressed in an upcoming refactor.

## Author

Henry Baker

## License

[Add license information]

## Citation

If you use this tool in your research, please cite:

```bibtex
[Add citation information]
```
