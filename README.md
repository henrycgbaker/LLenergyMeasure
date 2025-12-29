# LLM Energy Measure

A Python framework for measuring LLM inference efficiency, including energy consumption, throughput, and computational metrics (FLOPs). Designed for distributed GPU benchmarking using HuggingFace models.

## Features

- **Energy Measurement**: Track GPU, CPU, and RAM energy consumption via CodeCarbon
- **Throughput Metrics**: Tokens/second, latency per token, batch processing stats
- **FLOPs Estimation**: Multiple estimation methods (calflops, architecture-based, parameter-based)
- **Multi-GPU Support**: Distributed inference via `accelerate` with per-process result tracking
- **Late Aggregation**: Raw per-GPU results are saved separately and aggregated on demand
- **Flexible Configuration**: YAML/JSON configs with inheritance support
- **Docker Ready**: GPU-enabled containerization with docker-compose

## Installation

### Local (Poetry)

```bash
# Install dependencies
poetry install

# With development dependencies
poetry install --with dev
```

### Docker

```bash
# Build the image
docker compose build

# Run with GPU access
docker compose run --rm llm-energy-measure --help
```

## Quick Start

### 1. Create a Configuration

Create a YAML config file (e.g., `configs/my_experiment.yaml`):

```yaml
config_name: llama2-7b-benchmark
model_name: meta-llama/Llama-2-7b-hf

# Token limits
max_input_tokens: 512
max_output_tokens: 128

# Distributed setup
gpu_list: [0, 1]
num_processes: 2

# Precision
fp_precision: float16
```

### 2. Validate Configuration

```bash
llm-energy-measure config validate configs/my_experiment.yaml
```

### 3. Run Experiment

For multi-GPU experiments, use `accelerate launch`:

```bash
accelerate launch --num_processes 2 \
  -m llm_energy_measure.orchestration.launcher \
  --config configs/my_experiment.yaml \
  --prompts prompts.txt
```

### 4. Aggregate Results

After running, aggregate per-process results:

```bash
# Aggregate a specific experiment
llm-energy-measure aggregate exp_20240115_123456

# Aggregate all pending experiments
llm-energy-measure aggregate --all
```

### 5. View Results

```bash
# List all experiments
llm-energy-measure results list

# Show aggregated results
llm-energy-measure results show exp_20240115_123456

# Show per-process raw results
llm-energy-measure results show exp_20240115_123456 --raw

# Export as JSON
llm-energy-measure results show exp_20240115_123456 --json
```

## CLI Reference

```
llm-energy-measure [OPTIONS] COMMAND [ARGS]

Commands:
  run         Run an LLM efficiency experiment
  aggregate   Aggregate raw per-process results
  config      Configuration management
    validate  Validate a config file
    show      Display resolved configuration
  results     Results inspection
    list      List all experiments
    show      Show experiment results
```

### Global Options

| Option | Description |
|--------|-------------|
| `--version, -v` | Show version |
| `--verbose` | Enable debug logging |

## Configuration

### Config Structure

```yaml
# Identity
config_name: experiment-name  # Required
model_name: org/model-name    # Required (HuggingFace path)

# Model properties
is_encoder_decoder: false
task_type: text_generation    # text_generation | translation | summarisation
inference_type: pure_generative

# Token limits
max_input_tokens: 512
max_output_tokens: 128
min_output_tokens: 0

# Input
num_input_prompts: 100
save_outputs: false

# Distributed
gpu_list: [0, 1, 2, 3]
num_processes: 4

# Batching
batching_options:
  batch_size: 8
  dynamic_batching: false

# Generation
decoder_config:
  temperature: 1.0
  top_p: 1.0
  top_k: 50

# Precision & Quantization
fp_precision: float16         # float32 | float16 | bfloat16
quantization_config:
  quantization: false
  load_in_4bit: false
  load_in_8bit: false
```

### Config Inheritance

Configs can extend base configs using `_extends`:

```yaml
# configs/base.yaml
max_input_tokens: 512
fp_precision: float16

# configs/llama2-7b.yaml
_extends: base.yaml
config_name: llama2-7b
model_name: meta-llama/Llama-2-7b-hf
```

## Docker Usage

### Environment Variables

Copy `env.example` to `.env` and configure:

```bash
# Required for gated models (Llama, etc.)
HF_TOKEN=your_huggingface_token

# Optional: GPU selection
CUDA_VISIBLE_DEVICES=0,1

# Optional: Cache location
HF_HOME=/path/to/cache
```

### Running with Docker Compose

```bash
# Validate config
docker compose run --rm llm-energy-measure config validate /app/configs/experiment.yaml

# Run experiment (requires custom entrypoint for accelerate)
docker compose run --rm llm-energy-measure accelerate launch \
  --num_processes 2 \
  -m llm_energy_measure.orchestration.launcher \
  --config /app/configs/experiment.yaml

# View results
docker compose run --rm llm-energy-measure results list
```

### Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `~/.cache/huggingface` | `/home/app/.cache/huggingface` | Model cache |
| `./configs` | `/app/configs` (ro) | Experiment configs |
| `./results` | `/app/results` | Output results |

## Results Structure

```
results/
├── raw/
│   └── exp_20240115_123456/
│       ├── process_0.json    # GPU 0 results
│       ├── process_1.json    # GPU 1 results
│       └── ...
└── aggregated/
    └── exp_20240115_123456.json
```

### Metrics Collected

| Category | Metrics |
|----------|---------|
| **Inference** | total_tokens, tokens_per_second, latency_per_token_ms |
| **Energy** | total_energy_j, gpu_energy_j, cpu_energy_j, emissions_kg_co2 |
| **Compute** | flops_total, flops_per_token, peak_memory_mb |

## Development

```bash
# Install dev dependencies
make dev

# Run checks (format, lint, typecheck)
make check

# Run tests
make test              # Unit tests only
make test-integration  # Integration tests
make test-all          # All tests
```

## Project Structure

```
src/llm_energy_measure/
├── cli.py                 # Typer CLI application
├── config/                # Configuration loading & models
├── core/                  # Inference engine & metrics
├── domain/                # Domain models (Pydantic)
├── orchestration/         # Experiment lifecycle management
└── results/               # Results persistence & aggregation
```

See directory-level README.md files for detailed module documentation.

## License

MIT
