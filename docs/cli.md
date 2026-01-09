# CLI Reference

Complete command reference for `llm-energy-measure`.

## Usage

```
llm-energy-measure [OPTIONS] COMMAND [ARGS]
```

## Global Options

| Option | Description |
|--------|-------------|
| `--version, -v` | Show version |
| `--verbose` | Enable debug logging |

## Commands

### experiment

Run an experiment with automatic `accelerate launch` handling and result aggregation.

```bash
llm-energy-measure experiment <config.yaml> [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--dataset, -d` | Built-in dataset name (alpaca, sharegpt, gsm8k, mmlu) |
| `--prompts` | Path to prompts file (alternative to --dataset) |
| `-n` | Number of prompts to use |
| `--split` | Dataset split (default: train) |
| `--column` | Column to extract prompts from |
| `--preset` | Use a built-in preset |
| `--model` | Override model (useful with --preset) |
| `--no-aggregate` | Skip auto-aggregation |
| `--fresh` | Ignore incomplete experiments, start fresh |
| `--resume <id>` | Resume an interrupted experiment |

**Examples:**

```bash
# Basic experiment
llm-energy-measure experiment configs/llama2.yaml --dataset alpaca -n 100

# Using preset
llm-energy-measure experiment --preset quick-test --model meta-llama/Llama-2-7b-hf -d alpaca -n 10

# Resume interrupted
llm-energy-measure experiment --resume exp_20240115_123456

# Custom HuggingFace dataset
llm-energy-measure experiment config.yaml --dataset squad --split validation --column question -n 50
```

### run

Low-level inference command (called by `accelerate launch`). Use `experiment` instead for normal usage.

```bash
llm-energy-measure run --config <config.yaml> [OPTIONS]
```

### aggregate

Aggregate raw per-process results into final metrics.

```bash
llm-energy-measure aggregate <exp_id> [OPTIONS]
llm-energy-measure aggregate --all
```

| Option | Description |
|--------|-------------|
| `--force` | Aggregate even if incomplete (partial results) |
| `--all` | Aggregate all pending experiments |

### config

Configuration management commands.

#### config validate

Validate a configuration file.

```bash
llm-energy-measure config validate <config.yaml> [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--strict` | Enable strict validation (warnings become errors) |

#### config show

Display resolved configuration (with inheritance applied).

```bash
llm-energy-measure config show <config.yaml>
```

### results

Results inspection commands.

#### results list

List all experiments.

```bash
llm-energy-measure results list
```

#### results show

Show experiment results.

```bash
llm-energy-measure results show <exp_id> [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--raw` | Show per-process raw results |
| `--json` | Output as JSON |

### gpus

Show GPU topology including MIG instances.

```bash
llm-energy-measure gpus
```

**Example output:**

```
GPU Topology (4 device(s) detected)
├── [0] NVIDIA A100-PCIE-40GB (40GB) - Full GPU
├── [1] NVIDIA A100-PCIE-40GB (40GB) - MIG Enabled (3 instances)
├── [2] NVIDIA A100-PCIE-40GB (40GB) - MIG Enabled (3 instances)
└── [3] NVIDIA A100-PCIE-40GB (40GB) - MIG Enabled (3 instances)
```

### datasets

List available built-in datasets.

```bash
llm-energy-measure datasets
```

### presets

List built-in experiment presets.

```bash
llm-energy-measure presets
```

## Built-in Datasets

| Dataset | Source | Default Column |
|---------|--------|----------------|
| `alpaca` | tatsu-lab/alpaca | instruction |
| `sharegpt` | ShareGPT_Vicuna | conversations |
| `gsm8k` | gsm8k (main) | question |
| `mmlu` | cais/mmlu (all) | question |

## Built-in Presets

| Preset | Purpose |
|--------|---------|
| `quick-test` | Fast validation (batch=1, max_out=32) |
| `benchmark` | Formal measurements (fp16, deterministic) |
| `throughput` | Throughput testing (batch=8, dynamic batching) |

## Direct accelerate Usage

For advanced control, use `accelerate launch` directly:

```bash
accelerate launch --num_processes 2 \
  -m llm_energy_measure.orchestration.launcher \
  --config configs/my_experiment.yaml \
  --dataset alpaca -n 100
```
