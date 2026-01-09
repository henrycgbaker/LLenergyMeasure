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

---

## Commands

### experiment

Run an experiment with automatic `accelerate launch` handling and result aggregation.

```bash
llm-energy-measure experiment [config.yaml] [OPTIONS]
```

**Prompt Source Options:**

| Option | Description |
|--------|-------------|
| `--dataset, -d` | Built-in dataset (alpaca, sharegpt, gsm8k, mmlu) or HuggingFace path |
| `--prompts, -p` | Path to prompts file (one per line) |
| `--sample-size, -n` | Limit number of prompts |
| `--split` | Dataset split (default: train) |
| `--column` | Column to extract prompts from |

**Preset & Model Options:**

| Option | Description |
|--------|-------------|
| `--preset` | Built-in preset (quick-test, benchmark, throughput) |
| `--model, -m` | HuggingFace model name (required with --preset) |

**Workflow Parameters:**

| Option | Description |
|--------|-------------|
| `--max-tokens` | Override max_output_tokens |
| `--seed` | Random seed for reproducibility |
| `--cycles, -c` | Number of cycles for statistical robustness (1-10) |

**Workflow Control:**

| Option | Description |
|--------|-------------|
| `--no-aggregate` | Skip auto-aggregation after experiment |
| `--fresh` | Start fresh, ignore incomplete experiments |
| `--resume <id>` | Resume an interrupted experiment by ID |
| `--results-dir, -o` | Results directory |
| `--yes, -y` | Skip confirmation prompts |
| `--dry-run` | Show config and exit without running |
| `--force` | Run despite blocking config errors |

**Examples:**

```bash
# Basic experiment with dataset
llm-energy-measure experiment configs/llama2.yaml --dataset alpaca -n 100

# Using preset (no config file needed)
llm-energy-measure experiment --preset quick-test --model meta-llama/Llama-2-7b-hf -d alpaca -n 10

# Multiple cycles for statistical robustness
llm-energy-measure experiment configs/llama2.yaml --dataset alpaca -n 100 --cycles 5

# Resume interrupted experiment
llm-energy-measure experiment --resume exp_20240115_123456

# Dry run to check config
llm-energy-measure experiment configs/llama2.yaml --dataset alpaca -n 100 --dry-run
```

---

### batch

Run multiple experiment configs in batch.

```bash
llm-energy-measure batch <config_pattern> [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--dataset, -d` | HuggingFace dataset for all runs |
| `--sample-size, -n` | Limit prompts for all runs |
| `--parallel` | Run N configs in parallel (default: sequential) |
| `--dry-run` | List configs without running |

**Examples:**

```bash
# Sequential batch run
llm-energy-measure batch "configs/*.yaml" --dataset alpaca -n 100

# Parallel (4 at a time)
llm-energy-measure batch "configs/grid/*.yaml" --parallel 4 --dataset alpaca -n 100

# Preview what would run
llm-energy-measure batch "configs/*.yaml" --dry-run
```

---

### schedule

Run experiments on a schedule (daemon mode) for temporal variation studies.

```bash
llm-energy-measure schedule [config.yaml] [OPTIONS]
```

**Scheduling Options:**

| Option | Description |
|--------|-------------|
| `--interval, -i` | Interval between runs (e.g., '6h', '30m', '1d') |
| `--at` | Specific time of day (e.g., '09:00', '14:30') |
| `--days` | Days to run on (e.g., 'mon,wed,fri' or 'weekdays') |
| `--duration, -d` | Total daemon duration (default: '24h') |

**Other Options:**

| Option | Description |
|--------|-------------|
| `--dataset` | Dataset for all runs |
| `--sample-size, -n` | Number of prompts |
| `--model, -m` | HuggingFace model name |
| `--preset` | Built-in preset |
| `--results-dir, -o` | Results directory |

**Examples:**

```bash
# Run every 6 hours for 24 hours
llm-energy-measure schedule config.yaml --interval 6h --dataset alpaca -n 100

# Run at 9am on weekdays for a week
llm-energy-measure schedule config.yaml --at 09:00 --days weekdays --duration 7d --dataset alpaca -n 100

# Using preset
llm-energy-measure schedule --preset benchmark --model meta-llama/Llama-2-7b-hf --interval 12h
```

---

### run

Low-level inference command (called by `accelerate launch`). Use `experiment` instead.

```bash
llm-energy-measure run <config.yaml> [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--prompts, -p` | Path to prompts file |
| `--dataset, -d` | HuggingFace dataset |
| `--split` | Dataset split |
| `--column` | Dataset column |
| `--sample-size, -n` | Limit prompts |
| `--results-dir, -o` | Results directory |
| `--dry-run` | Validate without running |

---

### aggregate

Aggregate raw per-process results into final metrics.

```bash
llm-energy-measure aggregate [experiment_id] [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--all` | Aggregate all pending experiments |
| `--results-dir, -o` | Results directory |
| `--force, -f` | Re-aggregate even if result exists |
| `--strict/--no-strict` | Fail if results incomplete (default: strict) |

**Examples:**

```bash
# Aggregate specific experiment
llm-energy-measure aggregate exp_20240115_123456

# Aggregate all pending
llm-energy-measure aggregate --all

# Force re-aggregate
llm-energy-measure aggregate exp_20240115_123456 --force
```

---

### config

Configuration management commands.

#### config validate

Validate a configuration file.

```bash
llm-energy-measure config validate <config.yaml>
```

Checks for valid YAML syntax, required fields, value constraints, and inheritance resolution.

#### config show

Display resolved configuration (with inheritance applied).

```bash
llm-energy-measure config show <config.yaml>
```

#### config new

Interactive config builder for creating new experiment configs.

```bash
llm-energy-measure config new [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--output, -o` | Output path for config file |
| `--preset` | Start from a preset |

#### config generate-grid

Generate a grid of configs from a base config with parameter variations.

```bash
llm-energy-measure config generate-grid <base_config.yaml> [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--vary` | Parameter=values to vary (repeatable) |
| `--output-dir, -o` | Output directory (default: configs/grid) |
| `--validate` | Only generate valid configs (skip errors) |
| `--strict` | Fail if any config would be invalid |

**Examples:**

```bash
# Generate batch size × precision grid
llm-energy-measure config generate-grid base.yaml \
  --vary batch_size=1,2,4,8 \
  --vary fp_precision=float16,float32 \
  --output-dir ./grid/

# Only generate valid configs
llm-energy-measure config generate-grid base.yaml \
  --vary batch_size=1,2,4,8,16,32 \
  --validate
```

---

### results

Results inspection commands.

#### results list

List all experiments.

```bash
llm-energy-measure results list [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--results-dir, -o` | Results directory |
| `--all, -a` | Show all experiments (including pending) |

#### results show

Show experiment results.

```bash
llm-energy-measure results show <exp_id> [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--results-dir, -o` | Results directory |
| `--raw` | Show per-process raw results |
| `--json` | Output as JSON |

---

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

---

### datasets

List available built-in datasets.

```bash
llm-energy-measure datasets
```

---

### presets

List built-in experiment presets.

```bash
llm-energy-measure presets
```

---

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
