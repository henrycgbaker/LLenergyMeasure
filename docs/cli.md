# CLI Reference

Complete command reference for `lem`.

## Usage

```
lem [OPTIONS] COMMAND [ARGS]
```

## Global Options

| Option | Description |
|--------|-------------|
| `--version, -v` | Show version |
| `--verbose` | Enable debug logging |

---

## Commands

### experiment

Run an experiment with automatic launcher handling and result aggregation.

```bash
lem experiment [path/to/config.yaml] [OPTIONS]
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

**Backend & Streaming Options:**

| Option | Description |
|--------|-------------|
| `--backend` | Inference backend: pytorch (default), vllm, tensorrt |
| `--streaming` | Enable streaming latency measurement (TTFT/ITL) |
| `--streaming-warmup` | Warmup requests for streaming (default: 5) |

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
lem experiment configs/llama2.yaml --dataset alpaca -n 100

# Using preset (no config file needed)
lem experiment --preset quick-test --model meta-llama/Llama-2-7b-hf -d alpaca -n 10

# Multiple cycles for statistical robustness
lem experiment configs/llama2.yaml --dataset alpaca -n 100 --cycles 5

# Resume interrupted experiment
lem experiment --resume exp_20240115_123456

# Dry run to check config
lem experiment configs/llama2.yaml --dataset alpaca -n 100 --dry-run
```

---

### batch

Run multiple experiment configs in batch.

```bash
lem batch <config_pattern> [OPTIONS]
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
lem batch "configs/*.yaml" --dataset alpaca -n 100

# Parallel (4 at a time)
lem batch "configs/grid/*.yaml" --parallel 4 --dataset alpaca -n 100

# Preview what would run
lem batch "configs/*.yaml" --dry-run
```

---

### campaign

Run multi-config campaigns for statistical comparison. See [Campaign Guide](campaigns.md) for full documentation.

```bash
lem campaign CONFIG_PATHS... [OPTIONS]
```

**Arguments:**
- `CONFIG_PATHS`: Campaign YAML file OR multiple experiment config files

**Campaign Options:**

| Option | Description |
|--------|-------------|
| `--campaign-name` | Campaign name (required for multiple configs) |
| `--dataset, -d` | Dataset override for all experiments |
| `--sample-size, -n` | Sample size override |
| `--cycles, -c` | Number of cycles (default: 3) |
| `--structure` | Execution order: interleaved, shuffled, grouped |
| `--warmup-prompts` | Min warmup prompts per config |
| `--warmup-timeout` | Max warmup time in seconds |
| `--config-gap` | Gap between configs (seconds) |
| `--cycle-gap` | Gap between cycles (seconds) |
| `--seed` | Random seed for shuffled structure |
| `--results-dir, -o` | Results directory |
| `--dry-run` | Preview execution plan |
| `--yes, -y` | Skip confirmation |

**Examples:**

```bash
# Compare two configs with 5 cycles
lem campaign configs/pytorch.yaml configs/vllm.yaml \
  --campaign-name "backend-comparison" --cycles 5 -d alpaca -n 100

# Use campaign YAML
lem campaign configs/examples/campaign_example.yaml

# Preview execution plan
lem campaign configs/*.yaml --campaign-name "test" --dry-run
```

---

### schedule

Run experiments on a schedule (daemon mode) for temporal variation studies.

```bash
lem schedule [config.yaml] [OPTIONS]
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
lem schedule config.yaml --interval 6h --dataset alpaca -n 100

# Run at 9am on weekdays for a week
lem schedule config.yaml --at 09:00 --days weekdays --duration 7d --dataset alpaca -n 100

# Using preset
lem schedule --preset benchmark --model meta-llama/Llama-2-7b-hf --interval 12h
```

---

### aggregate

Aggregate raw per-process results into final metrics. Called by `experiment`.

```bash
lem aggregate [experiment_id] [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--all` | Aggregate all pending experiments |
| `--results-dir, -o` | Results directory |
| `--force, -f` | Re-aggregate even if result exists |
| `--strict/--no-strict` | Fail if results incomplete (default: strict) |
| `--allow-mixed-backends` | Allow aggregating results from different backends (not recommended) |

**Examples:**

```bash
# Aggregate specific experiment
lem aggregate exp_20240115_123456

# Aggregate all pending
lem aggregate --all

# Force re-aggregate
lem aggregate exp_20240115_123456 --force
```

---

### config

Configuration management commands.

#### config validate

Validate a configuration file.

```bash
lem config validate <config.yaml>
```

Checks for valid YAML syntax, required fields, value constraints, and inheritance resolution.

#### config show

Display resolved configuration (with inheritance applied).

```bash
lem config show <config.yaml>
```

#### config new

Interactive config builder for creating new experiment configs.

```bash
lem config new [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--output, -o` | Output path for config file |
| `--preset` | Start from a preset |

#### config generate-grid

Generate a grid of configs from a base config with parameter variations.

```bash
lem config generate-grid <base_config.yaml> [OPTIONS]
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
lem config generate-grid base.yaml \
  --vary batch_size=1,2,4,8 \
  --vary fp_precision=float16,float32 \
  --output-dir ./grid/

# Only generate valid configs
lem config generate-grid base.yaml \
  --vary batch_size=1,2,4,8,16,32 \
  --validate
```

---

### results

Results inspection commands.

#### results list

List all experiments.

```bash
lem results list [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--results-dir, -o` | Results directory |
| `--all, -a` | Show all experiments (including pending) |

#### results show

Show experiment results.

```bash
lem results show <exp_id> [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--results-dir, -o` | Results directory |
| `--raw` | Show per-process raw results |
| `--json` | Output as JSON |

---

### gpus

Show GPU topology (including MIG instances).

```bash
lem gpus
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
lem datasets
```

---

### presets

List built-in experiment presets.

```bash
lem presets
```

---

## Built-in Datasets

| Dataset | Source | Default Column | Notes |
|---------|--------|----------------|-------|
| `ai-energy-score` | AIEnergyScore/text_generation | text | **Default** when no `--dataset` specified |
| `alpaca` | tatsu-lab/alpaca | instruction | |
| `sharegpt` | ShareGPT_Vicuna | conversations | |
| `gsm8k` | gsm8k (main) | question | |
| `mmlu` | cais/mmlu (all) | question | |

## Built-in Presets

| Preset | Purpose |
|--------|---------|
| `quick-test` | Fast validation (batch=1, max_out=32) |
| `benchmark` | Formal measurements (fp16, deterministic) |
| `throughput` | Throughput testing (batch=8, dynamic batching) |

## Configuration Precedence

When parameters are specified in multiple places, they are resolved in this order (highest priority first):

```
CLI flags  >  Config file  >  Preset  >  Environment variables  >  Defaults
```

**Example:**
```bash
# Config file sets batch_size=4, CLI overrides to 8
lem experiment config.yaml --batch-size 8
# Result: batch_size=8
```

## Environment Variables

The tool loads `.env` files automatically via dotenv.

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_ENERGY_RESULTS_DIR` | Default results directory | `results/` |
| `HF_TOKEN` | HuggingFace token for gated models | — |
| `CUDA_VISIBLE_DEVICES` | GPU selection | All visible |
| `CODECARBON_LOG_LEVEL` | CodeCarbon logging (warning recommended) | `info` |

**Results directory precedence:**
```
--results-dir (CLI)  >  io.results_dir (config)  >  LLM_ENERGY_RESULTS_DIR (.env)  >  "results/"
```

**Example `.env` file:**
```bash
LLM_ENERGY_RESULTS_DIR=/data/experiments/results
HF_TOKEN=hf_xxxxxxxxxxxxx
CODECARBON_LOG_LEVEL=warning
```

## Deprecated CLI Flags

These flags still work but emit deprecation warnings. Use YAML config fields instead for reproducible experiments:

| Deprecated Flag | Replacement (YAML) | Notes |
|-----------------|-------------------|-------|
| `--batch-size, -b` | `batching.batch_size` | Use config for formal experiments |
| `--precision` | `fp_precision` | Use config for formal experiments |
| `--num-processes` | `num_processes` | Use config for formal experiments |
| `--gpu-list` | `gpus` | Use config for formal experiments |
| `--temperature` | `decoder.temperature` | Use config for formal experiments |
| `--quantization` | `quantization.quantization` | Use config for formal experiments |

**Why deprecated?** These flags encourage ad-hoc experiments that are harder to reproduce. For quick iteration they're fine, but formal experiments should use config files.

## Direct accelerate Usage

For advanced control, use `accelerate launch` directly:

```bash
accelerate launch --num_processes 2 \
  -m llenergymeasure.orchestration.launcher \
  --config configs/my_experiment.yaml \
  --dataset alpaca -n 100
```
