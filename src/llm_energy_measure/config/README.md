# config/ - Configuration System

Configuration loading, validation, and models for experiment setup.

## Purpose

Provides Pydantic-based configuration models and a loader that supports YAML/JSON files with inheritance via `_extends`. The system supports three configuration modes: config files, presets, and CLI overrides.

## Parameter Resolution

Configuration parameters are resolved with the following precedence (highest to lowest):

```
CLI flags  >  Config file  >  Preset  >  Defaults
```

This allows flexible workflows:
- **Formal experiments**: Full config files for reproducibility
- **Quick exploration**: Presets with minimal flags
- **Parameter sweeps**: Base config with CLI overrides

## CLI vs YAML Philosophy

**Core Principle**: CLI flags are for workflow control; YAML configs are for testable experiment parameters.

Think of it as: **CLI = "how to run"** vs **YAML = "what to measure"**

### Workflow Params (CLI Recommended)

These control execution workflow, not the experiment itself:

| Flag | Purpose |
|------|---------|
| `--model`, `-m` | Which model to benchmark |
| `--dataset`, `-d` | Which prompt source to use |
| `--preset` | Quick-start configuration template |
| `--cycles`, `-c` | Statistical repetition count |
| `--seed` | Reproducibility seed |
| `--max-tokens` | Generation limit |
| `--fresh` | Ignore incomplete experiments |
| `--resume` | Continue interrupted experiment |
| `--no-aggregate` | Skip auto-aggregation |
| `--results-dir` | Output location |

### Testable Params (YAML Required for Formal Experiments)

These define the experiment configuration that affects measurements:

| Config Field | What It Controls |
|--------------|------------------|
| `batching.batch_size` | Inference batch size |
| `batching.strategy` | static/dynamic/sorted_static/sorted_dynamic |
| `fp_precision` | float32/float16/bfloat16 |
| `num_processes` | Distributed workers |
| `gpus` | GPU allocation |
| `num_cycles` | Statistical repetition (YAML alternative to --cycles) |
| `decoder.temperature` | Sampling temperature |
| `quantization.*` | 4-bit/8-bit quantisation |
| `traffic_simulation.*` | Traffic patterns (Poisson/constant) |
| `sharding.*` | Tensor/pipeline parallelism |

### Deprecated CLI Flags

These CLI flags **still work** but emit deprecation warnings:

```
--batch-size, -b    → Use batching.batch_size in YAML
--precision         → Use fp_precision in YAML
--num-processes     → Use num_processes in YAML
--gpu-list          → Use gpus in YAML
--temperature       → Use decoder.temperature in YAML
--quantization      → Use quantization.quantization in YAML
```

**Why deprecated?** They encourage ad-hoc experiments harder to reproduce.
**Why keep working?** Backwards compatibility and quick iteration.

### Override Tracking

All CLI overrides are recorded in result metadata for traceability:

```json
{
  "effective_config": { "batch_size": 8, ... },
  "cli_overrides": {
    "batching.batch_size": { "original": 1, "new": 8 }
  }
}
```

### Workflow Examples

```bash
# 1. Formal experiment (fully reproducible)
llm-energy-measure experiment configs/llama2-7b-benchmark.yaml

# 2. Quick exploration (preset + model)
llm-energy-measure experiment --preset quick-test --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 -d alpaca

# 3. Parameter sweep (override tracked in metadata)
for batch in 1 2 4 8; do
  llm-energy-measure experiment config.yaml --batch-size $batch
done

# 4. Statistical robustness (CLI or YAML)
llm-energy-measure experiment config.yaml --cycles 5
# Or set num_cycles: 5 in YAML
```

## Key Files

### models.py
Pydantic configuration models.

**ExperimentConfig** - Main experiment configuration:
```python
from llm_energy_measure.config import ExperimentConfig

config = ExperimentConfig(
    config_name="my-experiment",
    model_name="meta-llama/Llama-2-7b-hf",
    max_input_tokens=512,
    max_output_tokens=128,
    gpus=[0, 1],
    num_processes=2,
    random_seed=42,  # For reproducibility
)
```

**Sub-configurations:**
- `BatchingConfig` - batch_size, strategy, max_tokens_per_batch
- `ShardingConfig` - tensor_parallel, pipeline_parallel
- `TrafficSimulation` - MLPerf-style Poisson/constant arrival simulation
- `DecoderConfig` - temperature, sampling presets, repetition control
- `QuantizationConfig` - 4-bit/8-bit BitsAndBytes
- `PromptSourceConfig` - File or HuggingFace dataset prompts

## Decoder Sampling Configuration

Industry-standard sampling parameters aligned with vLLM, HuggingFace, and MLPerf.

### Sampling Presets

Use `preset` for common configurations:

```yaml
decoder:
  preset: deterministic  # Greedy decoding (temp=0, do_sample=false)
```

| Preset | Temperature | do_sample | Notes |
|--------|-------------|-----------|-------|
| `deterministic` | 0.0 | false | Greedy decoding for reproducible benchmarks |
| `standard` | 1.0 | true | Balanced sampling (top_p=0.95, top_k=50) |
| `creative` | 0.8 | true | Higher variance (top_p=0.9, repetition_penalty=1.1) |
| `factual` | 0.3 | true | Lower variance (top_k=10) |

**Preset + override**: Explicit parameters override preset values:
```yaml
decoder:
  preset: deterministic
  temperature: 0.5  # This overrides the preset's temperature
```

### Full Parameter Reference

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| `temperature` | [0, 2] | 1.0 | 0=greedy, 1=standard, >1=more random |
| `do_sample` | bool | true | Enable sampling (ignored if temp=0) |
| `top_p` | [0, 1] | 1.0 | Nucleus sampling (1.0=disabled) |
| `top_k` | ≥0 | 50 | Top-k sampling (0=disabled) |
| `min_p` | [0, 1] | 0.0 | Min prob relative to top token (0=disabled) |
| `repetition_penalty` | [0.1, 10] | 1.0 | Repetition penalty (1.0=disabled) |
| `no_repeat_ngram_size` | ≥0 | 0 | Prevent n-gram repetition (0=disabled) |
| `preset` | enum | None | Shortcut: deterministic/standard/creative/factual |

### Example Configurations

```yaml
# Greedy decoding (most reproducible)
decoder:
  preset: deterministic

# Custom sampling
decoder:
  temperature: 0.7
  top_p: 0.9
  repetition_penalty: 1.2
  no_repeat_ngram_size: 3

# Preset with override
decoder:
  preset: creative
  repetition_penalty: 1.5  # Override preset value
```

### Seed Handling for Reproducibility

Seeds are applied per-batch for varied but reproducible outputs:
- Batch 0 uses `random_seed`
- Batch 1 uses `random_seed + 1`
- etc.

```yaml
random_seed: 42
decoder:
  preset: standard  # Sampling enabled but reproducible with seed
```

### Configuration Warnings

The validator (`validate_config()`) checks for problematic configurations and returns warnings at three severity levels:

| Severity | Behaviour |
|----------|-----------|
| **error** | Blocks experiment without `--force` flag |
| **warning** | Shows warning, prompts for confirmation (skip with `--yes`) |
| **info** | Informational, doesn't block |

#### All Configuration Warnings

| Category | Condition | Severity | Message |
|----------|-----------|----------|---------|
| Distributed | `num_processes > 1` with single GPU | info | Multiple processes with single GPU may not provide parallelism benefits |
| Tokens | `max_output_tokens > 2048` | warning | Very high max_output_tokens may cause memory issues |
| Quantization | `quantization=True` with `fp_precision=float32` | warning | Quantization typically uses float16 compute, not float32 |
| Quantization | `quantization=True` without 4bit/8bit specified | error | Must specify load_in_4bit or load_in_8bit |
| Batching | Dynamic strategy without `max_tokens_per_batch` | info | Will use max_input_tokens as token budget |
| Batching | Sorted strategy with `batch_size=1` | info | Sorting provides no benefit with batch_size=1 |
| Sharding | `num_shards > len(gpus)` | error | num_shards exceeds available GPUs |
| Sharding | Sharding strategy with single GPU | info | Sharding provides no benefit with single GPU |
| Decoder | Non-default sampling params in deterministic mode | error | Sampling params ignored when temp=0 or do_sample=False |
| Decoder | `do_sample=True` with `temperature=0` | info | do_sample has no effect when temperature=0 |
| Decoder | Both `temperature` and `top_p` modified | warning | Not recommended - alter one or the other |
| Traffic | `target_qps > 100` | warning | Very high QPS may not be achievable |

#### Handling Warnings

```bash
# Show warnings, prompt for confirmation
llm-energy-measure experiment config.yaml --dataset alpaca -n 100

# Skip confirmation prompts (auto-accept warnings)
llm-energy-measure experiment config.yaml --dataset alpaca -n 100 --yes

# Run despite blocking errors
llm-energy-measure experiment config.yaml --dataset alpaca -n 100 --force
```

### loader.py
Configuration loading with inheritance.

```python
from llm_energy_measure.config import load_config, validate_config

config = load_config("configs/experiment.yaml")
warnings = validate_config(config)
```

**Key functions:**
- `load_config(path)` - Load and validate config
- `validate_config(config)` - Return warnings (not errors)
- `load_config_dict(path)` - Load raw dict (YAML/JSON)
- `resolve_inheritance(dict, path)` - Apply `_extends`
- `deep_merge(base, overlay)` - Merge nested dicts

**Inheritance example:**
```yaml
# base.yaml
max_input_tokens: 512
fp_precision: float16

# experiment.yaml
_extends: base.yaml
config_name: my-experiment
model_name: meta-llama/Llama-2-7b-hf
```

## Built-in Presets

Presets provide sensible defaults for common scenarios. Use with `--preset <name>`:

| Preset | Purpose | Settings |
|--------|---------|----------|
| `quick-test` | Fast validation runs | batch=1, max_in=64, max_out=32, deterministic |
| `benchmark` | Formal measurements | batch=1, max_in=2048, max_out=512, fp16, deterministic |
| `throughput` | Throughput-optimised | batch=8, max_in=512, max_out=256, fp16, dynamic batching |

**Usage:**
```bash
# Preset with model (quick exploration)
llm-energy-measure experiment --preset quick-test --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 -d alpaca

# Start new config from preset
llm-energy-measure config new --preset benchmark
```

## Configuration Reference

### Parameter Table (CLI vs Config)

| Config Field | CLI Flag | Type | Default | Description |
|--------------|----------|------|---------|-------------|
| `config_name` | - | str | Required | Unique identifier |
| `model_name` | `--model / -m` | str | Required | HuggingFace model path |
| `max_input_tokens` | - | int | 512 | Max input tokens |
| `max_output_tokens` | `--max-tokens` | int | 128 | Max generated tokens |
| `min_output_tokens` | - | int | 0 | Min generated tokens |
| `fp_precision` | `--precision` | str | float16 | float32/float16/bfloat16 |
| `num_processes` | `--num-processes` | int | 1 | Worker processes |
| `gpus` | `--gpu-list` | list[int] | [0] | GPU indices |
| `random_seed` | `--seed` | int\|None | None | Random seed |
| `batching.batch_size` | `--batch-size / -b` | int | 1 | Batch size |
| `batching.strategy` | - | str | static | Batching strategy |
| `traffic_simulation.enabled` | - | bool | false | Enable traffic simulation |
| `traffic_simulation.mode` | - | str | poisson | Traffic mode (poisson/constant) |
| `traffic_simulation.target_qps` | - | float | 1.0 | Target queries per second |
| `decoder.temperature` | `--temperature` | float | 1.0 | Decoder temperature |
| `quantization.quantization` | `--quantization` | bool | false | Enable quantization |

### Required Fields
| Field | Type | Description |
|-------|------|-------------|
| `config_name` | str | Unique identifier |
| `model_name` | str | HuggingFace model path |

### Token Settings
| Field | Default | Description |
|-------|---------|-------------|
| `max_input_tokens` | 512 | Max input tokens |
| `max_output_tokens` | 128 | Max generated tokens |
| `min_output_tokens` | 0 | Min generated tokens |

### Distributed Settings
| Field | Default | Description |
|-------|---------|-------------|
| `gpus` | [0] | GPU indices |
| `num_processes` | 1 | Worker processes |

### Precision Settings
| Field | Default | Options |
|-------|---------|---------|
| `fp_precision` | float16 | float32, float16, bfloat16 |
| `backend` | pytorch | pytorch, tensorrt, vllm |

### Reproducibility
| Field | Default | Description |
|-------|---------|-------------|
| `random_seed` | None | Random seed (None = non-deterministic) |

## Prompt Source Configuration

Prompts can be loaded from files or HuggingFace datasets.

### File-based prompts
```yaml
prompts:
  type: file
  path: ./prompts.txt  # One prompt per line
```

### HuggingFace datasets
```yaml
prompts:
  type: huggingface
  dataset: alpaca          # Built-in alias or full HF path
  split: train             # Dataset split (default: train)
  column: instruction      # Column to extract (auto-detected if omitted)
  sample_size: 1000        # Limit prompts (optional)
  shuffle: true            # Shuffle before sampling (default: false)
  seed: 42                 # Random seed (default: 42)
```

**Built-in dataset aliases:**
| Alias | HuggingFace Path | Default Column |
|-------|-----------------|----------------|
| `alpaca` | tatsu-lab/alpaca | instruction |
| `sharegpt` | anon8231489123/ShareGPT_Vicuna_unfiltered | conversations |
| `gsm8k` | gsm8k (main subset) | question |
| `mmlu` | cais/mmlu (all subset) | question |

**Auto-detect columns:** text, prompt, question, instruction, input, content

## Batching Strategies (MLPerf/vLLM Terminology)

Industry-standard batching strategies for benchmarking:

```yaml
batching:
  batch_size: 4
  strategy: sorted_dynamic    # static | dynamic | sorted_static | sorted_dynamic
  max_tokens_per_batch: 512   # For dynamic strategies
```

| Strategy | Description |
|----------|-------------|
| `static` | Fixed batch size (default) |
| `dynamic` | Token-aware batching respecting `max_tokens_per_batch` |
| `sorted_static` | Sort prompts by length, then fixed batches |
| `sorted_dynamic` | Sort prompts by length, then token-aware batching |

**Length sorting** reduces padding waste by grouping similar-length prompts together.

## Traffic Simulation (MLPerf LoadGen Style)

Simulate realistic request arrival patterns for load testing:

```yaml
traffic_simulation:
  enabled: true
  mode: poisson             # poisson | constant
  target_qps: 2.0           # Target queries per second (arrival rate λ)
  seed: 42                  # For reproducibility
```

| Mode | Description |
|------|-------------|
| `poisson` | Exponential inter-arrival times (realistic traffic) |
| `constant` | Fixed inter-arrival = 1/target_qps |

**Poisson arrivals** model real-world traffic patterns where requests arrive randomly but at a known average rate.

## Scheduled Experiments (Daemon Mode)

Run experiments on a schedule for temporal variation studies (e.g., energy consumption at different times of day).

### CLI Usage
```bash
# Run every 6 hours for 24 hours
llm-energy-measure schedule config.yaml --interval 6h --duration 24h --dataset alpaca -n 100

# Run daily at 9am for a week
llm-energy-measure schedule config.yaml --at 09:00 --duration 7d

# Run at 9am on weekdays only
llm-energy-measure schedule config.yaml --at 09:00 --days weekdays --duration 14d

# Run every 12 hours on weekends
llm-energy-measure schedule config.yaml --interval 12h --days sat,sun --duration 48h
```

### YAML Configuration
```yaml
schedule:
  enabled: true
  interval: "6h"              # Run every 6 hours (alternative to 'at')
  at: "09:00"                 # Run at specific time (alternative to 'interval')
  days: ["mon", "wed", "fri"] # Filter to specific days (optional)
  total_duration: "7d"        # Stop daemon after 7 days
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | false | Enable scheduled mode |
| `interval` | str | None | Interval between runs (e.g., '6h', '30m') |
| `at` | str | None | Time of day to run (e.g., '09:00') |
| `days` | list | None | Days to run: ['mon','tue',...] or ['weekdays'] |
| `total_duration` | str | '24h' | How long to run daemon |

**Day aliases**: `weekdays` (mon-fri), `weekends` (sat-sun)

## Validation Rules

Pydantic validators enforce:
- `num_processes <= len(gpus)`
- `min_output_tokens <= max_output_tokens`
- `load_in_4bit` and `load_in_8bit` are mutually exclusive
- `target_qps > 0` (traffic simulation)
- `sample_size >= 1` (prompt source)

## Grid Generation

Generate configs for parameter sweeps using Cartesian product:

```bash
llm-energy-measure config generate-grid base.yaml \
    --vary batch_size=1,2,4,8 \
    --vary fp_precision=float16,float32 \
    --output-dir ./grid/
```

This creates 8 configs (4 batch sizes × 2 precisions) in `./grid/`.

### ⚠️ Grid Validation Flags

**Important**: Parameter sweeps can produce invalid combinations. For example, varying `temperature` and `top_k` together will create configs where sampling params are set in deterministic mode (temp=0), which is an error.

| Flag | Behaviour |
|------|-----------|
| (default) | Generate **all** configs, show warnings for invalid ones |
| `--validate` | Only generate **valid** configs (skip invalid) |
| `--strict` | **Fail** if any config would be invalid |

**Usage:**
```bash
# Default: generates all, warns about invalid (may produce unusable configs!)
llm-energy-measure config generate-grid base.yaml \
    --vary decoder.temperature=0.0,1.0 \
    --vary decoder.top_k=50,100

# Recommended: skip invalid configs
llm-energy-measure config generate-grid base.yaml \
    --vary decoder.temperature=0.0,1.0 \
    --vary decoder.top_k=50,100 \
    --validate

# CI/automation: fail if any would be invalid
llm-energy-measure config generate-grid base.yaml \
    --vary decoder.temperature=0.0,1.0 \
    --vary decoder.top_k=50,100 \
    --strict
```

**Common invalid combinations detected:**

| Combination | Error |
|-------------|-------|
| `temperature=0` + non-default `top_k`/`top_p`/`min_p` | Sampling params ignored in deterministic mode |
| `quantization=true` without bit mode | Missing `load_in_4bit` or `load_in_8bit` |
| `num_shards > len(gpus)` | More shards than GPUs |
| `do_sample=false` + sampling params | Sampling params ignored |

**Output example (default):**
```
Generating 6 configs from 2 parameters...
✓ Generated 6 configs in /tmp/grid/

⚠ 2 config(s) with blocking errors:
  base_temperature_0_0_top_k_100.yaml:
    [ERROR] decoder: Sampling params [top_k=100] have no effect in deterministic mode
  base_temperature_0_0_top_k_50.yaml:
    [ERROR] decoder: ... (if top_k=50 differs from default)
```

**Output example (--validate):**
```
Generating 6 configs from 2 parameters...
✓ Generated 4 configs in /tmp/grid/

Skipped 2 invalid configs due to --validate
```

**Best practice**: Use `--validate` when generating grids for batch experiments to avoid runtime failures.

## Interactive Config Builder

Create configs interactively with sensible defaults:

```bash
llm-energy-measure config new                    # Start from scratch
llm-energy-measure config new --preset benchmark # Start from preset
llm-energy-measure config new -o my-config.yaml  # Specify output path
```

## Reproducibility

Results include `effective_config` and `cli_overrides` fields for full reproducibility:

```json
{
  "effective_config": {
    "model_name": "meta-llama/Llama-2-7b-hf",
    "batch_size": 8,
    "fp_precision": "float16"
  },
  "cli_overrides": {
    "batching.batch_size": {"new": 8, "original": 1}
  }
}
```

## Related

- See `../cli.py` for CLI commands (`config validate`, `config show`, `config new`, `config generate-grid`)
- See `../core/inference.py` for config usage in inference
- See `../constants.py` for preset definitions
- See `../domain/experiment.py` for result models with config tracking
