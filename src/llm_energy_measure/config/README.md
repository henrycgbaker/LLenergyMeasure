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
    gpu_list=[0, 1],
    num_processes=2,
    random_seed=42,  # For reproducibility
)
```

**Sub-configurations:**
- `BatchingConfig` - batch_size, dynamic_batching
- `ShardingConfig` - tensor_parallel, pipeline_parallel
- `LatencySimulation` - Artificial delay injection
- `DecoderConfig` - temperature, top_p, top_k
- `QuantizationConfig` - 4-bit/8-bit BitsAndBytes
- `PromptSourceConfig` - File or HuggingFace dataset prompts

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
| `gpu_list` | `--gpu-list` | list[int] | [0] | GPU indices |
| `random_seed` | `--seed` | int\|None | None | Random seed |
| `batching_options.batch_size` | `--batch-size / -b` | int | 1 | Batch size |
| `decoder_config.temperature` | `--temperature` | float | 1.0 | Decoder temperature |
| `quantization_config.quantization` | `--quantization` | bool | false | Enable quantization |

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
| `gpu_list` | [0] | GPU indices |
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
prompt_source:
  type: file
  path: ./prompts.txt  # One prompt per line
```

### HuggingFace datasets
```yaml
prompt_source:
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

## Validation Rules

Pydantic validators enforce:
- `num_processes <= len(gpu_list)`
- `min_output_tokens <= max_output_tokens`
- `load_in_4bit` and `load_in_8bit` are mutually exclusive
- `delay_min_ms <= delay_max_ms` (latency simulation)
- `sample_size >= 1` (prompt source)

## Grid Generation

Generate configs for parameter sweeps using Cartesian product:

```bash
llm-energy-measure config generate-grid base.yaml \
    --vary batch_size=1,2,4,8 \
    --vary fp_precision=float16,float32 \
    --output-dir ./grid/
```

This creates 8 configs (4 batch sizes x 2 precisions) in `./grid/`.

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
    "batching_options.batch_size": {"new": 8, "original": 1}
  }
}
```

## Related

- See `../cli.py` for CLI commands (`config validate`, `config show`, `config new`, `config generate-grid`)
- See `../core/inference.py` for config usage in inference
- See `../constants.py` for preset definitions
- See `../domain/experiment.py` for result models with config tracking
