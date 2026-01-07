# config/ - Configuration System

Configuration loading, validation, and models for experiment setup.

## Purpose

Provides Pydantic-based configuration models and a loader that supports YAML/JSON files with inheritance via `_extends`.

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

## Configuration Reference

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

## Related

- See `../cli.py` for `config validate` and `config show` commands
- See `../core/inference.py` for config usage in inference
