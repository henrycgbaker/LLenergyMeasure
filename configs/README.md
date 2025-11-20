# Configuration System

This directory contains the configuration system for the LLM Efficiency Measurement Tool. The system is designed to enable systematic exploration of the parameter space through flexible, composable configurations.

## Overview

The configuration system consists of:
1. **Base Configuration**: Default parameters (`a_default_config.py`)
2. **Configuration Class**: Type-safe dataclass wrapper (`config_class.py`)
3. **Configuration Utilities**: Helper functions for manipulation (`config_utils.py`)
4. **Experiment-Specific Configurations**: Pre-defined configuration suites (b-e files)

## Files

### `a_default_config.py`
The single source of truth for all configuration parameters. Contains the `base_config` dictionary with ~50+ parameters organized into logical groups.

**Key Configuration Groups:**
- **Model Settings**: `model_name`, `is_encoder_decoder`, `backend`
- **Task Settings**: `task_type`, `inference_type`
- **Hardware**: `gpu_list`, `num_processes`
- **Inference Parameters**: `max_input_tokens`, `max_output_tokens`, `num_input_prompts`
- **Batching Options**: Fixed vs adaptive batching, batch sizes
- **Sharding Configuration**: FSDP settings
- **Precision**: `fp_precision` (float32/16/8, bfloat16)
- **Quantization**: 4-bit and 8-bit quantization settings
- **Decoder Configuration**: Decoding mode, temperature, top-k, top-p
- **Latency Simulation**: Network delay simulation (constant/bursty)
- **Output Settings**: `save_outputs`, `decode_token_to_text`

**Example Configuration:**
```python
base_config = {
    "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "task_type": "text_generation",
    "num_processes": 4,
    "fp_precision": "float32",
    "batching_options": {
        "batch_size___fixed_batching": 16,
        "adaptive_batching": False,
    },
    "quantization_config": {
        "quantization": None,
        "load_in_8bit": None,
        "load_in_4bit": None,
        "cached_flops_for_quantised_models": 52638582308864
    },
    # ... additional parameters
}
```

### `config_class.py`
Defines the `ExperimentConfig` dataclass for type-safe configuration handling. Provides validation and convenient attribute access.

**Key Features:**
- Automatic type conversion
- Attribute-style access (e.g., `config.model_name`)
- Dictionary compatibility
- Nested configuration support

**Usage:**
```python
from configs.config_class import ExperimentConfig

config = ExperimentConfig(base_config)
print(config.model_name)  # TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

### `config_utils.py`
Utility functions for configuration manipulation and variation generation.

**Key Functions:**

#### `nested_update(base_dict, update_dict)`
Recursively updates nested dictionaries without overwriting entire sub-dicts.

```python
base = {"a": {"b": 1, "c": 2}}
update = {"a": {"b": 99}}
result = nested_update(base, update)
# Result: {"a": {"b": 99, "c": 2}}
```

#### `update_configs_list(base_config_dict, multiple_updates_to_apply)`
Applies multiple updates to generate a list of configuration variations.

```python
base = {"batch_size": 16, "precision": "float32"}
updates = [
    {"batch_size": 32},
    {"precision": "float16"}
]
configs = update_configs_list(base, updates)
# Returns: [config_with_batch_32, config_with_fp16]
```

#### `generate_config_name_from_update(update_dict)`
Generates human-readable configuration names from update dictionaries.

```python
update = {
    "batching_options.batch_size___fixed_batching": 32,
    "fp_precision": "float16"
}
name = generate_config_name_from_update(update)
# Returns: "batching_32_precis_float16"
```

### `b_models_config.py`
Generates configurations for comparing different model sizes.

**Models Included:**
- TinyLlama-1.1B-Chat-v1.0
- Llama-3.2-1B
- Llama-3.2-3B
- Llama-3.1-8B

**Usage:**
```python
from configs.b_models_config import get_model_configs

model_configs = get_model_configs()
# Returns: [tinyllama_config, llama32_1b_config, llama32_3b_config, llama31_8b_config]
```

### `c_controlled_configs.py`
Generates ~250+ configurations for controlled parameter variation experiments. Each configuration varies ONE parameter while keeping all others constant (ceteris paribus).

**Parameter Variations:**

1. **Parallelization** (4 configs)
   - `num_processes`: 1, 2, 3, 4

2. **Batching** (13 configs)
   - Fixed batch sizes: 1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64
   - Adaptive batching variations

3. **Precision & Quantization** (7 configs)
   - float32 (baseline)
   - float16
   - bfloat16
   - float8
   - 8-bit quantization
   - 4-bit quantization
   - Mixed precision experiments

4. **Decoder Modes** (53 configs)
   - Greedy decoding (baseline)
   - Top-K sampling (k = 5, 10, 20, 50, 100, 200) × temperatures (0.1, 0.5, 0.7, 1.0, 1.5)
   - Top-P sampling (p = 0.9, 0.95, 0.99) × temperatures (0.1, 0.5, 0.7, 1.0, 1.5)

5. **Latency Simulation** (29 configs)
   - **Constant Delays**: 50ms, 100ms, 200ms, 500ms, 1000ms
   - **Bursty Traffic**: 24 combinations of burst patterns
     - Burst intervals: 1s, 2s, 5s
     - Burst sizes: 2, 5, 10, 20 requests
     - Various delay ranges

**Usage:**
```python
from configs.c_controlled_configs import get_controlled_configs

controlled_configs = get_controlled_configs()
print(f"Generated {len(controlled_configs)} controlled variations")

# Each config has metadata about the variation
config = controlled_configs[0]
print(config["controlled_variation"])
# Example: {"parameter": "num_processes", "value": 2, "category": "parallelization"}
```

### `d_scenario_configs.py`
Pre-defined realistic and ideal deployment scenarios.

**Scenarios:**

1. **Ideal Conditions** (Low latency, high performance)
   - No latency simulation
   - Large batch sizes
   - Greedy decoding
   - float16 precision

2. **Realistic Production** (Moderate latency, balanced performance)
   - 100ms constant latency
   - Medium batch sizes
   - Top-P sampling (p=0.95, temp=0.7)
   - float16 precision

3. **Constrained Edge Deployment** (High latency, resource-constrained)
   - 500ms constant latency
   - Small batch sizes
   - 4-bit quantization
   - Greedy decoding for speed

4. **Bursty Traffic** (Variable load)
   - Burst simulation enabled
   - Adaptive batching
   - Top-K sampling

**Usage:**
```python
from configs.d_scenario_configs import get_scenario_configs

scenarios = get_scenario_configs()
for scenario in scenarios:
    print(f"Scenario: {scenario['scenario_info']['name']}")
    print(f"Description: {scenario['scenario_info']['description']}")
```

### `e_grid_configs.py`
Generates comprehensive grid search configurations by computing the Cartesian product of parameter variations.

**Grid Dimensions:**
- **Precision**: float32, float16, bfloat16, 8-bit quantization, 4-bit quantization
- **Latency**: None, 50ms, 100ms, 200ms constant delays
- **Parallelization**: 1, 2, 4 processes
- **Batching**: 8, 16, 32, 64 batch sizes
- **Decoding**: Greedy, Top-K (k=50, temp=0.7), Top-P (p=0.95, temp=0.7)

**Total Configurations**: 5 × 4 × 3 × 4 × 3 = **720 configurations**

**Usage:**
```python
from configs.e_grid_configs import get_grid_search_configs

grid_configs = get_grid_search_configs()
print(f"Total grid search configurations: {len(grid_configs)}")

# Grid configs have metadata about all varied parameters
config = grid_configs[0]
print(config["grid_point"])
# Example: {"precision": "float16", "latency": "50ms", "num_processes": 2, ...}
```

## Configuration Naming Convention

Configurations are automatically named based on their variations:

**Format**: `{param1}_{value1}_{param2}_{value2}_...`

**Parameter Abbreviations:**
- `batching`: Batch size variations
- `precis`: Precision (float32/16/8, bfloat16)
- `quant`: Quantization (8bit, 4bit)
- `decode`: Decoding mode (greedy, topk, topp)
- `temp`: Temperature
- `latency`: Latency simulation (const, burst)
- `parallel`: Number of processes

**Examples:**
- `batching_32_precis_float16`
- `decode_topk50_temp_0.7`
- `latency_const_100ms_parallel_4`

## Configuration Validation

All configuration suites include validation to ensure:
1. Required keys are present
2. Types are correct
3. Values are within valid ranges
4. Dependent parameters are consistent

**Example Validation:**
```python
def validate_config(config):
    """Ensures configuration has all required keys and valid values."""
    required_keys = ["model_name", "task_type", "num_processes", ...]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key: {key}")

    if config["num_processes"] < 1:
        raise ValueError("num_processes must be >= 1")

    # Additional validation...
    return True
```

## Best Practices

### Creating Custom Configurations

1. **Start from Base Config:**
```python
from configs.a_default_config import base_config
from configs.config_utils import nested_update

custom_config = nested_update(base_config.copy(), {
    "model_name": "meta-llama/Llama-3.2-1B",
    "fp_precision": "float16",
    "batching_options.batch_size___fixed_batching": 32
})
```

2. **Generate Variations Systematically:**
```python
from configs.config_utils import update_configs_list

updates = [
    {"fp_precision": "float32"},
    {"fp_precision": "float16"},
    {"fp_precision": "bfloat16"}
]

precision_configs = update_configs_list(base_config, updates)
```

3. **Add Metadata:**
```python
config["suite"] = "precision_comparison"
config["controlled_variation"] = {
    "parameter": "fp_precision",
    "value": "float16",
    "category": "precision"
}
```

### Configuration Hierarchy

Configurations follow this priority order:
1. Experiment-specific configs (highest priority)
2. Suite-specific defaults
3. Base config (lowest priority)

### Working with Nested Configurations

Use dot notation for nested updates:
```python
update = {
    "batching_options.batch_size___fixed_batching": 64,
    "decoder_config.decoding_mode": "top_k",
    "decoder_config.decoder_top_k": 50
}
```

## Quantization Configuration

### 4-bit Quantization
```python
quantization_config = {
    "quantization": True,
    "load_in_4bit": True,
    "load_in_8bit": False,
    "cached_flops_for_quantised_models": 52638582308864
}
```

### 8-bit Quantization
```python
quantization_config = {
    "quantization": True,
    "load_in_4bit": False,
    "load_in_8bit": True,
    "cached_flops_for_quantised_models": 52638582308864
}
```

### Important Note on FLOPs
The `cached_flops_for_quantised_models` parameter is a workaround for computing FLOPs on quantized models. The `ptflops` library cannot accurately measure FLOPs for quantized models, so this hardcoded value is used. **This is a known limitation and will be addressed in a future refactor.**

## Decoder Configuration

### Greedy Decoding (Fastest, Deterministic)
```python
decoder_config = {
    "decoding_mode": "greedy",
    "decoder_temperature": 1.0,
    "decoder_top_k": None,
    "decoder_top_p": None
}
```

### Top-K Sampling
```python
decoder_config = {
    "decoding_mode": "top_k",
    "decoder_temperature": 0.7,
    "decoder_top_k": 50,
    "decoder_top_p": None
}
```

### Top-P (Nucleus) Sampling
```python
decoder_config = {
    "decoding_mode": "top_p",
    "decoder_temperature": 0.7,
    "decoder_top_k": None,
    "decoder_top_p": 0.95
}
```

## Latency Simulation

### Constant Latency
```python
latency_simulation = {
    "simulate": True,
    "delay_min": 0.1,      # 100ms
    "delay_max": 0.1,      # 100ms
    "simulate_burst": False
}
```

### Bursty Traffic
```python
latency_simulation = {
    "simulate": True,
    "delay_min": 0.05,     # 50ms
    "delay_max": 0.3,      # 300ms
    "simulate_burst": True,
    "burst_interval": 2.0,  # Burst every 2 seconds
    "burst_size": 10        # 10 requests per burst
}
```

## Usage in Main Scripts

### Single Experiment
```python
from configs.a_default_config import base_config
from experiment_orchestration_utils.a_experiment_runner_class import ExperimentRunner

runner = ExperimentRunner(base_config)
runner.run()
```

### Multiple Configurations
```python
from configs.c_controlled_configs import get_controlled_configs

configs = get_controlled_configs()
for config in configs:
    runner = ExperimentRunner(config)
    runner.run()
```

### Grid Search
```python
from configs.e_grid_configs import get_grid_search_configs

grid = get_grid_search_configs()
print(f"Running {len(grid)} configurations...")
for config in grid:
    # Execute experiment
    pass
```

## Adding New Configuration Suites

To create a new configuration suite:

1. **Create new file** (e.g., `f_custom_configs.py`)
2. **Import base config and utilities:**
```python
from configs.a_default_config import base_config
from configs.config_utils import update_configs_list, generate_config_name_from_update
```

3. **Define variations:**
```python
def get_custom_configs():
    updates = [
        {"model_name": "custom-model-1", "suite": "custom"},
        {"model_name": "custom-model-2", "suite": "custom"},
    ]
    return update_configs_list(base_config, updates)
```

4. **Add validation:**
```python
configs = get_custom_configs()
for config in configs:
    validate_config(config)
```

## Configuration Schema

For reference, here's the complete configuration schema:

```python
{
    "config_name": str,                    # Auto-generated
    "suite": str,                          # Configuration suite identifier
    "controlled_variation": dict,          # Metadata for controlled experiments
    "scenario_info": dict,                 # Metadata for scenarios
    "cycle_id": int,                       # Injected at runtime

    # Model
    "model_name": str,
    "is_encoder_decoder": bool,
    "backend": str,

    # Task
    "task_type": str,
    "inference_type": str,

    # Hardware
    "gpu_list": list[int],
    "num_processes": int,

    # Inference
    "max_input_tokens": int,
    "max_output_tokens": int,
    "min_output_tokens": int,
    "num_input_prompts": int,

    # Batching
    "batching_options": {
        "batch_size___fixed_batching": int,
        "adaptive_batching": bool,
        "adaptive_max_tokens": int,
        "max_batch_size___adaptive_batching": int
    },

    # Sharding
    "sharding_config": {
        "fsdp_config": {
            "use_orig_params": bool,
            "cpu_offload": bool
        },
        "sharding_strategy": str
    },

    # Decoding
    "decoder_config": {
        "decoding_mode": str,
        "decoder_temperature": float,
        "decoder_top_k": int | None,
        "decoder_top_p": float | None
    },

    # Precision & Quantization
    "fp_precision": str,
    "quantization_config": {
        "quantization": bool | None,
        "load_in_8bit": bool | None,
        "load_in_4bit": bool | None,
        "cached_flops_for_quantised_models": int
    },

    # Latency Simulation
    "latency_simulation": {
        "simulate": bool,
        "delay_min": float,
        "delay_max": float,
        "simulate_burst": bool,
        "burst_interval": float,
        "burst_size": int
    },

    # Output
    "query_rate": float,
    "save_outputs": bool,
    "decode_token_to_text": bool
}
```

## Troubleshooting

### Configuration Validation Errors
If you encounter validation errors:
1. Check that all required keys are present
2. Verify value types match expected types
3. Ensure nested dictionaries are properly structured
4. Use `nested_update()` for nested configurations

### Missing Configuration Keys
Use the `fill_missing_keys()` utility to add missing keys from base config:
```python
from configs.config_utils import fill_missing_keys

partial_config = {"model_name": "custom-model"}
complete_config = fill_missing_keys(partial_config, base_config)
```

### Configuration Name Collisions
If multiple configurations generate the same name:
1. Add more distinguishing parameters to the update
2. Manually set `config_name` field
3. Use suite identifier to namespace configs

## Future Improvements

Planned enhancements to the configuration system:
1. **Pydantic validation**: Replace manual validation with Pydantic models
2. **YAML/JSON configuration files**: Support external configuration files
3. **Configuration inheritance**: Allow configs to inherit from other configs
4. **Dynamic FLOPs calculation**: Properly compute FLOPs for quantized models
5. **Configuration versioning**: Track configuration schema versions

## Author

Henry Baker
