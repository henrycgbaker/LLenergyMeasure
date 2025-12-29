# configs/ - Legacy Configuration Files

Python-based configuration files from the original research implementation.

## Status

These are **legacy** configs from the research phase. The modern approach uses YAML/JSON configs loaded via `llm_energy_measure.config.loader`.

For new experiments, create YAML configs instead:
```yaml
# configs/experiment.yaml
config_name: my-experiment
model_name: meta-llama/Llama-2-7b-hf
max_input_tokens: 512
```

## Legacy Files

### a_default_config.py
Base configuration dictionary with all default values.
```python
base_config = {
    "config_name": None,
    "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "max_input_tokens": 128,
    "max_output_tokens": 128,
    # ... full defaults
}
```

### b_models_config.py
Model-specific configuration overrides.

### c_controlled_configs.py
Controlled experiment configurations with systematic variations for research experiments.

### d_scenario_configs.py
Scenario-based configs for specific experimental conditions.

### e_grid_configs.py
Grid search configurations for hyperparameter exploration.

### config_class.py
Legacy configuration class (predates Pydantic models).

### config_utils.py
Utilities for working with legacy Python configs.

## Migration Path

To use legacy configs with the new system:

1. Export to YAML:
```python
import yaml
from configs.a_default_config import base_config

with open("configs/base.yaml", "w") as f:
    yaml.dump(base_config, f)
```

2. Update field names to match `ExperimentConfig`:
   - `batch_size___fixed_batching` -> `batching_options.batch_size`
   - `decoder_temperature` -> `decoder_config.temperature`

3. Validate:
```bash
llm-energy-measure config validate configs/base.yaml
```

## Related

- See `src/llm_energy_measure/config/README.md` for modern config system
- See `src/llm_energy_measure/config/models.py` for `ExperimentConfig`
