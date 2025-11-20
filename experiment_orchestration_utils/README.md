# Experiment Orchestration Utilities

This directory contains high-level orchestration utilities for managing experiment execution, retry logic, and integration with Hugging Face Accelerate for distributed training/inference.

## Overview

The orchestration layer handles:
- **Experiment Execution**: Main `ExperimentRunner` class that coordinates the entire workflow
- **Retry Logic**: Automatic retry with exponential backoff for transient failures
- **Launcher Integration**: Seamless integration with `accelerate launch` CLI for distributed execution
- **Error Handling**: Comprehensive error handling and cleanup
- **Progress Tracking**: Integration with persistent progress trackers

## Modules

### `a_experiment_runner_class.py` - Main Experiment Runner

The `ExperimentRunner` class is the central coordinator for all experiments. It orchestrates the entire pipeline from setup to results saving.

**Class: `ExperimentRunner`**

#### Constructor
```python
runner = ExperimentRunner(experiment_config)
```

**Parameters:**
- `experiment_config` (dict): Configuration dictionary (from `configs/`)

**Attributes:**
- `config`: Experiment configuration
- `accelerator`: Hugging Face Accelerate object (initialized in `run_setup()`)
- `experiment_id`: Unique identifier (generated in `run_setup()`)
- `model`: Loaded model
- `tokenizer`: Loaded tokenizer
- `results`: Dictionary storing all results

#### Main Methods

##### `run_setup()`
Initializes the distributed environment and generates unique experiment ID.

```python
runner.run_setup()
```

**Operations:**
1. Calls `setup_distributed()` to initialize Accelerate
2. Generates unique experiment ID via `generate_unique_id()`
3. Broadcasts ID to all processes
4. Returns self for method chaining

##### `run_torch()`
Main execution method that runs the complete experiment workflow.

```python
runner.run_torch()
```

**Workflow:**
1. **Load Model & Tokenizer**
   ```python
   self.model, self.tokenizer = load_model_tokenizer(self.config)
   ```

2. **Prepare for Distributed Execution**
   ```python
   self.model = self.accelerator.prepare(self.model)
   ```

3. **Load and Process Prompts**
   ```python
   dataset = load_dataset("AIEnergyScore/text_generation")
   prompts = filter_prompts_by_length(dataset['text'], self.tokenizer, max_tokens)
   tokenized_prompts = tokenise_prompts(prompts, self.tokenizer)
   ```

4. **Warm-up Runs** (3 iterations)
   - Ensures GPU is fully initialized
   - Stabilizes timing measurements
   - Clears caches

5. **Start Energy Tracking**
   ```python
   tracker = start_energy_tracking(self.experiment_id)
   ```

6. **Run Inference**
   ```python
   inference_results = run_gen_inference(
       self.model, self.tokenizer, tokenized_prompts,
       self.config, self.accelerator
   )
   ```

7. **Stop Energy Tracking**
   ```python
   stop_energy_tracking(tracker)
   ```

8. **Collect Metadata**
   - Experiment setup info
   - Model architecture details
   - Experimental variables

9. **Compute Metrics**
   - Inference metrics (throughput, latency)
   - Compute metrics (FLOPs, memory, utilization)
   - Energy metrics (kWh, emissions)

10. **Save Results**
    - Raw results per component (JSON)
    - Aggregated results (JSON + CSV)

**Error Handling:**
- Catches and logs all exceptions
- Performs cleanup even on failure
- Re-raises exceptions for retry logic

##### `aggregate_results()`
Aggregates results across distributed processes and computes derived metrics.

```python
runner.aggregate_results()
```

**Operations:**
1. Loads per-process energy results
2. Computes aggregate statistics:
   - Sum of total energy across processes
   - Average energy per process
   - Peak memory usage
   - Total FLOPs
3. Derives efficiency metrics:
   - Tokens per joule
   - FLOPs per joule
   - Queries per kWh
   - Energy per token

**Returns:** Dictionary with aggregated metrics

##### `save_configuration_run_results_json()`
Saves complete experiment results as JSON.

```python
runner.save_configuration_run_results_json()
```

**Output Location:** `results/raw_results/{experiment_id}/`

**Files Created:**
- `{experiment_id}_1_experiment_setup.json`
- `{experiment_id}_2_experiment_variables.json`
- `{experiment_id}_3_model_architecture.json`
- `{experiment_id}_4_inference_metrics.json`
- `{experiment_id}_5_compute_metrics.json`
- `{experiment_id}_6_local_energy_results_process_{rank}.json` (per process)
- `{experiment_id}_7_global_energy_results.json`
- `{experiment_id}_8_text_output.json` (if enabled)
- `{experiment_id}_8_token_output.json` (if enabled)

##### `save_configuration_run_results_tabular()`
Appends experiment results to aggregated CSV file.

```python
runner.save_configuration_run_results_tabular()
```

**Output:** `results/{task_type}_results.csv`

**Features:**
- Flattens nested JSON structure
- Appends to existing CSV
- Creates CSV if it doesn't exist
- Ensures consistent column ordering

##### `teardown()`
Cleanup method for releasing resources.

```python
runner.teardown()
```

**Operations:**
- Clears CUDA cache
- Releases model from memory
- Closes file handles
- Performs distributed barrier synchronization

#### Complete Usage Example

```python
from experiment_orchestration_utils.a_experiment_runner_class import ExperimentRunner
from configs.a_default_config import base_config

# Create runner
runner = ExperimentRunner(base_config)

try:
    # Setup
    runner.run_setup()

    # Execute experiment
    runner.run_torch()

    # Aggregate and save results
    runner.aggregate_results()
    runner.save_configuration_run_results_json()
    runner.save_configuration_run_results_tabular()

except Exception as e:
    print(f"Experiment failed: {e}")

finally:
    # Always cleanup
    runner.teardown()
```

---

### `b_single_config_workflow.py` - Retry Logic

Provides robust retry logic for single configuration experiments.

**Key Functions:**

#### `run_single_config_with_retry(experiment_config, max_retries=3, backoff_base=2)`
Executes a single experiment with automatic retry on failure.

```python
success = run_single_config_with_retry(
    experiment_config=config,
    max_retries=3,
    backoff_base=2
)
```

**Parameters:**
- `experiment_config` (dict): Configuration for the experiment
- `max_retries` (int): Maximum number of retry attempts (default: 3)
- `backoff_base` (int): Base for exponential backoff in seconds (default: 2)

**Returns:**
- `True` if experiment succeeded
- `False` if all retries exhausted

**Retry Strategy:**
- **Attempt 1**: Immediate execution
- **Attempt 2**: Wait 2 seconds
- **Attempt 3**: Wait 4 seconds
- **Attempt 4**: Wait 8 seconds

**Backoff Formula:** `wait_time = backoff_base ** (attempt - 1)`

**Error Handling:**
1. Catches all exceptions
2. Logs error details
3. Waits according to backoff schedule
4. Retries up to `max_retries` times
5. Returns failure status if all retries exhausted

**Cleanup:**
- Always calls `runner.teardown()` after each attempt
- Clears CUDA memory between retries
- Releases distributed resources

**Usage Example:**
```python
from experiment_orchestration_utils.b_single_config_workflow import run_single_config_with_retry

configs = get_controlled_configs()

for config in configs:
    success = run_single_config_with_retry(config, max_retries=3)
    if success:
        print(f"✓ Config {config['config_name']} completed")
    else:
        print(f"✗ Config {config['config_name']} failed after retries")
```

**Common Failure Scenarios Handled:**
- CUDA out of memory
- Distributed process timeouts
- Network timeouts (Hugging Face Hub)
- Model loading failures
- Temporary file system issues

---

### `c_launcher_utils.py` - Accelerate Integration

Utilities for launching experiments via `accelerate launch` CLI.

**Key Functions:**

#### `launch_experiment_via_accelerate(experiment_config, temp_config_path="/tmp/exp_config.json")`
Launches an experiment as a subprocess using `accelerate launch`.

```python
success = launch_experiment_via_accelerate(
    experiment_config=config,
    temp_config_path="/tmp/exp_config_4459.json"
)
```

**Parameters:**
- `experiment_config` (dict): Configuration dictionary
- `temp_config_path` (str): Path for temporary config file

**Returns:**
- `True` if experiment succeeded (exit code 0)
- `False` if experiment failed

**Process:**
1. **Save Config to Temporary File**
   ```python
   with open(temp_config_path, 'w') as f:
       json.dump(experiment_config, f)
   ```

2. **Configure Accelerate Launch Arguments**
   ```python
   accelerate_args = [
       "accelerate", "launch",
       "--num_processes", str(num_processes),
       "--gpu_ids", ",".join(map(str, gpu_list)),
       "--mixed_precision", "fp16",  # or based on config
       "MAIN_a_single_experiment.py",
       "--config_path", temp_config_path
   ]
   ```

3. **Execute Subprocess**
   ```python
   result = subprocess.run(accelerate_args, capture_output=True, text=True)
   ```

4. **Handle Output**
   - Streams stdout/stderr in real-time
   - Captures exit code
   - Logs errors

5. **Cleanup**
   ```python
   os.remove(temp_config_path)
   ```

**Environment Variables Set:**
```python
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_list))
env["ACCELERATE_NUM_PROCESSES"] = str(num_processes)
```

#### `batch_launch_experiments(experiment_configs, max_parallel=1)`
Launches multiple experiments sequentially or in parallel.

```python
results = batch_launch_experiments(
    experiment_configs=configs,
    max_parallel=1  # Sequential execution
)
```

**Parameters:**
- `experiment_configs` (list[dict]): List of configurations
- `max_parallel` (int): Maximum parallel executions (default: 1 for sequential)

**Returns:**
- Dictionary mapping config names to success status

**Parallel Execution:**
```python
# Launch 4 experiments in parallel
results = batch_launch_experiments(configs, max_parallel=4)
```

**Features:**
- Progress tracking
- Failure logging
- Summary statistics
- Configurable parallelism

#### `log_failed_experiment(experiment_config, error_message, log_file="failed_experiments.csv")`
Logs failed experiments to CSV for analysis.

```python
log_failed_experiment(
    experiment_config=config,
    error_message="CUDA out of memory",
    log_file="failed_experiments.csv"
)
```

**CSV Columns:**
- `timestamp`: When failure occurred
- `experiment_id`: Unique ID (if generated)
- `config_name`: Configuration name
- `model_name`: Model being used
- `error_message`: Error description
- `config_json`: Full configuration (as JSON string)

**Usage:**
```python
try:
    runner.run_torch()
except Exception as e:
    log_failed_experiment(config, str(e))
    raise
```

---

## Workflow Integration

### Single Experiment (MAIN_a_single_experiment.py)

```python
from experiment_orchestration_utils.a_experiment_runner_class import ExperimentRunner
from configs.a_default_config import base_config

runner = ExperimentRunner(base_config)
runner.run_setup()
runner.run_torch()
runner.aggregate_results()
runner.save_configuration_run_results_json()
runner.save_configuration_run_results_tabular()
runner.teardown()
```

### Multiple Configurations with Retry (MAIN_c_controlled_experiments.py)

```python
from experiment_orchestration_utils.b_single_config_workflow import run_single_config_with_retry
from configs.c_controlled_configs import get_controlled_configs

configs = get_controlled_configs()

success_count = 0
failure_count = 0

for config in configs:
    if run_single_config_with_retry(config, max_retries=3):
        success_count += 1
    else:
        failure_count += 1

print(f"Completed: {success_count} succeeded, {failure_count} failed")
```

### Large-Scale Suite with Progress Tracking (MAIN_run_experimental_suite.py)

```python
from experiment_orchestration_utils.b_single_config_workflow import run_single_config_with_retry
from persistent_progress_trackers.progress_tracker import load_progress, save_progress
from configs.b_models_config import get_model_configs
from configs.c_controlled_configs import get_controlled_configs

# Load progress
progress = load_progress()

# Define suite
models = get_model_configs()
controlled = get_controlled_configs()

for cycle in range(1, 6):  # 5 cycles
    for model_config in models:
        for controlled_config in controlled:
            # Merge configs
            config = {**model_config, **controlled_config, "cycle_id": cycle}

            # Check if already completed
            key = f"{config['model_name']}::{config['suite']}::{config['config_name']}::{cycle}"
            if progress.get(key):
                print(f"Skipping {key} (already completed)")
                continue

            # Run experiment
            success = run_single_config_with_retry(config, max_retries=3)

            # Update progress
            if success:
                progress[key] = runner.experiment_id
                save_progress(progress)
```

---

## Distributed Execution Flow

### When Running Directly (Without Accelerate Launch)

```python
# MAIN_a_single_experiment.py
from experiment_orchestration_utils.a_experiment_runner_class import ExperimentRunner

runner = ExperimentRunner(config)

# Check if already launched with accelerate
if not os.environ.get("ACCELERATE_TORCH_DEVICE"):
    # Not launched via accelerate - relaunch
    print("Relaunching with accelerate...")
    launch_experiment_via_accelerate(config)
    sys.exit(0)

# Already launched via accelerate - proceed
runner.run_setup()
runner.run_torch()
# ...
```

### When Using Accelerate Launch Directly

```bash
accelerate launch \
    --num_processes 4 \
    --gpu_ids 0,1,2,3 \
    --mixed_precision fp16 \
    MAIN_a_single_experiment.py
```

**Process Flow:**

1. **Process 0 (Main):**
   - Loads model
   - Coordinates experiment
   - Saves results
   - Aggregates metrics

2. **Processes 1-3 (Workers):**
   - Load model replicas
   - Process subset of prompts
   - Report local metrics
   - Wait at barriers

3. **Synchronization Points:**
   - After model loading
   - Before inference
   - After inference
   - Before results saving

---

## Error Handling Best Practices

### 1. CUDA Out of Memory

```python
try:
    runner.run_torch()
except RuntimeError as e:
    if "out of memory" in str(e):
        torch.cuda.empty_cache()
        # Reduce batch size
        config["batching_options"]["batch_size___fixed_batching"] //= 2
        # Retry
        runner = ExperimentRunner(config)
        runner.run_torch()
    else:
        raise
```

### 2. Distributed Timeout

```python
try:
    runner.run_torch()
except TimeoutError:
    # Cleanup distributed processes
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    # Retry with increased timeout
    os.environ["NCCL_TIMEOUT"] = "3600"  # 1 hour
    runner = ExperimentRunner(config)
    runner.run_torch()
```

### 3. Model Loading Failure

```python
try:
    runner.run_torch()
except OSError as e:
    if "not found" in str(e):
        print("Model not found, downloading...")
        # Will auto-download on next attempt
        time.sleep(5)
        runner = ExperimentRunner(config)
        runner.run_torch()
    else:
        raise
```

---

## Performance Optimization

### Memory Optimization

```python
# Use gradient checkpointing for large models
config["model_config"] = {
    "use_cache": False,
    "gradient_checkpointing": True
}

# Enable CPU offloading if needed
config["sharding_config"]["fsdp_config"]["cpu_offload"] = True
```

### Batch Size Tuning

```python
# Start with large batch size, reduce on OOM
for batch_size in [64, 32, 16, 8, 4, 2, 1]:
    config["batching_options"]["batch_size___fixed_batching"] = batch_size
    try:
        runner = ExperimentRunner(config)
        runner.run_torch()
        break  # Success
    except RuntimeError as e:
        if "out of memory" in str(e):
            torch.cuda.empty_cache()
            continue
        raise
```

### Warm-up Optimization

```python
# Adjust warm-up iterations based on model size
if model_params > 10e9:  # > 10B parameters
    config["num_warmup_iterations"] = 5
else:
    config["num_warmup_iterations"] = 3
```

---

## Progress Tracking Integration

The orchestration layer integrates with persistent progress trackers:

```python
from persistent_progress_trackers.progress_tracker import (
    load_progress,
    save_progress,
    get_next_experiment_id
)

# Load existing progress
progress = load_progress()

# Get next experiment ID
experiment_id = get_next_experiment_id()

# Update progress after successful run
key = f"{model_name}::{suite}::{config_name}::{cycle_id}"
progress[key] = experiment_id
save_progress(progress)
```

---

## Signal Handling

For graceful shutdown on SIGINT/SIGTERM:

```python
import signal

def signal_handler(signum, frame):
    print("Received interrupt signal, cleaning up...")
    if runner:
        runner.teardown()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
```

---

## Logging Best Practices

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'experiment_{experiment_id}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

logger.info(f"Starting experiment {experiment_id}")
logger.info(f"Config: {config['config_name']}")
logger.info(f"Model: {config['model_name']}")
```

---

## Testing

Recommended testing strategy:

```python
# Test with small config
test_config = base_config.copy()
test_config["num_input_prompts"] = 10
test_config["max_output_tokens"] = 16

runner = ExperimentRunner(test_config)
runner.run_setup()
runner.run_torch()

assert runner.experiment_id is not None
assert runner.results["inference_metrics"]["num_prompts"] == 10
```

---

## Future Improvements

1. **Checkpoint/Resume:** Save intermediate state for long experiments
2. **Real-time Monitoring:** Stream metrics to dashboard
3. **Dynamic Resource Allocation:** Adjust num_processes based on load
4. **Better Parallelization:** Run multiple configs simultaneously
5. **Cloud Integration:** Support for cloud GPU providers (AWS, GCP, Azure)
6. **Container Support:** Docker/Singularity integration
7. **Fault Tolerance:** Automatic recovery from node failures

---

## Author

Henry Baker
