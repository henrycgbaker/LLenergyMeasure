# Experiment Core Utilities

This directory contains the core utilities for executing LLM inference experiments and collecting comprehensive metrics. These modules handle distributed setup, model loading, inference execution, and metrics collection.

## Overview

The core utilities are organized in a pipeline structure, indicated by the alphabetical prefixes (a-l) that suggest the typical execution order:

```
a_distributed.py          → Setup distributed environment
b_model_loader.py         → Load models and tokenizers
c_prompt_processing.py    → Prepare input prompts
d_energy_tracking.py      → Initialize energy monitoring
e_inference.py            → Execute inference
f_experiment_info.py      → Collect experiment metadata
g_metrics_inference.py    → Compute inference metrics
h_metrics_compute.py      → Calculate computational metrics (FLOPs)
i_metrics_energy.py       → Aggregate energy metrics
j_results_saving.py       → Save results to disk
k_results_aggregation.py  → Aggregate multi-process results
l_results_csv_cleaning.py → Format results as CSV
```

## Modules

### `a_distributed.py` - Distributed Setup

Handles distributed computing setup using Hugging Face Accelerate.

**Key Functions:**

#### `setup_distributed(experiment_config)`
Initializes the Accelerate distributed environment.

```python
accelerator = setup_distributed(config)
print(f"Process {accelerator.process_index} of {accelerator.num_processes}")
```

**Returns:**
- `Accelerator` object configured for distributed execution

**Features:**
- Automatic device assignment
- Process synchronization
- Mixed precision support
- Gradient accumulation (if needed)

#### `generate_unique_id(accelerator)`
Generates a unique identifier for the experiment, ensuring consistency across all distributed processes.

```python
unique_id = generate_unique_id(accelerator)
# Example: "4459"
```

**Process:**
1. Main process reads/increments counter from `persistent_progress_trackers/experiment_id.txt`
2. ID is broadcast to all processes via distributed communication
3. All processes use the same ID for result correlation

**Usage:**
```python
from experiment_core_utils.a_distributed import setup_distributed, generate_unique_id

accelerator = setup_distributed(config)
experiment_id = generate_unique_id(accelerator)
```

---

### `b_model_loader.py` - Model Loading

Loads models and tokenizers from Hugging Face with support for various precision levels and quantization methods.

**Key Functions:**

#### `detect_supported_quant_types()`
Detects available quantization capabilities based on installed BitsAndBytes version.

```python
qsupport = detect_supported_quant_types()
print(f"4-bit support: {qsupport['supports_4bit']}")
print(f"8-bit support: {qsupport['supports_8bit']}")
```

**Returns:**
```python
{
    "supports_4bit": bool,
    "supports_8bit": bool,
    "default_4bit_quant_type": str | None,  # e.g., "nf4"
    "default_8bit_quant_type": str | None   # e.g., "fp8" or "int8"
}
```

#### `load_model_tokenizer(configs)`
Main function to load model and tokenizer with specified precision and quantization.

```python
model, tokenizer = load_model_tokenizer(config)
```

**Supported Precisions:**
- `float32`: Full precision (default for training)
- `float16`: Half precision (most common for inference)
- `bfloat16`: Brain floating point (better numerical stability)
- `float8`: 8-bit floating point

**Supported Quantization:**
- **8-bit quantization**: Reduces memory by ~75%, slight quality loss
- **4-bit quantization**: Reduces memory by ~87.5%, moderate quality loss
- **NF4 quantization**: 4-bit NormalFloat, optimized for neural networks

**Example Usage:**
```python
# Load with float16 precision
config.fp_precision = "float16"
model, tokenizer = load_model_tokenizer(config)

# Load with 4-bit quantization
config.quantization_config = {
    "quantization": True,
    "load_in_4bit": True,
    "load_in_8bit": False
}
model, tokenizer = load_model_tokenizer(config)
```

**Tokenizer Configuration:**
- `pad_token` set to `eos_token` (standard for causal LMs)
- `padding_side` set to `'left'` (for batched generation)

**Classes:**

#### `ModelWrapper`
Wraps Hugging Face models in a standard `nn.Module` interface for compatibility with FLOPs calculation tools.

```python
wrapped_model = ModelWrapper(hf_model)
output = wrapped_model(input_ids)
```

---

### `c_prompt_processing.py` - Prompt Processing

Prepares and processes input prompts for inference.

**Key Functions:**

#### `filter_prompts_by_length(prompts, tokenizer, max_tokens)`
Filters prompts that exceed the maximum token length.

```python
filtered_prompts = filter_prompts_by_length(
    prompts=dataset['text'],
    tokenizer=tokenizer,
    max_tokens=128
)
```

#### `sort_prompts_by_length(prompts, tokenizer)`
Sorts prompts by token length (useful for batching optimization).

```python
sorted_prompts = sort_prompts_by_length(prompts, tokenizer)
# Shorter prompts first, reduces padding
```

#### `tokenise_prompts(prompts, tokenizer)`
Batch tokenizes prompts with proper padding and attention masks.

```python
tokenized = tokenise_prompts(prompts, tokenizer)
# Returns: {"input_ids": tensor, "attention_mask": tensor}
```

**Typical Pipeline:**
```python
# 1. Load prompts from dataset
prompts = dataset['text']

# 2. Filter by length
valid_prompts = filter_prompts_by_length(prompts, tokenizer, max_tokens=128)

# 3. Sort by length (optional, for efficiency)
sorted_prompts = sort_prompts_by_length(valid_prompts, tokenizer)

# 4. Tokenize
tokenized_input = tokenise_prompts(sorted_prompts[:num_prompts], tokenizer)
```

---

### `d_energy_tracking.py` - Energy Tracking

Manages energy consumption tracking using CodeCarbon.

**Key Functions:**

#### `start_energy_tracking(experiment_id)`
Initializes and starts a CodeCarbon emissions tracker.

```python
tracker = start_energy_tracking(experiment_id="4459")
# Begins monitoring CPU, GPU, and RAM power consumption
```

**Tracked Metrics:**
- CPU energy consumption
- GPU energy consumption (per device)
- RAM energy consumption
- Total energy in kWh
- Carbon emissions (based on regional grid)

#### `stop_energy_tracking(tracker)`
Stops the tracker and saves emissions data.

```python
stop_energy_tracking(tracker)
# Saves results to: emissions_4459.csv
```

**Output Files:**
- `emissions_{experiment_id}.csv`: Detailed energy breakdown

**CodeCarbon Configuration:**
- Logging suppressed (set to ERROR level only)
- Saves data per experiment
- Tracks entire inference duration

**Usage Example:**
```python
from experiment_core_utils.d_energy_tracking import (
    start_energy_tracking,
    stop_energy_tracking
)

tracker = start_energy_tracking(experiment_id)

# ... run inference ...

stop_energy_tracking(tracker)
```

---

### `e_inference.py` - Inference Execution

Core inference execution with support for batching, latency simulation, and distributed processing.

**Key Functions:**

#### `run_gen_inference(model, tokeniser, tokenised_prompts, experiment_config, accelerator)`
Main inference function that executes text generation with comprehensive tracking.

```python
results = run_gen_inference(
    model=model,
    tokeniser=tokenizer,
    tokenised_prompts=tokenized_input,
    experiment_config=config,
    accelerator=accelerator
)
```

**Returns:**
```python
{
    "output_tokens": tensor,              # Generated token IDs
    "output_text": list[str],             # Decoded text (optional)
    "total_inference_time_s": float,      # Total time in seconds
    "total_input_tokens": int,
    "total_output_tokens": int,
    "num_prompts_processed": int,
    "tokens_per_second": float,
    "queries_per_second": float,
    "latency_per_query_s": float
}
```

**Features:**

1. **Batching Support**
   - Fixed batch size
   - Adaptive batching (based on token count)
   - Automatic batch splitting

2. **Latency Simulation**
   - Constant delays
   - Bursty traffic patterns
   - Configurable delay ranges

3. **Distributed Processing**
   - Splits prompts across processes
   - Synchronization via barriers
   - Per-process and global metrics

4. **Decoding Strategies**
   - Greedy decoding (fastest)
   - Top-K sampling
   - Top-P (nucleus) sampling
   - Temperature-based sampling

**Inference Flow:**
```
1. Split prompts across distributed processes
2. Prepare batches based on config
3. For each batch:
   a. Apply latency simulation (if enabled)
   b. Generate outputs using model.generate()
   c. Track timing and token counts
4. Aggregate metrics across batches
5. Synchronize across processes
6. Return comprehensive results
```

**Configuration Parameters:**

```python
# Batching
config.batching_options = {
    "batch_size___fixed_batching": 16,
    "adaptive_batching": False,
    "adaptive_max_tokens": 2048,
    "max_batch_size___adaptive_batching": 32
}

# Decoder
config.decoder_config = {
    "decoding_mode": "top_k",
    "decoder_temperature": 0.7,
    "decoder_top_k": 50,
    "decoder_top_p": None
}

# Latency Simulation
config.latency_simulation = {
    "simulate": True,
    "delay_min": 0.05,
    "delay_max": 0.1,
    "simulate_burst": False
}
```

**Usage Example:**
```python
# Basic inference
results = run_gen_inference(model, tokenizer, tokenized_prompts, config, accelerator)
print(f"Throughput: {results['tokens_per_second']:.2f} tokens/s")

# With latency simulation
config.latency_simulation["simulate"] = True
config.latency_simulation["delay_min"] = 0.1
results = run_gen_inference(model, tokenizer, tokenized_prompts, config, accelerator)
```

---

### `f_experiment_info.py` - Experiment Metadata

Collects comprehensive metadata about the experiment, system, and model.

**Key Functions:**

#### `gather_experiment_info(experiment_config, accelerator, experiment_id)`
Collects experimental setup information.

```python
setup_info = gather_experiment_info(config, accelerator, experiment_id)
```

**Returns:**
```python
{
    "experiment_id": str,
    "cycle_id": int,
    "timestamp": str,
    "distributed_config": {
        "num_processes": int,
        "local_process_index": int,
        "device": str
    },
    "hardware_info": {
        "cuda_available": bool,
        "num_gpus": int,
        "gpu_names": list[str],
        "cpu_count": int,
        "total_ram_gb": float
    }
}
```

#### `get_model_architecture_info(model)`
Extracts detailed model architecture information.

```python
arch_info = get_model_architecture_info(model)
```

**Returns:**
```python
{
    "model_class": str,
    "total_parameters": int,
    "trainable_parameters": int,
    "model_size_mb": float,
    "architecture": {
        "num_layers": int,
        "hidden_size": int,
        "num_attention_heads": int,
        "vocab_size": int,
        "max_position_embeddings": int
    },
    "dtype": str,
    "device": str
}
```

#### `gather_experimental_variables(experiment_config)`
Extracts all experimental variables for result tracking.

```python
variables = gather_experimental_variables(config)
```

**Returns:**
```python
{
    "model_name": str,
    "precision": str,
    "quantization": dict,
    "batch_size": int,
    "num_processes": int,
    "decoding_config": dict,
    "latency_config": dict,
    # ... all experimental parameters
}
```

---

### `g_metrics_inference.py` - Inference Metrics

Computes inference-related performance metrics.

**Key Functions:**

#### `combine_inference_metrics(inference_results)`
Aggregates and computes derived metrics from inference results.

```python
metrics = combine_inference_metrics(inference_results)
```

**Returns:**
```python
{
    "total_input_tokens": int,
    "total_output_tokens": int,
    "total_tokens": int,
    "num_prompts": int,
    "total_time_s": float,
    "tokens_per_second": float,
    "queries_per_second": float,
    "latency_per_query_s": float,
    "avg_input_tokens_per_prompt": float,
    "avg_output_tokens_per_prompt": float
}
```

---

### `h_metrics_compute.py` - Computational Metrics

Calculates FLOPs (Floating Point Operations) and other computational metrics.

**Key Functions:**

#### `get_flops(model, input_ids, timeout_per_sample=10)`
Computes total FLOPs for inference using the `ptflops` library.

```python
flops = get_flops(model, tokenized_input_ids)
print(f"Total FLOPs: {flops:,}")
```

**Features:**
- Optimized computation for uniform-length batches
- Per-sample fallback for variable-length inputs
- Timeout protection (10s default per sample)
- Thread-based execution for safety

**Process:**
1. Check if all samples have the same length
2. If yes: Compute FLOPs for one sample, multiply by batch size
3. If no: Compute FLOPs per sample and sum

**Known Limitation:**
The `ptflops` library cannot accurately compute FLOPs for quantized models. The code falls back to cached values (see `combine_comp_metrics`).

#### `get_memory(device)`
Retrieves GPU memory usage statistics.

```python
memory_stats = get_memory(device)
```

**Returns:**
```python
{
    "gpu_current_memory_allocated_bytes": int,
    "gpu_max_memory_allocated_bytes": int,
    "gpu_current_memory_reserved_bytes": int,
    "gpu_max_memory_reserved_bytes": int
}
```

#### `get_gpu_cpu_utilisation(device)`
Retrieves GPU and CPU utilization metrics.

```python
utilization = get_gpu_cpu_utilisation(device)
```

**Returns:**
```python
{
    "gpu_utilization_percent": list[float],  # Per-GPU utilization
    "cpu_usage_percent": float,
    "cpu_memory_usage_bytes": int
}
```

#### `combine_comp_metrics(model, device, tokenised_input_ids, accelerator, experiment_config)`
Main function that combines all computational metrics.

```python
comp_metrics = combine_comp_metrics(
    model, device, tokenized_input, accelerator, config
)
```

**Returns:**
```python
{
    "flops": float,
    "memory": {
        "gpu_current_memory_allocated_bytes": int,
        "gpu_max_memory_allocated_bytes": int,
        "gpu_current_memory_reserved_bytes": int,
        "gpu_max_memory_reserved_bytes": int
    },
    "compute_utilisation": {
        "gpu_utilization_percent": list[float],
        "cpu_usage_percent": float,
        "cpu_memory_usage_bytes": int
    }
}
```

**Quantization Handling:**
If `experiment_config.quantization_config["quantization"]` is `True`, uses cached FLOPs value instead of computing (due to `ptflops` limitations).

---

### `i_metrics_energy.py` - Energy Metrics

Aggregates energy consumption metrics from CodeCarbon.

**Key Functions:**

#### `aggregate_energy_metrics(experiment_id, accelerator)`
Loads and aggregates energy metrics from CodeCarbon output.

```python
energy_metrics = aggregate_energy_metrics(experiment_id, accelerator)
```

**Returns:**
```python
{
    "total_energy_kwh": float,
    "cpu_energy_kwh": float,
    "gpu_energy_kwh": float,
    "ram_energy_kwh": float,
    "total_emissions_kg_co2": float,
    "duration_seconds": float
}
```

**Process:**
1. Reads `emissions_{experiment_id}.csv`
2. Parses energy data from CodeCarbon
3. Converts units to standard format
4. Aggregates across all monitoring periods

---

### `j_results_saving.py` - Results Persistence

Saves experimental results in multiple formats.

**Key Functions:**

#### `save_raw_results_json(results_dict, experiment_id, result_type, accelerator)`
Saves raw results as JSON files.

```python
save_raw_results_json(
    results_dict=setup_info,
    experiment_id="4459",
    result_type="experiment_setup",
    accelerator=accelerator
)
```

**Output Location:**
```
results/raw_results/{experiment_id}/{experiment_id}_{priority}_{result_type}.json
```

**Priority Ordering:**
1. `experiment_setup`
2. `experiment_variables`
3. `model_architecture`
4. `inference_metrics`
5. `compute_metrics`
6. `local_energy_results_process_{rank}`
7. `global_energy_results`
8. `text_output` / `token_output` (optional)

#### `save_aggregated_results_json(results_dict, task_type)`
Saves aggregated results across experiments.

```python
save_aggregated_results_json(
    results_dict={"experiment_4459": {...}},
    task_type="text_generation"
)
```

**Output:** `results/{task_type}_results.json`

#### `flatten_results_to_csv(results_dict, task_type)`
Converts nested JSON results to flat CSV format.

```python
flatten_results_to_csv(results_dict, task_type="text_generation")
```

**Output:** `results/{task_type}_results.csv`

**Features:**
- Flattens nested dictionaries with dot notation
- Handles lists by creating indexed columns
- Preserves all data in tabular format

---

### `k_results_aggregation.py` - Results Aggregation

Aggregates results from multiple distributed processes.

**Key Functions:**

#### `load_and_aggregate_per_process_energy_results(experiment_id, num_processes)`
Loads energy results from each process and aggregates them.

```python
aggregated = load_and_aggregate_per_process_energy_results(
    experiment_id="4459",
    num_processes=4
)
```

**Returns:**
```python
{
    "total_energy_kwh": float,
    "avg_energy_per_process_kwh": float,
    "per_process_breakdown": list[dict],
    # ... other aggregated metrics
}
```

---

### `l_results_csv_cleaning.py` - CSV Formatting

Formats and cleans CSV output for analysis.

**Key Functions:**

#### `json_to_csv(json_file_path, csv_file_path)`
Converts JSON results to clean CSV format.

```python
json_to_csv(
    json_file_path="results/text_generation_results.json",
    csv_file_path="results/text_generation_results.csv"
)
```

#### `reorder_csv_columns(csv_file_path, priority_columns)`
Reorders CSV columns for better readability.

```python
priority_columns = [
    "experiment_id",
    "model_name",
    "precision",
    "batch_size",
    "tokens_per_second",
    "total_energy_kwh"
]

reorder_csv_columns("results/text_generation_results.csv", priority_columns)
```

**Features:**
- Places important columns first
- Alphabetizes remaining columns
- Preserves all data
- Handles missing columns gracefully

---

## Typical Workflow

Here's how these modules work together in a typical experiment:

```python
# 1. Setup distributed environment
accelerator = setup_distributed(config)
experiment_id = generate_unique_id(accelerator)

# 2. Load model and tokenizer
model, tokenizer = load_model_tokenizer(config)

# 3. Prepare prompts
prompts = load_dataset_prompts()
valid_prompts = filter_prompts_by_length(prompts, tokenizer, max_tokens=128)
tokenized_prompts = tokenise_prompts(valid_prompts, tokenizer)

# 4. Prepare model for distributed execution
model = accelerator.prepare(model)

# 5. Start energy tracking
tracker = start_energy_tracking(experiment_id)

# 6. Run inference
inference_results = run_gen_inference(
    model, tokenizer, tokenized_prompts, config, accelerator
)

# 7. Stop energy tracking
stop_energy_tracking(tracker)

# 8. Collect metrics
setup_info = gather_experiment_info(config, accelerator, experiment_id)
arch_info = get_model_architecture_info(model)
inference_metrics = combine_inference_metrics(inference_results)
comp_metrics = combine_comp_metrics(model, device, tokenized_prompts, accelerator, config)
energy_metrics = aggregate_energy_metrics(experiment_id, accelerator)

# 9. Save results
if accelerator.is_main_process:
    save_raw_results_json(setup_info, experiment_id, "experiment_setup", accelerator)
    save_raw_results_json(arch_info, experiment_id, "model_architecture", accelerator)
    save_raw_results_json(inference_metrics, experiment_id, "inference_metrics", accelerator)
    save_raw_results_json(comp_metrics, experiment_id, "compute_metrics", accelerator)
    save_raw_results_json(energy_metrics, experiment_id, "global_energy_results", accelerator)

    # Aggregate and save CSV
    all_results = load_all_experiments()
    flatten_results_to_csv(all_results, task_type="text_generation")
```

## Performance Considerations

### Memory Management
- Use appropriate precision (`float16` recommended for inference)
- Enable quantization for large models
- Clear CUDA cache between experiments: `torch.cuda.empty_cache()`

### FLOPs Calculation
- Timeout set to 10s per sample (configurable)
- Batch optimization: computes once for uniform-length batches
- Thread-based execution prevents hangs

### Distributed Synchronization
- Barriers ensure all processes complete before aggregation
- Main process handles file I/O
- Results broadcast to all processes for validation

### Energy Tracking
- CodeCarbon adds minimal overhead (~1-2% performance impact)
- Tracks full experiment duration
- Per-experiment CSV files for detailed analysis

## Known Issues and Limitations

### 1. Quantized Model FLOPs (Critical)
**Issue:** `ptflops` cannot compute FLOPs for quantized models.

**Current Workaround:** Uses hardcoded `cached_flops_for_quantised_models` value (52638582308864).

**Problem:** Same value used for ALL quantized models, regardless of size.

**Solution in Progress:** Compute FLOPs on full-precision model before quantization, or implement custom FLOPs calculation based on model architecture.

### 2. Energy Tracking Granularity
**Issue:** CodeCarbon tracks at process level, not per-batch.

**Impact:** Cannot measure energy per query, only per experiment.

### 3. Variable-Length Batching
**Issue:** FLOPs computation for variable-length batches is slow (per-sample calculation).

**Workaround:** Sort prompts by length to create more uniform batches.

## Future Improvements

1. **Dynamic FLOPs Calculation:**
   - Compute FLOPs before quantization
   - Store per-model FLOPs cache
   - Implement architecture-based FLOPs estimation

2. **Enhanced Energy Tracking:**
   - Per-batch energy measurement
   - Per-layer energy profiling
   - Real-time energy monitoring

3. **Better Quantization Support:**
   - GPTQ support
   - AWQ support
   - Mixed-precision quantization

4. **Improved Metrics:**
   - Token-level latency breakdown
   - Memory timeline tracking
   - Batch efficiency metrics

5. **Modular Refactoring:**
   - Extract metrics into separate classes
   - Plugin system for custom metrics
   - Better error handling and recovery

## Testing

Currently, these utilities lack automated tests. Future work includes:
- Unit tests for each function
- Integration tests for workflows
- Mocking for expensive operations (model loading, inference)
- Validation tests for metric calculations

## Author

Henry Baker
