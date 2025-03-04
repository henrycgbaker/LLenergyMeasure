import os
import json
import sys
import time
import logging
import uuid
from datetime import datetime
from contextlib import redirect_stdout
from io import StringIO

import torch
import torch.nn as nn
import torch.distributed as dist
import psutil

from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator, notebook_launcher
from codecarbon import EmissionsTracker
from fvcore.nn import FlopCountAnalysis

# Optional: if using optimum benchmark branch
from optimum_benchmark import Benchmark, BenchmarkConfig, TorchrunConfig, InferenceConfig, PyTorchConfig
from optimum_benchmark.logging_utils import setup_logging

# -----------------------------------------------------------------------------
# For the FLOP counter:
# -----------------------------------------------------------------------------
class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, input_ids):
        return self.model(input_ids=input_ids)
    
# -----------------------------------------------------------------------------
# Backend model loading
# -----------------------------------------------------------------------------
def load_model_tokenizer_backend(model_name, backend="pytorch"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    )
    model.eval()
    return model, tokenizer

# -----------------------------------------------------------------------------
# Accelerator (Distributed Environment Setup)
# -----------------------------------------------------------------------------
def prep_distributed_env(model, tokenizer, gpu_list=[0, 1]):
    """Prepares model and tokenizer for distributed inference using Accelerate."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_list))
    accelerator = Accelerator(device_placement=True)
    local_rank = accelerator.local_process_index  # Get local process index
    selected_device = gpu_list[local_rank % len(gpu_list)]  # Cycle through provided GPUs
    device = torch.device(f"cuda:{selected_device}")
    accelerator.print(f"Using device: {device} (Local Rank: {local_rank})")
    model, tokenizer = accelerator.prepare(model, tokenizer)
    accelerator.print(f"Using {accelerator.num_processes} GPUs: {gpu_list}")
    print(f"Model is on {next(model.parameters()).device}") 
    return model, tokenizer, accelerator

# -----------------------------------------------------------------------------
# Energy tracking functions 
# -----------------------------------------------------------------------------

def detect_cpu_vendor():
    """Detects the CPU vendor by reading /proc/cpuinfo."""
    try:
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read()
        if "AuthenticAMD" in cpuinfo:
            return "AMD"
        elif "GenuineIntel" in cpuinfo:
            return "Intel"
    except Exception as e:
        print("Error reading /proc/cpuinfo:", e)
    return "Unknown"

def start_energy_tracking():
    """Starts CodeCarbon energy tracking."""
    tracker = EmissionsTracker(measure_power_secs=1, allow_multiple_runs=True)
    tracker.start()
    return tracker

def stop_energy_tracking(tracker):
    """Stops energy tracking and returns the final emissions data."""
    tracker.stop()  
    codecarbon_data = tracker.final_emissions_data
    return codecarbon_data

# -----------------------------------------------------------------------------
# Inference function that measures performance metrics.
# -----------------------------------------------------------------------------
def run_gen_inference_with_metrics(model, tokenizer, accelerator, prompts, 
                                   max_input_tokens, max_output_tokens, batch_size):
    """
    Runs inference and returns performance metrics.
    """
    truncated_prompts = [
        tokenizer.decode(
            tokenizer(p, truncation=True, max_length=max_input_tokens, return_tensors="pt").input_ids[0],
            skip_special_tokens=True
        )
        for p in prompts
    ]
    
    # Sort prompts by token length for efficient batching
    sorted_prompts = sorted(truncated_prompts, key=lambda x: len(tokenizer.tokenize(x)))
    latencies = []
    total_tokens = 0
    total_input_tokens = 0  # Track input tokens
    device = accelerator.device
    num_batches = (len(sorted_prompts) + batch_size - 1) // batch_size

    for i in range(num_batches):
        batch = sorted_prompts[i * batch_size: (i + 1) * batch_size]

        # Tokenize batch
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_input_tokens)
        input_ids = encoded.input_ids.to(device)
        total_input_tokens += input_ids.numel()  # Count input tokens

        # Generate outputs with DistributedDataParallel fix
        start_time = time.perf_counter()
        if hasattr(model, "module"):
            outputs = model.module.generate(input_ids, max_new_tokens=max_output_tokens, do_sample=False)
        else:
            outputs = model.generate(input_ids, max_new_tokens=max_output_tokens, do_sample=False)
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000.0)

        # Count generated tokens per prompt
        for j in range(len(batch)):
            prompt_len = input_ids[j].shape[0]
            gen_len = outputs[j].shape[0] - prompt_len
            total_tokens += gen_len

    avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0
    total_time_sec = sum(latencies) / 1000.0
    throughput_qps = len(sorted_prompts) / total_time_sec if total_time_sec > 0 else 0.0
    tokens_per_sec = total_tokens / total_time_sec if total_time_sec > 0 else 0.0

    return {
        "avg_latency_ms": avg_latency_ms,
        "throughput_qps": throughput_qps,
        "tokens_per_sec": tokens_per_sec,
        "total_generated_tokens": total_tokens,
        "num_runs": len(sorted_prompts),
        "total_time": total_time_sec,
        "total_input_tokens": total_input_tokens  
    }

# -----------------------------------------------------------------------------
# Compute performance metrics collection.
# -----------------------------------------------------------------------------
def get_compute_performance_metrics(model=None, tokenizer=None, device=None, input_length=128):
    """
    Returns a dictionary of low-level compute metrics.
    Attempts to compute FLOPs using fvcore if possible.
    Also reports GPU memory usage, GPU utilization (as a list), and CPU usage.
    """
    compute_metrics = {}
    
    # CPU vendor detection (already in your code)
    cpu_vendor = detect_cpu_vendor()
    compute_metrics["cpu_vendor"] = cpu_vendor

    # Always set a "gpu" key based on the provided device.
    compute_metrics["gpu"] = str(device) if device is not None else "No device provided"

    # GPU memory metrics (if CUDA is available)
    if torch.cuda.is_available() and device is not None:
        torch.cuda.reset_peak_memory_stats(device)
        compute_metrics["current_memory_allocated_bytes"] = torch.cuda.memory_allocated(device)
        compute_metrics["max_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        compute_metrics["current_memory_reserved_bytes"] = torch.cuda.memory_reserved(device)
        compute_metrics["max_memory_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        # GPU utilization using nvidia-smi:
        import subprocess
        try:
            result = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]
            )
            # Split the result into a list of lines, then convert each to a float.
            lines = result.decode("utf-8").strip().splitlines()
            gpu_utils = [float(line.strip()) for line in lines if line.strip()]
            compute_metrics["gpu_utilization_percent"] = gpu_utils
        except Exception as e:
            compute_metrics["gpu_utilization_percent"] = f"Error: {str(e)}"
    # CPU metrics using psutil
    try:
        process = psutil.Process(os.getpid())
        cpu_usage = process.cpu_percent(interval=1.0)
        compute_metrics["cpu_usage_percent"] = cpu_usage
        cpu_mem = process.memory_info().rss
        compute_metrics["cpu_memory_usage_bytes"] = cpu_mem
    except ImportError:
        compute_metrics["cpu_usage_percent"] = "psutil not installed"

    # FLOPs estimation using fvcore 
    if model is not None and tokenizer is not None and device is not None:
        try:
            from fvcore.nn import FlopCountAnalysis
            model_to_trace = model.module if hasattr(model, "module") else model
            model_to_trace.eval()
            wrapped_model = ModelWrapper(model_to_trace)
            dummy_input_ids = torch.ones((1, input_length), dtype=torch.long).to(device)
            flops_analyzer = FlopCountAnalysis(wrapped_model, dummy_input_ids)
            compute_metrics["flops_forward_pass"] = flops_analyzer.total()
        except ImportError:
            compute_metrics["flops_forward_pass"] = "fvcore not installed"
        except Exception as e:
            compute_metrics["flops_forward_pass"] = f"Error: {str(e)}"
    else:
        compute_metrics["flops_forward_pass"] = "model, tokenizer, or device not provided"

    return compute_metrics

# -----------------------------------------------------------------------------
# Experiment extraction and result saving functions.
# -----------------------------------------------------------------------------
def extract_experiment_setup(model_name, codecarbon_data, accelerator, task_type):
    """Extracts experiment configuration and environment details."""
    return {
        "model": model_name,
        "task_type": task_type,
        "date": datetime.now().strftime("%B %d, %Y at %I:%M:%S %p"),
        "cpu_count": codecarbon_data.cpu_count,
        "cpu_model": codecarbon_data.cpu_model,
        "gpu_count": codecarbon_data.gpu_count,
        "gpu_model": codecarbon_data.gpu_model,
        "os": codecarbon_data.os,
        "python_version": sys.version,
        "accelerate_config": {
            "distributed_type": str(accelerator.distributed_type),
            "num_processes": accelerator.num_processes,
            "local_process_index": accelerator.local_process_index
        },
        "country": codecarbon_data.country_name,
        "region": codecarbon_data.region,
    }

def extract_experiment_results(metrics, codecarbon_data, model=None, tokenizer=None, device=None):
    """
    Extracts performance and energy results from the inference metrics and
    CodeCarbon energy data. Also extracts low-level compute performance metrics.
    """
    energy_kwh = codecarbon_data.energy_consumed
    energy_joules = energy_kwh * 3.6e6  # Convert kWh to Joules
    tokens_per_joule = (metrics["total_generated_tokens"] / energy_joules) if energy_joules > 0 else 0

    # Placeholder for task-specific performance
    task_specific_performance = {}

    compute_metrics = get_compute_performance_metrics(model=model, tokenizer=tokenizer, device=device)

    return {
        "inference_performance": {
            "total_inference_time_sec": metrics["total_time"],
            "average_latency_ms_per_batch": metrics["avg_latency_ms"],
            "throughput_queries_per_sec": metrics["throughput_qps"],
            "throughput_tokens_per_sec": metrics["tokens_per_sec"],
        },
        "energy_performance": {
            "cpu_power": codecarbon_data.cpu_power,
            "gpu_power": codecarbon_data.gpu_power,
            "ram_power": codecarbon_data.ram_power,
            "cpu_energy": codecarbon_data.cpu_energy,
            "gpu_energy": codecarbon_data.gpu_energy,
            "ram_energy": codecarbon_data.ram_energy,
            "total_energy_consumed_kwh": energy_kwh,
            "total_energy_consumed_joules": energy_joules,
            "energy_efficiency_tokens_per_joule": tokens_per_joule,
            "final_emissions": codecarbon_data.emissions
        },
        "compute_performance": compute_metrics,
        "task-specific_performance": task_specific_performance
    }

def save_results(task_type, benchmark_results):
    """
    Saves the benchmark results to a JSON file. Appends new experiments if the file exists.
    """
    output_dir = "benchmark_results"
    os.makedirs(output_dir, exist_ok=True)
    output_json_path = os.path.join(output_dir, f"{task_type}_results.json")
    
    if os.path.exists(output_json_path):
        with open(output_json_path, "r") as json_file:
            try:
                existing_data = json.load(json_file)
                if not isinstance(existing_data, list):
                    existing_data = [existing_data]
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []
    
    existing_data.append(benchmark_results)
    with open(output_json_path, "w") as json_file:
        json.dump(existing_data, json_file, indent=4)
    
    return output_json_path

# -----------------------------------------------------------------------------
# Aggregate the metrics (provided earlier)
# -----------------------------------------------------------------------------
def aggregate_experiments(results):
    """
    Aggregates results across multiple processes.
    Assumes `results` is a list of dictionaries, one per process.
    """
    aggregated = {}

    aggregated["experiment_setup"] = results[0]["experiment_setup"].copy()
    if "accelerate_config" in aggregated["experiment_setup"]:
        aggregated["experiment_setup"]["accelerate_config"].pop("local_process_index", None)
    aggregated["experiment_variables"] = results[0]["experiment_variables"]

    # Aggregate inference performance by averaging
    inf_keys = ["total_inference_time_sec", "average_latency_ms_per_batch", "throughput_queries_per_sec", "throughput_tokens_per_sec"]
    agg_inference = {key: sum(proc["experiment_results"]["inference_performance"][key] for proc in results) / len(results)
                     for key in inf_keys}

    # Aggregate energy performance (averaging some keys, summing others)
    energy_keys_avg = ["cpu_power", "gpu_power", "ram_power", "energy_efficiency_tokens_per_joule"]
    energy_keys_sum = ["cpu_energy", "gpu_energy", "ram_energy", "total_energy_consumed_kwh", "total_energy_consumed_joules", "final_emissions"]
    agg_energy = {}
    for key in energy_keys_avg:
        agg_energy[key] = sum(proc["experiment_results"]["energy_performance"][key] for proc in results) / len(results)
    for key in energy_keys_sum:
        agg_energy[key] = sum(proc["experiment_results"]["energy_performance"][key] for proc in results)

    # Aggregate compute performance.
    compute_setup = results[0]["experiment_results"]["compute_performance"]
    agg_compute = {
        "gpu": compute_setup.get("gpu", "N/A"),
        "flops_forward_pass": compute_setup.get("flops_forward_pass", "N/A"),
    }
    # Define numeric keys to average; note that gpu_utilization_percent is now a list.
    numeric_compute_keys = [
        "cpu_usage_percent",
        "cpu_memory_usage_bytes",
        "gpu_utilization_percent",
        "current_memory_allocated_bytes",
        "max_memory_allocated_bytes",
        "current_memory_reserved_bytes",
        "max_memory_reserved_bytes"
    ]
    for key in numeric_compute_keys:
        values = []
        for proc in results:
            val = proc["experiment_results"]["compute_performance"].get(key)
            if isinstance(val, list):
                values.append(val)
            elif isinstance(val, (int, float)):
                values.append(val)
        if values:
            # If the first value is a list, assume all are lists of the same length and average element-wise.
            if isinstance(values[0], list):
                list_length = len(values[0])
                averaged = [sum(v[i] for v in values) / len(values) for i in range(list_length)]
                agg_compute[key] = averaged
            else:
                agg_compute[key] = sum(values) / len(values)
    
    for extra_key in ["cpu_vendor", "AMD_CONSTANT_POWER"]:
        if extra_key in compute_setup:
            agg_compute[extra_key] = compute_setup[extra_key]
    
    task_perf = results[0]["experiment_results"].get("task-specific_performance", {})

    aggregated["experiment_results"] = {
        "inference_performance": agg_inference,
        "energy_performance": agg_energy,
        "compute_performance": agg_compute,
        "task-specific_performance": task_perf
    }
    return aggregated

# -----------------------------------------------------------------------------
# Persistent Experiment ID
# -----------------------------------------------------------------------------
ID_FILE = "experiment_id.txt"
def get_persistent_unique_id():
    if os.path.exists(ID_FILE):
        with open(ID_FILE, "r") as f:
            last_id = int(f.read().strip())
    else:
        last_id = 0
    new_id = last_id + 1
    with open(ID_FILE, "w") as f:
        f.write(str(new_id))
    return f"{new_id:04d}"

# -----------------------------------------------------------------------------
# Experiment runner with aggregation integration.
# -----------------------------------------------------------------------------
def run_experiment(model_name, prompts, inference_fn, task_type, 
                   backend="pytorch", use_optimum=False, **inference_kwargs):
    """
    Runs an experiment in one of two modes.
    
    For the standard mode (use_optimum=False):
      1. Loads the model.
      2. Prepares the distributed environment.
      3. Starts energy tracking, runs inference, and stops tracking.
      4. Extracts per-process experiment results.
      5. Gathers & aggregates per-process results across all processes.
      6. Saves the aggregated benchmark results.
    """
    if use_optimum:
        # --- Optimum benchmark branch ---
        setup_logging(level="INFO")
        launcher_config = TorchrunConfig(nproc_per_node=1)
        scenario_config = InferenceConfig(latency=True, memory=True, input_shapes={"sequence_length": 128})
        backend_config = PyTorchConfig(model=model_name, device="cuda", device_ids="0", no_weights=True)
        benchmark_config = BenchmarkConfig(
            name=f"{backend}_{model_name}",
            scenario=scenario_config,
            launcher=launcher_config,
            backend=backend_config,
        )
        benchmark_report = Benchmark.launch(benchmark_config)
        benchmark_results = benchmark_report.to_dict()
        print(json.dumps({
            "model": model_name,
            "optimum_benchmark_results": benchmark_results
        }, indent=4))
        return benchmark_results
    else:
        # --- Standard experiment branch ---
        model, tokenizer = load_model_tokenizer_backend(model_name, backend=backend)
        model, tokenizer, accelerator = prep_distributed_env(model, tokenizer)
        tracker = start_energy_tracking()
        
        # Run inference (this runs on each process)
        inference_metrics = inference_fn(model, tokenizer, accelerator, prompts, **inference_kwargs)
        codecarbon_data = stop_energy_tracking(tracker)
        experiment_results = extract_experiment_results(inference_metrics, codecarbon_data, model=model, tokenizer=tokenizer, device=accelerator.device)
        
        # Extract common experimental setup and variables
        experiment_setup = extract_experiment_setup(model_name, codecarbon_data, accelerator, task_type)
        experiment_variables = {
            "total_token_inputted": inference_metrics["total_input_tokens"],
            "total_tokens_outputted": inference_metrics["total_generated_tokens"],
            "number_runs": inference_metrics["num_runs"]
        }
        
        # Build the local result dictionary for this process
        local_result = {
            "experiment_setup": experiment_setup,
            "experiment_variables": experiment_variables,
            "experiment_results": experiment_results
        }
        
        # Gather results from all processes if running distributed
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            all_results = [None] * world_size
            # Gather local_result from each process into all_results
            dist.all_gather_object(all_results, local_result)
        else:
            all_results = [local_result]
        
        # Only the main process (local rank 0) aggregates and saves the results.
        if accelerator.local_process_index == 0:
            aggregated_result = aggregate_experiments(all_results)
            unique_id = get_persistent_unique_id()
            current_time = datetime.now().strftime("%B %d, %Y at %I:%M:%S %p")
            experiment_title = f"EXPERIMENT #{unique_id}"
            
            benchmark_results = {
                experiment_title: aggregated_result
            }
            output_json_path = save_results(task_type, benchmark_results)
            print(f"Aggregated benchmark results saved to {output_json_path}")
            return benchmark_results
        else:
            return None